import pytest

from backend.app.api.ai import get_image_ai_client
from backend.app.core.security import encrypt_text
from backend.app.main import app
from backend.app.models import AiDraft, AiGeneratedAsset, DraftAsset, ModelConfig, Task
from test_creative import env  # isolated database + client fixture

pytestmark = pytest.mark.usefixtures("env")


def _task_with_plans(env):
    result = {"ideas": [{
        "idea_id": "idea_1", "title": "标题一", "variants": [
            {"variant_id": "copy_0_0", "label": "v1", "body": "正文一", "tags": [],
             "image_plan": [{"purpose": "封面", "prompt": "prompt-a"}, {"purpose": "细节", "prompt": "prompt-b"}]},
            {"variant_id": "copy_0_1", "label": "v2", "body": "正文二", "tags": [],
             "image_plan": [{"purpose": "封面", "prompt": "prompt-c"}]},
        ],
    }]}
    with env.factory() as db:
        task = Task(user_id=env.user_id, platform="xhs", task_type="creative_analysis",
                    status="completed", progress=100, payload={"result": result})
        db.add(task)
        db.commit()
        return task.id


def _image_model(env):
    with env.factory() as db:
        config = ModelConfig(user_id=env.user_id, name="图片模型", model_type="image", provider="openai",
                             model_name="img-1", encrypted_api_key=encrypt_text("key"), is_default=True)
        db.add(config)
        db.commit()
        return config.id


class FakeImageClient:
    def __init__(self, failing=()):
        self.calls = []
        self.failing = set(failing)

    def generate_image(self, *, model_config, api_key, prompt, reference_images=None):
        self.calls.append(prompt)
        if prompt in self.failing:
            raise ValueError("image backend error")
        return {"url": f"https://img.example/{prompt}.png", "raw": {"ok": True}}


def _override_client(client):
    app.dependency_overrides[get_image_ai_client] = lambda: client


def test_image_generation_is_optional_without_model(env):
    task_id = _task_with_plans(env)
    response = env.client.post(f"/api/xhs/creative/runs/{task_id}/generate-images",
                               headers=env.headers, json={"scope": "cover"})
    assert response.status_code == 400
    assert "手动上传图片" in response.json()["detail"]
    status = env.client.get("/api/ai/images/status", headers=env.headers).json()
    assert status["ready"] is False and status["model_name"] == ""


def test_cover_scope_generates_one_image_per_variant_and_saves_assets(env):
    task_id = _task_with_plans(env)
    model_id = _image_model(env)
    client = FakeImageClient()
    _override_client(client)
    try:
        status = env.client.get("/api/ai/images/status", headers=env.headers).json()
        assert status["ready"] is True and status["model_name"] == "img-1"
        response = env.client.post(f"/api/xhs/creative/runs/{task_id}/generate-images",
                                   headers=env.headers, json={"scope": "cover"})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["planned"] == 2
        assert [item["variant_id"] for item in data["created"]] == ["copy_0_0", "copy_0_1"]
        assert client.calls == ["prompt-a", "prompt-c"]
        with env.factory() as db:
            assets = db.query(AiGeneratedAsset).all()
            assert len(assets) == 2 and assets[0].model_name == "img-1"
            assert {asset.prompt for asset in assets} == {"prompt-a", "prompt-c"}
    finally:
        app.dependency_overrides.pop(get_image_ai_client, None)


def test_generated_image_attaches_to_matching_draft(env):
    task_id = _task_with_plans(env)
    _image_model(env)
    with env.factory() as db:
        draft = AiDraft(user_id=env.user_id, platform="xhs", title="标题一", body="正文一",
                        source_type="creative", source_task_id=task_id, source_variant_id="copy_0_0")
        db.add(draft)
        db.commit()
        draft_id = draft.id
    _override_client(FakeImageClient())
    try:
        data = env.client.post(f"/api/xhs/creative/runs/{task_id}/generate-images",
                               headers=env.headers, json={"variant_ids": ["copy_0_0"], "scope": "cover"}).json()
        assert data["created"][0]["draft_id"] == draft_id
        assert data["created"][0]["draft_asset_id"] is not None
        with env.factory() as db:
            asset = db.query(DraftAsset).filter(DraftAsset.draft_id == draft_id).one()
            assert asset.asset_type == "image" and asset.source == "generated"
            assert asset.prompt == "prompt-a"
    finally:
        app.dependency_overrides.pop(get_image_ai_client, None)


def test_partial_failures_do_not_abort_the_batch(env):
    task_id = _task_with_plans(env)
    _image_model(env)
    _override_client(FakeImageClient(failing={"prompt-a"}))
    try:
        data = env.client.post(f"/api/xhs/creative/runs/{task_id}/generate-images",
                               headers=env.headers, json={"scope": "cover"}).json()
        assert len(data["created"]) == 1 and data["created"][0]["variant_id"] == "copy_0_1"
        assert data["failed"][0]["variant_id"] == "copy_0_0"
        assert "image backend error" in data["failed"][0]["error"]
    finally:
        app.dependency_overrides.pop(get_image_ai_client, None)
