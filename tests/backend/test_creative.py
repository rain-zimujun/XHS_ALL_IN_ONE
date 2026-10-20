import json
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from urllib3.exceptions import ReadTimeoutError

from backend.app.api.platforms.xhs.pc import get_xhs_pc_api_adapter_factory
from backend.app.api import files as files_api
from backend.app.core.database import Base, get_db
from backend.app.core.security import create_access_token, encrypt_text
from backend.app.main import app
from backend.app.models import AccountCookieVersion, ModelConfig, Note, PlatformAccount, PublishJob, Task, User
from backend.app.schemas.creative import CreativeSearchRequest
from backend.app.services import creative_report, creative_runner, creative_service
from backend.app.services.creative_analysis import get_creative_text_client_factory
from backend.app.services.creative_runner import recover_interrupted_runs
from backend.app.services.ai_service import OpenAICompatibleTextClient


def valid_analysis(count=1, ids=None):
    return {
        "summary": "样本主要关注 AI 服装流程，不能代表全站趋势。", "patterns": ["步骤拆解", "效果对比"],
        "ideas": [{
            "title": f"AI 服装设计流程 {i + 1}", "alternative_titles": ["从草图到服装", "AI 试衣流程", "设计师的 AI 工具箱"],
            "body": "先准备服装草图，再比较不同方案。具体效果需自行验证。", "tags": ["AI服装", "服装设计"],
            "reason": "多篇样本都关注可重复的设计步骤。", "source_note_ids": ids or ["note-0", "note-1"],
            "cover_text": "AI 服装设计流程", "image_plan": [{
                "purpose": "封面对比", "composition": "左侧草图，右侧服装，3:4 竖图，顶部留白。",
                "prompt": "Editorial fashion concept sheet, sketch and garment side by side, neutral background, vertical 3:4, no text",
                "negative_prompt": "watermark, logo, text, distorted anatomy",
            }],
        } for i in range(count)],
    }


class FakeTextClient:
    calls = []

    def _complete(self, **kwargs):
        self.calls.append(kwargs)
        prompt = json.loads(kwargs["user_prompt"].split("\n上次输出")[0])
        return json.dumps(valid_analysis(prompt["required_idea_count"]), ensure_ascii=False)


class FakeAdapter:
    calls = []

    def __init__(self, cookies):
        pass

    def search_note(self, keyword, **options):
        self.calls.append((keyword, options))
        page = options["page"]
        # Adjacent pages share a result; the service must deduplicate it.
        first = (page - 1) * 19
        items = [{"id": f"note-{i}", "note_card": {
            "title": f"AI 服装案例 {i}", "desc": "搜索摘要",
            "interact_info": {"liked_count": str(i)},
        }} for i in range(first, first + 20)]
        return True, "", {"data": {"items": items, "has_more": page < 4}}

    def get_note_info(self, url):
        note_id = url.rsplit("/", 1)[-1]
        return True, "", {"data": {"items": [{"id": note_id, "note_card": {
            "title": f"AI 服装案例 {note_id}", "desc": "这是一篇完整正文，介绍服装设计流程与 AI 试衣效果。",
            "time": 1792306914000,
            "interact_info": {"liked_count": 120, "collected_count": 60},
        }}]}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'creative.db'}", connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with factory() as db:
        user = User(username="creative-user", password_hash="unused")
        other = User(username="other-user", password_hash="unused")
        db.add_all([user, other]); db.flush()
        account = PlatformAccount(user_id=user.id, platform="xhs", sub_type="pc", nickname="测试账号", external_user_id="test")
        db.add(account); db.flush()
        db.add(AccountCookieVersion(platform_account_id=account.id, encrypted_cookies=encrypt_text("test=fixture")))
        model = ModelConfig(user_id=user.id, name="测试文字模型", model_type="text", provider="mock", model_name="test-model",
                            base_url="https://model.example.invalid/v1", encrypted_api_key=encrypt_text("fixture-key-private"))
        db.add(model); db.flush()
        db.commit()
        user_id, other_id, account_id, model_id = user.id, other.id, account.id, model.id

    def override_db():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_xhs_pc_api_adapter_factory] = lambda: FakeAdapter
    app.dependency_overrides[get_creative_text_client_factory] = lambda: FakeTextClient
    monkeypatch.setattr(creative_service.time, "sleep", lambda _: None)
    report_settings = SimpleNamespace(storage_dir=tmp_path / "storage")
    monkeypatch.setattr(creative_report, "get_settings", lambda: report_settings)
    monkeypatch.setattr(files_api, "get_settings", lambda: report_settings)
    FakeAdapter.calls = []
    FakeTextClient.calls = []
    yield SimpleNamespace(
        client=TestClient(app), factory=factory, account_id=account_id, user_id=user_id, model_id=model_id,
        headers={"Authorization": f"Bearer {create_access_token(user_id)}"},
        other_headers={"Authorization": f"Bearer {create_access_token(other_id)}"},
    )
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_xhs_pc_api_adapter_factory, None)
    app.dependency_overrides.pop(get_creative_text_client_factory, None)
    engine.dispose()


@pytest.mark.parametrize("limit", [20, 30, 50])
def test_search_paginates_deduplicates_and_preserves_filters(env, limit):
    response = env.client.post("/api/xhs/creative/search", headers=env.headers, json={
        "account_id": env.account_id, "keyword": "AI 服装", "limit": limit,
        "sort_type_choice": 4, "note_time": 1, "note_type": 2,
    })
    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "completed"
    notes = run["payload"]["items"]
    assert len(notes) == len({n["note_id"] for n in notes}) == limit
    assert all(n["raw"] == {} for n in notes)
    assert all(options["sort_type_choice"] == 4 and options["note_time"] == 1 for _, options in FakeAdapter.calls)
    assert len({options["search_id"] for _, options in FakeAdapter.calls}) == 1
    assert env.client.get(f"/api/xhs/creative/runs/{run['id']}", headers=env.other_headers).status_code == 404
    assert env.client.get("/api/xhs/creative/runs", headers=env.other_headers).json()["items"] == []


def test_insufficient_search_results_are_reported(monkeypatch):
    class ShortAdapter:
        def search_note(self, *_args, **_kwargs):
            return True, "", {"data": {"items": [{"id": "only-one", "title": "案例"}], "has_more": False}}

    notes, warnings = creative_service.search_samples(ShortAdapter(), CreativeSearchRequest(account_id=1, keyword="AI", limit=30))
    assert len(notes) == 1
    assert "实际获得 1 篇" in warnings[0]


def test_search_rejects_foreign_account_and_invalid_input(env):
    body = {"account_id": env.account_id, "keyword": "AI", "limit": 20}
    assert env.client.post("/api/xhs/creative/search", headers=env.other_headers, json=body).status_code == 404
    assert env.client.post("/api/xhs/creative/search", headers=env.headers, json={**body, "keyword": "   "}).status_code == 422
    assert env.client.post("/api/xhs/creative/search", headers=env.headers, json={**body, "limit": 500}).status_code == 422
    assert FakeAdapter.calls == []


def test_failed_search_is_saved_without_sensitive_error(env, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("cookie=do-not-expose")
    monkeypatch.setattr(FakeAdapter, "search_note", fail)
    result = env.client.post("/api/xhs/creative/search", headers=env.headers, json={"account_id": env.account_id, "keyword": "AI"})
    assert result.status_code == 502
    assert "do-not-expose" not in result.text
    with env.factory() as db:
        task = db.scalar(select(Task))
        assert task.status == "failed"
        assert task.finished_at is not None


def search_for_collection(env):
    return env.client.post("/api/xhs/creative/search", headers=env.headers, json={
        "account_id": env.account_id, "keyword": "AI 服装", "limit": 20,
    }).json()


def test_collection_persists_only_selected_full_text_and_never_publishes(env):
    search = search_for_collection(env)
    response = env.client.post("/api/xhs/creative/collect", headers=env.headers, json={
        "search_task_id": search["id"], "note_ids": ["note-0", "note-1", "note-2"],
    })
    assert response.status_code == 202
    result = env.client.get(f"/api/xhs/creative/runs/{response.json()['id']}", headers=env.headers).json()
    assert result["status"] == "completed"
    assert len(result["payload"]["sources"]) == 3
    with env.factory() as db:
        assert len(db.scalars(select(Note)).all()) == 3
        assert not db.scalars(select(PublishJob)).all()


def test_collection_retains_successes_when_one_note_fails(env, monkeypatch):
    original = FakeAdapter.get_note_info
    def sometimes_fail(self, url):
        if url.endswith("note-1"):
            raise TimeoutError("internal details")
        return original(self, url)
    monkeypatch.setattr(FakeAdapter, "get_note_info", sometimes_fail)
    search = search_for_collection(env)
    run = env.client.post("/api/xhs/creative/collect", headers=env.headers, json={
        "search_task_id": search["id"], "note_ids": [f"note-{i}" for i in range(4)],
    }).json()
    result = env.client.get(f"/api/xhs/creative/runs/{run['id']}", headers=env.headers).json()
    assert result["status"] == "completed"
    assert len(result["payload"]["sources"]) == 3
    assert len(result["payload"]["failures"]) == 1
    assert "internal details" not in str(result)


def test_collection_cancellation_is_observed_after_network_call(env, monkeypatch):
    original = FakeAdapter.get_note_info
    def cancel_during_request(self, url):
        with env.factory() as db:
            task = db.scalar(select(Task).where(Task.task_type == "creative_collect"))
            task.status = "cancelled"
            db.commit()
        return original(self, url)
    monkeypatch.setattr(FakeAdapter, "get_note_info", cancel_during_request)
    search = search_for_collection(env)
    run = env.client.post("/api/xhs/creative/collect", headers=env.headers, json={
        "search_task_id": search["id"], "note_ids": [f"note-{i}" for i in range(3)],
    }).json()
    result = env.client.get(f"/api/xhs/creative/runs/{run['id']}", headers=env.headers).json()
    assert result["status"] == "cancelled"
    with env.factory() as db:
        assert not db.scalars(select(Note)).all()


def test_collection_rejects_tampered_selection(env):
    search = search_for_collection(env)
    response = env.client.post("/api/xhs/creative/collect", headers=env.headers, json={
        "search_task_id": search["id"], "note_ids": ["note-0", "note-1", "foreign-id"],
    })
    assert response.status_code == 400
    response = env.client.post("/api/xhs/creative/collect", headers=env.other_headers, json={
        "search_task_id": search["id"], "note_ids": ["note-0", "note-1", "note-2"],
    })
    assert response.status_code == 404


def collect_for_analysis(env, options=None):
    search = search_for_collection(env)
    response = env.client.post("/api/xhs/creative/collect", headers=env.headers, json={
        "search_task_id": search["id"], "note_ids": ["note-0", "note-1", "note-2"], "analysis": options,
    })
    assert response.status_code == 202
    return env.client.get(f"/api/xhs/creative/runs/{response.json()['id']}", headers=env.headers).json()


@pytest.mark.parametrize("count", [1, 3, 5])
def test_one_click_collects_and_generates_validated_ideas(env, count):
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": count})
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert len(result["payload"]["result"]["ideas"]) == count
    assert result["payload"]["analysis_metadata"]["sample_count"] == 3
    assert "fixture-key-private" not in str(result)
    assert len(FakeTextClient.calls) == 1
    assert "不可信" in FakeTextClient.calls[0]["system_prompt"]
    assert "base_url" not in FakeTextClient.calls[0]["user_prompt"]
    with env.factory() as db:
        assert not db.scalars(select(PublishJob)).all()


def test_analysis_can_retry_existing_sources_without_another_crawl(env):
    collection = collect_for_analysis(env)
    before = len(FakeAdapter.calls)
    response = env.client.post("/api/xhs/creative/analyze", headers=env.headers, json={
        "collection_task_id": collection["id"], "model_id": env.model_id, "idea_count": 1,
    })
    result = env.client.get(f"/api/xhs/creative/runs/{response.json()['id']}", headers=env.headers).json()
    assert result["status"] == "completed"
    assert result["parent_task_id"] == collection["id"]
    assert len(FakeAdapter.calls) == before
    assert env.client.post("/api/xhs/creative/analyze", headers=env.other_headers, json={
        "collection_task_id": collection["id"], "model_id": env.model_id,
    }).status_code == 404


@pytest.mark.parametrize("malformed", ["not json", json.dumps(valid_analysis(ids=["note-0", "foreign-id"])), json.dumps(valid_analysis(ids=["note-0", "note-0"]))])
def test_invalid_ai_output_is_retried_once_and_never_accepted(env, monkeypatch, malformed):
    calls = []
    def bad_response(self, **kwargs):
        calls.append(kwargs)
        return malformed
    monkeypatch.setattr(FakeTextClient, "_complete", bad_response)
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    assert result["status"] == "failed"
    assert len(calls) == 2
    assert len(result["payload"]["sources"]) == 3
    assert "result" not in result["payload"]


def test_ai_invalid_output_can_be_repaired_on_second_attempt(env, monkeypatch):
    responses = iter(["not json", "```json\n" + json.dumps(valid_analysis()) + "\n```"])
    monkeypatch.setattr(FakeTextClient, "_complete", lambda *_a, **_kw: next(responses))
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    assert result["status"] == "completed"


def test_ai_provider_errors_are_masked_and_sources_retained(env, monkeypatch):
    def fail(*_a, **_kw):
        raise ValueError("Authorization: Bearer fixture-key-private")
    monkeypatch.setattr(FakeTextClient, "_complete", fail)
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    assert result["status"] == "failed"
    assert len(result["payload"]["sources"]) == 3
    assert "fixture-key-private" not in str(result)


def test_model_ownership_is_checked_before_collection(env):
    with env.factory() as db:
        model = db.get(ModelConfig, env.model_id)
        other = db.scalar(select(User).where(User.username == "other-user"))
        model.user_id = other.id
        db.commit()
    search = search_for_collection(env)
    response = env.client.post("/api/xhs/creative/collect", headers=env.headers, json={
        "search_task_id": search["id"], "note_ids": ["note-0", "note-1", "note-2"],
        "analysis": {"model_id": env.model_id},
    })
    assert response.status_code == 400
    assert not FakeTextClient.calls
    with env.factory() as db:
        assert not db.scalars(select(Note)).all()


def test_cancelling_during_ai_request_prevents_saving_generated_result(env, monkeypatch):
    def cancel_during_model(*_a, **_kw):
        with env.factory() as db:
            task = db.scalar(select(Task).where(Task.task_type == "creative_collect"))
            task.status = "cancelled"
            db.commit()
        return json.dumps(valid_analysis())
    monkeypatch.setattr(FakeTextClient, "_complete", cancel_during_model)
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    assert result["status"] == "cancelled"
    assert "result" not in result["payload"]
    assert len(result["payload"]["sources"]) == 3


def test_startup_marks_interrupted_creative_runs_without_touching_other_tasks(env):
    with env.factory() as db:
        creative = Task(user_id=env.user_id, platform="xhs", task_type="creative_analysis", status="running", payload={"sources": [{"note_id": "saved"}]})
        unrelated = Task(user_id=env.user_id, platform="xhs", task_type="other", status="pending", payload={})
        db.add_all([creative, unrelated]); db.commit()
        creative_id, unrelated_id = creative.id, unrelated.id
    recover_interrupted_runs(env.factory)
    with env.factory() as db:
        assert db.get(Task, creative_id).status == "failed"
        assert db.get(Task, creative_id).payload["sources"] == [{"note_id": "saved"}]
        assert db.get(Task, unrelated_id).status == "pending"


def test_reports_are_generated_with_full_sources_and_owner_only_downloads(env):
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    files = result["payload"]["report_files"]
    assert len(files) == 2
    for file in files:
        response = env.client.get(file["download_url"], headers=env.headers)
        assert response.status_code == 200
        assert "fixture-key-private" not in response.text
        assert env.client.get(file["download_url"], headers=env.other_headers).status_code == 404
        assert env.client.get(file["download_url"]).status_code == 401
        if file["file_name"].endswith(".json"):
            document = response.json()
            assert document["result"] == result["payload"]["result"]
            assert len(document["sources"]) == 3
            assert "完整正文" in document["sources"][0]["content"]
            assert "raw" not in document["sources"][0]
        else:
            assert response.headers["content-type"].startswith("text/html")
            assert "生图提示词" in response.text


def test_html_report_escapes_untrusted_ai_and_source_text(env, monkeypatch):
    malicious = valid_analysis()
    malicious["ideas"][0]["body"] = '<script>alert("unsafe")</script><img src=x onerror=alert(1)>'
    monkeypatch.setattr(FakeTextClient, "_complete", lambda *_a, **_kw: json.dumps(malicious))
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    report = next(f for f in result["payload"]["report_files"] if f["file_name"].endswith(".html"))
    text = env.client.get(report["download_url"], headers=env.headers).text
    assert "<script>" not in text
    assert "<img src=x" not in text
    assert "&lt;script&gt;" in text
    assert "Content-Security-Policy" in text
    assert "javascript:" not in creative_report.source_link({"note_id": "id", "title": '<img src=x>', "note_url": 'javascript:alert(1)'})


def test_export_failure_does_not_lose_analysis_and_can_be_retried(env, monkeypatch):
    def disk_fail(_task):
        raise PermissionError("private-directory-detail")
    monkeypatch.setattr(creative_runner, "export_report", disk_fail)
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    assert result["status"] == "completed"
    assert result["payload"]["result"]
    assert result["payload"]["export_error"]
    assert "private-directory-detail" not in str(result)
    url = f"/api/xhs/creative/runs/{result['id']}/export"
    assert env.client.post(url, headers=env.other_headers).status_code == 404
    retry = env.client.post(url, headers=env.headers)
    assert retry.status_code == 200
    assert len(retry.json()["payload"]["report_files"]) == 2
    assert retry.json()["payload"]["export_error"] is None


def test_export_rejects_runs_without_analysis(env):
    result = collect_for_analysis(env)
    assert env.client.post(f"/api/xhs/creative/runs/{result['id']}/export", headers=env.headers).status_code == 400


def test_later_page_exception_preserves_first_page(env, monkeypatch):
    original = FakeAdapter.search_note
    def page_failure(self, keyword, **options):
        if options["page"] > 1:
            raise TimeoutError("private transport detail")
        return original(self, keyword, **options)
    monkeypatch.setattr(FakeAdapter, "search_note", page_failure)
    response = env.client.post("/api/xhs/creative/search", headers=env.headers, json={
        "account_id": env.account_id, "keyword": "AI", "limit": 50,
    })
    assert response.status_code == 200
    result = response.json()
    assert len(result["payload"]["items"]) == 20
    assert any("实际获得 20 篇" in w for w in result["payload"]["warnings"])
    assert "private transport detail" not in str(result)


def test_generic_task_retry_cannot_leave_creative_task_pending_forever(env):
    with env.factory() as db:
        task = Task(user_id=env.user_id, platform="xhs", task_type="creative_collect", status="failed", payload={})
        db.add(task); db.commit()
        task_id = task.id
    assert env.client.post(f"/api/tasks/{task_id}/retry", headers=env.headers).status_code == 400
    with env.factory() as db:
        assert db.get(Task, task_id).status == "failed"


def test_login_challenge_stops_collection_without_calling_model(env, monkeypatch):
    calls = []
    def challenge(_self, url):
        calls.append(url)
        return False, "请登录完成验证码验证", {}
    monkeypatch.setattr(FakeAdapter, "get_note_info", challenge)
    result = collect_for_analysis(env, {"model_id": env.model_id})
    assert result["status"] == "failed"
    assert len(calls) == 1
    assert "重新验证" in result["payload"]["error"]
    assert not FakeTextClient.calls


def model_response(content=None, status=200):
    response = requests.Response()
    response.status_code = status
    response.encoding = "utf-8"
    response._content = json.dumps({"choices": [{"message": {"content": content or json.dumps(valid_analysis())}}]}).encode("utf-8")
    return response


def test_creative_request_uses_three_minute_read_timeout_and_allows_slow_provider(env, monkeypatch):
    calls = []
    def slow_provider(_url, **kwargs):
        calls.append(kwargs)
        timeout = kwargs["timeout"]
        read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
        # Simulate a 90-second provider without waiting or making a billable call.
        if read_timeout < 90:
            raise requests.ReadTimeout("fixture-key-private")
        return model_response()
    monkeypatch.setattr("backend.app.services.ai_service.requests.post", slow_provider)
    app.dependency_overrides[get_creative_text_client_factory] = lambda: OpenAICompatibleTextClient
    result = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1})
    assert result["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["timeout"] == (10, 180)
    assert len(result["payload"]["result"]["ideas"]) == 1


def test_other_text_workflows_keep_their_existing_timeout(monkeypatch):
    calls = []
    def post(_url, **kwargs):
        calls.append(kwargs)
        return model_response("普通文本")
    monkeypatch.setattr("backend.app.services.ai_service.requests.post", post)
    result = OpenAICompatibleTextClient()._complete(
        model_config=SimpleNamespace(base_url="https://example.invalid/v1", model_name="mock"),
        api_key="fixture", system_prompt="system", user_prompt="user",
    )
    assert result == "普通文本"
    assert calls[0]["timeout"] == 60


@pytest.mark.parametrize("separate_analysis", [False, True])
@pytest.mark.parametrize("kind,expected_type,expected_message", [
    ("read", "ai_read_timeout", "3 分钟 / 180 秒"),
    ("wrapped_read", "ai_read_timeout", "3 分钟 / 180 秒"),
    ("connect", "ai_connect_timeout", "10 秒"),
    ("network", "ai_network_error", "连接中断"),
    ("http", "ai_http_error", "HTTP 429"),
    ("html", "ai_response_invalid", "HTTP 200 不代表有效生成结果"),
])
def test_transport_failures_are_safe_specific_and_not_retried(env, monkeypatch, separate_analysis, kind, expected_type, expected_message):
    calls = []
    def bad_provider(_url, **kwargs):
        calls.append(kwargs)
        if kind == "read":
            raise requests.ReadTimeout("fixture-key-private")
        if kind == "wrapped_read":
            raise requests.ConnectionError(ReadTimeoutError(None, "/private", "fixture-key-private"))
        if kind == "connect":
            raise requests.ConnectTimeout("fixture-key-private")
        if kind == "network":
            raise requests.ConnectionError("fixture-key-private")
        response = model_response(status=429 if kind == "http" else 200)
        response._content = b"<!doctype html><html>fixture-key-private</html>"
        return response
    monkeypatch.setattr("backend.app.services.ai_service.requests.post", bad_provider)
    app.dependency_overrides[get_creative_text_client_factory] = lambda: OpenAICompatibleTextClient
    options = {"model_id": env.model_id, "idea_count": 1}
    if separate_analysis:
        collection = collect_for_analysis(env)
        pending = env.client.post("/api/xhs/creative/analyze", headers=env.headers, json={
            "collection_task_id": collection["id"], **options,
        }).json()
        result = env.client.get(f"/api/xhs/creative/runs/{pending['id']}", headers=env.headers).json()
    else:
        result = collect_for_analysis(env, options)
    assert result["status"] == "failed"
    assert result["error_type"] == expected_type
    assert result["payload"]["stage"] == "任务失败"
    assert expected_message in result["payload"]["error"]
    assert "fixture-key-private" not in str(result)
    assert len(result["payload"]["sources"]) == 3
    assert len(calls) == 1


def test_search_history_backfills_existing_detail_times_without_rewriting_snapshot(env):
    collection = collect_for_analysis(env)
    parent_id = collection["parent_task_id"]
    with env.factory() as db:
        snapshot = db.get(Task, parent_id)
        assert all(n["timestamp"] == 0 for n in snapshot.payload["items"])
    result = env.client.get(f"/api/xhs/creative/runs/{parent_id}", headers=env.headers).json()
    notes = result["payload"]["items"]
    assert all(n["timestamp"] == 1792306914000 for n in notes[:3])
    assert all(n["timestamp"] == 0 for n in notes[3:])
    assert [n["note_id"] for n in notes] == [f"note-{i}" for i in range(20)]
    # Enrichment also applies to a new search of already-collected notes.
    assert search_for_collection(env)["payload"]["items"][0]["timestamp"] == 1792306914000
    with env.factory() as db:
        assert all(n["timestamp"] == 0 for n in db.get(Task, parent_id).payload["items"])


def test_time_backfill_never_uses_another_users_notes_or_overwrites_search_time(env):
    search = search_for_collection(env)
    with env.factory() as db:
        other = db.scalar(select(User).where(User.username == "other-user"))
        other_account = PlatformAccount(user_id=other.id, platform="xhs", sub_type="pc", nickname="他人账号", external_user_id="other")
        db.add(other_account); db.flush()
        db.add(Note(user_id=other.id, platform_account_id=other_account.id, platform="xhs", note_id="note-0", raw_json={"timestamp": 1792306914000}))
        db.add(Note(user_id=env.user_id, platform_account_id=env.account_id, platform="xhs", note_id="note-1", raw_json={"timestamp": 1792306914000}))
        task = db.get(Task, search["id"])
        items = [dict(n) for n in task.payload["items"]]
        items[1]["timestamp"] = 1792200000000
        task.payload = {**task.payload, "items": items}
        db.commit()
    result = env.client.get(f"/api/xhs/creative/runs/{search['id']}", headers=env.headers).json()
    assert result["payload"]["items"][0]["timestamp"] == 0
    assert result["payload"]["items"][1]["timestamp"] == 1792200000000
