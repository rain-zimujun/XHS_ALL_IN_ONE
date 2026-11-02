import pytest

from backend.app.models import AiDraft, DraftAsset, PlatformAccount
from test_creative import env  # reuse the isolated database + client fixture

pytestmark = pytest.mark.usefixtures("env")


def _create_draft(env, **kwargs):
    with env.factory() as db:
        draft = AiDraft(user_id=env.user_id, platform="xhs", **kwargs)
        db.add(draft)
        db.commit()
        return draft.id


def _add_image(env, draft_id):
    with env.factory() as db:
        db.add(DraftAsset(draft_id=draft_id, asset_type="image", url="https://img/x.jpg", local_path="x.jpg", sort_order=0))
        db.commit()


def _add_creator_account(env):
    with env.factory() as db:
        account = PlatformAccount(user_id=env.user_id, platform="xhs", sub_type="creator",
                                  nickname="创作者账号", status="active")
        db.add(account)
        db.commit()
        return account.id


def test_validate_reports_title_assets_and_account_issues(env):
    draft_id = _create_draft(env, title="这是一个超过二十个字的小红书测试标题用来验证校验逻辑", body="正文")
    response = env.client.post(f"/api/drafts/{draft_id}/validate", headers=env.headers)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ok"] is False
    fields = {issue["field"] for issue in data["issues"]}
    assert fields == {"title", "assets", "account"}
    title_issue = next(issue for issue in data["issues"] if issue["field"] == "title")
    assert "上限 20 字" in title_issue["message"]
    assert data["creator_accounts"] == []


def test_validate_passes_with_creator_account_and_one_image(env):
    draft_id = _create_draft(env, title="短标题", body="正文内容",
                             tags=[{"id": "", "name": "AI服装"}, {"id": "", "name": "设计"}])
    _add_image(env, draft_id)
    account_id = _add_creator_account(env)

    data = env.client.post(f"/api/drafts/{draft_id}/validate", headers=env.headers).json()
    assert data["ok"] is True, data["issues"]
    assert data["images"] == 1
    assert data["tags"] == ["AI服装", "设计"]
    assert [account["id"] for account in data["creator_accounts"]] == [account_id]


def test_validate_rejects_video_and_too_many_tags(env):
    draft_id = _create_draft(env, title="短标题", body="正文",
                             tags=[{"id": "", "name": f"话题{i}"} for i in range(11)])
    with env.factory() as db:
        db.add(DraftAsset(draft_id=draft_id, asset_type="video", url="https://v/1.mp4", local_path="1.mp4", sort_order=0))
        db.commit()
    data = env.client.post(f"/api/drafts/{draft_id}/validate", headers=env.headers).json()
    fields = {issue["field"] for issue in data["issues"]}
    assert {"assets", "tags"} <= fields
    assert any("只支持图文" in issue["message"] for issue in data["issues"])


def test_send_to_publish_requires_a_title(env):
    draft_id = _create_draft(env, title="", body="正文")
    response = env.client.post(f"/api/drafts/{draft_id}/send-to-publish", headers=env.headers, json={})
    assert response.status_code == 400
    assert "标题" in response.json()["detail"]


def test_send_to_publish_copies_draft_tags_as_publish_options(env):
    draft_id = _create_draft(env, title="短标题", body="正文",
                             tags=[{"id": "", "name": "AI服装"}, {"id": "", "name": "设计"}])
    account_id = _add_creator_account(env)
    response = env.client.post(f"/api/drafts/{draft_id}/send-to-publish", headers=env.headers,
                               json={"platform_account_id": account_id, "publish_mode": "immediate"})
    assert response.status_code in (200, 201, 202), response.text
    job = response.json()
    assert job["title"] == "短标题" and job["body"] == "正文"
    assert job["status"] == "pending"
