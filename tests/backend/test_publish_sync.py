import pytest

from backend.app.core.security import encrypt_text
from backend.app.models import AccountCookieVersion, PlatformAccount, PublishJob
from backend.app.services.publish_sync import _posted_note_ids, sync_scheduled_jobs
from test_creative import env  # isolated database + client fixture

pytestmark = pytest.mark.usefixtures("env")


def _creator_account(env, with_cookies=True):
    with env.factory() as db:
        account = PlatformAccount(user_id=env.user_id, platform="xhs", sub_type="creator",
                                  nickname="创作者", status="active")
        db.add(account)
        db.flush()
        if with_cookies:
            db.add(AccountCookieVersion(platform_account_id=account.id, encrypted_cookies=encrypt_text("sid=abc")))
        db.commit()
        return account.id


def _scheduled_job(env, account_id, note_id="note-1"):
    with env.factory() as db:
        job = PublishJob(user_id=env.user_id, platform_account_id=account_id, platform="xhs",
                         title="标题", body="正文", publish_mode="scheduled", status="scheduled",
                         scheduled_at=None, external_note_id=note_id)
        db.add(job)
        db.commit()
        return job.id


class FakeAdapter:
    def __init__(self, notes):
        self.notes = notes

    def get_published_notes(self):
        return True, "", self.notes


def test_posted_note_ids_handles_both_shapes():
    assert _posted_note_ids([{"id": "a"}, {"note_id": "b"}, {"note_card": {"id": "c"}}]) == {"a", "b", "c"}
    assert _posted_note_ids({"data": {"notes": [{"id": "d"}]}}) == {"d"}
    assert _posted_note_ids(None) == set()


def test_sync_marks_matching_scheduled_job_as_published(env):
    account_id = _creator_account(env)
    job_id = _scheduled_job(env, account_id, note_id="note-1")
    with env.factory() as db:
        from backend.app.models import User
        user = db.get(User, env.user_id)
        summary = sync_scheduled_jobs(db, user, lambda cookies: FakeAdapter([{"id": "note-1"}]))
        job = db.get(PublishJob, job_id)
        assert job.status == "published"
        assert job.published_at is not None
    assert summary == {"checked": 1, "published": 1, "still_scheduled": 0, "errors": []}


def test_sync_keeps_unlisted_jobs_scheduled(env):
    account_id = _creator_account(env)
    job_id = _scheduled_job(env, account_id, note_id="note-2")
    with env.factory() as db:
        from backend.app.models import User
        user = db.get(User, env.user_id)
        summary = sync_scheduled_jobs(db, user, lambda cookies: FakeAdapter([{"id": "other"}]))
        assert db.get(PublishJob, job_id).status == "scheduled"
    assert summary["still_scheduled"] == 1 and summary["published"] == 0


def test_sync_reports_missing_cookies_and_adapter_errors(env):
    no_cookie_account = _creator_account(env, with_cookies=False)
    _scheduled_job(env, no_cookie_account, note_id="note-3")
    account_id = _creator_account(env)
    _scheduled_job(env, account_id, note_id="note-4")

    def boom(cookies):
        raise RuntimeError("网络中断")

    with env.factory() as db:
        from backend.app.models import User
        user = db.get(User, env.user_id)
        summary = sync_scheduled_jobs(db, user, boom)
    assert any("没有可用 Cookie" in error for error in summary["errors"])
    assert any("网络中断" in error for error in summary["errors"])


def test_sync_endpoint_is_idempotent_and_owned(env):
    account_id = _creator_account(env)
    _scheduled_job(env, account_id, note_id="note-5")
    from backend.app.api.publish import get_creator_publish_adapter_factory
    from backend.app.main import app
    app.dependency_overrides[get_creator_publish_adapter_factory] = lambda: (lambda cookies: FakeAdapter([{"id": "note-5"}]))
    try:
        first = env.client.post("/api/publish/jobs/sync", headers=env.headers)
        assert first.status_code == 200, first.text
        assert first.json()["published"] == 1
        second = env.client.post("/api/publish/jobs/sync", headers=env.headers)
        assert second.json()["checked"] == 0  # nothing left in scheduled state
        assert env.client.post("/api/publish/jobs/sync", headers=env.other_headers).json()["checked"] == 0
    finally:
        app.dependency_overrides.pop(get_creator_publish_adapter_factory, None)
