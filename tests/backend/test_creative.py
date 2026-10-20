from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.api.platforms.xhs.pc import get_xhs_pc_api_adapter_factory
from backend.app.core.database import Base, get_db
from backend.app.core.security import create_access_token, encrypt_text
from backend.app.main import app
from backend.app.models import AccountCookieVersion, PlatformAccount, Task, User
from backend.app.schemas.creative import CreativeSearchRequest
from backend.app.services import creative_service


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
        db.commit()
        user_id, other_id, account_id = user.id, other.id, account.id

    def override_db():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_xhs_pc_api_adapter_factory] = lambda: FakeAdapter
    monkeypatch.setattr(creative_service.time, "sleep", lambda _: None)
    FakeAdapter.calls = []
    yield SimpleNamespace(
        client=TestClient(app), factory=factory, account_id=account_id, user_id=user_id,
        headers={"Authorization": f"Bearer {create_access_token(user_id)}"},
        other_headers={"Authorization": f"Bearer {create_access_token(other_id)}"},
    )
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_xhs_pc_api_adapter_factory, None)
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
