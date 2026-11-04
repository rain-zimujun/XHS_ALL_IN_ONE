import pytest

from backend.app.models import TopicCache
from backend.app.services.topic_lookup import cached_topic_map, entities_for_topics, store_topics
from test_creative import env  # isolated database + client fixture

pytestmark = pytest.mark.usefixtures("env")


def test_store_topics_is_idempotent_and_updates_links(env):
    with env.factory() as db:
        assert store_topics(db, env.user_id, [{"id": "t1", "name": "穿搭", "link": "xhsdiscover://topic/v2/t1"}]) == 1
        assert store_topics(db, env.user_id, [{"id": "t1", "name": "穿搭", "link": "xhsdiscover://topic/v2/t1b"}]) == 1
        rows = db.query(TopicCache).all()
        assert len(rows) == 1
        assert rows[0].topic_id == "t1"
        assert rows[0].link.endswith("t1b")


def test_entities_for_topics_replaces_known_names_only(env):
    with env.factory() as db:
        store_topics(db, env.user_id, [{"id": "t1", "name": "穿搭", "link": "xhsdiscover://topic/v2/t1"}])
        resolved = entities_for_topics(db, env.user_id, ["穿搭", "不存在的话题", {"id": "tz", "name": "已有实体"}])
        assert resolved[0] == {"id": "t1", "name": "穿搭", "link": "xhsdiscover://topic/v2/t1", "type": "topic"}
        assert resolved[1] == "不存在的话题"
        assert resolved[2] == {"id": "tz", "name": "已有实体"}


def test_cached_topic_map_ignores_other_users(env):
    with env.factory() as db:
        store_topics(db, env.user_id, [{"id": "t1", "name": "穿搭"}])
        assert cached_topic_map(db, env.user_id, ["穿搭"])["穿搭"]["id"] == "t1"
        assert cached_topic_map(db, 9999, ["穿搭"]) == {}


def test_resolve_topics_accepts_prebuilt_entities():
    from backend.app.adapters.xhs.creator_api_adapter import resolve_topics

    class NoLookup:
        def get_topic(self, keyword):
            raise AssertionError("lookup should be skipped for entities")

    hash_tags, plain, warnings = resolve_topics(
        NoLookup(), [{"id": "t1", "name": "穿搭", "link": "xhsdiscover://topic/v2/t1", "type": "topic"}]
    )
    assert hash_tags == [{"id": "t1", "link": "xhsdiscover://topic/v2/t1", "name": "穿搭", "type": "topic"}]
    assert plain == [] and warnings == []
