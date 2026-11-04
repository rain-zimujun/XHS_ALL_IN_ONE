import pytest

from backend.app.adapters.xhs.creator_api_adapter import resolve_topics


class FakeApi:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get_topic(self, keyword):
        self.calls.append(keyword)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_resolves_known_topic_into_entity():
    api = FakeApi((True, "", {"data": {"topic_info_dtos": [{"id": "t1", "name": "AI服装", "link": "xhs://t1"}]}}))
    hash_tags, plain, warnings = resolve_topics(api, ["AI服装"])
    assert hash_tags == [{"id": "t1", "link": "xhs://t1", "name": "AI服装", "type": "topic"}]
    assert plain == [] and warnings == []


def test_unresolved_topic_falls_back_to_plain_text():
    api = FakeApi((False, "'msg'", None))
    hash_tags, plain, warnings = resolve_topics(api, ["测试"])
    assert hash_tags == []
    assert plain == ["测试"]
    assert warnings and "普通文字" in warnings[0]


def test_missing_entity_list_also_falls_back():
    api = FakeApi((True, "", {"data": {"topic_info_dtos": []}}))
    hash_tags, plain, warnings = resolve_topics(api, ["测试"])
    assert hash_tags == [] and plain == ["测试"] and warnings


def test_topic_lookup_exception_does_not_abort():
    api = FakeApi(RuntimeError("network down"))
    hash_tags, plain, warnings = resolve_topics(api, ["测试", "AI服装"])
    assert hash_tags == []
    assert plain == ["测试", "AI服装"]
    assert len(warnings) == 2


def test_blank_topics_are_ignored():
    api = FakeApi((True, "", {"data": {"topic_info_dtos": [{"id": "t1", "name": "x"}]}}))
    hash_tags, plain, warnings = resolve_topics(api, ["", "   ", None])
    assert hash_tags == [] and plain == [] and warnings == []
    assert api.calls == []


def test_creator_rap_fingerprint_template_is_loadable():
    """The template path used to point one directory too deep, breaking every publish."""
    from xhs_utils.xhs_creator_util import load_creator_rap_fingerprint_hex

    value = load_creator_rap_fingerprint_hex()
    assert value.startswith("03ea00000010")
    assert len(value) == 872

def test_parse_topic_ref_accepts_link_id_and_name():
    from backend.app.adapters.xhs.creator_api_adapter import parse_topic_ref

    topic_id = "6aeaa74800000000030009b1"
    assert parse_topic_ref(f"https://www.xiaohongshu.com/page/topics/{topic_id}?name=AI服装") == (topic_id, "AI服装")
    assert parse_topic_ref(f"https://www.xiaohongshu.com/topic/{topic_id}") == (topic_id, "")
    assert parse_topic_ref(f"https://x.com/a?topic_id={topic_id}") == (topic_id, "")
    assert parse_topic_ref(topic_id) == (topic_id, "")
    assert parse_topic_ref("AI服装设计") == ("", "AI服装设计")
    assert parse_topic_ref("") == ("", "")


def test_resolve_topics_uses_id_from_link_without_lookup():
    class NoLookup:
        def get_topic(self, keyword):
            raise AssertionError("lookup should be skipped when an id is present")

    topic_id = "6aeaa74800000000030009b1"
    hash_tags, plain, warnings = resolve_topics(
        NoLookup(), [f"https://www.xiaohongshu.com/page/topics/{topic_id}?name=AI服装"]
    )
    assert hash_tags == [{"id": topic_id, "link": "", "name": "AI服装", "type": "topic"}]
    assert plain == [] and warnings == []