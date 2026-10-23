import copy
import json

import pytest

from backend.app.services.creative_agents.topic_selection import prepare_topic_input
from backend.app.services.creative_agents.title_generation import prepare_title_input
from backend.app.services.creative_agents.copy_generation import prepare_copy_input
from backend.app.services.creative_graph.context import production_context
from test_creative_topic_selection import sample_analysis, audience_brief
from test_creative_titles import inputs as title_inputs
from test_creative_copy_agent import inputs as copy_inputs
from test_creative_graph_workflow import GraphCompletion, workflow


def payload_for(role):
    if role == "topic":
        return prepare_topic_input("AI 服装", "简洁原创", 3, sample_analysis(), audience_brief())
    if role == "title":
        return prepare_title_input(**title_inputs(3))
    return prepare_copy_input(**copy_inputs(3, 3))


def all_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(all_keys(v) for v in value), set())
    return set()


@pytest.mark.parametrize("role", ["topic", "title", "copy"])
def test_producer_context_allowlist_removes_quotes_raw_material_and_unused_fields(role):
    payload = payload_for(role)
    original = copy.deepcopy(payload)
    payload["sources"] = [{"content": "绝不发给后续模型的完整采集正文"}]
    payload["unused_future_field"] = "不要默认传递将来新增的状态字段"
    compact = production_context(role, payload)
    assert not {"quote", "body_excerpt", "sources", "unused_future_field", "evidence"} & all_keys(compact)
    assert len(json.dumps(compact, ensure_ascii=False)) < len(json.dumps(original, ensure_ascii=False))
    assert payload["sample_analysis"] == original["sample_analysis"]
    assert compact["keyword"] == payload["keyword"]
    assert compact["requirements"] == payload["requirements"]
    if role == "topic":
        assert len(compact["sample_analysis"]["observations"]) == len(payload["sample_analysis"]["observations"])
        assert compact["audience_segment_ids"] == payload["audience_segment_ids"]
        assert "style_patterns" not in compact["sample_analysis"]
    else:
        assert "observations" not in compact["sample_analysis"]
        assert compact["sample_analysis"]["style_summary"] == payload["sample_analysis"]["style_summary"]
        assert compact["topic_brief"]["ideas"][0]["must_cover"] == payload["topic_brief"]["ideas"][0]["must_cover"]
        assert compact["topic_brief"]["ideas"][0]["boundaries"] == payload["topic_brief"]["ideas"][0]["boundaries"]
        assert compact["input_fingerprint"] == payload["input_fingerprint"]
        if role == "title":
            assert all(p["dimension"] == "title" for p in compact["sample_analysis"]["style_patterns"])
        else:
            assert compact["variant_count"] == 3
            assert compact["locked_titles"][0]["promise"] == payload["locked_titles"][0]["promise"]
            assert [p["finding"] for p in compact["sample_analysis"]["style_patterns"]] == [p["finding"] for p in payload["sample_analysis"]["style_patterns"]]


def test_full_graph_sends_compact_production_inputs_native_schema_and_keeps_evidence_for_judges():
    model = GraphCompletion()
    w = workflow(model, count=3, variants=3)
    result, _ = w.run()
    assert result
    for call in model.wire_calls:
        role = call["response_format"]["json_schema"]["name"]
        prompt = json.loads(call["user_prompt"])
        payload = prompt["input"]
        if role in ("TopicResponse", "TitleResponse", "CopyResponse"):
            assert "output_schema" not in prompt
            assert "quote" not in all_keys(payload)
            assert "sources" not in all_keys(payload)
            assert call["response_format"]["json_schema"]["schema"]["properties"]
        elif role in ("TopicReviewResponse", "CopyReviewResponse", "FinalReviewResponse"):
            assert "output_schema" not in prompt
            context = payload.get("context", payload)
            assert "quote" in all_keys(context["sample_analysis"])
        if role != "SampleAnalysisResponse":
            assert "body_excerpt" not in all_keys(payload)
    assert w.state["sources"]
    assert w.state["sample_analysis"]["observations"][0]["quote"]
