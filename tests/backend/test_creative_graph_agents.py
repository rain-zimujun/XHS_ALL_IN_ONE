import json
from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError
from typing_extensions import is_typeddict

from backend.app.schemas.creative_graph import SampleAnalysisResponse
from backend.app.services.ai_service import OpenAICompatibleTextClient
from backend.app.services.creative_agents.sample_analysis import prepare_sample_input, validate_sample_evidence
from backend.app.services.creative_analysis import request_analysis
from backend.app.services.creative_graph.agents import GraphOutputError, build_agents
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter
from backend.app.services.creative_graph.review_schema import item_review_response_type
from backend.app.services.creative_agents.title_generation import bind_title_ids, title_response_type
from test_creative_sample_agent import OPTIONS, SEARCH, sources, valid_output
from test_structured_output import assert_strict_contract


def test_all_ten_agents_bind_distinct_handwritten_typeddict_responses():
    agents = build_agents(CompatibleChatModel(lambda **_: pytest.fail("unexpected model request")))
    assert len(agents) == len({a.response_format for a in agents.values()}) == 10
    for agent in agents.values():
        assert is_typeddict(agent.response_format)
        schema = response_parameter(agent.response_format)["json_schema"]["schema"]
        assert schema["properties"]
        assert_strict_contract(schema, schema)
    schema = response_parameter(SampleAnalysisResponse)["json_schema"]["schema"]
    assert schema["properties"]["observations"]["minItems"] == 3


def test_native_schema_reaches_actual_http_body_and_returns_one_validated_response(monkeypatch):
    payload, seen = prepare_sample_input(SEARCH, sources(), OPTIONS), []

    def post(url, **kwargs):
        seen.append(kwargs)
        assert url == "https://example.invalid/v1/chat/completions"
        body = kwargs["json"]
        assert kwargs["timeout"] == (10, 180)
        assert body["response_format"] == response_parameter(SampleAnalysisResponse)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"choices": [{"message": {"content": json.dumps(valid_output(payload))}}]}).encode()
        return response

    monkeypatch.setattr("backend.app.services.ai_service.requests.post", post)
    def complete(**kwargs):
        return request_analysis(OpenAICompatibleTextClient(), model_config=SimpleNamespace(
            base_url="https://example.invalid/v1", model_name="mock",
        ), api_key="fixture-private", **kwargs)

    model = CompatibleChatModel(complete)
    agent = build_agents(model)["sample_analysis"]
    result = agent.invoke(payload, validate_output=lambda r: validate_sample_evidence(r, payload))
    assert len(seen) == len(result.attempts) == 1
    assert len(result.output.observations) == 3
    assert "fixture-private" not in json.dumps(model.to_json())


@pytest.mark.parametrize("kind", ["schema", "handoff"])
def test_schema_and_business_failures_are_distinct_and_bounded(kind):
    payload, calls = prepare_sample_input(SEARCH, sources(), OPTIONS), []
    def complete(**kwargs):
        calls.append(kwargs)
        output = valid_output(payload)
        if kind == "schema":
            output["observations"][0]["note_id"] = 123
        else:
            output["observations"][0]["quote"] = "这句话不存在于原文"
        return json.dumps(output)
    agent = build_agents(CompatibleChatModel(complete))["sample_analysis"]
    with pytest.raises(GraphOutputError) as caught:
        agent.invoke(payload, validate_output=lambda r: validate_sample_evidence(r, payload))
    assert caught.value.code == f"creative_{kind}_invalid"
    assert len(calls) == len(caught.value.attempts) == 2
    assert calls[0]["response_format"] == calls[1]["response_format"]


def test_missing_nullable_field_is_invalid_and_provider_rejection_is_not_retried():
    from pydantic import TypeAdapter
    from backend.app.schemas.creative_graph import ItemIssue
    with pytest.raises(ValidationError):
        TypeAdapter(ItemIssue).validate_python({
            "criterion": "fit", "severity": "blocking", "field": "body", "quote": "原文",
            "problem": "问题", "revision_instruction": "修改", "note_ids": [], "related_item_ids": [],
        })
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        raise requests.HTTPError("fixture-private")
    agent = build_agents(CompatibleChatModel(complete))["sample_analysis"]
    with pytest.raises(requests.HTTPError):
        agent.invoke({}, validate_output=lambda _: None)
    assert len(calls) == 1


def test_incomplete_json_is_not_accepted_by_structured_output_parser():
    model = CompatibleChatModel(lambda **_: '{"summary":"unfinished')
    agent = build_agents(model)["sample_analysis"]
    with pytest.raises(GraphOutputError) as caught:
        agent.invoke({}, validate_output=lambda _: None)
    assert caught.value.code == "creative_schema_invalid"


def test_item_review_schema_narrows_dynamic_references_to_current_batch():
    payload = {
        "kind": "copy",
        "rubric": [{"key": "publishability"}, {"key": "fit"}],
        "items": [
            {"item_id": "copy_a", "fields": {"body": "正文", "tags": ["AI"]}},
            {"item_id": "copy_b", "fields": {"body": "另一篇", "tags": ["服装"]}},
        ],
    }
    schema = response_parameter(item_review_response_type(payload))["json_schema"]["schema"]
    score = schema["$defs"]["DynamicItemScore"]["properties"]
    issue = schema["$defs"]["DynamicItemIssue"]["properties"]
    judgement = schema["$defs"]["DynamicItemJudgement"]["properties"]
    assert score["criterion"]["enum"] == ["publishability", "fit"]
    assert score["fields"]["items"]["enum"] == ["body", "tags"]
    assert issue["field"]["enum"] == ["body", "tags"]
    assert judgement["item_id"]["enum"] == ["copy_a", "copy_b"]


def test_title_schema_uses_program_owned_idea_keys_and_adapter_binds_candidate_ids():
    payload = {
        "title_max_length": 20,
        "candidate_count": 2,
        "topic_brief": {"ideas": [
            {"idea_id": "idea_1"}, {"idea_id": "idea_2"},
        ]},
    }
    response_type = title_response_type(payload)
    schema = response_parameter(response_type)["json_schema"]["schema"]
    ideas = schema["$defs"]["TitleIdeasByProgramId"]
    assert ideas["required"] == ["idea_1", "idea_2"]
    assert ideas["additionalProperties"] is False
    candidate = schema["$defs"]["TitleCandidateContent"]["properties"]
    assert "candidate_id" not in candidate
    adapted = bind_title_ids({"ideas": {
        "idea_1": {"candidates": [{"title": "标题\n一", "angle": "角度1", "reader_value": "价值", "promise": "承诺", "support_points": ["依据"], "rationale": "理由"},
                                    {"title": "标题二", "angle": "角度2", "reader_value": "价值", "promise": "承诺", "support_points": ["依据"], "rationale": "理由"}]},
        "idea_2": {"candidates": [{"title": "标题三", "angle": "角度1", "reader_value": "价值", "promise": "承诺", "support_points": ["依据"], "rationale": "理由"},
                                    {"title": "标题四", "angle": "角度2", "reader_value": "价值", "promise": "承诺", "support_points": ["依据"], "rationale": "理由"}]},
    }, "limitations": ["局限"]}, payload)
    assert [idea["idea_id"] for idea in adapted["ideas"]] == ["idea_1", "idea_2"]
    assert [candidate["candidate_id"] for idea in adapted["ideas"] for candidate in idea["candidates"]] == [
        "title_0_0", "title_0_1", "title_1_0", "title_1_1",
    ]
    assert adapted["ideas"][0]["candidates"][0]["title"] == "标题 一"
