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
