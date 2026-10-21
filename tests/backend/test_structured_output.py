import json
from types import SimpleNamespace

import pytest
import requests

from backend.app.schemas.creative import CreativeAnalysisResult
from backend.app.schemas.creative_agents import (
    AudienceBrief, CopyVariants, ImagePlans, ItemReviewResponse, SampleAnalysis,
    TitleCandidates, TopicBrief, TopicReviewResponse,
)
from backend.app.services.ai_service import OpenAICompatibleTextClient
from backend.app.services.creative_analysis import CreativeError, request_analysis
from backend.app.services.structured_output import json_schema_response_format
from test_creative_workflow import WorkflowCompletion, workflow


OUTPUT_MODELS = [
    SampleAnalysis, AudienceBrief, TopicBrief, TopicReviewResponse,
    TitleCandidates, ItemReviewResponse, CopyVariants, ImagePlans,
    CreativeAnalysisResult,
]


def assert_strict_contract(node, root):
    if isinstance(node, dict):
        assert "default" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        if "$ref" in node:
            assert node["$ref"].startswith("#/$defs/")
            assert node["$ref"].removeprefix("#/$defs/") in root["$defs"]
        for child in node.values():
            assert_strict_contract(child, root)
    elif isinstance(node, list):
        for child in node:
            assert_strict_contract(child, root)


@pytest.mark.parametrize("output_model", OUTPUT_MODELS)
def test_all_creative_schemas_meet_strict_api_contract_without_changing_local_models(output_model):
    original = output_model.model_json_schema()
    response_format = json_schema_response_format(output_model)
    assert response_format["type"] == "json_schema"
    config = response_format["json_schema"]
    assert config["name"] == output_model.__name__
    assert config["strict"] is True
    assert_strict_contract(config["schema"], config["schema"])
    assert output_model.model_json_schema() == original


def test_defaulted_fields_are_required_and_nullable_review_owner_remains_nullable():
    review_schema = json_schema_response_format(ItemReviewResponse)["json_schema"]["schema"]
    issue = review_schema["$defs"]["ItemReviewIssue"]
    assert "responsible_stage" in issue["required"]
    owner = issue["properties"]["responsible_stage"]
    assert {"type": "null"} in owner["anyOf"]
    assert next(item for item in owner["anyOf"] if item.get("type") == "string")["enum"] == ["topic", "title", "copy", "image"]
    audience_schema = json_schema_response_format(AudienceBrief)["json_schema"]["schema"]
    evidence = audience_schema["$defs"]["AudienceEvidence"]
    assert "note_ids" in evidence["required"]
    assert evidence["properties"]["note_ids"]["type"] == "array"
    sample_schema = json_schema_response_format(SampleAnalysis)["json_schema"]["schema"]
    quote = sample_schema["$defs"]["SampleEvidence"]["properties"]["quote"]
    assert quote["minLength"] == 2 and quote["maxLength"] == 180


@pytest.mark.parametrize("fail_stage,expected_calls", [(None, 10), ("copy_review", 12)])
def test_every_producer_judge_and_quality_revision_sends_schema_in_actual_http_body(monkeypatch, fail_stage, expected_calls):
    model, requests_seen = WorkflowCompletion(fail_stage=fail_stage), []

    def post(url, **kwargs):
        assert url == "https://example.invalid/v1/chat/completions"
        assert kwargs["timeout"] == (10, 180)
        body = kwargs["json"]
        requests_seen.append(body)
        prompt, _ = json.JSONDecoder().raw_decode(body["messages"][1]["content"])
        response_format = body["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == prompt["output_schema"]
        assert response_format["json_schema"]["name"] == prompt["output_schema"]["title"]
        content = model(
            system_prompt=body["messages"][0]["content"],
            user_prompt=body["messages"][1]["content"],
            temperature=body["temperature"], response_format=response_format,
        )
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        return response

    monkeypatch.setattr("backend.app.services.ai_service.requests.post", post)
    config = SimpleNamespace(base_url="https://example.invalid/v1", model_name="mock")

    def complete(**kwargs):
        return request_analysis(OpenAICompatibleTextClient(), model_config=config, api_key="fixture", **kwargs)

    result, metadata = workflow(complete).run()
    assert result["ideas"]
    assert metadata["request_count"] == len(requests_seen) == expected_calls
    assert {body["response_format"]["json_schema"]["name"] for body in requests_seen} == {
        model.__name__ for model in OUTPUT_MODELS if model is not CreativeAnalysisResult
    }
    if fail_stage:
        assert any("quality_revision" in json.JSONDecoder().raw_decode(body["messages"][1]["content"])[0] for body in requests_seen)


@pytest.mark.parametrize("status", [400, 422])
def test_provider_rejecting_schema_stops_without_silent_fallback_or_extra_requests(monkeypatch, status):
    calls = []

    def post(_url, **kwargs):
        calls.append(kwargs["json"])
        response = requests.Response()
        response.status_code = status
        response._content = b'{"error":{"message":"unsupported json_schema fixture-private"}}'
        return response

    monkeypatch.setattr("backend.app.services.ai_service.requests.post", post)
    config = SimpleNamespace(base_url="https://example.invalid/v1", model_name="mock")

    def complete(**kwargs):
        return request_analysis(OpenAICompatibleTextClient(), model_config=config, api_key="fixture", **kwargs)

    with pytest.raises(CreativeError) as caught:
        workflow(complete).run()
    assert caught.value.code == "ai_http_error"
    assert f"HTTP {status}" in str(caught.value)
    assert "fixture-private" not in str(caught.value)
    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"
