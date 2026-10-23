import copy
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from backend.app.schemas.creative import CreativeAnalysisResult
from backend.app.schemas.creative_agents import CopyVariants
from backend.app.schemas.creative_graph import CopyResponse
from backend.app.services.creative_agents.copy_generation import run_copy_generation
from backend.app.services.creative_analysis import get_creative_text_client_factory
from backend.app.services.creative_graph.agents import build_agents, GraphOutputError
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter
from backend.app.services.creative_orchestrator import AgentRunError
from backend.app.services.structured_output import json_schema_response_format
from backend.app.main import app
from test_creative import env, valid_analysis, collect_for_analysis
from test_creative_copy_agent import inputs, CopyCompletion, valid_copies
from backend.app.services.creative_agents.copy_generation import prepare_copy_input, validate_copy_variants


@pytest.mark.parametrize("length", [500, 501])
def test_graph_copy_native_schema_and_runtime_share_500_character_limit(length):
    payload = prepare_copy_input(**inputs())
    raw = valid_copies(payload)
    raw.pop("input_fingerprint")
    raw["ideas"][0]["variants"][0]["body"] = "服" * (length - 2) + "\n✨"
    assert len(raw["ideas"][0]["variants"][0]["body"]) == length
    schema = response_parameter(CopyResponse)["json_schema"]["schema"]
    assert schema["$defs"]["CopyVariant"]["properties"]["body"]["maxLength"] == 500
    if length == 500:
        assert TypeAdapter(CopyResponse).validate_python(raw)
    else:
        with pytest.raises(ValidationError):
            TypeAdapter(CopyResponse).validate_python(raw)


@pytest.mark.parametrize("always", [False, True])
def test_graph_overlong_copy_repairs_once_without_silent_truncation(always):
    payload = prepare_copy_input(**inputs())
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        raw = valid_copies(payload)
        raw.pop("input_fingerprint")
        if len(calls) == 1 or always:
            raw["ideas"][0]["variants"][0]["body"] = "服" * 501
        return json.dumps(raw, ensure_ascii=False)
    agent = build_agents(CompatibleChatModel(complete))["copy"]
    if always:
        with pytest.raises(GraphOutputError):
            agent.invoke(payload, validate_output=lambda r: validate_copy_variants(r, payload))
    else:
        output = agent.invoke(payload, validate_output=lambda r: validate_copy_variants(r, payload)).output
        assert output.ideas[0].variants[0].body == valid_copies(payload)["ideas"][0]["variants"][0]["body"]
    assert len(calls) == 2
    assert "500" in calls[1]["user_prompt"]


@pytest.mark.parametrize("always", [False, True])
def test_original_multi_agent_generation_enforces_limit_in_schema_and_local_checks(always):
    def transform(output, call):
        if call == 1 or always:
            output["ideas"][0]["variants"][0]["body"] = "文" * 501
    complete = CopyCompletion(transform)
    if always:
        with pytest.raises(AgentRunError):
            run_copy_generation(**inputs(), complete=complete)
    else:
        run, _ = run_copy_generation(**inputs(), complete=complete)
        assert all(len(v.body) <= 500 for g in run.output.ideas for v in g.variants)
    assert len(complete.calls) == 2
    schema = complete.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["CopyVariant"]["properties"]["body"]["maxLength"] == 500
    assert "500" in complete.calls[1]["user_prompt"]


@pytest.mark.parametrize("always", [False, True])
def test_legacy_generation_repairs_or_rejects_long_body_with_schema_limit(env, always):
    calls = []
    class Client:
        def _complete(self, **kwargs):
            calls.append(kwargs)
            raw = valid_analysis()
            if len(calls) == 1 or always:
                raw["ideas"][0]["variants"][0]["body"] = "服" * 501
            return json.dumps(raw, ensure_ascii=False)
    app.dependency_overrides[get_creative_text_client_factory] = lambda: Client
    run = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1, "workflow": "legacy"})
    assert run["status"] == ("failed" if always else "completed")
    assert len(calls) == 2
    assert calls[0]["response_format"]["json_schema"]["schema"]["$defs"]["CreativeVariant"]["properties"]["body"]["maxLength"] == 500
    assert "body 超过 500 字" in calls[1]["user_prompt"]


def test_historical_long_copy_models_and_unconstrained_schema_remain_readable():
    historical = valid_analysis()
    historical["ideas"][0]["variants"][0]["body"] = "历史正文" * 200
    assert CreativeAnalysisResult.model_validate(historical)
    payload = prepare_copy_input(**inputs())
    old = valid_copies(payload)
    old["ideas"][0]["variants"][0]["body"] = "历史正文" * 200
    assert CopyVariants.model_validate(old)
    original = copy.deepcopy(CopyVariants.model_json_schema())
    assert json_schema_response_format(CopyVariants, max_body_chars=500)["json_schema"]["schema"]["$defs"]["CopyVariant"]["properties"]["body"]["maxLength"] == 500
    assert CopyVariants.model_json_schema() == original
    assert original["$defs"]["CopyVariant"]["properties"]["body"]["maxLength"] == 3000
