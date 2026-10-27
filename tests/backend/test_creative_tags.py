import copy
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from backend.app.schemas.creative import CreativeAnalysisResult
from backend.app.schemas.creative_agents import CopyVariants
from backend.app.schemas.creative_graph import CopyResponse
from backend.app.services.creative_agents.copy_generation import prepare_copy_input, run_copy_generation, validate_copy_variants
from backend.app.services.creative_graph.agents import build_agents, GraphOutputError
from backend.app.services.creative_graph.model import CompatibleChatModel
from backend.app.services.creative_orchestrator import AgentRunError
from backend.app.services.creative_analysis import get_creative_text_client_factory
from backend.app.services.creative_copy import validate_generated_tags
from backend.app.services.structured_output import json_schema_response_format
from backend.app.main import app
from test_creative import env, valid_analysis, collect_for_analysis
from test_creative_copy_agent import inputs, valid_copies


@pytest.mark.parametrize("engine", ["langgraph", "multi_agent", "legacy"])
def test_flexible_tag_range_retries_only_when_below_minimum(env, engine):
    calls = []
    payload = prepare_copy_input(**inputs()) if engine != "legacy" else None
    expected = ["服装配色", "米色穿搭", "浅蓝色穿搭", "服装廓形", "配色灵感", "服装设计"]

    def complete(**kwargs):
        calls.append(kwargs)
        raw = valid_analysis() if engine == "legacy" else valid_copies(payload)
        if engine == "langgraph":
            raw.pop("input_fingerprint")
        for idea in raw["ideas"]:
            for variant in idea["variants"]:
                variant["tags"] = expected[:4] if len(calls) == 1 else expected
        return json.dumps(raw, ensure_ascii=False)

    if engine == "legacy":
        class Client:
            def _complete(self, **kwargs):
                return complete(**kwargs)
        app.dependency_overrides[get_creative_text_client_factory] = lambda: Client
        task = collect_for_analysis(env, {"model_id": env.model_id, "idea_count": 1, "workflow": engine})
        assert task["status"] == "completed"
        result = task["payload"].get("result")
    else:
        def run():
            if engine == "langgraph":
                return build_agents(CompatibleChatModel(complete))["copy"].invoke(
                    payload, validate_output=lambda r: validate_copy_variants(r, payload)).output.model_dump()
            return run_copy_generation(**inputs(), complete=complete)[0].output.model_dump()
        result = run()
    assert len(calls) == 2
    schema = calls[0]["response_format"]["json_schema"]["schema"]
    variant_type = "CreativeVariant" if engine == "legacy" else "BoundCopyVariant" if engine == "langgraph" else "CopyVariant"
    tags = schema["$defs"][variant_type]["properties"]["tags"]
    assert tags["minItems"] == 5
    assert tags["maxItems"] == 8
    assert calls[0]["response_format"] == calls[1]["response_format"]
    assert "最终正文最相关" in calls[0]["system_prompt"]
    assert result["ideas"][0]["variants"][0]["tags"] == expected


@pytest.mark.parametrize("count", [5, 6, 7, 8])
def test_graph_count_boundaries_and_historical_models_remain_compatible(count):
    raw = valid_copies(prepare_copy_input(**inputs()))
    raw["ideas"][0]["variants"][0]["tags"] = [f"话题{i}" for i in range(count)]
    assert CopyVariants.model_validate(raw)
    raw.pop("input_fingerprint")
    assert TypeAdapter(CopyResponse).validate_python(raw)


@pytest.mark.parametrize("count", [4, 9])
def test_graph_count_outside_generation_range_is_rejected(count):
    raw = valid_copies(prepare_copy_input(**inputs()))
    raw["ideas"][0]["variants"][0]["tags"] = [f"话题{i}" for i in range(count)]
    raw.pop("input_fingerprint")
    with pytest.raises(ValidationError):
        TypeAdapter(CopyResponse).validate_python(raw)


def test_generation_validation_rejects_below_minimum_after_cleaning():
    with pytest.raises(ValueError, match="5～8 个"):
        validate_generated_tags(["AI", "服装", "设计", "配色"])


@pytest.mark.parametrize("count", [5, 6, 7, 8])
def test_generation_accepts_five_to_eight_clean_tags(count):
    validate_generated_tags([f"话题{i}" for i in range(count)])


def test_generation_schema_overrides_do_not_rewrite_historical_tag_constraints():
    for model, variant in [(CreativeAnalysisResult, "CreativeVariant"), (CopyVariants, "CopyVariant")]:
        original = copy.deepcopy(model.model_json_schema())
        schema = json_schema_response_format(model, tag_min_count=5, tag_max_count=8)["json_schema"]["schema"]
        assert schema["$defs"][variant]["properties"]["tags"]["minItems"] == 5
        assert schema["$defs"][variant]["properties"]["tags"]["maxItems"] == 8
        assert model.model_json_schema() == original
        assert original["$defs"][variant]["properties"]["tags"]["minItems"] == 1
