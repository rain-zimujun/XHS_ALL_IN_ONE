import copy
import json

import pytest

from backend.app.schemas.creative_agents import ImagePlans
from backend.app.services.creative_agents import image_planning
from backend.app.services.creative_agents.copy_review import run_copy_review
from backend.app.services.creative_agents.image_planning import run_image_planning
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph import visual_batches
from backend.app.services.creative_graph.model import response_parameter
from backend.app.services.creative_orchestrator import AgentValidationError
from test_creative_copy_review import inputs as review_inputs
from test_creative_title_review import JudgeCompletion, block


def inputs(count=1, variants=2):
    args = review_inputs(count, variants)
    run, _ = run_copy_review(**args, complete=JudgeCompletion())
    return {**args, "copy_review": run.output.model_dump(mode="json")}


def valid_images(payload):
    return {
        "input_fingerprint": payload["input_fingerprint"],
        "variants": [{
            "idea_id": v["idea_id"], "candidate_id": v["candidate_id"], "variant_id": v["variant_id"],
            "cover_text": "浅蓝和米色的配色观察", "image_plan": [{
                "purpose": "展示正文提到的配色。", "composition": "米色服装为主体，浅蓝背景，顶部留白。",
                "aspect_ratio": "3:4", "search_description": "米色服装浅蓝背景配色参考",
                "prompt": "服装概念视觉，米色服装搭浅蓝背景，柔光，竖版构图，顶部留白，无文字。",
                "negative_prompt": "文字水印，畸形服装结构，商标",
            }], "material_notes": ["真实商品需提供实物图，概念图不作为效果证据。"],
        } for v in payload["locked_copies"]],
        "limitations": ["未分析样本原图，不保证还原真实商品。"],
    }


class ImageCompletion:
    def __init__(self, transform=None):
        self.calls, self.transform = [], transform

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        result = valid_images(prompt["input"])
        if self.transform:
            self.transform(result, len(self.calls))
        return json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 3)])
def test_image_plan_covers_each_variant_without_mutating_text(count, variants):
    args = inputs(count, variants)
    before = copy.deepcopy(args)
    run, meta = run_image_planning(**args, complete=ImageCompletion())
    assert args == before
    assert len(run.output.variants) == count * variants
    assert meta["quality_review_status"] == "not_reviewed"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_idea", "wrong_title", "old", "rewrite_body", "empty_prompt"])
def test_invalid_image_bindings_repair_once(fault):
    def transform(result, call):
        if call > 1:
            return
        v = result["variants"][0]
        if fault == "missing":
            result["variants"].pop()
        elif fault == "duplicate":
            result["variants"][1] = copy.deepcopy(v)
        elif fault == "wrong_idea":
            v["idea_id"] = "idea_other"
        elif fault == "wrong_title":
            v["candidate_id"] = "title_other"
        elif fault == "old":
            result["input_fingerprint"] = "0" * 64
        elif fault == "rewrite_body":
            v["body"] = "擅自改写"
        else:
            v["image_plan"][0]["prompt"] = ""
    complete = ImageCompletion(transform)
    run, _ = run_image_planning(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    assert "body" not in run.output.variants[0].model_dump()


def test_image_count_range_and_visual_brief_reach_the_plan():
    args = inputs()
    payload = image_planning.prepare_image_input(
        **args, image_count_min=3, image_count_max=5, visual_preferences="简洁，全图不要文字，不留白",
    )
    assert payload["image_count_min"] == 3 and payload["image_count_max"] == 5
    assert payload["visual_preferences"] == "简洁，全图不要文字，不留白"

    # The response schema binds the requested range so the model cannot drift.
    schema = response_parameter(visual_batches.image_response_type(payload))["json_schema"]["schema"]
    plan = schema["$defs"]["BoundVariantImages"]["properties"]["image_plan"]
    assert plan["minItems"] == 3 and plan["maxItems"] == 5


def test_image_plan_outside_the_requested_range_is_rejected():
    args = inputs()
    payload = image_planning.prepare_image_input(**args, image_count_min=3, image_count_max=5)
    output = valid_images(payload)
    assert len(output["variants"][0]["image_plan"]) == 1
    result = ImagePlans.model_validate(output)

    with pytest.raises(AgentValidationError) as caught:
        image_planning.validate_image_plans(result, payload)
    assert "3～5 张" in str(caught.value)


def test_image_count_range_must_be_ordered():
    from pydantic import ValidationError
    from backend.app.schemas.creative import CreativeAnalysisOptions
    with pytest.raises(ValidationError):
        CreativeAnalysisOptions(model_id=1, image_count_min=6, image_count_max=3)


def test_unapproved_copy_or_changed_body_blocks_image_call():
    args, complete = inputs(), ImageCompletion()
    def transform(result, payload, _):
        block(result["reviews"][0], payload["items"][0], criterion="publishability", field="body")
    raw_args = {key: value for key, value in args.items() if key != "copy_review"}
    run, _ = run_copy_review(**raw_args, complete=JudgeCompletion(transform))
    args["copy_review"] = run.output
    with pytest.raises(CreativeError):
        run_image_planning(**args, complete=complete)
    args = inputs()
    args["copy_variants"]["ideas"][0]["variants"][0]["body"] += "新段落"
    with pytest.raises(CreativeError):
        run_image_planning(**args, complete=complete)
    assert not complete.calls
