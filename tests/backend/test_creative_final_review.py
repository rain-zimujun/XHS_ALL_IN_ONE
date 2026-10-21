import copy

import pytest

from backend.app.services.creative_agents.final_review import prepare_final_review_input, run_final_review
from backend.app.services.creative_agents.image_planning import run_image_planning
from backend.app.services.creative_agents.reviewing import verify_review
from backend.app.services.creative_analysis import CreativeError
from test_creative_image_agent import ImageCompletion, inputs as image_inputs
from test_creative_title_review import JudgeCompletion, block


def inputs(count=1, variants=2):
    args = image_inputs(count, variants)
    run, _ = run_image_planning(**args, complete=ImageCompletion())
    return {**args, "image_plans": run.output.model_dump(mode="json"), "final_revision": 1}


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 3)])
def test_final_review_checks_all_bundles_without_rewriting(count, variants):
    args = inputs(count, variants)
    before = copy.deepcopy(args)
    run, meta = run_final_review(**args, complete=JudgeCompletion())
    assert args == before
    assert len(run.output.reviews) == count * variants
    assert meta["downstream_ready"]
    assert verify_review(run.output, prepare_final_review_input(**args)) == run.output


@pytest.mark.parametrize("field,owner", [("title", "title"), ("body", "copy"), ("cover_text", "image")])
def test_final_issues_are_routed_to_actual_field_owner(field, owner):
    def transform(result, payload, _):
        block(result["reviews"][0], payload["items"][0], criterion="safety", field=field)
        result["reviews"][0]["issues"][0]["responsible_stage"] = owner
    complete = JudgeCompletion(transform)
    run, meta = run_final_review(**inputs(), complete=complete)
    assert not meta["downstream_ready"]
    assert run.output.reviews[0].judgement.issues[0].responsible_stage == owner
    assert len(complete.calls) == 1


def test_wrong_owner_and_missing_owner_repair_without_rewriting():
    for wrong in (None, "image"):
        def transform(result, payload, call):
            block(result["reviews"][0], payload["items"][0], criterion="title_body_fit", field="body")
            result["reviews"][0]["issues"][0]["responsible_stage"] = wrong if call == 1 else "copy"
        complete = JudgeCompletion(transform)
        run, _ = run_final_review(**inputs(), complete=complete)
        assert len(complete.calls) == 2
        assert run.output.status == "revision_required"


def test_low_safety_blocks_even_with_high_total():
    def transform(result, payload, _):
        review = result["reviews"][0]
        for score in review["scores"]:
            score["score"] = 6 if score["criterion"] == "safety" else 10
        block(review, payload["items"][0], criterion="safety", field="body")
        review["issues"][0]["responsible_stage"] = "copy"
    run, _ = run_final_review(**inputs(), complete=JudgeCompletion(transform))
    assert run.output.reviews[0].quality.overall_score > 9
    assert not run.output.reviews[0].quality.passed


@pytest.mark.parametrize("change", ["prompt", "cover", "material", "revision"])
def test_final_review_invalidates_on_image_changes(change):
    args = inputs()
    run, _ = run_final_review(**args, complete=JudgeCompletion())
    v = args["image_plans"]["variants"][0]
    if change == "prompt":
        v["image_plan"][0]["prompt"] += "新的构图"
    elif change == "cover":
        v["cover_text"] += "新封面"
    elif change == "material":
        v["material_notes"].append("需提供产品图片")
    else:
        args["final_revision"] += 1
    with pytest.raises(CreativeError):
        verify_review(run.output, prepare_final_review_input(**args))
