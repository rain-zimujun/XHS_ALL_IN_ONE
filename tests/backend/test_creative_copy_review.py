import pytest

from backend.app.services.creative_agents.copy_generation import run_copy_generation
from backend.app.services.creative_agents.copy_review import prepare_copy_review_input, run_copy_review
from backend.app.services.creative_agents.reviewing import verify_review
from backend.app.services.creative_analysis import CreativeError
from test_creative_copy_agent import CopyCompletion, inputs as copy_inputs
from test_creative_title_review import JudgeCompletion, block


def inputs(count=1, variants=2):
    args = copy_inputs(count, variants)
    run, _ = run_copy_generation(**args, complete=CopyCompletion())
    return {**args, "copy_variants": run.output.model_dump(mode="json"), "copy_revision": 1}


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 3)])
def test_all_variants_are_reviewed_and_context_includes_title_and_style(count, variants):
    args = inputs(count, variants)
    run, meta = run_copy_review(**args, complete=JudgeCompletion())
    assert len(run.output.reviews) == count * variants
    assert meta["downstream_ready"]
    payload = prepare_copy_review_input(**args)
    assert payload["context"]["locked_titles"]
    assert payload["context"]["sample_analysis"]["style_patterns"]
    assert verify_review(run.output, payload) == run.output


def test_one_failed_variant_blocks_handoff_without_dropping_or_retry():
    def transform(result, payload, _):
        block(result["reviews"][0], payload["items"][0], criterion="publishability", field="body")
    complete = JudgeCompletion(transform)
    run, meta = run_copy_review(**inputs(3, 3), complete=complete)
    assert len(run.output.reviews) == 9
    assert run.output.revision_required_ids == ["copy_0_0"]
    assert not meta["downstream_ready"]
    assert len(complete.calls) == 1
    with pytest.raises(CreativeError):
        verify_review(run.output, prepare_copy_review_input(**inputs(3, 3)))


def test_many_explicit_and_rubric_blockers_do_not_overflow_review_schema():
    def transform(result, payload, _):
        review = result["reviews"][0]
        for score in review["scores"]:
            score["score"] = 1
        review["issues"] = [{
            "criterion": c["key"], "severity": "blocking", "field": "body", "quote": "米色",
            "problem": f"需要修改的问题{i}", "revision_instruction": "补足具体信息。",
        } for i, c in enumerate(payload["rubric"])]
        review["issues"].extend([{**review["issues"][0], "problem": f"附加阻断{i}"} for i in range(2)])
    complete = JudgeCompletion(transform)
    run, _ = run_copy_review(**inputs(), complete=complete)
    assert len(run.output.reviews[0].quality.blocking_issues) == 13
    assert len(complete.calls) == 1


@pytest.mark.parametrize("field", ["body", "tags", "covered_points"])
def test_feedback_must_quote_current_field(field):
    def transform(result, payload, call):
        block(result["reviews"][0], payload["items"][0], criterion="value", field="body")
        if call == 1:
            result["reviews"][0]["issues"][0].update(field=field, quote="不存在的句子")
    complete = JudgeCompletion(transform)
    run, _ = run_copy_review(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    assert run.output.status == "revision_required"


@pytest.mark.parametrize("change", ["body", "tags", "version", "score"])
def test_review_becomes_invalid_when_copy_or_version_changes(change):
    args = inputs()
    run, _ = run_copy_review(**args, complete=JudgeCompletion())
    result = run.output.model_dump(mode="json")
    if change == "body":
        args["copy_variants"]["ideas"][0]["variants"][0]["body"] += "新的正文"
    elif change == "tags":
        args["copy_variants"]["ideas"][0]["variants"][0]["tags"].append("视觉")
    elif change == "version":
        args["copy_revision"] += 1
    else:
        result["reviews"][0]["quality"]["passed"] = False
    with pytest.raises(CreativeError):
        verify_review(result, prepare_copy_review_input(**args))
