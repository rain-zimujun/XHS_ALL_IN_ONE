import copy
import json

import pytest

from backend.app.services.creative_agents.title_generation import run_title_generation
from backend.app.services.creative_agents.title_review import (
    prepare_title_review_input, run_title_review, verified_title_review,
)
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_orchestrator import AgentRunError
from test_creative_titles import TitleCompletion, inputs as title_inputs


def inputs(count=1):
    args = title_inputs(count)
    run, _ = run_title_generation(**args, complete=TitleCompletion())
    return {**args, "title_candidates": run.output.model_dump(mode="json"), "title_revision": 1}


def valid_review(payload):
    return {
        "revision": payload["revision"], "input_fingerprint": payload["input_fingerprint"],
        "summary": "候选有明确对象和内容支撑。",
        "reviews": [{
            "item_id": item["item_id"],
            "scores": [{"criterion": c["key"], "score": 8.0, "comment": "具体而自然。", "fields": [next(iter(item["fields"]))], "note_ids": []} for c in payload["rubric"]],
            "issues": [], "preserve": ["明确的场景。"], "comparison": "与同组其他方案在开头不同。",
        } for item in payload["items"]],
        "limitations": ["模型评价需要人工校准。"],
    }


def block(review, item, criterion="hook", field="title"):
    review["issues"] = [{
        "criterion": criterion, "severity": "blocking", "field": field,
        "quote": item["fields"][field], "problem": "存在无法支撑的承诺。",
        "revision_instruction": "收窄承诺到已有材料。", "note_ids": [], "related_item_ids": [],
    }]


class JudgeCompletion:
    def __init__(self, transform=None):
        self.calls, self.transform = [], transform

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        payload = prompt["input"]
        result = valid_review(payload)
        if self.transform:
            self.transform(result, payload, len(self.calls))
        return json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("count", [1, 3, 5])
def test_scores_select_one_per_idea_and_ties_are_stable(count):
    args = inputs(count)
    before = copy.deepcopy(args)
    run, meta = run_title_review(**args, complete=JudgeCompletion())
    assert args == before
    assert run.output.selected_candidate_ids == {f"idea_{i + 1}": f"title_{i}_0" for i in range(count)}
    assert meta["downstream_ready"] is True
    assert verified_title_review(run.output, prepare_title_review_input(**args)) == run.output


def test_best_passing_candidate_wins_even_if_another_has_high_score_and_blocker():
    def transform(result, payload, _):
        for score in result["reviews"][0]["scores"]:
            score["score"] = 10
        block(result["reviews"][0], payload["items"][0])
        for score in result["reviews"][2]["scores"]:
            score["score"] = 9
    complete = JudgeCompletion(transform)
    run, meta = run_title_review(**inputs(), complete=complete)
    assert run.output.selected_candidate_ids == {"idea_1": "title_0_2"}
    assert run.output.review.status == "revision_required"
    assert run.output.status == "passed" and meta["downstream_ready"]
    assert len(complete.calls) == 1


def test_no_passing_candidate_preserves_requested_idea_count_without_retry():
    def transform(result, payload, _):
        for review, item in zip(result["reviews"][:5], payload["items"][:5]):
            block(review, item)
    complete = JudgeCompletion(transform)
    run, meta = run_title_review(**inputs(3), complete=complete)
    assert run.output.unresolved_idea_ids == ["idea_1"]
    assert len(run.output.review.reviews) == 15
    assert len(run.output.selected_candidate_ids) == 2
    assert not meta["downstream_ready"]
    assert len(complete.calls) == 1


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown_field", "unknown_score", "duplicate_score", "false_quote", "unknown_note", "self_ref", "old_fingerprint", "old_revision", "forged_pass", "low_no_issue", "low_suggestion"])
def test_invalid_judgements_repair_once(fault):
    def transform(result, payload, call):
        if call > 1:
            return
        review = result["reviews"][0]
        score = review["scores"][0]
        if fault == "missing":
            result["reviews"].pop()
        elif fault == "duplicate":
            result["reviews"][1] = copy.deepcopy(review)
        elif fault == "unknown_field":
            score["fields"] = ["private-data"]
        elif fault == "unknown_score":
            score["criterion"] = "private-data"
        elif fault == "duplicate_score":
            review["scores"][1] = copy.deepcopy(score)
        elif fault == "unknown_note":
            score["note_ids"] = ["private-data"]
        elif fault == "old_fingerprint":
            result["input_fingerprint"] = "0" * 64
        elif fault == "old_revision":
            result["revision"] += 1
        elif fault == "forged_pass":
            review["passed"] = True
        elif fault.startswith("low"):
            score["score"] = 2
            if fault == "low_suggestion":
                block(review, payload["items"][0])
                review["issues"][0]["severity"] = "suggestion"
        else:
            block(review, payload["items"][0])
            if fault == "false_quote":
                review["issues"][0]["quote"] = "private-data"
            else:
                review["issues"][0]["related_item_ids"] = [review["item_id"]]
    complete = JudgeCompletion(transform)
    run, _ = run_title_review(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    assert "private-data" not in str(run.attempts)
    assert "private-data" not in complete.calls[1]["user_prompt"]


@pytest.mark.parametrize("score", [True, "8", float("nan"), float("inf"), -1, 11])
def test_invalid_numeric_scores_exhaust_bounded_repairs(score):
    complete = JudgeCompletion(lambda result, _, __: result["reviews"][0]["scores"][0].update(score=score))
    with pytest.raises(AgentRunError):
        run_title_review(**inputs(), complete=complete)
    assert len(complete.calls) == 2


@pytest.mark.parametrize("change", ["selection", "scores", "title", "revision"])
def test_handoff_rejects_tampered_selection_or_stale_review(change):
    args = inputs()
    run, _ = run_title_review(**args, complete=JudgeCompletion())
    stored = run.output.model_dump(mode="json")
    if change == "selection":
        stored["selected_candidate_ids"]["idea_1"] = "title_unknown"
    elif change == "scores":
        stored["review"]["reviews"][0]["quality"]["overall_score"] = 10
    elif change == "title":
        args["title_candidates"]["ideas"][0]["candidates"][0]["title"] = "改过的标题"
    else:
        args["title_revision"] += 1
    with pytest.raises(CreativeError) as caught:
        verified_title_review(stored, prepare_title_review_input(**args))
    assert caught.value.code == "title_review_not_approved"


def test_cancel_and_transport_error_never_repeat_requests():
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise CreativeError("读取超时", "ai_read_timeout")
    with pytest.raises(CreativeError):
        run_title_review(**inputs(), complete=fail)
    assert len(calls) == 1
    def cancel(_):
        raise RuntimeError("已取消")
    with pytest.raises(RuntimeError, match="已取消"):
        run_title_review(**inputs(), complete=fail, on_progress=cancel)
    assert len(calls) == 1
