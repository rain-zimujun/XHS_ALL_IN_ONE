import copy
import json

import pytest
import requests

from backend.app.services.creative_agents.title_generation import analyze_titles, prepare_title_input, run_title_generation
from backend.app.services.creative_agents.topic_review import run_topic_review
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_orchestrator import AgentRunError
from test_creative_topic_review import FakeCompletion as TopicJudge, inputs as topic_inputs, model_db


def inputs(count=1, candidate_count=5):
    args = topic_inputs(count)
    run, _ = run_topic_review(**args, complete=TopicJudge())
    return {**args, "topic_review": run.output.model_dump(mode="json"), "candidate_count": candidate_count}


def valid_titles(payload):
    return {
        "topic_revision": payload["topic_revision"], "input_fingerprint": payload["input_fingerprint"],
        "ideas": [{
            "idea_id": idea["idea_id"],
            "candidates": [{
                "candidate_id": f"title_{i}_{j}", "title": f"配色{i}，换个角度看{j}",
                "angle": f"切入角度{j}", "reader_value": "获得具体配色参考。",
                "promise": "提供配色观察。", "support_points": ["具体颜色与轮廓"],
                "rationale": "以具体观察吸引目标读者。",
            } for j in range(payload["candidate_count"])],
        } for i, idea in enumerate(payload["topic_brief"]["ideas"])],
        "limitations": ["仅参考文字样本，实际吸引力需人工确认。"],
    }


class TitleCompletion:
    def __init__(self, transform=None):
        self.calls, self.transform = [], transform

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        result = valid_titles(prompt["input"])
        if self.transform:
            self.transform(result, len(self.calls))
        return json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("count,candidates", [(1, 5), (3, 6), (5, 8)])
def test_approved_topics_generate_all_candidates_without_mutation(count, candidates):
    args, complete = inputs(count, candidates), TitleCompletion()
    before = copy.deepcopy(args)
    run, meta = run_title_generation(**args, complete=complete)
    assert args == before
    assert len(run.output.ideas) == count
    assert all(len(idea.candidates) == candidates for idea in run.output.ideas)
    assert meta["quality_review_status"] == "not_reviewed"
    assert meta["request_count"] == 1
    assert "topic_review" not in json.loads(complete.calls[0]["user_prompt"])["input"]


@pytest.mark.parametrize("change", ["revision", "content", "requirements", "status", "score", "approved", "blocking"])
def test_stale_or_tampered_review_prevents_title_request(change):
    args, complete = inputs(), TitleCompletion()
    if change == "revision":
        args["topic_revision"] += 1
    elif change == "content":
        args["topic_brief"]["ideas"][0]["reader_promise"] = "新的承诺"
    elif change == "requirements":
        args["requirements"] += "新的用户要求"
    elif change == "status":
        args["topic_review"]["status"] = "revision_required"
    elif change == "score":
        args["topic_review"]["reviews"][0]["quality"]["overall_score"] = 10
    elif change == "approved":
        args["topic_review"]["approved_idea_ids"] = []
    else:
        args["topic_review"]["reviews"][0]["judgement"]["issues"] = [{
            "criterion": "reader_value", "severity": "blocking", "field": "reader_promise",
            "quote": args["topic_brief"]["ideas"][0]["reader_promise"],
            "problem": "承诺缺少依据。", "revision_instruction": "收窄承诺。",
        }]
    with pytest.raises(CreativeError) as error:
        run_title_generation(**args, complete=complete)
    assert error.value.code == "topic_review_not_approved"
    assert not complete.calls


@pytest.mark.parametrize("value", [4, 9, True, 5.0, "5"])
def test_invalid_candidate_budget_fails_without_request(value):
    args, complete = inputs(), TitleCompletion()
    args["candidate_count"] = value
    with pytest.raises(CreativeError):
        run_title_generation(**args, complete=complete)
    assert not complete.calls


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate_id", "duplicate_title", "wrong_idea", "old", "angles", "long", "editorial", "newline"])
def test_candidate_errors_repair_once(fault):
    def transform(result, call):
        if call > 1:
            return
        group = result["ideas"][0]
        first, second = group["candidates"][:2]
        if fault == "missing":
            group["candidates"].pop()
        elif fault == "extra":
            result["private-secret"] = "private-secret"
        elif fault == "duplicate_id":
            second["candidate_id"] = first["candidate_id"]
        elif fault == "duplicate_title":
            second["title"] = " " + first["title"].replace("0", "０")
        elif fault == "wrong_idea":
            group["idea_id"] = "idea_missing"
        elif fault == "old":
            result["input_fingerprint"] = "0" * 64
        elif fault == "angles":
            for c in group["candidates"]:
                c["angle"] = "同一角度"
        elif fault == "long":
            first["title"] = "长" * 21
        elif fault == "editorial":
            first["title"] = "标题：可以直接复制"
        else:
            first["title"] = "配色\n观察"
    complete = TitleCompletion(transform)
    run, _ = run_title_generation(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    assert [trace.status for trace in run.attempts] == ["failed", "completed"]
    assert "private-secret" not in complete.calls[1]["user_prompt"]


def test_structural_repairs_bounded_and_cancellation_does_not_send():
    complete = TitleCompletion(lambda result, _: result.update(ideas=[]))
    with pytest.raises(AgentRunError):
        run_title_generation(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    def cancel(_):
        raise RuntimeError("已取消")
    with pytest.raises(RuntimeError, match="已取消"):
        run_title_generation(**inputs(), complete=complete, on_progress=cancel)
    assert len(complete.calls) == 2


def test_transport_ownership_timeout_and_no_retry(model_db):
    db, owner, other, model = model_db
    complete = TitleCompletion()
    class Client:
        def _complete(self, **kwargs):
            return complete(**kwargs)
    analyze_titles(db, owner, **inputs(), options={"model_id": model}, client_factory=Client)
    assert complete.calls[0]["timeout"] == (10, 180)
    assert "fixture-only-key" not in complete.calls[0]["user_prompt"]
    with pytest.raises(CreativeError, match="自己的文字模型"):
        analyze_titles(db, other, **inputs(), options={"model_id": model}, client_factory=Client)
    calls = []
    class TimeoutClient:
        def _complete(self, **kwargs):
            calls.append(kwargs)
            raise requests.ReadTimeout("private-error")
    with pytest.raises(CreativeError) as caught:
        analyze_titles(db, owner, **inputs(), options={"model_id": model}, client_factory=TimeoutClient)
    assert caught.value.code == "ai_read_timeout"
    assert len(calls) == 1


def test_title_length_budget_is_bound_to_input_fingerprint():
    args = inputs()
    original = prepare_title_input(**args)
    longer = prepare_title_input(**args, title_max_length=30)
    assert original["input_fingerprint"] != longer["input_fingerprint"]


def test_joined_emoji_is_valid_title_text():
    complete = TitleCompletion(lambda result, _: result["ideas"][0]["candidates"][0].update(title="👩‍🎨换个配色思路"))
    run, _ = run_title_generation(**inputs(), complete=complete)
    assert run.output.ideas[0].candidates[0].title == "👩‍🎨换个配色思路"
