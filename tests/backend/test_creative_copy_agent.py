import json

import pytest

from backend.app.services.creative_agents.copy_generation import prepare_copy_input, run_copy_generation
from backend.app.services.creative_agents.title_review import run_title_review
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_orchestrator import AgentRunError
from test_creative_title_review import JudgeCompletion, inputs as review_inputs


def inputs(count=1, variants=2):
    args = review_inputs(count)
    run, _ = run_title_review(**args, complete=JudgeCompletion())
    return {**args, "title_review": run.output.model_dump(mode="json"), "variant_count": variants}


def valid_copies(payload):
    topics = {t["idea_id"]: t for t in payload["topic_brief"]["ideas"]}
    return {
        "input_fingerprint": payload["input_fingerprint"],
        "ideas": [{
            "idea_id": title["idea_id"], "candidate_id": title["candidate_id"],
            "variants": [{
                "variant_id": f"copy_{i}_{j}", "label": f"观察角度{j}",
                "body": f"米色搭浅蓝，颜色轻一些，轮廓就更清晰。\n配色观察{i}·{j}。",
                "tags": ["服装灵感", "AI设计"], "covered_points": topics[title["idea_id"]]["must_cover"],
                "source_note_ids": topics[title["idea_id"]]["source_note_ids"],
            } for j in range(payload["variant_count"])],
        } for i, title in enumerate(payload["locked_titles"])],
        "limitations": ["需人工确认表达风格。"],
    }


class CopyCompletion:
    def __init__(self, transform=None):
        self.calls, self.transform = [], transform

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        result = valid_copies(prompt["input"])
        if self.transform:
            self.transform(result, len(self.calls))
        return json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 2)])
def test_copy_has_locked_titles_and_exact_variant_count(count, variants):
    run, meta = run_copy_generation(**inputs(count, variants), complete=CopyCompletion())
    assert len(run.output.ideas) == count
    assert all(len(i.variants) == variants for i in run.output.ideas)
    assert meta["quality_review_status"] == "not_reviewed"


@pytest.mark.parametrize("fault", ["wrong_title", "missing_idea", "missing_variant", "duplicate_id", "duplicate_body", "editorial", "placeholder", "tags_in_body", "duplicate_tag", "foreign_source", "missing_point", "fingerprint"])
def test_copy_content_and_handoff_validation_repairs_once(fault):
    def transform(result, call):
        if call > 1:
            return
        idea = result["ideas"][0]
        variant = idea["variants"][0]
        if fault == "wrong_title":
            idea["candidate_id"] = "title_unselected"
        elif fault == "missing_idea":
            result["ideas"] = []
        elif fault == "missing_variant":
            idea["variants"].pop()
        elif fault == "duplicate_id":
            idea["variants"][1]["variant_id"] = variant["variant_id"]
        elif fault == "duplicate_body":
            idea["variants"][1]["body"] = variant["body"]
        elif fault == "editorial":
            variant["body"] = "配图建议：使用浅蓝底色"
        elif fault == "placeholder":
            variant["body"] = "【请填写案例】"
        elif fault == "tags_in_body":
            variant["body"] += "\n#服装灵感"
        elif fault == "duplicate_tag":
            variant["tags"] = ["AI", "ＡＩ"]
        elif fault == "foreign_source":
            variant["source_note_ids"] = ["sample-0", "unknown"]
        elif fault == "missing_point":
            variant["covered_points"].pop()
        else:
            result["input_fingerprint"] = "0" * 64
    complete = CopyCompletion(transform)
    run, _ = run_copy_generation(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    assert [a.status for a in run.attempts] == ["failed", "completed"]


def test_changed_title_requires_new_review_before_copy():
    args, complete = inputs(), CopyCompletion()
    args["title_candidates"]["ideas"][0]["candidates"][0]["title"] = "新的标题"
    with pytest.raises(CreativeError):
        run_copy_generation(**args, complete=complete)
    assert not complete.calls


@pytest.mark.parametrize("count", [1, 4, True, "2", 2.0])
def test_invalid_variant_count_is_not_coerced(count):
    args = inputs()
    args["variant_count"] = count
    with pytest.raises(CreativeError):
        prepare_copy_input(**args)


def test_format_failures_bounded_but_transport_failure_not_repeated():
    complete = CopyCompletion(lambda result, _: result.update(ideas=[]))
    with pytest.raises(AgentRunError):
        run_copy_generation(**inputs(), complete=complete)
    assert len(complete.calls) == 2
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise CreativeError("读取超时", "ai_read_timeout")
    with pytest.raises(CreativeError):
        run_copy_generation(**inputs(), complete=fail)
    assert len(calls) == 1
