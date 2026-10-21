import copy
import json

import pytest

from backend.app.schemas.creative_agents import CopyVariants
from backend.app.services.creative_agents.copy_generation import COPY_AGENT, prepare_copy_input
from backend.app.services.creative_agents.revision import RequestBudget, ensure_targeted_revision, revision_completion
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_orchestrator import AgentValidationError, run_agent
from test_creative_copy_agent import inputs, valid_copies


def drafts():
    payload = prepare_copy_input(**inputs())
    return payload, valid_copies(payload)


def test_targeted_revision_can_change_only_failed_version():
    _, old = drafts()
    new = copy.deepcopy(old)
    new["ideas"][0]["variants"][0]["body"] = "修改后的纯正文"
    ensure_targeted_revision("copy", old, new, ["copy_0_0"])
    with pytest.raises(AgentValidationError):
        ensure_targeted_revision("copy", old, new, ["copy_0_1"])


@pytest.mark.parametrize("change", ["rename", "remove", "move", "duplicate", "unknown_target"])
def test_revision_cannot_hide_failed_items_or_change_binding(change):
    _, old = drafts()
    new = copy.deepcopy(old)
    targets = ["copy_0_0"]
    if change == "rename":
        new["ideas"][0]["variants"][0]["variant_id"] = "copy_other"
    elif change == "remove":
        new["ideas"][0]["variants"].pop()
    elif change == "move":
        new["ideas"][0]["candidate_id"] = "title_other"
    elif change == "duplicate":
        new["ideas"][0]["variants"][1] = new["ideas"][0]["variants"][0]
    else:
        targets = ["copy_unknown"]
    with pytest.raises(AgentValidationError):
        ensure_targeted_revision("copy", old, new, targets)


def test_out_of_scope_revision_uses_one_existing_structural_repair():
    payload, old = drafts()
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        new = copy.deepcopy(old)
        new["ideas"][0]["variants"][0]["body"] = "针对问题修正后的正文"
        if len(calls) == 1:
            new["ideas"][0]["variants"][1]["body"] = "擅自改动已经通过的正文"
        return json.dumps(new, ensure_ascii=False)
    wrapped = revision_completion(complete, stage="copy", previous=old, target_ids=["copy_0_0"], feedback={"issues": ["读者价值不具体"]})
    run = run_agent(COPY_AGENT, input_payload=payload, output_model=CopyVariants, complete=wrapped)
    assert len(calls) == 2
    prompt, _ = json.JSONDecoder().raw_decode(calls[1]["user_prompt"])
    assert prompt["quality_revision"]["target_ids"] == ["copy_0_0"]
    assert "未列入返工范围" in calls[1]["user_prompt"]
    assert run.output.ideas[0].variants[1].body == old["ideas"][0]["variants"][1]["body"]


def test_request_budget_counts_attempted_calls_and_never_retries_transport():
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise CreativeError("模型超时", "ai_read_timeout")
    budget = RequestBudget(fail, max_requests=1)
    with pytest.raises(CreativeError) as error:
        budget()
    assert error.value.code == "ai_read_timeout"
    with pytest.raises(CreativeError) as error:
        budget()
    assert error.value.code == "creative_budget_exhausted"
    assert budget.request_count == len(calls) == 1


def test_elapsed_budget_and_cancellation_prevent_request():
    calls = []
    times = iter([0, 11])
    budget = RequestBudget(lambda **_: calls.append(1), max_seconds=10, clock=lambda: next(times))
    with pytest.raises(CreativeError, match="上限"):
        budget()
    def cancel():
        raise RuntimeError("停止")
    budget = RequestBudget(lambda **_: calls.append(1), before_request=cancel)
    with pytest.raises(RuntimeError, match="停止"):
        budget()
    assert not calls
