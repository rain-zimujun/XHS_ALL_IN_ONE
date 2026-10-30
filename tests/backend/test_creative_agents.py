import json

import pytest
from pydantic import BaseModel, Field

from backend.app.services.creative_orchestrator import AgentRunError, AgentSpec, run_agent
from backend.app.services.creative_pipeline import begin_stage, complete_stage, fail_stage, new_pipeline_state, validate_pipeline_state
from backend.app.services.creative_quality import QualityScore, assess_review, rubric_for, weighted_score


class TinyOutput(BaseModel):
    answer: str = Field(min_length=1)


def test_quality_rubric_is_fixed_and_weighted():
    criteria = rubric_for("title")
    assert round(sum(item.weight for item in criteria), 2) == 1.0
    scores = [QualityScore(criterion=item.key, score=8, comment="够具体") for item in criteria]
    assert weighted_score("title", scores) == 8.0
    review = assess_review("title", scores)
    assert review.passed is True
    low = assess_review("title", [QualityScore(criterion="hook", score=3, comment="太平")])
    assert low.passed is False
    assert low.blocking_issues


def test_quality_review_cannot_claim_pass_with_blocking_issues():
    with pytest.raises(ValueError):
        from backend.app.services.creative_quality import QualityReview
        QualityReview(kind="copy", scores=[QualityScore(criterion="opening", score=8, comment="ok")], overall_score=8, passed=True, blocking_issues=["编辑说明混入正文"])


def test_pipeline_state_is_serializable_and_tracks_attempts():
    from backend.app.services.creative_agents.revision import MAX_QUALITY_REVISIONS
    state = new_pipeline_state()
    # The recorded budget must match what the workflow actually enforces.
    assert state["max_quality_revisions"] == MAX_QUALITY_REVISIONS
    assert validate_pipeline_state(state).current_stage is None
    state = begin_stage(state, "title")
    assert state["current_stage"] == "title"
    assert state["stages"]["title"]["attempts"][0]["attempt"] == 1
    state = complete_stage(state, "title", {"candidates": ["标题一"]})
    parsed = validate_pipeline_state(state)
    assert parsed.stages["title"].status == "completed"
    assert parsed.stages["title"].output == {"candidates": ["标题一"]}
    state = begin_stage(state, "title")
    state = fail_stage(state, "title", "评审未通过")
    parsed = validate_pipeline_state(state)
    assert parsed.stages["title"].attempts[-1].attempt == 2
    assert parsed.stages["title"].attempts[-1].error == "评审未通过"


def test_pipeline_requires_stage_start_before_completion():
    with pytest.raises(ValueError):
        complete_stage(new_pipeline_state(), "copy", {"body": "正文"})


def test_orchestrator_validates_json_and_retries_with_targeted_context():
    prompts = []
    responses = iter(["不是 JSON", json.dumps({"answer": "通过"}, ensure_ascii=False)])
    def complete(**kwargs):
        prompts.append(kwargs)
        return next(responses)
    run = run_agent(
        AgentSpec("title", "标题代理", "只返回 JSON", max_attempts=2),
        input_payload={"topic": "AI 服装"}, output_model=TinyOutput, complete=complete,
    )
    assert run.output.answer == "通过"
    assert len(run.attempts) == 2
    assert "上一次输出未通过" in prompts[1]["user_prompt"]
    assert '"topic": "AI 服装"' in prompts[0]["user_prompt"]
    assert prompts[0]["response_format"] == prompts[1]["response_format"]
    response_format = prompts[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == json.loads(prompts[0]["user_prompt"])["output_schema"]


def test_orchestrator_returns_trace_when_attempts_exhausted():
    def complete(**_kwargs):
        return json.dumps({"wrong": "field"})
    with pytest.raises(AgentRunError) as error:
        run_agent(AgentSpec("copy", "文案代理", "只返回 JSON", max_attempts=2), input_payload={}, output_model=TinyOutput, complete=complete)
    assert error.value.stage == "copy"
    assert len(error.value.attempts) == 2
    assert all(item.status == "failed" for item in error.value.attempts)


def test_orchestrator_does_not_retry_transport_errors():
    calls = []
    def complete(**_kwargs):
        calls.append(1)
        raise RuntimeError("模型连接失败")
    with pytest.raises(RuntimeError, match="模型连接失败"):
        run_agent(AgentSpec("copy", "文案代理", "只返回 JSON", max_attempts=2), input_payload={}, output_model=TinyOutput, complete=complete)
    assert len(calls) == 1


def test_review_requires_all_criteria_before_passing():
    review = assess_review("title", [QualityScore(criterion="hook", score=10, comment="好")])
    assert review.passed is False
    assert any("缺少评分项" in issue for issue in review.blocking_issues)


def test_orchestrator_emits_stage_progress():
    stages = []
    run_agent(AgentSpec("sample_analysis", "样本分析代理", "只返回 JSON"), input_payload={}, output_model=TinyOutput, complete=lambda **_: '{"answer":"ok"}', on_progress=stages.append)
    assert stages == ["样本分析代理（第 1/2 次）"]
