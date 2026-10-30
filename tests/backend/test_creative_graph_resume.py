import copy
import json
from types import SimpleNamespace

import pytest
import requests

from backend.app.models import Task
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.workflow import LangGraphWorkflow, estimate_graph_request_budget
from backend.app.services.creative_agents.revision import RequestBudget
from test_creative import env, collect_for_analysis
from test_creative_graph_workflow import GraphCompletion, workflow
from test_creative_workflow import install_client


ROLES = ["SampleAnalysisResponse", "SampleReviewResponse", "AudienceGroupingResponse", "AudienceNeedsResponse", "TopicResponse", "TopicReviewResponse", "TitleResponse",
         "TitleReviewResponse", "CopyResponse", "CopyReviewResponse", "ImageResponse", "FinalReviewResponse"]
EXECUTION_ROLES = ROLES[:-2] + ["ImageResponse"] * 2 + ["FinalReviewResponse"] * 2


@pytest.mark.parametrize("failed_role", ROLES)
def test_resume_starts_at_exact_agent_preserving_every_upstream_artifact(failed_role):
    model = GraphCompletion()
    def complete(**kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == failed_role:
            raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)
    original = workflow(complete)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    # Fixtures use default legacy; production requests explicitly set langgraph.
    task.payload["analysis"]["workflow"] = "langgraph"
    task.payload["graph_state"]["options"]["workflow"] = "langgraph"
    before = copy.deepcopy(task.payload)
    restored, role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=role)
    result, metadata = resumed.run()
    assert result["ideas"]
    assert recovery.wire_calls[0]["response_format"]["json_schema"]["name"] == failed_role
    expected_calls = len(EXECUTION_ROLES) - EXECUTION_ROLES.index(failed_role)
    if EXECUTION_ROLES.index(failed_role) <= EXECUTION_ROLES.index("AudienceNeedsResponse"):
        expected_calls += 1  # Reader-needs analysis now runs once per audience group.
    assert len(recovery.calls) == expected_calls
    assert metadata["request_count"] == saved["pipeline"]["request_count"] + len(recovery.calls)
    for key in ("sample_analysis", "sample_review", "audience", "topic", "topic_review", "title", "title_review", "copy", "copy_review", "image", "final_review"):
        if saved[key] is not None:
            assert resumed.state[key] == saved[key]
    assert task.payload == before


def test_paused_snapshot_resumes_without_an_error_whitelist():
    model = GraphCompletion()
    def complete(**kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "TitleResponse":
            raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)
    original = workflow(complete)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    # A user pause keeps the same snapshot shape but is not a failure.
    pipeline.update(status="paused", error_code="creative_paused")
    stage = pipeline["current_stage"]
    pipeline["stages"][stage]["status"] = "paused"
    task = SimpleNamespace(status="paused", error_type=None, payload={
        "analysis": copy.deepcopy(saved["options"]), "search": saved["search"],
        "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)
    assert role == "title"
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"],
                                options=restored["options"], complete=GraphCompletion(),
                                resume_state=restored, resume_role=role)
    result, _ = resumed.run()
    assert result["ideas"]


def test_paused_snapshot_without_a_paused_stage_is_rejected():
    model = GraphCompletion()
    def complete(**kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "TitleResponse":
            raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)
    original = workflow(complete)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    task = SimpleNamespace(status="paused", error_type=None, payload={
        "analysis": copy.deepcopy(saved["options"]), "search": saved["search"],
        "sources": saved["sources"], "graph_state": saved,
    })
    with pytest.raises(CreativeError):
        prepare_resume(task)


def test_resume_api_preserves_original_and_is_idempotent_with_ownership(env):
    failing = GraphCompletion(timeout_stage="title_review")
    install_client(failing)
    original = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2})
    assert original["status"] == "failed"
    assert original["resume_available"] is True
    from backend.app.services.creative_retries import NETWORK_ATTEMPTS
    assert failing.counts["title_review"] == NETWORK_ATTEMPTS
    failed_pipeline = copy.deepcopy(original["payload"]["pipeline"])
    failed_network = failed_pipeline["stages"]["title"]["metadata"]["network_attempts"]
    assert len(failed_network) == NETWORK_ATTEMPTS
    assert [r["will_retry"] for r in failed_network] == [True] * (NETWORK_ATTEMPTS - 1) + [False]
    assert env.client.post(f"/api/xhs/creative/runs/{original['id']}/resume", headers=env.other_headers).status_code == 404
    recovery = GraphCompletion()
    install_client(recovery)
    response = env.client.post(f"/api/xhs/creative/runs/{original['id']}/resume", headers=env.headers)
    assert response.status_code == 202, response.text
    child_id = response.json()["id"]
    child = env.client.get(f"/api/xhs/creative/runs/{child_id}", headers=env.headers).json()
    assert child["status"] == "completed", child["payload"].get("error")
    assert child["parent_task_id"] == original["id"]
    assert child["payload"]["resume"]["role"] == "title_review"
    assert child["payload"]["sources"] == original["payload"]["sources"]
    assert child["payload"]["pipeline"]["request_count"] == 21
    assert child["payload"]["graph_state"]["pipeline"] == child["payload"]["pipeline"]
    assert len(recovery.calls) == 7
    assert recovery.wire_calls[0]["response_format"]["json_schema"]["name"] == "TitleReviewResponse"
    old = env.client.get(f"/api/xhs/creative/runs/{original['id']}", headers=env.headers).json()
    assert old["status"] == "failed" and old["payload"]["pipeline"] == failed_pipeline
    assert env.client.post(f"/api/xhs/creative/runs/{original['id']}/resume", headers=env.headers).json()["id"] == child_id
    assert len(recovery.calls) == 7
    assert "fixture-key-private" not in json.dumps(child)


def test_unknown_copy_source_is_rejected_by_bound_schema(env):
    invalid_model = GraphCompletion()

    def invalid_copy(**kwargs):
        raw = invalid_model(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "CopyResponse":
            value = json.loads(raw)
            value["ideas"][0]["variants"][0]["source_note_ids"] = ["sample-0", "unknown-note"]
            return json.dumps(value, ensure_ascii=False)
        return raw

    install_client(invalid_copy)
    original = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2})
    assert original["status"] == "failed"
    assert original["error_type"] == "creative_schema_invalid"
    # Source IDs are now enums in the actual provider request, so an unknown
    # ID is rejected before business handoff validation and is not resumable.
    assert original["resume_available"] is False


def test_transient_failure_recovers_in_place_without_repeating_upstream(env):
    model, failures = GraphCompletion(), []
    def complete(**kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "TitleReviewResponse" and len(failures) < 3:
            failures.append(1)
            raise requests.ReadTimeout("private-provider-details")
        return model(**kwargs)
    install_client(complete)
    result = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2})
    assert result["status"] == "completed"
    assert result["payload"]["pipeline"]["request_count"] == 18
    assert len(model.calls) == 15
    stage = result["payload"]["pipeline"]["stages"]["title"]
    assert len(stage["metadata"]["network_attempts"]) == 3
    assert stage["history"][-1]["metadata"]["request_count"] == 4
    assert "network_attempts" not in stage["history"][0]["metadata"]
    assert "private-provider-details" not in json.dumps(result)


@pytest.mark.parametrize("owner", ["topic", "title", "copy", "image"])
def test_resume_during_targeted_quality_revision_keeps_revision_scope(owner):
    schema = {"topic": "TopicResponse", "title": "TitleResponse", "copy": "CopyResponse", "image": "ImageResponse"}[owner]
    model = GraphCompletion(fail_stage="final_review", final_owner=owner)
    def complete(**kwargs):
        p, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        if kwargs["response_format"]["json_schema"]["name"] == schema and p.get("quality_revision"):
            raise CreativeError("模型服务错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)
    original = workflow(complete)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, role = prepare_resume(task)
    model.fail_stage = None
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=model, resume_state=restored, resume_role=role)
    result, _ = resumed.run()
    assert result["ideas"]
    assert role == owner
    assert resumed.state["pipeline"]["stages"][owner]["quality_revisions"] == 1


def test_old_snapshot_without_role_still_resumes_title_judge():
    model = GraphCompletion()
    def complete(**kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "TitleReviewResponse":
            raise CreativeError("HTTP 524", "ai_http_error")
        return model(**kwargs)
    w = workflow(complete)
    with pytest.raises(CreativeError):
        w.run()
    saved = copy.deepcopy(w.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    saved["pipeline"]["stages"]["title"]["metadata"].pop("failed_role")
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    _, role = prepare_resume(task)
    assert role == "title_review"


@pytest.mark.parametrize("failure", [
    "observations[0].quote：observations.quote 必须来自对应样本标题或已提供正文片段",
    "topic_patterns[0].evidence[0].quote：evidence.quote 必须来自对应标题或正文片段",
])
def test_sample_source_quote_failure_can_resume_sample_analysis_only(failure):
    saved = copy.deepcopy(workflow(GraphCompletion(), count=1, variants=2).state)
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="sample_analysis", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["sample_analysis"]
    stage.update(status="failed", error=failure)
    stage["metadata"].update(failed_role="sample_analysis", failed_attempts=[
        {"attempt": 1, "error": failure}, {"attempt": 2, "error": failure},
    ])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)

    assert role == "sample_analysis"
    assert restored["sample_analysis"] is None
    assert restored["pipeline"]["stages"]["sample_analysis"]["status"] == "pending"

    unrelated = copy.deepcopy(task)
    unrelated.payload["graph_state"]["pipeline"]["stages"]["sample_analysis"]["metadata"]["failed_attempts"][-1]["error"] = "任意来源或数量错误"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(unrelated)


def test_sample_rejected_claim_reference_failure_can_resume_sample_review_only():
    completed = workflow(GraphCompletion(), count=1, variants=2)
    completed.run()
    saved = copy.deepcopy(completed.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="sample_analysis", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["sample_analysis"]
    failure = "claim_001_003 判为无依据时 note_ids 必须为空"
    stage.update(status="failed", error=failure)
    stage["metadata"].update(failed_role="sample_review", failed_attempts=[
        {"attempt": 1, "error": failure}, {"attempt": 2, "error": failure},
    ])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)

    assert role == "sample_review"
    assert restored["sample_analysis"] is not None
    assert restored["pipeline"]["stages"]["sample_analysis"]["status"] == "pending"


def test_audience_claim_source_binding_failure_can_resume_audience_only():
    completed = workflow(GraphCompletion(), count=1, variants=2)
    completed.run()
    saved = copy.deepcopy(completed.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="audience", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["audience"]
    failure = "segments 分组校验失败；缺少受众词=[]；错误=[{'field': 'segments[0]', 'reason': 'note_id 必须属于本条 evidence 引用的获批 claim_id'}]；最多8组"
    stage.update(status="failed", error=failure)
    stage["metadata"].update(failed_role="audience", failed_attempts=[
        {"attempt": 1, "error": failure}, {"attempt": 2, "error": failure},
    ])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)

    assert role == "audience"
    assert restored["sample_analysis"] is not None
    assert restored["sample_review"] is not None


def test_legacy_snapshot_without_sample_review_is_migrated_and_reviewed_before_audience():
    model = GraphCompletion()
    completed = workflow(model, count=1, variants=2)
    completed.run()
    saved = copy.deepcopy(completed.state)
    saved.pop("sample_review")
    saved["pipeline"]["stages"]["sample_analysis"]["metadata"].pop("sample_review_batches", None)
    saved["pipeline"]["stages"]["sample_analysis"]["metadata"]["quality_review_status"] = "not_reviewed"
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    saved["pipeline"].update(status="failed", current_stage="audience", error_code="ai_http_error")
    audience = saved["pipeline"]["stages"]["audience"]
    audience.update(status="failed", error="旧任务失败")
    audience["metadata"]["failed_role"] = "audience"
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=role)
    result, _ = resumed.run()

    assert role == "sample_review"
    assert result["ideas"]
    assert recovery.counts["SampleAnalysis"] == 0
    assert recovery.counts["sample_review"] == 1


def test_restart_interrupted_snapshot_resumes_exact_incomplete_batch():
    model = GraphCompletion()
    w = workflow(model, count=3, variants=3)
    # Build a realistic resumable snapshot with an in-progress judge and two
    # validated cached batches; recovery must not re-run any earlier agent.
    saved = copy.deepcopy(w.state)
    saved["options"]["workflow"] = "langgraph"
    saved["pipeline"].update(status="failed", current_stage="title", error_code="creative_interrupted")
    saved["pipeline"]["request_count"] = 40
    saved["pipeline"]["max_requests"] = estimate_graph_request_budget(len(saved["sources"]), saved["options"])
    stage = saved["pipeline"]["stages"]["title"]
    stage.update(status="failed", output=saved["title"])
    stage["metadata"].update(failed_role="title_review")
    task_payload = {
        "analysis": saved["options"], "search": saved["search"],
        "sources": saved["sources"], "graph_state": saved, "pipeline": saved["pipeline"],
    }
    task = SimpleNamespace(status="failed", error_type="creative_interrupted", payload=task_payload)

    # Incomplete/mismatched snapshots and exhausted budgets remain forbidden.
    broken = copy.deepcopy(task)
    broken.payload["pipeline"]["error_code"] = "ai_http_error"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(broken)
    broken = copy.deepcopy(task)
    broken.payload["graph_state"]["pipeline"]["max_requests"] = 512
    broken.payload["graph_state"]["pipeline"]["request_count"] = 512
    broken.payload["pipeline"]["request_count"] = broken.payload["graph_state"]["pipeline"]["request_count"]
    broken.payload["pipeline"]["max_requests"] = 512
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(broken)

    restored, role = prepare_resume(task)
    assert role == "title_review"
    assert restored["pipeline"]["request_count"] == 40
    assert restored["pipeline"]["max_requests"] > 72
    assert restored["pipeline"]["error_code"] is None


def test_title_revision_scope_failure_is_resumable_but_other_handoff_failures_are_not():
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    result, _ = w.run()
    assert result["ideas"]
    saved = copy.deepcopy(w.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="title", error_code="creative_handoff_invalid")
    title_stage = pipeline["stages"]["title"]
    title_stage.update(status="failed", error="未列入返工范围的已通过内容必须原样保留")
    title_stage["metadata"].update(
        failed_role="title",
        failed_attempts=[{"attempt": 1, "status": "creative_handoff_invalid", "error": "未列入返工范围的已通过内容必须原样保留"}],
    )
    saved["revision_plan"] = {
        "title": {"target_ids": [saved["title"]["ideas"][0]["candidates"][0]["candidate_id"]], "direct": True, "feedback": {}}
    }
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, role = prepare_resume(task)
    assert role == "title"
    assert restored["pipeline"]["status"] == "pending"

    rejected = copy.deepcopy(task)
    rejected.payload["graph_state"]["pipeline"]["stages"]["title"]["metadata"]["failed_attempts"][0]["error"] = "任意其他交接错误"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(rejected)


def test_audience_coverage_failure_resumes_without_repeating_sample_stages():
    model = GraphCompletion()
    def incomplete_audience(**kwargs):
        raw = model(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "AudienceGroupingResponse":
            value = json.loads(raw)
            value["segments"][-1]["audience_terms"] = value["segments"][-1]["audience_terms"][:-1]
            return json.dumps(value, ensure_ascii=False)
        return raw

    original = workflow(incomplete_audience, count=1, variants=2)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)
    recovery_model = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery_model, resume_state=restored, resume_role=role)
    result, _ = resumed.run()

    assert role == "audience"
    assert result["ideas"]
    assert recovery_model.calls[0][2]["response_format"]["json_schema"]["name"] == "AudienceNeedsResponse"
    assert recovery_model.counts["sample_analysis"] == 0
    assert recovery_model.counts["sample_review"] == 0
    assert "分组归属属于技术性补齐" in " ".join(resumed.state["audience"]["limitations"])


def test_resume_rejects_audience_handoff_failure_with_source_errors():
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    w.run()
    saved = copy.deepcopy(w.state)
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="audience", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["audience"]
    stage.update(status="failed", error="读者分组引用来源失败")
    stage["metadata"].update(failed_role="audience", failed_attempts=[
        {"error": "segments 分组校验失败；缺少受众词=['AI']；错误=[来源引用失败]；最多8组"},
        {"error": "segments 分组校验失败；缺少受众词=['AI']；错误=[来源引用失败]；最多8组"},
    ])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(task)


@pytest.mark.parametrize("failure", [
    "同一条 evidence.note_ids 不可重复",
    "同一条 evidence.claim_ids 不可重复",
    "source_note_ids 不可重复",
])
def test_topic_duplicate_source_reference_failure_is_resumable_only_for_exact_error(failure):
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    w.run()
    saved = copy.deepcopy(w.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="topic", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["topic"]
    stage.update(status="failed", error=failure)
    stage["metadata"].update(failed_role="topic", failed_attempts=[{"error": failure}, {"error": failure}])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)

    assert role == "topic"
    assert restored["pipeline"]["stages"]["topic"]["status"] == "pending"

    task.payload["graph_state"]["pipeline"]["stages"]["topic"]["metadata"]["failed_attempts"][-1]["error"] = "另一个来源错误"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(task)


def test_topic_source_binding_failure_allows_mixed_known_errors_only():
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    w.run()
    saved = copy.deepcopy(w.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="topic", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["topic"]
    stage.update(status="failed")
    stage["metadata"].update(failed_role="topic", failed_attempts=[
        {"error": "note_id 必须属于本条 evidence 引用的获批 claim_id"},
        {"error": "source_note_ids 必须与 evidence 的样本来源合集一致"},
    ])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)
    assert role == "topic"
    assert restored["sample_analysis"] == saved["sample_analysis"]
    assert restored["audience"] == saved["audience"]

    task.payload["graph_state"]["pipeline"]["stages"]["topic"]["metadata"]["failed_attempts"][-1]["error"] = "未知 ID"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(task)


@pytest.mark.parametrize("error_type", ["creative_handoff_invalid", "creative_schema_invalid"])
def test_topic_revision_no_progress_failure_resumes_same_quality_round_only(error_type):
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    w.run()
    saved = copy.deepcopy(w.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    saved["revision_plan"]["topic"] = {"target_ids": ["idea_1"], "direct": True, "feedback": {
        "reviews": [{"judgement": {"idea_id": "idea_1", "scores": [], "issues": [], "preserve": []}}],
    }}
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="topic", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["topic"]
    stage.update(status="failed", quality_revisions=2)
    stage["metadata"].update(failed_role="topic", failed_attempts=[
        {"error": "选题 idea_1 未落实评审意见：目标内容字段没有任何变化"},
        {"error": "选题 idea_1 未落实评审意见：必改字段未变化=reader_promise、must_cover"},
    ])
    task = SimpleNamespace(status="failed", error_type=error_type, payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)
    assert role == "topic"
    assert restored["pipeline"]["stages"]["topic"]["quality_revisions"] == 2

    task.payload["graph_state"]["pipeline"]["stages"]["topic"]["metadata"]["failed_attempts"][-1]["error"] = "任意其他错误"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(task)


def test_topic_review_issue_quote_failure_is_resumable_only_for_exact_error():
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    w.run()
    saved = copy.deepcopy(w.state)
    saved["result"] = None
    saved["metadata"] = {}
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    pipeline.update(status="failed", current_stage="topic", error_code="creative_handoff_invalid")
    stage = pipeline["stages"]["topic"]
    stage.update(status="failed", error="问题 quote 必须逐字出现在当前选题的对应 field 中")
    stage["metadata"].update(failed_role="topic_review", failed_attempts=[
        {"error": "问题 quote 必须逐字出现在当前选题的对应 field 中"},
        {"error": "问题 quote 必须逐字出现在当前选题的对应 field 中"},
    ])
    task = SimpleNamespace(status="failed", error_type="creative_handoff_invalid", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })

    restored, role = prepare_resume(task)

    assert role == "topic_review"
    assert restored["pipeline"]["stages"]["topic"]["status"] == "pending"

    task.payload["graph_state"]["pipeline"]["stages"]["topic"]["metadata"]["failed_attempts"][-1]["error"] = "其他 topic 错误"
    with pytest.raises(CreativeError, match="没有可继续"):
        prepare_resume(task)


@pytest.mark.parametrize("failure", ["schema", "handoff", "quality", "legacy", "incomplete"])
def test_resume_rejects_business_failures_and_missing_checkpoints(env, failure):
    model = GraphCompletion(fail_stage="final_review" if failure == "quality" else None, always=True)
    def complete(**kwargs):
        value = json.loads(model(**kwargs))
        if failure == "schema":
            value["observations"][0]["note_id"] = 123
        elif failure == "handoff" and kwargs["response_format"]["json_schema"]["name"] == "CopyResponse":
            value["ideas"][0]["variants"][0]["source_note_ids"] = ["sample-0", "unknown-note"]
        return json.dumps(value)
    install_client(complete)
    original = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2})
    with env.factory() as db:
        task = db.get(Task, original["id"])
        if failure in ("legacy", "incomplete"):
            task.status, task.error_type = "failed", "ai_read_timeout"
            if failure == "legacy":
                task.payload = {**task.payload, "analysis": {**task.payload["analysis"], "workflow": "legacy"}}
            else:
                task.payload = {**task.payload, "graph_state": {}}
            db.commit()
    before = len(model.calls)
    response = env.client.post(f"/api/xhs/creative/runs/{original['id']}/resume", headers=env.headers)
    assert response.status_code == 400
    assert len(model.calls) == before


@pytest.mark.parametrize("error_code", ["creative_time_exhausted", "creative_budget_exhausted"])
def test_expired_window_resumes_missing_parallel_final_batches_with_cumulative_request_limit(error_code):
    now, model = [0], GraphCompletion()
    def complete(**kwargs):
        raw = model(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "FinalReviewResponse":
            batch = json.loads(kwargs["user_prompt"])["input"]["batch"]
            if batch["index"] == 4:
                now[0] = 1801
        return raw
    original = workflow(complete, count=3, variants=3)
    assert original.budget.max_seconds == 90 * 60
    original.budget.clock, original.budget.started, original.budget.max_seconds = lambda: now[0], 0, 1800
    with pytest.raises(CreativeError) as caught:
        original.run()
    assert caught.value.code == "creative_time_exhausted"
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    pipeline = saved["pipeline"]
    metadata = pipeline["stages"]["final_review"]["metadata"]
    expected_budget = estimate_graph_request_budget(len(saved["sources"]), saved["options"])
    completed_batches = metadata["final_review_completed_batches"]
    assert 1 <= completed_batches < 9
    assert metadata["budget_exhaustion"] == {
        "reason": "time", "elapsed_seconds": 1801, "max_seconds": 1800,
        "request_count": pipeline["request_count"], "max_requests": expected_budget,
    }
    if error_code == "creative_budget_exhausted":
        # Historical snapshots did not distinguish request and time limits.
        pipeline["error_code"] = error_code
        metadata.pop("budget_exhaustion")
    task = SimpleNamespace(status="failed", error_type=error_code, payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    before = copy.deepcopy(task.payload)
    for fault in ("count", "code", "reason"):
        broken = copy.deepcopy(task)
        bp = broken.payload["graph_state"]["pipeline"]
        if fault == "count":
            bp["request_count"] = bp["max_requests"]
        elif fault == "code":
            bp["error_code"] = "ai_http_error"
        else:
            bp["stages"]["final_review"]["metadata"]["budget_exhaustion"] = {"reason": "requests"}
        with pytest.raises(CreativeError, match="没有可继续"):
            prepare_resume(broken)
    restored, role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=role)
    result, meta = resumed.run()
    assert len(result["ideas"]) == 3
    assert resumed.budget.max_seconds == 90 * 60
    assert meta["request_count"] == pipeline["request_count"] + len(recovery.wire_calls)
    assert resumed.budget.max_requests == expected_budget
    assert len(recovery.wire_calls) == 9 - completed_batches
    resumed_indices = [json.loads(c["user_prompt"])["input"]["batch"]["index"] for c in recovery.wire_calls]
    assert len(set(resumed_indices)) == len(resumed_indices)
    assert all(c["response_format"]["json_schema"]["name"] == "FinalReviewResponse" for c in recovery.wire_calls)
    assert task.payload == before


def test_time_exhausted_task_api_exposes_resume_and_reuses_saved_final_batch(env, monkeypatch):
    now, model = [0], GraphCompletion()
    monkeypatch.setattr("backend.app.services.creative_graph.workflow.RequestBudget",
                        lambda *args, **kwargs: RequestBudget(*args, **kwargs, clock=lambda: now[0]))
    def complete(**kwargs):
        raw = model(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "FinalReviewResponse":
            now[0] = 90 * 60 + 1
        return raw
    install_client(complete)
    original = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 3, "variant_count": 2})
    assert original["status"] == "failed" and original["error_type"] == "creative_time_exhausted"
    assert original["resume_available"] is True
    completed_batches = original["payload"]["pipeline"]["stages"]["final_review"]["metadata"]["final_review_completed_batches"]
    assert 1 <= completed_batches <= 3
    recovery = GraphCompletion()
    install_client(recovery)
    response = env.client.post(f"/api/xhs/creative/runs/{original['id']}/resume", headers=env.headers)
    assert response.status_code == 202, response.text
    child = env.client.get(f"/api/xhs/creative/runs/{response.json()['id']}", headers=env.headers).json()
    assert child["status"] == "completed"
    assert child["payload"]["pipeline"]["request_count"] == original["payload"]["pipeline"]["request_count"] + len(recovery.wire_calls)
    assert len(recovery.wire_calls) == 6 - completed_batches
    resumed_indices = [json.loads(call["user_prompt"])["input"]["batch"]["index"] for call in recovery.wire_calls]
    assert len(set(resumed_indices)) == len(resumed_indices)
    assert child["payload"]["pipeline"]["stages"]["final_review"]["metadata"]["final_review_reused_batches"] == completed_batches
