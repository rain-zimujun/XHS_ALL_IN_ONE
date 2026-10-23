import copy
import json
from types import SimpleNamespace

import pytest
import requests

from backend.app.models import Task
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.workflow import LangGraphWorkflow
from test_creative import env, collect_for_analysis
from test_creative_graph_workflow import GraphCompletion, workflow
from test_creative_workflow import install_client


ROLES = ["SampleAnalysisResponse", "AudienceGroupingResponse", "AudienceNeedsResponse", "TopicResponse", "TopicReviewResponse", "TitleResponse",
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
    assert len(recovery.calls) == len(EXECUTION_ROLES) - EXECUTION_ROLES.index(failed_role)
    assert metadata["request_count"] == 14  # one original failure + thirteen successes
    for key in ("sample_analysis", "audience", "topic", "topic_review", "title", "title_review", "copy", "copy_review", "image", "final_review"):
        if saved[key] is not None:
            assert resumed.state[key] == saved[key]
    assert task.payload == before


def test_resume_api_preserves_original_and_is_idempotent_with_ownership(env):
    failing = GraphCompletion(timeout_stage="title_review")
    install_client(failing)
    original = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2})
    assert original["status"] == "failed"
    assert original["resume_available"] is True
    assert failing.counts["title_review"] == 4
    failed_pipeline = copy.deepcopy(original["payload"]["pipeline"])
    failed_network = failed_pipeline["stages"]["title"]["metadata"]["network_attempts"]
    assert len(failed_network) == 4
    assert [r["will_retry"] for r in failed_network] == [True, True, True, False]
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
    assert child["payload"]["pipeline"]["request_count"] == 17
    assert child["payload"]["graph_state"]["pipeline"] == child["payload"]["pipeline"]
    assert len(recovery.calls) == 7
    assert recovery.wire_calls[0]["response_format"]["json_schema"]["name"] == "TitleReviewResponse"
    old = env.client.get(f"/api/xhs/creative/runs/{original['id']}", headers=env.headers).json()
    assert old["status"] == "failed" and old["payload"]["pipeline"] == failed_pipeline
    assert env.client.post(f"/api/xhs/creative/runs/{original['id']}/resume", headers=env.headers).json()["id"] == child_id
    assert len(recovery.calls) == 7
    assert "fixture-key-private" not in json.dumps(child)


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
    assert result["payload"]["pipeline"]["request_count"] == 16
    assert len(model.calls) == 13
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
    saved["options"]["workflow"] = "langgraph"
    saved["pipeline"]["stages"]["title"]["metadata"].pop("failed_role")
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    _, role = prepare_resume(task)
    assert role == "title_review"


@pytest.mark.parametrize("failure", ["schema", "quality", "legacy", "incomplete"])
def test_resume_rejects_business_failures_and_missing_checkpoints(env, failure):
    model = GraphCompletion(fail_stage="final_review" if failure == "quality" else None, always=True)
    def complete(**kwargs):
        value = json.loads(model(**kwargs))
        if failure == "schema":
            value["observations"][0]["note_id"] = 123
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
