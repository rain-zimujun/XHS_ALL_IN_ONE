import copy
import json
from types import SimpleNamespace

import pytest
import requests
from sqlalchemy import select

from backend.app.main import app
from backend.app.models import Task
from backend.app.services.ai_service import OpenAICompatibleTextClient
from backend.app.services.creative_analysis import CreativeError, get_creative_text_client_factory, request_analysis
from backend.app.services.creative_runner import recover_interrupted_runs
from test_creative import env, collect_for_analysis, search_for_collection
from test_creative_graph_workflow import GraphCompletion, workflow
from test_creative_workflow import install_client
from test_structured_output import assert_strict_contract


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3)])
def test_langgraph_api_saves_global_state_and_compatible_reports_with_ownership(env, count, variants):
    model = GraphCompletion()
    install_client(model)
    result = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": count, "variant_count": variants})
    assert result["status"] == "completed", result["payload"].get("error")
    payload = result["payload"]
    assert payload["analysis"]["workflow"] == payload["analysis_metadata"]["workflow"] == "langgraph"
    assert payload["graph_state"]["result"] == payload["result"]
    assert payload["graph_state"]["pipeline"] == payload["pipeline"]
    assert payload["graph_state"]["final_review"]["status"] == "passed"
    assert len(payload["result"]["ideas"]) == count
    assert len(model.wire_calls) == payload["pipeline"]["request_count"] == 10
    assert all(c["timeout"] == (10, 180) for c in model.wire_calls)
    assert "fixture-key-private" not in json.dumps(result)
    assert "test=fixture" not in json.dumps(result)
    assert env.client.get(f"/api/xhs/creative/runs/{result['id']}", headers=env.other_headers).status_code == 404
    for file in payload["report_files"]:
        response = env.client.get(file["download_url"], headers=env.headers)
        assert response.status_code == 200
        assert env.client.get(file["download_url"], headers=env.other_headers).status_code == 404
        if file["file_name"].endswith(".json"):
            document = response.json()
            assert document["schema_version"] == 3
            assert document["analysis"]["workflow"] == "langgraph"
            assert document["pipeline"] == payload["pipeline"]
        else:
            assert "终审评分：8.00 分 · 通过" in response.text
    assert env.client.post(f"/api/xhs/creative/runs/{result['id']}/export", headers=env.headers).status_code == 200
    assert len(model.calls) == 10


def test_saved_collection_can_switch_to_graph_without_recrawling_or_changing_old_task(env):
    original = collect_for_analysis(env)
    assert original["status"] == "completed"
    model = GraphCompletion()
    install_client(model)
    response = env.client.post("/api/xhs/creative/analyze", headers=env.headers, json={
        "collection_task_id": original["id"], "model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2,
    })
    assert response.status_code == 202
    run_id = response.json()["id"]
    result = env.client.get(f"/api/xhs/creative/runs/{run_id}", headers=env.headers).json()
    assert result["status"] == "completed", result["payload"].get("error")
    assert result["id"] != original["id"]
    assert result["payload"]["sources"] == original["payload"]["sources"]
    assert env.client.get(f"/api/xhs/creative/runs/{original['id']}", headers=env.headers).json()["payload"] == original["payload"]


@pytest.mark.parametrize("failure,code", [("timeout", "ai_read_timeout"), ("quality", "creative_quality_exhausted"), ("schema", "creative_schema_invalid"), ("handoff", "creative_handoff_invalid")])
def test_failed_api_task_keeps_global_drafts_and_specific_error_code(env, failure, code):
    model = GraphCompletion(timeout_stage="CopyVariants" if failure == "timeout" else None,
                            fail_stage="final_review" if failure == "quality" else None, always=failure == "quality")
    def complete(**kwargs):
        output = json.loads(model(**kwargs))
        if failure == "schema":
            output["observations"][0]["note_id"] = 123
        elif failure == "handoff":
            output["observations"][0]["quote"] = "不属于原文的引文"
        return json.dumps(output)
    install_client(complete)
    result = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1, "variant_count": 2})
    assert result["status"] == "failed"
    assert result["error_type"] == result["payload"]["pipeline"]["error_code"] == code
    assert "result" not in result["payload"]
    assert result["payload"]["graph_state"]["result"] is None
    assert result["payload"]["graph_state"]["pipeline"] == result["payload"]["pipeline"]
    assert len(model.calls) == (14 if failure == "quality" else 7 if failure == "timeout" else 2)
    if failure in ("quality", "timeout"):
        assert result["payload"]["graph_state"]["title_review"]["status"] == "passed"
    else:
        assert "结构或交接校验" not in result["payload"]["error"]


def test_cancel_during_model_request_updates_both_saved_snapshots(env):
    model = GraphCompletion()
    class Client:
        def _complete(self, **kwargs):
            if len(model.calls) == 2:
                with env.factory() as db:
                    task = db.scalar(select(Task).where(Task.task_type == "creative_collect"))
                    task.status = "cancelled"
                    db.commit()
            return model(**kwargs)
    app.dependency_overrides[get_creative_text_client_factory] = lambda: Client
    result = collect_for_analysis(env, {"model_id": env.model_id, "workflow": "langgraph", "idea_count": 1})
    assert result["status"] == result["payload"]["graph_state"]["pipeline"]["status"] == "cancelled"
    assert result["payload"]["graph_state"]["topic"] is None
    assert len(model.calls) == 3


def test_restart_marks_graph_snapshot_interrupted_without_model_calls(env):
    model = GraphCompletion()
    w = workflow(model)
    snapshot = copy.deepcopy(w.state)
    snapshot["pipeline"].update(status="running", current_stage="topic")
    snapshot["pipeline"]["stages"]["topic"]["status"] = "running"
    with env.factory() as db:
        task = Task(user_id=env.user_id, platform="xhs", task_type="creative_analysis", status="running",
                    payload={"pipeline": snapshot["pipeline"], "graph_state": snapshot})
        db.add(task)
        db.commit()
        task_id = task.id
    recover_interrupted_runs(env.factory)
    with env.factory() as db:
        task = db.get(Task, task_id)
        assert task.status == task.payload["graph_state"]["pipeline"]["status"] == "failed"
        assert task.payload["graph_state"]["pipeline"]["error_code"] == "creative_interrupted"
        assert task.payload["graph_state"]["pipeline"] == task.payload["pipeline"]
    assert model.calls == []


def test_all_graph_roles_and_revision_send_native_schema_in_actual_http_body(monkeypatch):
    model, seen = GraphCompletion(fail_stage="copy_review"), []
    def post(url, **kwargs):
        assert url == "https://example.invalid/v1/chat/completions"
        assert kwargs["timeout"] == (10, 180)
        body = kwargs["json"]
        seen.append(body)
        parameter = body["response_format"]
        assert parameter["type"] == "json_schema" and parameter["json_schema"]["strict"] is True
        assert_strict_contract(parameter["json_schema"]["schema"], parameter["json_schema"]["schema"])
        assert "tools" not in body
        raw = model(system_prompt=body["messages"][0]["content"], user_prompt=body["messages"][1]["content"],
                    temperature=body["temperature"], response_format=parameter)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"choices": [{"message": {"content": raw}}]}).encode()
        return response
    monkeypatch.setattr("backend.app.services.ai_service.requests.post", post)
    def complete(**kwargs):
        return request_analysis(OpenAICompatibleTextClient(), model_config=SimpleNamespace(
            base_url="https://example.invalid/v1", model_name="mock",
        ), api_key="fixture-private", **kwargs)
    _, metadata = workflow(complete).run()
    assert len(seen) == metadata["request_count"] == 12
    assert {c["response_format"]["json_schema"]["name"] for c in seen} == set(GraphCompletion.NAMES)


def test_provider_rejecting_graph_schema_does_not_trigger_fallback(monkeypatch):
    seen = []
    def post(_url, **kwargs):
        seen.append(kwargs["json"])
        response = requests.Response()
        response.status_code = 400
        response._content = b'{"error":{"message":"unsupported json_schema private"}}'
        return response
    monkeypatch.setattr("backend.app.services.ai_service.requests.post", post)
    def complete(**kwargs):
        return request_analysis(OpenAICompatibleTextClient(), model_config=SimpleNamespace(
            base_url="https://example.invalid/v1", model_name="mock",
        ), api_key="fixture-private", **kwargs)
    w = workflow(complete)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == "ai_http_error"
    assert "private" not in json.dumps(w.state)
    assert len(seen) == 1
