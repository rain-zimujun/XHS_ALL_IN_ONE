import copy
import json
from types import SimpleNamespace

import pytest
import requests

from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_agents.revision import PipelineCancelled
from backend.app.services.creative_graph.state import SampleUpdate
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.workflow import LangGraphWorkflow
from backend.app.services.creative_pipeline import validate_pipeline_state
from test_creative_sample_agent import SEARCH, sources
from test_creative_workflow import WorkflowCompletion


class GraphCompletion(WorkflowCompletion):
    """Use the existing independent synthetic outputs with the new wire schema."""
    NAMES = {
        "SampleAnalysisResponse": "SampleAnalysis", "AudienceGroupingResponse": "AudienceBrief", "AudienceNeedsResponse": "AudienceBrief",
        "TopicResponse": "TopicBrief", "TopicReviewResponse": "TopicReviewResponse",
        "TitleResponse": "TitleCandidates", "CopyResponse": "CopyVariants", "ImageResponse": "ImagePlans",
        "TitleReviewResponse": "ItemReviewResponse", "CopyReviewResponse": "ItemReviewResponse", "FinalReviewResponse": "ItemReviewResponse",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wire_calls = []

    def __call__(self, **kwargs):
        self.wire_calls.append(copy.deepcopy(kwargs))
        prompt, end = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        wire_name = kwargs["response_format"]["json_schema"]["name"]
        prompt.setdefault("output_schema", copy.deepcopy(kwargs["response_format"]["json_schema"]["schema"]))
        prompt["output_schema"]["title"] = self.NAMES[prompt["output_schema"]["title"]]
        raw = super().__call__(**{**kwargs, "user_prompt": json.dumps(prompt, ensure_ascii=False) + kwargs["user_prompt"][end:]})
        output = json.loads(raw)
        if wire_name == "AudienceGroupingResponse":
            output = {"segments": [{k: v for k, v in group.items() if k != "segment_id"} for group in output["segments"]]}
        elif wire_name == "AudienceNeedsResponse":
            mapping = {group["segment_id"]: prompt["input"]["segments"][i]["segment_id"] for i, group in enumerate(output["segments"])}
            for item in output["shared_needs"]:
                item["applies_to_segment_ids"] = [mapping[ref] for ref in item["applies_to_segment_ids"]]
            for item in output["format_recommendations"]:
                item["for_segment_ids"] = [mapping[ref] for ref in item["for_segment_ids"]]
            for key in ("segments", "requested_audience_terms"):
                output.pop(key)
        elif wire_name == "TopicResponse":
            for idea in output["ideas"]:
                idea["primary_segment_id"] = prompt["input"]["audience_brief"]["segments"][0]["segment_id"]
        elif wire_name == "TitleResponse":
            output = {
                "ideas": {
                    idea["idea_id"]: {
                        "candidates": [
                            {key: value for key, value in candidate.items() if key != "candidate_id"}
                            for candidate in idea["candidates"]
                        ]
                    }
                    for idea in output["ideas"]
                },
                "limitations": output["limitations"],
            }
        for key in ("input_fingerprint", "revision", "topic_revision"):
            output.pop(key, None)
        for review in output.get("reviews", []):
            if "item_id" in review:
                for issue in review["issues"]:
                    issue.setdefault("responsible_stage", None)
                if wire_name == "TitleReviewResponse":
                    review["comment"] = review["scores"][0]["comment"]
                    review["scores"] = {score["criterion"]: score["score"] for score in review["scores"]}
                    for issue in review["issues"]:
                        issue.pop("responsible_stage")
                        issue.pop("note_ids")
        return json.dumps(output, ensure_ascii=False)


def workflow(model, count=1, variants=2, **kwargs):
    return LangGraphWorkflow(search=SEARCH, sources=sources(), options={"model_id": 1, "idea_count": count, "variant_count": variants}, complete=model, **kwargs)


@pytest.mark.parametrize("count", [1, 3, 5])
@pytest.mark.parametrize("variants", [2, 3])
def test_graph_executes_ten_roles_and_persists_serializable_global_state(count, variants):
    model, saved = GraphCompletion(), []
    w = workflow(model, count, variants, on_graph_state=saved.append)
    result, metadata = w.run()
    assert len(result["ideas"]) == count
    assert all(len(i["variants"]) == variants for i in result["ideas"])
    assert metadata["workflow"] == "langgraph"
    assert len(model.calls) == metadata["request_count"] == 7 + 2 * count + 2 * count * variants
    assert saved[-1]["result"] == result
    assert saved[-1]["pipeline"]["status"] == "completed"
    assert saved[-1]["revision_plan"] == {}
    for state in saved:
        validate_pipeline_state(json.loads(json.dumps(state))["pipeline"])
    assert {c["response_format"]["json_schema"]["name"] for c in model.wire_calls} == set(GraphCompletion.NAMES)
    assert "fixture-private" not in json.dumps(w.state)


def test_graph_full_chain_normalizes_title_review_blocking_severity_without_retry():
    class LowScoreSuggestion(GraphCompletion):
        def __call__(self, **kwargs):
            prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
            raw = super().__call__(**kwargs)
            if kwargs["response_format"]["json_schema"]["name"] != "TitleReviewResponse":
                return raw
            output = json.loads(raw)
            review = output["reviews"][0]
            review["scores"]["reader_benefit"] = 5.0
            review["issues"] = [{
                "criterion": "reader_benefit", "severity": "suggestion", "field": "title",
                "quote": prompt["input"]["items"][0]["fields"]["title"],
                "problem": "读者收益不够具体。", "revision_instruction": "补出明确的观察收益。",
                "related_item_ids": [],
            }]
            return json.dumps(output, ensure_ascii=False)

    model = LowScoreSuggestion()
    w = workflow(model, count=1, variants=2)
    result, metadata = w.run()
    assert len(result["ideas"]) == 1
    assert all(len(idea["variants"]) == 2 for idea in result["ideas"])
    assert model.counts["title_review"] == 1
    assert metadata["request_count"] == 13


@pytest.mark.parametrize("cache_fault", [None, "version", "fingerprint", "review"])
def test_title_review_resume_reuses_only_valid_batches_and_keeps_context_compact(cache_fault):
    model = GraphCompletion()

    def complete(**kwargs):
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        if kwargs["response_format"]["json_schema"]["name"] == "TitleReviewResponse":
            batch = prompt["input"]["batch"]
            if batch["index"] == 2:
                raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
            assert "output_schema" not in prompt
            assert kwargs["response_format"]["json_schema"]["schema"]["properties"]["reviews"]
            context = prompt["input"]["context"]
            assert set(context) == {"keyword", "requirements", "title_max_length", "topic_brief", "audience", "title_style", "style_summary"}
            assert len(prompt["input"]["items"]) == 5
            assert len(context["topic_brief"]["ideas"]) == 1
        return model(**kwargs)

    original = workflow(complete, count=3)
    with pytest.raises(CreativeError):
        original.run()
    cache = original.state["pipeline"]["stages"]["title"]["metadata"]["title_review_batches"]
    assert len(cache["results"]) == 1
    assert original.state["pipeline"]["stages"]["title"]["metadata"]["failed_role"] == "title_review"

    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    stored_cache = saved["pipeline"]["stages"]["title"]["metadata"]["title_review_batches"]
    if cache_fault == "version":
        stored_cache["version"] = "outdated"
    elif cache_fault == "fingerprint":
        stored_cache["input_fingerprint"] = "0" * 64
    elif cache_fault == "review":
        next(iter(stored_cache["results"].values()))["reviews"][0]["item_id"] = "title_unknown"
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    before = copy.deepcopy(task.payload)
    restored, role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=role)
    result, _ = resumed.run()
    assert len(result["ideas"]) == 3
    assert recovery.counts["title_review"] == (2 if cache_fault is None else 3)
    assert [json.loads(c["user_prompt"])["input"]["batch"]["index"] for c in recovery.wire_calls
            if c["response_format"]["json_schema"]["name"] == "TitleReviewResponse"] == ([2, 3] if cache_fault is None else [1, 2, 3])
    metadata = resumed.state["pipeline"]["stages"]["title"]["metadata"]
    assert metadata["title_review_reused_batches"] == (1 if cache_fault is None else 0)
    assert metadata["title_review_completed_batches"] == 3
    assert task.payload == before


@pytest.mark.parametrize("stage", ["TopicReviewResponse", "title_review", "copy_review"])
def test_conditional_review_edge_returns_to_producer_and_only_revises_failed_items(stage):
    model = GraphCompletion(fail_stage=stage)
    w = workflow(model, 3, 3)
    result, _ = w.run()
    assert len(result["ideas"]) == 3
    assert len(model.calls) == (33 if stage == "TopicReviewResponse" else 35)
    producer = {"TopicReviewResponse": "topic", "title_review": "title", "copy_review": "copy"}[stage]
    item = w.state["pipeline"]["stages"][producer]
    assert item["quality_revisions"] == 1
    history = [h for h in item["history"] if h["role"] == "producer"]
    assert history[0]["output"]["ideas"][-1] == history[-1]["output"]["ideas"][-1]


@pytest.mark.parametrize("owner", ["topic", "title", "copy", "image"])
def test_final_conditional_route_invalidates_and_rebuilds_only_dependencies(owner):
    model, states = GraphCompletion(fail_stage="final_review", final_owner=owner), []
    w = workflow(model, 3, 2, on_graph_state=states.append)
    result, _ = w.run()
    assert result and model.counts["final_review"] == 12
    assert any(s["pipeline"]["stages"]["final_review"]["status"] == "invalidated" and s["final_review"] is None for s in states)
    assert w.state["pipeline"]["stages"][owner]["quality_revisions"] == 1
    for stage in ("topic", "title", "copy", "image"):
        history = [h["output"] for h in w.state["pipeline"]["stages"][stage]["history"] if h["role"] == "producer"]
        if len(history) > 1:
            field = "variants" if stage == "image" else "ideas"
            assert history[0][field][-1] == history[-1][field][-1]


@pytest.mark.parametrize("stage,owner", [("TopicReviewResponse", "topic"), ("title_review", "title"), ("copy_review", "copy"), ("final_review", "image")])
def test_exhausted_review_cycle_stops_with_drafts_and_no_final_result(stage, owner):
    model = GraphCompletion(fail_stage=stage, always=True, final_owner=owner)
    w = workflow(model)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == "creative_quality_exhausted"
    assert w.state["result"] is None
    assert w.state["pipeline"]["status"] == "revision_required"
    assert w.state["pipeline"]["stages"][owner]["quality_revisions"] == 2
    assert model.counts[stage] == (6 if stage == "final_review" else 3)


def test_budget_and_timeout_stop_without_further_requests_and_preserve_global_outputs():
    model = GraphCompletion()
    w = workflow(model, max_requests=3)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == "creative_budget_exhausted"
    assert len(model.calls) == w.state["pipeline"]["request_count"] == 3
    assert w.state["audience"] and w.state["topic"] is None and w.state["result"] is None
    model = GraphCompletion(timeout_stage="CopyVariants")
    w = workflow(model)
    with pytest.raises(requests.ReadTimeout):
        w.run()
    assert model.counts["CopyVariants"] == 1
    assert w.state["title_review"]["status"] == "passed"
    assert w.state["copy"] is None
    assert "secret-transport-details" not in json.dumps(w.state)


def test_sample_analysis_batches_sequentially_and_resume_reuses_completed_sample_batches():
    notes = sources(7)
    def fail_second_batch(**kwargs):
        prompt = json.loads(kwargs["user_prompt"])
        if (kwargs["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"
                and prompt["input"]["batch"]["index"] == 2):
            raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)

    model = GraphCompletion()
    original = LangGraphWorkflow(search=SEARCH, sources=notes,
                                 options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                                 complete=fail_second_batch)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    metadata = saved["pipeline"]["stages"]["sample_analysis"]["metadata"]
    assert metadata["sample_batch_count"] == 3
    assert metadata["sample_completed_batches"] == 1
    assert len(metadata["sample_analysis_batches"]["results"]) == 1
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, role = prepare_resume(task)
    resumed_model = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=resumed_model, resume_state=restored, resume_role=role)
    result, pipeline_metadata = resumed.run()
    assert len(result["ideas"]) == 1
    sample = resumed.state["pipeline"]["stages"]["sample_analysis"]["metadata"]
    assert sample["sample_completed_batches"] == 3
    assert sample["sample_reused_batches"] == 1
    sample_calls = [call for call in resumed_model.wire_calls
                    if call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"]
    assert len(sample_calls) == 2
    assert [json.loads(call["user_prompt"])["input"]["batch"]["index"] for call in sample_calls] == [2, 3]
    assert pipeline_metadata["request_count"] == 16


def test_sample_relevance_filter_preserves_raw_sources_and_batches_only_selected_notes():
    notes = sources(7)
    notes.append({"note_id": "weak-note", "title": "AI绘图工具记录", "content": ("AI绘图工具使用记录。" + "补充。" * 300)})
    notes.append({"note_id": "unrelated-note", "title": "周末咖啡探店", "content": "這家咖啡店很安靜，甜品也不错。"})
    original = copy.deepcopy(notes)
    model = GraphCompletion()
    w = LangGraphWorkflow(search=SEARCH, sources=notes,
                          options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                          complete=model)

    update = w.sample_node(copy.deepcopy(w.state))

    assert notes == original
    sent = [json.loads(call["user_prompt"])["input"]
            for call in model.wire_calls if call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"]
    sent_ids = [source["note_id"] for batch in sent for source in batch["sources"]]
    assert "unrelated-note" not in sent_ids
    assert "weak-note" in sent_ids
    weak_input = next(source for batch in sent for source in batch["sources"] if source["note_id"] == "weak-note")
    assert weak_input["relevance_tier"] == "weak"
    assert weak_input["relevance_body_truncated"] is True
    assert weak_input["original_body_chars"] > 500
    assert weak_input["relevance_original_body_chars"] == weak_input["original_body_chars"]
    relevance = update["pipeline"]["stages"]["sample_analysis"]["metadata"]["sample_relevance"]
    assert relevance["total_count"] == 9
    assert relevance["selected_count"] == 8
    assert relevance["excluded_note_ids"] == ["unrelated-note"]
    assert relevance["weak_note_ids"] == ["weak-note"]
    assert relevance["version"] == "sample-relevance-v1"
    assert len(sent) == 3


def test_sample_relevance_filter_rejects_too_few_effective_samples():
    notes = sources(2)
    notes.append({"note_id": "unrelated-note", "title": "周末咖啡", "content": "咖啡很香。"})
    model = GraphCompletion()
    w = LangGraphWorkflow(search=SEARCH, sources=notes,
                          options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                          complete=model)
    with pytest.raises(CreativeError) as caught:
        w.sample_node(copy.deepcopy(w.state))
    assert caught.value.code == "sample_relevance_insufficient"
    assert "仅剩 2 篇可用样本" in str(caught.value)
    assert model.calls == []


@pytest.mark.parametrize("race", [False, True])
def test_cancellation_at_request_boundary_keeps_results_uncommitted(race):
    model, states = GraphCompletion(timeout_stage="TopicBrief" if race else None), []
    def save(state):
        if len(model.calls) == 3 and state["pipeline"]["status"] != "cancelled":
            raise PipelineCancelled()
        states.append(state)
    w = workflow(model, on_graph_state=save)
    with pytest.raises(PipelineCancelled):
        w.run()
    assert len(model.calls) == 3
    assert states[-1]["pipeline"]["status"] == "cancelled"
    assert states[-1]["topic"] is None


def test_changed_title_selection_rebinds_copy_and_image_groups():
    model = GraphCompletion(fail_stage="final_review", final_owner="title")
    def complete(**kwargs):
        output = json.loads(model(**kwargs))
        if model.calls[-1][0] == "title_review" and model.counts["title_review"] == 4:
            output["reviews"][1]["scores"] = {key: 9.0 for key in output["reviews"][1]["scores"]}
        return json.dumps(output)
    w = workflow(complete, 3, 2)
    result, _ = w.run()
    assert result["ideas"][0]["candidate_id"] == "title_0_1"
    assert w.state["copy"]["ideas"][0]["candidate_id"] == "title_0_1"
    assert w.state["image"]["variants"][0]["candidate_id"] == "title_0_1"


def test_graph_has_visible_conditional_edges_and_node_write_scope_is_enforced():
    w = workflow(GraphCompletion())
    diagram = w.graph.get_graph()
    assert any(e.source == "topic_review" and e.target == "topic" and e.conditional for e in diagram.edges)
    assert any(e.source == "plan_revision" and e.target == "image" and e.conditional for e in diagram.edges)
    original = copy.deepcopy(w.state)
    def out_of_scope(state):
        state["search"]["keyword"] = "mutated"
        return {"copy": {}}
    with pytest.raises(CreativeError) as caught:
        w._node("sample_analysis", out_of_scope, SampleUpdate)(w.state)
    assert caught.value.code == "creative_node_scope_invalid"
    assert w.state["search"] == original["search"]


def test_model_does_not_echo_server_owned_versions_and_every_artifact_binds_current_state():
    model = GraphCompletion()
    w = workflow(model)
    w.run()
    for call in model.wire_calls:
        properties = call["response_format"]["json_schema"]["schema"]["properties"]
        assert not set(properties).intersection({"revision", "topic_revision", "input_fingerprint"})
        assert "原样回填" not in call["system_prompt"]
    assert w.state["title"]["topic_revision"] == w.state["pipeline"]["stages"]["topic"]["revision"]
    assert len(w.state["copy"]["input_fingerprint"]) == 64
    assert w.state["final_review"]["revision"] == 1


def test_cancellation_before_final_result_persistence_does_not_publish_result():
    states = []
    def save(state):
        if state["result"] is not None:
            raise PipelineCancelled()
        states.append(state)
    w = workflow(GraphCompletion(), on_graph_state=save)
    with pytest.raises(PipelineCancelled):
        w.run()
    assert states[-1]["pipeline"]["status"] == "cancelled"
    assert states[-1]["result"] is None


@pytest.mark.parametrize("kind", ["schema", "handoff"])
def test_distinct_output_failure_reason_is_saved_in_global_state(kind):
    model = GraphCompletion()
    def complete(**kwargs):
        output = json.loads(model(**kwargs))
        if kind == "schema":
            output["observations"][0]["note_id"] = 123
        else:
            output["observations"][0]["quote"] = "未出现在样本中的引文"
        return json.dumps(output)
    w = workflow(complete)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == w.state["pipeline"]["error_code"] == f"creative_{kind}_invalid"
    assert len(model.calls) == 2 and w.state["audience"] is None
    assert len(w.state["pipeline"]["stages"]["sample_analysis"]["metadata"]["failed_attempts"]) == 2
