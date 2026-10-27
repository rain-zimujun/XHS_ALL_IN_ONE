import copy
import json
from types import SimpleNamespace

import pytest
import requests

from backend.app.schemas.creative_agents import ItemReviewResponse
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_agents.copy_review import prepare_copy_review_input
from backend.app.services.creative_agents.reviewing import decide_review
from backend.app.services.creative_graph.copy_review import prepare_copy_review_batches, merge_copy_review_results
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.workflow import LangGraphWorkflow
from backend.app.services.creative_orchestrator import AgentValidationError
from test_creative_copy_review import inputs
from test_creative_title_review import valid_review, block
from test_creative_graph_workflow import GraphCompletion, workflow


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 2), (5, 3)])
def test_copy_batches_keep_all_title_peers_and_only_relevant_context(count, variants):
    payload = prepare_copy_review_input(**inputs(count, variants))
    before = copy.deepcopy(payload)
    batches = prepare_copy_review_batches(payload)
    assert len(batches) == count
    assert [len(b["items"]) for b in batches] == [variants] * count
    assert [item for b in batches for item in b["items"]] == payload["items"]
    for batch in batches:
        context = batch["context"]
        assert set(context) == {"keyword", "requirements", "variant_count", "locked_titles", "topic_brief", "audience", "sample_analysis"}
        assert len(context["locked_titles"]) == len(context["topic_brief"]["ideas"]) == 1
        title, idea = context["locked_titles"][0], context["topic_brief"]["ideas"][0]
        assert {(i["idea_id"], i["candidate_id"]) for i in batch["items"]} == {(title["idea_id"], title["candidate_id"])}
        assert title["idea_id"] == idea["idea_id"]
        assert set(batch["sample_note_ids"]).issubset(idea["source_note_ids"])
        for field in ("style_patterns", "value_patterns", "avoid_patterns"):
            for finding in context["sample_analysis"][field]:
                assert finding["evidence"]
                assert all(e["note_id"] in batch["sample_note_ids"] for e in finding["evidence"])
        assert all(o["note_id"] in batch["sample_note_ids"] for o in context["sample_analysis"]["observations"])
    results = [ItemReviewResponse.model_validate(valid_review(b)) for b in batches]
    merged = merge_copy_review_results(list(reversed(results)), payload)
    assert [r.item_id for r in merged.reviews] == [i["item_id"] for i in payload["items"]]
    assert decide_review(merged, payload).status == "passed"
    assert payload == before
    assert prepare_copy_review_batches(payload) == batches


def test_batch_merge_preserves_blocker_and_all_requested_versions():
    payload = prepare_copy_review_input(**inputs(3, 3))
    batches = prepare_copy_review_batches(payload)
    responses = [valid_review(batch) for batch in batches]
    block(responses[1]["reviews"][0], batches[1]["items"][0], criterion="publishability", field="body")
    merged = merge_copy_review_results([ItemReviewResponse.model_validate(r) for r in responses], payload)
    decision = decide_review(merged, payload)
    assert len(decision.reviews) == 9
    assert decision.revision_required_ids == [batches[1]["items"][0]["item_id"]]


@pytest.mark.parametrize("change", ["body", "title", "style", "version"])
def test_input_changes_invalidate_batch_fingerprints(change):
    payload = prepare_copy_review_input(**inputs(3, 3))
    batches = prepare_copy_review_batches(payload)
    updated = copy.deepcopy(payload)
    if change == "body":
        updated["items"][0]["fields"]["body"] += "新增正文内容"
    elif change == "title":
        updated["context"]["locked_titles"][0]["title"] += "新"
    elif change == "style":
        updated["context"]["sample_analysis"]["style_summary"] += "，更新表达风格"
    else:
        updated["revision"] += 1
    refreshed = prepare_copy_review_batches(updated)
    assert batches[0]["input_fingerprint"] != refreshed[0]["input_fingerprint"]


def test_same_title_references_work_and_batch_coverage_is_checked_before_merge():
    payload = prepare_copy_review_input(**inputs(3, 3))
    batches = prepare_copy_review_batches(payload)
    response = valid_review(batches[0])
    block(response["reviews"][0], batches[0]["items"][0], criterion="publishability", field="body")
    response["reviews"][0]["issues"][0]["related_item_ids"] = [batches[0]["items"][1]["item_id"]]
    assert decide_review(ItemReviewResponse.model_validate(response), batches[0]).status == "revision_required"
    with pytest.raises(AgentValidationError):
        merge_copy_review_results([ItemReviewResponse.model_validate(response)], payload)


def test_compact_requests_do_not_repeat_full_audience_or_unrelated_source_findings():
    payload = prepare_copy_review_input(**inputs(3, 3))
    batches = prepare_copy_review_batches(payload)
    original_chars = len(json.dumps(payload, ensure_ascii=False))
    assert all(len(json.dumps(batch, ensure_ascii=False)) < original_chars for batch in batches)
    for batch in batches:
        assert "audience_brief" not in batch["context"]
        assert "topic_patterns" not in batch["context"]["sample_analysis"]
        assert len(batch["rubric"]) == 6


def test_split_sample_analysis_adapts_copy_reviews_to_one_version_with_short_peer_summaries():
    payload = prepare_copy_review_input(**inputs(3, 3))
    batches = prepare_copy_review_batches(payload, sample_batches=4)

    assert len(batches) == 9
    assert all(len(batch["items"]) == 1 for batch in batches)
    assert all(batch["batch_policy"]["sample_batch_count"] == 4 for batch in batches)
    assert [batch["items"][0]["item_id"] for batch in batches] == [item["item_id"] for item in payload["items"]]
    for batch in batches:
        peers = batch["context"]["peer_summaries"]
        assert len(peers) == 2
        assert all(len(peer["opening"]) <= 120 for peer in peers)
        assert all(set(peer) == {"label", "opening"} for peer in peers)
        assert not any("variant_id" in peer or "item_id" in peer for peer in peers)
    results = [ItemReviewResponse.model_validate(valid_review(batch)) for batch in batches]
    merged = merge_copy_review_results(results, payload)
    assert [review.item_id for review in merged.reviews] == [item["item_id"] for item in payload["items"]]


@pytest.mark.parametrize("fault", ["cross_title", "quote", "score", "coverage"])
def test_per_batch_checks_reject_bad_references_quotes_scores_and_missing_versions(fault):
    batches = prepare_copy_review_batches(prepare_copy_review_input(**inputs(3, 3)))
    raw = valid_review(batches[0])
    block(raw["reviews"][0], batches[0]["items"][0], criterion="publishability", field="body")
    if fault == "cross_title":
        raw["reviews"][0]["issues"][0]["related_item_ids"] = [batches[1]["items"][0]["item_id"]]
    elif fault == "quote":
        raw["reviews"][0]["issues"][0]["quote"] = "不在当前正文的句子"
    elif fault == "score":
        raw["reviews"][0]["scores"].pop()
    else:
        raw["reviews"].pop()
    with pytest.raises(AgentValidationError):
        decide_review(ItemReviewResponse.model_validate(raw), batches[0])


@pytest.mark.parametrize("cache_fault", [None, "version", "fingerprint", "review", "shape", "results_shape"])
def test_failed_batch_resume_reuses_only_current_valid_judgements(cache_fault):
    model = GraphCompletion()
    def complete(**kwargs):
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        if kwargs["response_format"]["json_schema"]["name"] == "CopyReviewResponse":
            assert "output_schema" not in prompt
            batch = prompt["input"]
            assert len(batch["items"]) == 3
            if batch["batch"]["index"] == 2:
                raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)
    original = workflow(complete, count=3, variants=3)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    metadata = saved["pipeline"]["stages"]["copy"]["metadata"]
    cache = metadata["copy_review_batches"]
    assert len(cache["results"]) == 1
    assert metadata["copy_review_completed_batches"] == 1
    assert metadata["failed_role"] == "copy_review"
    if cache_fault in ("version", "fingerprint"):
        cache["version" if cache_fault == "version" else "input_fingerprint"] = "stale"
    elif cache_fault == "review":
        next(iter(cache["results"].values()))["reviews"][0]["item_id"] = "copy_unknown"
    elif cache_fault == "shape":
        metadata["copy_review_batches"] = None
    elif cache_fault == "results_shape":
        cache["results"] = []
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
    assert all(len(i["variants"]) == 3 for i in result["ideas"])
    calls = [c for c in recovery.wire_calls if c["response_format"]["json_schema"]["name"] == "CopyReviewResponse"]
    assert [json.loads(c["user_prompt"])["input"]["batch"]["index"] for c in calls] == ([2, 3] if cache_fault is None else [1, 2, 3])
    assert recovery.counts["CopyVariants"] == 0
    for field in ("sample_analysis", "audience", "topic", "title", "title_review", "copy"):
        assert resumed.state[field] == saved[field]
    final_metadata = resumed.state["pipeline"]["stages"]["copy"]["metadata"]
    assert final_metadata["copy_review_reused_batches"] == (1 if cache_fault is None else 0)
    assert final_metadata["copy_review_completed_batches"] == 3
    assert task.payload == before


def test_old_failed_copy_review_snapshot_without_cache_can_resume_with_batching():
    model = GraphCompletion(timeout_stage="copy_review")
    original = workflow(model, count=3)
    with pytest.raises(requests.ReadTimeout):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    saved["pipeline"]["stages"]["copy"]["metadata"].pop("copy_review_batches", None)
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=role)
    resumed.run()
    assert recovery.counts["copy_review"] == 3
    assert recovery.counts["CopyVariants"] == 0
