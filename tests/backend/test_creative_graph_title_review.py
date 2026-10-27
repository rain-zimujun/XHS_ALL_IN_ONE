import copy

import pytest

from backend.app.schemas.creative_agents import ItemReviewResponse
from backend.app.services.creative_agents.title_generation import run_title_generation
from backend.app.services.creative_agents.title_review import prepare_title_review_input, select_titles
from backend.app.services.creative_agents.reviewing import decide_review
from backend.app.services.creative_graph.title_review import (
    expand_title_review, prepare_title_review_batches, merge_title_review_results,
)
from test_creative_title_review import valid_review
from test_creative_titles import inputs, TitleCompletion


@pytest.mark.parametrize("count,sizes", [(5, [5]), (6, [3, 3]), (7, [4, 3]), (8, [4, 4])])
def test_batches_bound_work_keep_topic_peers_and_merge_in_original_order(count, sizes):
    args = inputs(count=3, candidate_count=count)
    run, _ = run_title_generation(**args, complete=TitleCompletion())
    payload = prepare_title_review_input(**args, title_candidates=run.output, title_revision=1)
    before = copy.deepcopy(payload)
    batches = prepare_title_review_batches(payload)
    assert [len(batch["items"]) for batch in batches] == sizes * 3
    batched = [item for batch in batches for item in batch["items"]]
    assert [item["item_id"] for item in batched] == [item["item_id"] for item in payload["items"]]
    assert all("claim_ids" not in item["fields"] and "source_note_ids" not in item["fields"] for item in batched)
    assert all("evidence" in item for item in batched)
    for batch in batches:
        idea_id = batch["context"]["topic_brief"]["ideas"][0]["idea_id"]
        assert {item["idea_id"] for item in batch["items"]} == {idea_id}
    assert prepare_title_review_batches(payload) == batches
    results = [ItemReviewResponse.model_validate(valid_review(batch)) for batch in batches]
    merged = merge_title_review_results(list(reversed(results)), payload)
    assert [review.item_id for review in merged.reviews] == [item["item_id"] for item in payload["items"]]
    selected = select_titles(decide_review(merged, payload), payload)
    assert selected.status == "passed"
    assert len(selected.selected_candidate_ids) == 3
    assert payload == before


def test_title_judge_receives_relevant_approved_claims_and_candidate_bindings():
    args = inputs(count=1, candidate_count=5, with_claims=True)
    run, _ = run_title_generation(**args, complete=TitleCompletion())
    payload = prepare_title_review_input(**args, title_candidates=run.output, title_revision=1)
    batch = prepare_title_review_batches(payload)[0]
    idea = batch["context"]["topic_brief"]["ideas"][0]
    assert len(idea["approved_claims"]) == 2
    assert idea["approved_claims"][0]["text"].startswith("服装配色观察")
    assert batch["items"][0]["evidence"]["claim_ids"]
    assert batch["items"][0]["evidence"]["source_note_ids"]
    assert "claim_ids" not in batch["items"][0]["fields"]


def test_compact_title_review_promotes_existing_low_score_issue_to_blocking():
    args = inputs(count=1, candidate_count=5)
    run, _ = run_title_generation(**args, complete=TitleCompletion())
    payload = prepare_title_review_input(**args, title_candidates=run.output, title_revision=1)
    batch = prepare_title_review_batches(payload)[0]
    compact = valid_review(batch)
    review = compact["reviews"][0]
    for current in compact["reviews"]:
        current["comment"] = "标题切入明确，但读者收益需要更具体。"
        current["scores"] = {score["criterion"]: score["score"] for score in current["scores"]}
    review["scores"]["reader_benefit"] = 5.0
    review["issues"] = [{
        "criterion": "reader_benefit", "severity": "suggestion", "field": "title",
        "quote": batch["items"][0]["fields"]["title"], "problem": "读者收益不够具体。",
        "revision_instruction": "补出明确的观察收益。", "related_item_ids": [],
    }]
    normalized = expand_title_review(compact)
    assert normalized["reviews"][0]["issues"][0]["severity"] == "blocking"
    from backend.app.schemas.creative_agents import ItemReviewResponse
    checked = ItemReviewResponse.model_validate(normalized)
    decision = decide_review(checked, batch)
    assert decision.status == "revision_required"


def test_compact_title_review_synthesizes_missing_low_score_issue_from_candidate():
    args = inputs(count=1, candidate_count=5)
    run, _ = run_title_generation(**args, complete=TitleCompletion())
    payload = prepare_title_review_input(**args, title_candidates=run.output, title_revision=1)
    batch = prepare_title_review_batches(payload)[0]
    compact = valid_review(batch)
    for review in compact["reviews"]:
        review["scores"] = {
            score["criterion"]: 6.5 for score in review["scores"]
        }
        review["comment"] = "主题明确，但措辞读起来不顺。"
    review = compact["reviews"][0]
    review["scores"]["naturalness"] = 3.5

    normalized = expand_title_review(compact, batch)
    item = normalized["reviews"][0]
    issue = next(issue for issue in item["issues"] if issue["criterion"] == "naturalness")
    assert issue["severity"] == "suggestion"
    assert issue["quote"] == batch["items"][0]["fields"]["title"]
    assert "3.5" in issue["problem"]
    assert "自然" in issue["revision_instruction"]
    decision = decide_review(ItemReviewResponse.model_validate(normalized), batch)
    assert decision.status == "revision_required"
    assert decision.reviews[0].quality.passed is False


def test_split_sample_analysis_adapts_title_reviews_to_one_title_with_id_free_peer_summaries():
    args = inputs(count=3, candidate_count=5)
    run, _ = run_title_generation(**args, complete=TitleCompletion())
    payload = prepare_title_review_input(**args, title_candidates=run.output, title_revision=1)
    batches = prepare_title_review_batches(payload, sample_batches=4)

    assert len(batches) == 15
    assert all(len(batch["items"]) == 1 for batch in batches)
    assert all(batch["batch_policy"] == {
        "version": "adaptive-batches-v1", "sample_batch_count": 4,
        "mode": "per_item", "batch_size": 1,
    } for batch in batches)
    for batch in batches:
        peers = batch["context"]["peer_summaries"]
        assert 1 <= len(peers) <= 4
        assert all(set(peer) == {"title", "angle"} for peer in peers)
        assert not any("candidate_id" in peer or "item_id" in peer for peer in peers)
    results = [ItemReviewResponse.model_validate(valid_review(batch)) for batch in batches]
    merged = merge_title_review_results(results, payload)
    assert [review.item_id for review in merged.reviews] == [item["item_id"] for item in payload["items"]]
