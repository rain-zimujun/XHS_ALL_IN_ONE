import copy

import pytest

from backend.app.schemas.creative_agents import ItemReviewResponse
from backend.app.services.creative_agents.title_generation import run_title_generation
from backend.app.services.creative_agents.title_review import prepare_title_review_input, select_titles
from backend.app.services.creative_agents.reviewing import decide_review
from backend.app.services.creative_graph.title_review import prepare_title_review_batches, merge_title_review_results
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
    assert [item for batch in batches for item in batch["items"]] == payload["items"]
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
