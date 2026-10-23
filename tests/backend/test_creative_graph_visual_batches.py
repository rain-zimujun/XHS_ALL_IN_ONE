import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.app.schemas.creative_agents import ImagePlans, ItemReviewResponse
from backend.app.services.creative_agents.final_review import prepare_final_review_input
from backend.app.services.creative_agents.image_planning import prepare_image_input, validate_image_plans
from backend.app.services.creative_agents.reviewing import decide_review
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.visual_batches import (
    ImageBatchOutput, image_batch_revision, merge_image_batches, merge_final_batches,
    prepare_image_batches, prepare_final_batches,
)
from backend.app.services.creative_graph.workflow import LangGraphWorkflow
from backend.app.services.creative_orchestrator import AgentValidationError
from test_creative_image_agent import inputs as image_inputs, valid_images
from test_creative_final_review import inputs as final_inputs
from test_creative_title_review import valid_review, block
from test_creative_graph_workflow import GraphCompletion, workflow


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 3)])
def test_visual_batches_cover_each_copy_keep_full_fields_and_merge_in_original_order(count, variants):
    payload = prepare_image_input(**image_inputs(count, variants))
    before = copy.deepcopy(payload)
    batches = prepare_image_batches(payload)
    assert len(batches) == count * variants
    for batch in batches:
        assert len(batch["locked_copies"]) == len(batch["locked_titles"]) == len(batch["topic_brief"]["ideas"]) == 1
        assert batch["locked_copies"][0] in payload["locked_copies"]
        assert batch["locked_copies"][0]["idea_id"] == batch["locked_titles"][0]["idea_id"]
    results = [ImageBatchOutput.model_validate(valid_images(b)) for b in batches]
    merged = merge_image_batches(list(reversed(results)), payload)
    validate_image_plans(merged, payload)
    assert [v.variant_id for v in merged.variants] == [v["variant_id"] for v in payload["locked_copies"]]
    assert payload == before
    with pytest.raises(ValidationError):
        ImagePlans.model_validate(results[0].model_dump())
    with pytest.raises(CreativeError):
        merge_image_batches(results[:-1], payload)


@pytest.mark.parametrize("count,variants", [(1, 2), (3, 3), (5, 3)])
def test_final_batches_preserve_full_note_and_relevant_evidence_only(count, variants):
    payload = prepare_final_review_input(**final_inputs(count, variants))
    before = copy.deepcopy(payload)
    batches = prepare_final_batches(payload)
    assert len(batches) == count * variants
    assert [b["items"][0] for b in batches] == payload["items"]
    for batch in batches:
        assert len(batch["items"]) == 1
        assert "locked_copies" not in batch["context"]
        for field in ("topic_patterns", "style_patterns", "value_patterns", "avoid_patterns"):
            for pattern in batch["context"]["sample_analysis"][field]:
                assert pattern["evidence"]
                assert all(e["note_id"] in batch["sample_note_ids"] for e in pattern["evidence"])
    results = [ItemReviewResponse.model_validate(valid_review(b)) for b in batches]
    decision = decide_review(merge_final_batches(list(reversed(results)), payload), payload)
    assert decision.status == "passed"
    assert [d.judgement.item_id for d in decision.reviews] == [i["item_id"] for i in payload["items"]]
    assert payload == before


@pytest.mark.parametrize("fault", ["quote", "owner", "cross_item", "missing_score", "missing_item"])
def test_final_batch_still_checks_quotes_owners_references_and_scores(fault):
    batch = prepare_final_batches(prepare_final_review_input(**final_inputs()))[0]
    raw = valid_review(batch)
    review = raw["reviews"][0]
    block(review, batch["items"][0], criterion="safety", field="body")
    review["issues"][0]["responsible_stage"] = "copy"
    if fault == "quote":
        review["issues"][0]["quote"] = "这里没有这句话"
    elif fault == "owner":
        review["issues"][0]["responsible_stage"] = "image"
    elif fault == "cross_item":
        review["issues"][0]["related_item_ids"] = ["copy_unknown"]
    elif fault == "missing_score":
        review["scores"].pop()
    else:
        raw["reviews"] = []
    with pytest.raises((ValidationError, AgentValidationError)):
        decide_review(ItemReviewResponse.model_validate(raw), batch)


@pytest.mark.parametrize("role,prefix,stage", [("ImageResponse", "image", "image"), ("FinalReviewResponse", "final_review", "final_review")])
@pytest.mark.parametrize("fault", [None, "version", "fingerprint", "shape", "result"])
def test_failed_visual_batch_resume_reuses_valid_results_and_skips_upstream(role, prefix, stage, fault):
    model = GraphCompletion()
    def complete(**kwargs):
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        if kwargs["response_format"]["json_schema"]["name"] == role:
            assert "output_schema" not in prompt
            schema = kwargs["response_format"]["json_schema"]["schema"]
            key = "variants" if stage == "image" else "reviews"
            assert schema["properties"][key]["minItems"] == schema["properties"][key]["maxItems"] == 1
            if prompt["input"]["batch"]["index"] == 2:
                raise CreativeError("HTTP 401", "ai_http_error", http_status=401)
        return model(**kwargs)
    original = workflow(complete, count=3, variants=3)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    metadata = saved["pipeline"]["stages"][stage]["metadata"]
    cache = metadata[f"{prefix}_batches"]
    assert len(cache["results"]) == 1
    if fault == "version":
        cache["version"] = "old"
    elif fault == "fingerprint":
        cache["input_fingerprint"] = "old"
    elif fault == "shape":
        cache["results"] = []
    elif fault == "result":
        next(iter(cache["results"].values()))["variants" if stage == "image" else "reviews"] = []
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, resume_role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=resume_role)
    result, _ = resumed.run()
    calls = [c for c in recovery.wire_calls if c["response_format"]["json_schema"]["name"] == role]
    assert [json.loads(c["user_prompt"])["input"]["batch"]["index"] for c in calls] == list(range(2 if fault is None else 1, 10))
    assert len(result["ideas"]) == 3
    assert recovery.counts["CopyVariants"] == 0
    assert resumed.state["copy"] == saved["copy"]
    if stage == "final_review":
        assert recovery.counts["ImagePlans"] == 0
        assert resumed.state["image"] == saved["image"]


def test_image_revision_calls_only_target_variant_with_local_feedback_and_preserves_other_images():
    model = GraphCompletion(fail_stage="final_review", final_owner="image")
    w = workflow(model, count=3, variants=3)
    result, _ = w.run()
    assert result
    calls = [c for c in model.wire_calls if c["response_format"]["json_schema"]["name"] == "ImageResponse"]
    assert len(calls) == 10  # Nine first-pass variants, one targeted revision.
    revision = json.loads(calls[-1]["user_prompt"])["quality_revision"]
    assert len(revision["target_ids"]) == len(revision["previous_output"]["variants"]) == len(revision["review_feedback"]["reviews"]) == 1
    history = [h["output"] for h in w.state["pipeline"]["stages"]["image"]["history"] if h["role"] == "producer"]
    assert history[0]["variants"][1:] == history[1]["variants"][1:]


def test_image_revision_cache_identity_includes_feedback_and_prior_output():
    payload = prepare_image_input(**image_inputs())
    previous = valid_images(payload)
    revision = {"stage": "image", "target_ids": [previous["variants"][0]["variant_id"]],
                "previous_output": previous, "review_feedback": {"summary": "封面应具体", "reviews": []}, "rules": "返工"}
    batch = prepare_image_batches(payload, revision)[0]
    local = image_batch_revision(revision, batch)
    assert len(local["previous_output"]["variants"]) == 1
    result = ImageBatchOutput.model_validate(valid_images(batch))
    merged = merge_image_batches([result], payload, revision)
    assert merged.variants[1].model_dump() == previous["variants"][1]


def test_final_blocker_survives_batch_merge_with_responsible_stage():
    payload = prepare_final_review_input(**final_inputs(3, 3))
    batches = prepare_final_batches(payload)
    results = [valid_review(b) for b in batches]
    block(results[3]["reviews"][0], batches[3]["items"][0], criterion="safety", field="cover_text")
    results[3]["reviews"][0]["issues"][0]["responsible_stage"] = "image"
    decision = decide_review(merge_final_batches([ItemReviewResponse.model_validate(r) for r in results], payload), payload)
    assert decision.revision_required_ids == [batches[3]["items"][0]["item_id"]]
    assert decision.reviews[3].judgement.issues[0].responsible_stage == "image"
