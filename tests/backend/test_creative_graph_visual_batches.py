import copy
import json
from threading import Barrier, Lock, RLock
from time import sleep
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.app.schemas.creative_agents import ImagePlans, ItemReviewResponse
from backend.app.services.creative_agents.final_review import prepare_final_review_input
from backend.app.services.creative_agents.image_planning import prepare_image_input, validate_image_plans
from backend.app.services.creative_agents.reviewing import decide_review
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.batch_runner import run_batches
from backend.app.services.creative_graph import workflow as workflow_module
from backend.app.services.creative_graph.adaptive_batches import MAX_PARALLEL_BATCHES
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
    idea_id = payload["locked_copies"][0]["idea_id"]
    payload["approved_claims_by_idea"] = {idea_id: [{
        "claim_id": "claim_image", "text": "冷色配色与轮廓观察。", "note_ids": payload["locked_copies"][0]["source_note_ids"],
    }]}
    payload["locked_copies"][0]["claim_ids"] = ["claim_image"]
    before = copy.deepcopy(payload)
    batches = prepare_image_batches(payload)
    assert len(batches) == count * variants
    for batch in batches:
        assert len(batch["locked_copies"]) == len(batch["locked_titles"]) == len(batch["topic_brief"]["ideas"]) == 1
        assert batch["locked_copies"][0] in payload["locked_copies"]
        assert batch["locked_copies"][0]["idea_id"] == batch["locked_titles"][0]["idea_id"]
        assert set(batch["approved_claims_by_idea"]) == {batch["locked_copies"][0]["idea_id"]}
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
    for item in payload["items"]:
        item["evidence"] = {"claim_ids": ["claim_final"], "source_note_ids": item["fields"]["source_note_ids"],
                            "approved_claims": [{"claim_id": "claim_final", "text": "获批配色观察。",
                                                 "note_ids": item["fields"]["source_note_ids"]}]}
    before = copy.deepcopy(payload)
    batches = prepare_final_batches(payload)
    assert len(batches) == count * variants
    assert [b["items"][0] for b in batches] == payload["items"]
    for batch in batches:
        assert len(batch["items"]) == 1
        assert batch["items"][0]["evidence"]["approved_claims"][0]["text"] == "获批配色观察。"
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
    total_batches = 9
    assert 1 <= len(cache["results"]) < total_batches
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
    discarded_cache = fault in ("version", "fingerprint", "shape")
    valid_cached_batches = 0 if discarded_cache else len(cache["results"])
    invalid_cached_batch = 1 if fault == "result" else 0
    assert len(calls) == total_batches - valid_cached_batches + invalid_cached_batch
    assert len(result["ideas"]) == 3
    assert recovery.counts["CopyVariants"] == 0
    assert resumed.state["copy"] == saved["copy"]
    if stage == "final_review":
        assert recovery.counts["ImagePlans"] == 0
        assert resumed.state["image"] == saved["image"]


def test_run_batches_executes_at_most_three_concurrently_and_merges_in_input_order():
    class Output:
        def __init__(self, value):
            self.value = value

        @classmethod
        def model_validate(cls, value):
            return value if isinstance(value, cls) else cls(value)

        def model_dump(self, mode=None):
            return self.value

    entered = Barrier(3)
    lock = Lock()
    active = peak = 0
    metadata = {}
    checkpoint_indices = []

    def checkpoint(_text):
        checkpoint_indices.append(metadata["image_batch"]["index"])

    class Agent:
        output_model = Output
        spec = SimpleNamespace(stage="image", name="图片策划 Agent")

        def invoke(self, batch, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            entered.wait(timeout=5)
            if batch["order"] == 0:
                sleep(0.02)
            with lock:
                active -= 1
            return SimpleNamespace(output=Output(batch["order"]), attempts=())

    batches = [{"order": i, "input_fingerprint": str(i)} for i in range(6)]
    result = run_batches(
        agent=Agent(), payload={"input_fingerprint": "payload"}, batches=batches,
        metadata=metadata, prefix="image", version="test", validate=lambda *_: None,
        merge=lambda values: values, checkpoint=checkpoint, max_workers=3, state_lock=RLock(),
    )

    assert peak == 3
    assert [item.value for item in result.output] == list(range(6))
    assert set(checkpoint_indices) == set(range(1, 7))
    assert 1 <= metadata["image_batch"]["index"] <= 6


def test_run_batches_resume_skips_every_successful_parallel_batch():
    class Output:
        def __init__(self, value):
            self.value = value

        @classmethod
        def model_validate(cls, value):
            return value if isinstance(value, cls) else cls(value)

        def model_dump(self, mode=None):
            return self.value

    batches = [{"order": i, "input_fingerprint": str(i)} for i in range(6)]
    cache_metadata = {}

    class Agent:
        output_model = Output
        spec = SimpleNamespace(stage="image", name="图片策划 Agent")

        def __init__(self, fail):
            self.fail = fail
            self.called = []

        def invoke(self, batch, **kwargs):
            self.called.append(batch["order"])
            if self.fail and batch["order"] == 2:
                raise RuntimeError("batch failed")
            return SimpleNamespace(output=Output(batch["order"]), attempts=())

    first = Agent(fail=True)
    with pytest.raises(RuntimeError, match="batch failed"):
        run_batches(
            agent=first, payload={"input_fingerprint": "payload"}, batches=batches,
            metadata=cache_metadata, prefix="image", version="test", validate=lambda *_: None,
            merge=lambda values: values, checkpoint=lambda *_: None, max_workers=3,
        )
    cached = set(cache_metadata["image_batches"]["results"])
    assert cached
    assert "2" not in cached

    resumed = Agent(fail=False)
    result = run_batches(
        agent=resumed, payload={"input_fingerprint": "payload"}, batches=batches,
        metadata=cache_metadata, prefix="image", version="test", validate=lambda *_: None,
        merge=lambda values: values, checkpoint=lambda *_: None, max_workers=3,
    )

    assert set(resumed.called) == {int(key) for key in {b["input_fingerprint"] for b in batches} - cached}
    assert [item.value for item in result.output] == list(range(6))


def test_independent_review_and_visual_nodes_use_shared_parallel_limit(monkeypatch):
    configured = {}
    original = workflow_module.run_batches

    def capture_parallelism(**kwargs):
        configured[kwargs["prefix"]] = kwargs["max_workers"]
        return original(**kwargs)

    monkeypatch.setattr(workflow_module, "run_batches", capture_parallelism)
    result, _ = workflow(GraphCompletion()).run()

    assert result
    assert configured == {
        "title_review": MAX_PARALLEL_BATCHES,
        "copy_review": MAX_PARALLEL_BATCHES,
        "image": MAX_PARALLEL_BATCHES,
        "final_review": MAX_PARALLEL_BATCHES,
    }


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
