from copy import deepcopy

import pytest

from backend.app.schemas.creative_agents import SampleAnalysis, SampleReviewResponse
from backend.app.services.creative_agents.sample_review import (
    prepare_sample_review_input, validate_sample_claim_coverage, validate_sample_review,
)
from backend.app.services.creative_graph.sample_review_schema import sample_review_response_type
from backend.app.services.creative_graph.model import response_parameter
from backend.app.services.creative_orchestrator import AgentValidationError
from test_creative_sample_agent import SEARCH, sources, valid_output
from backend.app.services.creative_agents.sample_analysis import prepare_sample_input


def prepared(count=3):
    payload = prepare_sample_input(SEARCH, sources(count), {"model_id": 1, "idea_count": 1})
    analysis = SampleAnalysis.model_validate(valid_output(payload))
    return payload, analysis


def test_review_input_contains_claims_and_only_current_batch_source_text():
    payload, analysis = prepared()
    review = prepare_sample_review_input(analysis, payload)
    assert review["sources"] == [{"note_id": item["note_id"], "title": item["title"],
                                  "body_segments": item["body_segments"], "relevance_tier": item["relevance_tier"]}
                                 for item in payload["sources"]]
    assert len(review["claims"]) > len(analysis.observations)
    assert [claim["claim_id"] for claim in review["claims"]] == [f"claim_{i:03d}" for i in range(1, len(review["claims"]) + 1)]


def test_response_schema_binds_claim_ids_and_sources_as_literals():
    payload, _ = prepared()
    review_payload = prepare_sample_review_input(SampleAnalysis.model_validate(valid_output(payload)), payload)
    schema = response_parameter(sample_review_response_type(review_payload))["json_schema"]["schema"]
    serialized = str(schema)
    assert "claim_001" in serialized
    assert "sample-0" in serialized
    assert "unknown" not in serialized


def test_valid_semantic_review_is_accepted():
    payload, analysis = prepared()
    review_payload = prepare_sample_review_input(analysis, payload)
    decisions = [{"claim_id": claim["claim_id"], "supported": True,
                  "note_ids": claim["candidate_note_ids"][:1], "reason": "有对应样本支持。"}
                 for claim in review_payload["claims"]]
    validate_sample_review(SampleReviewResponse(decisions=decisions), review_payload)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "source", "unsupported"])
def test_review_rejects_incomplete_or_invalid_decisions(mutation):
    payload, analysis = prepared()
    review_payload = prepare_sample_review_input(analysis, payload)
    decisions = [{"claim_id": claim["claim_id"], "supported": True,
                  "note_ids": claim["candidate_note_ids"][:1], "reason": "有对应样本支持。"}
                 for claim in review_payload["claims"]]
    if mutation == "missing":
        decisions.pop()
    elif mutation == "duplicate":
        decisions[-1] = deepcopy(decisions[0])
    elif mutation == "unknown":
        decisions[-1]["claim_id"] = "claim_unknown"
    elif mutation == "source":
        decisions[0]["note_ids"] = ["not-in-claim"]
    else:
        decisions[0].update(supported=False, note_ids=[], reason="原文不支持该结论。")
    with pytest.raises(AgentValidationError):
        validate_sample_review(SampleReviewResponse(decisions=decisions), review_payload)


def test_claim_source_ids_are_locally_bound_to_the_batch():
    payload, analysis = prepared()
    broken = analysis.model_copy(deep=True)
    broken.topic_patterns[0].evidence[0].note_id = "unknown"
    with pytest.raises(AgentValidationError):
        validate_sample_claim_coverage(broken, payload)
