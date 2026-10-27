"""Per-batch TypedDict contract binding review IDs and source IDs in code."""
from typing import Annotated
from typing_extensions import TypedDict

from pydantic import ConfigDict, Field, with_config

from backend.app.services.creative_graph.source_schema import literal_values


def sample_review_response_type(payload: dict) -> type:
    claims = payload.get("claims", [])
    if not claims:
        raise ValueError("样本语义评审需要至少一条 claim")
    claim_ids = literal_values([claim["claim_id"] for claim in claims], "样本评审 claim_id")
    source_ids = literal_values(
        sorted({note_id for claim in claims for note_id in claim["candidate_note_ids"]}),
        "样本评审 note_id",
    )
    decision = with_config(ConfigDict(extra="forbid", str_strip_whitespace=True))(
        TypedDict("SampleClaimDecisionResponse", {
            "claim_id": claim_ids,
            "supported": bool,
            "note_ids": Annotated[list[source_ids], Field(max_length=8)],
            "reason": Annotated[str, Field(min_length=1, max_length=600)],
        }, total=True)
    )
    return with_config(ConfigDict(extra="forbid", str_strip_whitespace=True))(
        TypedDict("SampleReviewResponse", {
            "decisions": Annotated[list[decision], Field(min_length=len(claims), max_length=len(claims))],
        }, total=True)
    )
