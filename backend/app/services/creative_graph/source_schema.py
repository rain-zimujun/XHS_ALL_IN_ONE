"""Helpers for binding model-owned source references to current graph state."""
from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import ConfigDict, Field, with_config

from backend.app.schemas import creative_graph as responses
from backend.app.schemas.creative_graph import NoteId
from backend.app.schemas.creative import COPY_TAG_MAX_COUNT, COPY_TAG_MIN_COUNT, MAX_COPY_BODY_CHARS


def literal_values(values: list[str], label: str):
    """Create a JSON-Schema enum for a non-empty state-owned value list."""
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"动态 Schema 缺少合法的 {label}")
    return Literal[unique]


def source_note_ids_type(note_ids: list[str], *, max_length: int = 8, min_length: int = 0):
    """Return a list type whose values can only be current sample note IDs.

    An empty list is a valid state for compact reviews which intentionally do
    not receive source notes. Its max length is zero so the model cannot invent
    a sentinel value just to satisfy an enum.
    """
    if not note_ids:
        return Annotated[list[NoteId], Field(min_length=min_length, max_length=0)]
    source_id = literal_values(note_ids, "样本 note_id")
    return Annotated[list[source_id], Field(min_length=min_length, max_length=max_length)]


def _strict(name: str, annotations: dict) -> type:
    return with_config(ConfigDict(extra="forbid", str_strip_whitespace=True))(
        TypedDict(name, annotations, total=True)
    )


def topic_response_type(payload: dict) -> type:
    """Bind topic evidence and audience references to the current state."""
    segment_type = literal_values(payload["audience_segment_ids"], "受众分组")
    note_type = Annotated[list[literal_values(payload["sample_note_ids"], "样本 note_id")],
                          Field(max_length=8)]
    claim_ids = payload.get("approved_claim_ids", [])
    claim_type = (Annotated[list[literal_values(claim_ids, "获批 claim_id")], Field(max_length=8)]
                  if claim_ids else Annotated[list[str], Field(max_length=0)])
    evidence = _strict("BoundTopicEvidence", {
        "basis": Literal["sample_pattern", "sample_observation", "user_requirement"],
        "note_ids": note_type,
        "claim_ids": claim_type,
        "statement": responses.Text,
    })
    idea = _strict("BoundTopicIdea", {
        "primary_segment_id": segment_type,
        "topic": Annotated[str, Field(min_length=1, max_length=120)],
        "content_type": responses.ContentType,
        "reader_promise": responses.Text,
        "core_angle": responses.Text,
        "must_cover": Annotated[list[responses.Text], Field(min_length=1, max_length=8)],
        "evidence": Annotated[list[evidence], Field(min_length=1, max_length=5)],
        "boundaries": Annotated[list[responses.Text], Field(min_length=1, max_length=6)],
        "differentiator": responses.Text,
    })
    requested = payload["requested_idea_count"]
    return _strict("TopicResponse", {
        "summary": responses.Summary,
        "ideas": Annotated[list[idea], Field(min_length=requested, max_length=requested)],
        "selection_limits": Annotated[list[responses.Text], Field(min_length=1, max_length=8)],
    })


def copy_response_type(payload: dict) -> type:
    """Bind locked title IDs and approved claim/source IDs for this request."""
    idea_ids = [item["idea_id"] for item in payload["locked_titles"]]
    candidate_ids = [item["candidate_id"] for item in payload["locked_titles"]]
    claims_by_idea = payload.get("approved_claims_by_idea", {})
    claim_ids = list(dict.fromkeys(claim["claim_id"] for idea_id in idea_ids
                                   for claim in claims_by_idea.get(idea_id, [])))
    source_ids = list(dict.fromkeys(note_id for idea_id in idea_ids
                                    for claim in claims_by_idea.get(idea_id, []) for note_id in claim["note_ids"]))
    claim_field = (Annotated[list[literal_values(claim_ids, "获批 claim_id")], Field(max_length=8)]
                   if claim_ids else Annotated[list[str], Field(max_length=0)])
    variant = _strict("BoundCopyVariant", {
        "variant_id": responses.CopyId,
        "label": Annotated[str, Field(min_length=1, max_length=40)],
        "body": Annotated[str, Field(min_length=1, max_length=MAX_COPY_BODY_CHARS)],
        "tags": Annotated[list[str], Field(min_length=COPY_TAG_MIN_COUNT, max_length=COPY_TAG_MAX_COUNT)],
        "claim_ids": claim_field,
        "covered_points": Annotated[list[responses.Text], Field(min_length=1, max_length=8)],
        "source_note_ids": source_note_ids_type(source_ids or payload["sample_note_ids"], min_length=2, max_length=8),
    })
    group = _strict("BoundCopyGroup", {
        "idea_id": literal_values(idea_ids, "选题"),
        "candidate_id": literal_values(candidate_ids, "标题"),
        "variants": Annotated[list[variant], Field(min_length=payload["variant_count"], max_length=payload["variant_count"])],
    })
    return _strict("CopyResponse", {
        "ideas": Annotated[list[group], Field(min_length=len(idea_ids), max_length=len(idea_ids))],
        "limitations": Annotated[list[responses.Text], Field(min_length=1, max_length=8)],
    })
