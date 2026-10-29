"""Dynamic TypedDict contracts for item-level review requests.

The review rubric and item fields are request data, so a static response type
cannot express which criterion or field is valid for the current batch. These
contracts keep the explicit TypedDict/JSON-Schema path while narrowing those
references per request.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, with_config
from typing_extensions import TypedDict

from backend.app.schemas.creative_graph import Summary, Text, TitleId
from backend.app.services.creative_graph.model import response_parameter
from backend.app.services.creative_graph.source_schema import literal_values, source_note_ids_type


def _strict(name: str, annotations: dict) -> type:
    response = TypedDict(name, annotations, total=True)
    return with_config(ConfigDict(extra="forbid", str_strip_whitespace=True))(response)


def _literal(values: list[str], label: str):
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"动态评审 Schema 缺少合法的 {label}")
    return Literal[unique]


def _peer_refs(item_ids: list[str], *, max_length: int):
    """References to other review items in the same batch.

    A single-item batch has no legal peer: the business check rejects self
    references, so the contract must forbid the field instead of offering an
    enum whose only member would always be rejected. The empty form mirrors
    ``source_note_ids_type`` for compact reviews without source notes.
    """
    if len(item_ids) < 2:
        return Annotated[list[str], Field(max_length=0)]
    return Annotated[list[_literal(item_ids, "关联对象")], Field(max_length=max_length)]


def _batch_values(payload: dict) -> tuple[list[str], list[str], list[str]]:
    criteria = [item["key"] for item in payload["rubric"]]
    item_ids = [item["item_id"] for item in payload["items"]]
    fields = [field for item in payload["items"] for field in item["fields"]]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("动态评审 Schema 的 item_id 不得重复")
    return criteria, item_ids, fields


def _note_ids(payload: dict):
    return source_note_ids_type(payload.get("sample_note_ids", []), max_length=8)


def item_review_response_type(payload: dict) -> type:
    """Return a native JSON Schema with criterion/field/item enums for a batch."""
    criteria, item_ids, fields = _batch_values(payload)
    criterion = _literal(criteria, "评分项")
    field = _literal(fields, "字段")
    item_id = _literal(item_ids, "评审对象")
    score = _strict("DynamicItemScore", {
        "criterion": criterion,
        "score": Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)],
        "comment": Text,
        "fields": Annotated[list[field], Field(min_length=1, max_length=8)],
        "note_ids": _note_ids(payload),
    })
    issue = _strict("DynamicItemIssue", {
        "criterion": criterion,
        "severity": Literal["blocking", "suggestion"],
        "field": field,
        "quote": Annotated[str, Field(min_length=1, max_length=180)],
        "problem": Text,
        "revision_instruction": Text,
        "note_ids": _note_ids(payload),
        "related_item_ids": _peer_refs(item_ids, max_length=8),
        "responsible_stage": Literal["topic", "title", "copy", "image", None],
    })
    judgement = _strict("DynamicItemJudgement", {
        "item_id": item_id,
        "scores": Annotated[list[score], Field(min_length=1, max_length=12)],
        "issues": Annotated[list[issue], Field(max_length=8)],
        "preserve": Annotated[list[Text], Field(min_length=1, max_length=5)],
        "comparison": Text,
    })
    response_name = {"copy": "CopyReviewResponse", "final": "FinalReviewResponse"}.get(payload.get("kind"), "ItemReviewResponse")
    response = _strict(response_name, {
        # LangGraph binds revision and fingerprint from state after parsing;
        # they are deliberately not model-owned response fields.
        "summary": Summary,
        "reviews": Annotated[list[judgement], Field(min_length=len(item_ids), max_length=len(item_ids))],
        "limitations": Annotated[list[Text], Field(min_length=1, max_length=8)],
    })
    return response


def title_review_response_type(payload: dict) -> type:
    """Return the compact title-review contract with batch-specific references."""
    criteria, item_ids, fields = _batch_values(payload)
    criterion = _literal(criteria, "评分项")
    field = _literal(fields, "字段")
    item_id = _literal(item_ids, "评审对象")
    scores = _strict("DynamicTitleScores", {
        key: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)] for key in criteria
    })
    issue = _strict("DynamicTitleIssue", {
        "criterion": criterion,
        "severity": Literal["blocking", "suggestion"],
        "field": field,
        "quote": Annotated[str, Field(min_length=1, max_length=180)],
        "problem": Annotated[str, Field(min_length=1, max_length=120)],
        "revision_instruction": Annotated[str, Field(min_length=1, max_length=120)],
        "related_item_ids": _peer_refs(item_ids, max_length=4),
    })
    judgement = _strict("DynamicTitleJudgement", {
        "item_id": item_id,
        "scores": scores,
        "comment": Annotated[str, Field(min_length=1, max_length=120)],
        "issues": Annotated[list[issue], Field(max_length=8)],
        "preserve": Annotated[list[Annotated[str, Field(min_length=1, max_length=80)]], Field(min_length=1, max_length=2)],
        "comparison": Annotated[str, Field(min_length=1, max_length=100)],
    })
    response = _strict("TitleReviewResponse", {
        "summary": Annotated[str, Field(min_length=1, max_length=200)],
        "reviews": Annotated[list[judgement], Field(min_length=len(item_ids), max_length=len(item_ids))],
        "limitations": Annotated[list[Annotated[str, Field(min_length=1, max_length=120)]], Field(min_length=1, max_length=3)],
    })
    return response


def topic_review_response_type(payload: dict) -> type:
    """Bind topic-review criteria, fields, ideas and sample IDs to the batch."""
    criteria = [item["key"] for item in payload["rubric"]]
    item_ids = [idea["idea_id"] for idea in payload["topic_brief"]["ideas"]]
    fields = [
        "topic", "primary_segment_id", "content_type", "reader_promise", "core_angle",
        "must_cover", "source_note_ids", "evidence", "boundaries", "differentiator",
    ]
    criterion = literal_values(criteria, "评分项")
    field = literal_values(fields, "字段")
    idea_id = literal_values(item_ids, "选题")
    score = _strict("DynamicTopicScore", {
        "criterion": criterion,
        "score": Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)],
        "comment": Text,
        "fields": Annotated[list[field], Field(min_length=1, max_length=5)],
        "note_ids": _note_ids(payload),
    })
    issue = _strict("DynamicTopicIssue", {
        "criterion": criterion,
        "severity": Literal["blocking", "suggestion"],
        "field": field,
        "quote": Annotated[str, Field(min_length=1, max_length=180)],
        "problem": Text,
        "revision_instruction": Text,
        "note_ids": _note_ids(payload),
        "related_idea_ids": _peer_refs(item_ids, max_length=4),
    })
    judgement = _strict("DynamicTopicJudgement", {
        "idea_id": idea_id,
        "scores": Annotated[list[score], Field(min_length=len(criteria), max_length=len(criteria))],
        "issues": Annotated[list[issue], Field(max_length=8)],
        "preserve": Annotated[list[Text], Field(min_length=1, max_length=5)],
    })
    return _strict("TopicReviewResponse", {
        "summary": Summary,
        "reviews": Annotated[list[judgement], Field(min_length=len(item_ids), max_length=len(item_ids))],
        "limitations": Annotated[list[Text], Field(min_length=1, max_length=8)],
    })
