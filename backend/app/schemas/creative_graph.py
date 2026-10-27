"""Handwritten model responses for LangGraph, independent of legacy models.

TypedDict describes each response; TypeAdapter supplies runtime validation and
JSON Schema. Every model property is required, including nullable values.
"""
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, with_config
from typing_extensions import TypedDict
from backend.app.schemas.creative import COPY_TAG_MAX_COUNT, COPY_TAG_MIN_COUNT, MAX_COPY_BODY_CHARS


strict_response = with_config(ConfigDict(extra="forbid", str_strip_whitespace=True))
Text = Annotated[str, Field(min_length=1, max_length=600)]
NoteId = Annotated[str, Field(min_length=1, max_length=120)]
Quote = Annotated[str, Field(min_length=2, max_length=180)]
IdeaId = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^idea_[a-z0-9_]{1,39}$")]
TitleId = Annotated[str, Field(pattern=r"^title_[a-z0-9_]{1,60}$")]
CopyId = Annotated[str, Field(pattern=r"^copy_[a-z0-9_]{1,60}$")]
SegmentId = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]{1,39}$")]
AudienceTerm = Annotated[str, Field(min_length=1, max_length=60)]
Summary = Annotated[str, Field(min_length=1, max_length=1500)]
ContentType = Literal["作品展示", "灵感分享", "行业观察", "方法说明", "教程", "案例拆解"]
TopicCriterion = Literal["reader_value", "specificity", "fit", "novelty", "evidence"]
TopicField = Literal["topic", "primary_segment_id", "content_type", "reader_promise", "core_angle", "must_cover", "source_note_ids", "evidence", "boundaries", "differentiator"]


@strict_response
class Observation(TypedDict):
    note_id: NoteId
    content_type: Literal["作品展示", "灵感分享", "行业观察", "工具体验", "教程", "其他"]
    topic: Annotated[str, Field(min_length=1, max_length=100)]
    reader_value: Text
    quote: Quote


@strict_response
class Evidence(TypedDict):
    note_id: NoteId
    field: Literal["title", "body"]
    quote: Quote


@strict_response
class Finding(TypedDict):
    finding: Text
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=8)]


@strict_response
class StyleFinding(Finding):
    dimension: Literal["title", "opening", "tone", "paragraphs", "emoji", "tags", "ending"]


@strict_response
class SampleAnalysisResponse(TypedDict):
    """Analyze only the supplied text samples and retain verifiable evidence."""

    summary: Summary
    # A 2-note tail batch is valid internally; the public sample input still
    # requires at least 3 collected notes before batching starts.
    observations: Annotated[list[Observation], Field(min_length=2, max_length=50)]
    topic_patterns: Annotated[list[Finding], Field(min_length=1, max_length=8)]
    style_summary: Summary
    style_patterns: Annotated[list[StyleFinding], Field(min_length=1, max_length=7)]
    value_patterns: Annotated[list[Finding], Field(min_length=1, max_length=6)]
    avoid_patterns: Annotated[list[Finding], Field(max_length=6)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class SampleClaimDecisionResponse(TypedDict):
    claim_id: Annotated[str, Field(min_length=1, max_length=40)]
    supported: bool
    note_ids: Annotated[list[NoteId], Field(max_length=8)]
    reason: Text


@strict_response
class SampleReviewResponse(TypedDict):
    """Per-claim semantic evidence review; IDs are batch-bound at runtime."""

    decisions: Annotated[list[SampleClaimDecisionResponse], Field(min_length=1, max_length=80)]


@strict_response
class AudienceEvidence(TypedDict):
    basis: Literal["sample_evidence", "user_requirement", "hypothesis"]
    note_ids: Annotated[list[NoteId], Field(max_length=8)]
    statement: Text


@strict_response
class AudienceGroup(TypedDict):
    """Semantic grouping only; identifiers are assigned by the program."""
    audience_terms: Annotated[list[AudienceTerm], Field(min_length=1, max_length=4)]
    role_context: Text
    primary_scenario: Text
    attention_points: Annotated[list[Text], Field(min_length=1, max_length=6)]
    value_opportunities: Annotated[list[Text], Field(min_length=1, max_length=5)]
    suitable_formats: Annotated[list[ContentType], Field(min_length=1, max_length=4)]
    evidence: Annotated[list[AudienceEvidence], Field(min_length=1, max_length=5)]
    uncertainties: Annotated[list[Text], Field(max_length=5)]


@strict_response
class AudienceSegment(AudienceGroup):
    segment_id: SegmentId


@strict_response
class AudienceGroupingResponse(TypedDict):
    # A partial repair may need no new groups; merged coverage must still
    # include all audience terms before this step can complete.
    segments: Annotated[list[AudienceGroup], Field(max_length=8)]


@strict_response
class AudienceNeed(TypedDict):
    need: Text
    applies_to_segment_ids: Annotated[list[str], Field(min_length=1, max_length=8)]
    confidence: Literal["evidence", "hypothesis"]
    evidence: Annotated[list[AudienceEvidence], Field(min_length=1, max_length=5)]


@strict_response
class AudienceFormat(TypedDict):
    format: ContentType
    for_segment_ids: Annotated[list[str], Field(min_length=1, max_length=8)]
    reason: Text
    caution: Text


@strict_response
class AudienceResponse(TypedDict):
    """Map all user-supplied audience terms to concrete reader needs."""

    summary: Summary
    requested_audience_terms: Annotated[list[AudienceTerm], Field(min_length=1, max_length=20)]
    segments: Annotated[list[AudienceSegment], Field(min_length=1, max_length=8)]
    shared_needs: Annotated[list[AudienceNeed], Field(min_length=1, max_length=8)]
    format_recommendations: Annotated[list[AudienceFormat], Field(min_length=1, max_length=8)]
    avoid_assumptions: Annotated[list[Text], Field(max_length=8)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class TopicEvidence(TypedDict):
    basis: Literal["sample_pattern", "sample_observation", "user_requirement"]
    note_ids: Annotated[list[NoteId], Field(max_length=8)]
    statement: Text


@strict_response
class TopicIdea(TypedDict):
    idea_id: IdeaId
    primary_segment_id: SegmentId
    topic: Annotated[str, Field(min_length=1, max_length=120)]
    content_type: ContentType
    reader_promise: Text
    core_angle: Text
    must_cover: Annotated[list[Text], Field(min_length=1, max_length=8)]
    source_note_ids: Annotated[list[NoteId], Field(min_length=2, max_length=8)]
    evidence: Annotated[list[TopicEvidence], Field(min_length=1, max_length=5)]
    boundaries: Annotated[list[Text], Field(min_length=1, max_length=6)]
    differentiator: Text


@strict_response
class TopicResponse(TypedDict):
    """Select content directions, with source and audience bindings."""

    summary: Summary
    ideas: Annotated[list[TopicIdea], Field(min_length=1, max_length=5)]
    selection_limits: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class TopicScore(TypedDict):
    criterion: TopicCriterion
    score: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    comment: Text
    fields: Annotated[list[TopicField], Field(min_length=1, max_length=5)]
    note_ids: Annotated[list[NoteId], Field(max_length=8)]


@strict_response
class TopicIssue(TypedDict):
    criterion: TopicCriterion
    severity: Literal["blocking", "suggestion"]
    field: TopicField
    quote: Annotated[str, Field(min_length=1, max_length=180)]
    problem: Text
    revision_instruction: Text
    note_ids: Annotated[list[NoteId], Field(max_length=8)]
    related_idea_ids: Annotated[list[str], Field(max_length=4)]


@strict_response
class TopicJudgement(TypedDict):
    idea_id: IdeaId
    scores: Annotated[list[TopicScore], Field(min_length=5, max_length=5)]
    issues: Annotated[list[TopicIssue], Field(max_length=8)]
    preserve: Annotated[list[Text], Field(min_length=1, max_length=5)]


@strict_response
class TopicReviewResponse(TypedDict):
    """Evaluate topics; the application calculates pass/fail."""

    summary: Summary
    reviews: Annotated[list[TopicJudgement], Field(min_length=1, max_length=5)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class TitleCandidate(TypedDict):
    candidate_id: TitleId
    title: Annotated[str, Field(min_length=1, max_length=40)]
    angle: Text
    reader_value: Text
    promise: Text
    support_points: Annotated[list[Text], Field(min_length=1, max_length=8)]
    rationale: Text


@strict_response
class TitleGroup(TypedDict):
    idea_id: IdeaId
    candidates: Annotated[list[TitleCandidate], Field(min_length=5, max_length=8)]


@strict_response
class TitleResponse(TypedDict):
    """Create distinct candidates for the approved topics."""

    ideas: Annotated[list[TitleGroup], Field(min_length=1, max_length=5)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class ItemScore(TypedDict):
    criterion: Annotated[str, Field(min_length=1, max_length=60)]
    score: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    comment: Text
    fields: Annotated[list[str], Field(min_length=1, max_length=8)]
    note_ids: Annotated[list[NoteId], Field(max_length=8)]


@strict_response
class ItemIssue(TypedDict):
    criterion: Annotated[str, Field(min_length=1, max_length=60)]
    severity: Literal["blocking", "suggestion"]
    field: Annotated[str, Field(min_length=1, max_length=80)]
    quote: Annotated[str, Field(min_length=1, max_length=180)]
    problem: Text
    revision_instruction: Text
    note_ids: Annotated[list[NoteId], Field(max_length=8)]
    related_item_ids: Annotated[list[str], Field(max_length=8)]
    responsible_stage: Literal["topic", "title", "copy", "image"] | None


@strict_response
class ItemJudgement(TypedDict):
    item_id: Annotated[str, Field(min_length=1, max_length=80)]
    scores: Annotated[list[ItemScore], Field(min_length=1, max_length=12)]
    issues: Annotated[list[ItemIssue], Field(max_length=8)]
    preserve: Annotated[list[Text], Field(min_length=1, max_length=5)]
    comparison: Text


@strict_response
class TitleScores(TypedDict):
    hook: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    specificity: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    reader_benefit: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    relevance: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    difference: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]
    naturalness: Annotated[float, Field(ge=0, le=10, allow_inf_nan=False)]


@strict_response
class TitleIssue(TypedDict):
    criterion: Literal["hook", "specificity", "reader_benefit", "relevance", "difference", "naturalness"]
    severity: Literal["blocking", "suggestion"]
    field: Literal["title", "angle", "reader_value", "promise", "support_points", "rationale"]
    quote: Annotated[str, Field(min_length=1, max_length=180)]
    problem: Annotated[str, Field(min_length=1, max_length=120)]
    revision_instruction: Annotated[str, Field(min_length=1, max_length=120)]
    related_item_ids: Annotated[list[TitleId], Field(max_length=4)]


@strict_response
class TitleJudgement(TypedDict):
    item_id: TitleId
    scores: TitleScores
    comment: Annotated[str, Field(min_length=1, max_length=120)]
    issues: Annotated[list[TitleIssue], Field(max_length=8)]
    preserve: Annotated[list[Annotated[str, Field(min_length=1, max_length=80)]], Field(min_length=1, max_length=2)]
    comparison: Annotated[str, Field(min_length=1, max_length=100)]


@strict_response
class TitleReviewResponse(TypedDict):
    """Briefly score at most five candidates; the application calculates totals."""

    summary: Annotated[str, Field(min_length=1, max_length=200)]
    reviews: Annotated[list[TitleJudgement], Field(min_length=1, max_length=5)]
    limitations: Annotated[list[Annotated[str, Field(min_length=1, max_length=120)]], Field(min_length=1, max_length=3)]


@strict_response
class CopyVariant(TypedDict):
    variant_id: CopyId
    label: Annotated[str, Field(min_length=1, max_length=40)]
    body: Annotated[str, Field(min_length=1, max_length=MAX_COPY_BODY_CHARS)]
    tags: Annotated[list[str], Field(min_length=COPY_TAG_MIN_COUNT, max_length=COPY_TAG_MAX_COUNT, description="与本版最终正文最相关的 5～8 个话题，按相关性降序，可带 #")]
    covered_points: Annotated[list[Text], Field(min_length=1, max_length=8)]
    source_note_ids: Annotated[list[NoteId], Field(min_length=2, max_length=8)]


@strict_response
class CopyGroup(TypedDict):
    idea_id: IdeaId
    candidate_id: TitleId
    variants: Annotated[list[CopyVariant], Field(min_length=2, max_length=3)]


@strict_response
class CopyResponse(TypedDict):
    """Write directly usable copy variants for each locked title."""

    ideas: Annotated[list[CopyGroup], Field(min_length=1, max_length=5)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class CopyReviewResponse(TypedDict):
    """Evaluate the 2-3 bodies under one title, not self-reported coverage."""

    summary: Summary
    reviews: Annotated[list[ItemJudgement], Field(min_length=2, max_length=3)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class ImagePlan(TypedDict):
    purpose: Text
    composition: Annotated[str, Field(min_length=1, max_length=1000)]
    aspect_ratio: Literal["3:4", "1:1", "4:3", "9:16", "16:9"]
    search_description: Text
    prompt: Annotated[str, Field(min_length=1, max_length=3000)]
    negative_prompt: Annotated[str, Field(min_length=1, max_length=1000)]


@strict_response
class VariantImages(TypedDict):
    idea_id: IdeaId
    candidate_id: TitleId
    variant_id: CopyId
    cover_text: Annotated[str, Field(min_length=1, max_length=80)]
    image_plan: Annotated[list[ImagePlan], Field(min_length=1, max_length=6)]
    material_notes: Annotated[list[Text], Field(min_length=1, max_length=6)]


@strict_response
class ImageResponse(TypedDict):
    """Plan images for exactly one locked copy variant."""

    variants: Annotated[list[VariantImages], Field(min_length=1, max_length=1)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]


@strict_response
class FinalReviewResponse(TypedDict):
    """Review each complete note and identify the owner of every issue."""

    summary: Summary
    reviews: Annotated[list[ItemJudgement], Field(min_length=1, max_length=1)]
    limitations: Annotated[list[Text], Field(min_length=1, max_length=8)]
