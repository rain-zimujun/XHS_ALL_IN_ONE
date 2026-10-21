"""Structured, evidence-linked outputs for the creative agents (M3–M6)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


NoteId = Annotated[str, Field(min_length=1, max_length=120)]
FindingText = Annotated[str, Field(min_length=1, max_length=600)]
EvidenceQuote = Annotated[str, Field(min_length=2, max_length=180)]


class AgentOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SampleObservation(AgentOutputModel):
    note_id: NoteId
    content_type: Literal["作品展示", "灵感分享", "行业观察", "工具体验", "教程", "其他"]
    topic: str = Field(min_length=1, max_length=100)
    reader_value: FindingText
    quote: EvidenceQuote = Field(description="该笔记已提供标题或正文片段中的连续原文，不拼接或改写")


class SampleEvidence(AgentOutputModel):
    note_id: NoteId
    field: Literal["title", "body"]
    quote: EvidenceQuote = Field(description="对应字段中可以逐字核对的连续短摘录")


class SampleFinding(AgentOutputModel):
    finding: FindingText
    evidence: list[SampleEvidence] = Field(min_length=1, max_length=8)


class SampleStyleFinding(SampleFinding):
    dimension: Literal["title", "opening", "tone", "paragraphs", "emoji", "tags", "ending"]


class SampleAnalysis(AgentOutputModel):
    summary: str = Field(min_length=1, max_length=1500)
    observations: list[SampleObservation] = Field(
        min_length=3, max_length=50, description="逐篇记录，每个输入 note_id 恰好出现一次，不能遗漏样本",
    )
    topic_patterns: list[SampleFinding] = Field(min_length=1, max_length=8)
    style_summary: str = Field(min_length=1, max_length=1500)
    style_patterns: list[SampleStyleFinding] = Field(min_length=1, max_length=7)
    value_patterns: list[SampleFinding] = Field(min_length=1, max_length=6)
    avoid_patterns: list[SampleFinding] = Field(
        max_length=6, description="有样本依据、但不适合本次需求的表达模式；没有依据时返回空数组",
    )
    limitations: list[FindingText] = Field(min_length=1, max_length=8)


AudienceTerm = Annotated[str, Field(min_length=1, max_length=60)]


class AudienceEvidence(AgentOutputModel):
    basis: Literal["sample_evidence", "user_requirement", "hypothesis"]
    note_ids: list[NoteId] = Field(default_factory=list, max_length=8)
    statement: FindingText


class AudienceSegment(AgentOutputModel):
    segment_id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]{1,39}$")
    audience_terms: list[AudienceTerm] = Field(min_length=1, max_length=4)
    role_context: FindingText
    primary_scenario: FindingText
    attention_points: list[FindingText] = Field(min_length=1, max_length=6)
    value_opportunities: list[FindingText] = Field(min_length=1, max_length=5)
    suitable_formats: list[Literal["作品展示", "灵感分享", "行业观察", "方法说明", "教程", "案例拆解"]] = Field(min_length=1, max_length=4)
    evidence: list[AudienceEvidence] = Field(min_length=1, max_length=5)
    uncertainties: list[FindingText] = Field(default_factory=list, max_length=5)


class AudienceNeed(AgentOutputModel):
    need: FindingText
    applies_to_segment_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: Literal["evidence", "hypothesis"]
    evidence: list[AudienceEvidence] = Field(min_length=1, max_length=5)


class AudienceFormatRecommendation(AgentOutputModel):
    format: Literal["作品展示", "灵感分享", "行业观察", "方法说明", "教程", "案例拆解"]
    for_segment_ids: list[str] = Field(min_length=1, max_length=8)
    reason: FindingText
    caution: FindingText


class AudienceBrief(AgentOutputModel):
    """A reader brief, not a market-research or demographic claim."""

    summary: str = Field(min_length=1, max_length=1500)
    requested_audience_terms: list[AudienceTerm] = Field(min_length=1, max_length=20)
    segments: list[AudienceSegment] = Field(min_length=1, max_length=8)
    shared_needs: list[AudienceNeed] = Field(min_length=1, max_length=8)
    format_recommendations: list[AudienceFormatRecommendation] = Field(min_length=1, max_length=8)
    avoid_assumptions: list[FindingText] = Field(default_factory=list, max_length=8)
    limitations: list[FindingText] = Field(min_length=1, max_length=8)


TopicContentType = Literal["作品展示", "灵感分享", "行业观察", "方法说明", "教程", "案例拆解"]


class TopicEvidence(AgentOutputModel):
    basis: Literal["sample_pattern", "sample_observation", "user_requirement"]
    note_ids: list[NoteId] = Field(default_factory=list, max_length=8, description="本条依据的不同来源；用户要求不附样本 ID")
    statement: FindingText


class TopicIdea(AgentOutputModel):
    idea_id: str = Field(min_length=1, max_length=40, pattern=r"^idea_[a-z0-9_]{1,39}$")
    primary_segment_id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]{1,39}$")
    topic: str = Field(min_length=1, max_length=120)
    content_type: TopicContentType
    reader_promise: FindingText
    core_angle: FindingText
    must_cover: list[FindingText] = Field(min_length=1, max_length=8)
    source_note_ids: list[NoteId] = Field(min_length=2, max_length=8, description="至少两个不同来源，与 evidence 的来源合集一致")
    evidence: list[TopicEvidence] = Field(min_length=1, max_length=5)
    boundaries: list[FindingText] = Field(min_length=1, max_length=6)
    differentiator: FindingText


class TopicBrief(AgentOutputModel):
    summary: str = Field(min_length=1, max_length=1500)
    ideas: list[TopicIdea] = Field(min_length=1, max_length=5)
    selection_limits: list[FindingText] = Field(min_length=1, max_length=8)


TopicCriterion = Literal["reader_value", "specificity", "fit", "novelty", "evidence"]
TopicReviewField = Literal[
    "topic", "primary_segment_id", "content_type", "reader_promise", "core_angle",
    "must_cover", "source_note_ids", "evidence", "boundaries", "differentiator",
]


class TopicCriterionScore(AgentOutputModel):
    criterion: TopicCriterion
    score: float = Field(ge=0, le=10, strict=True, allow_inf_nan=False)
    comment: FindingText
    fields: list[TopicReviewField] = Field(min_length=1, max_length=5)
    note_ids: list[NoteId] = Field(default_factory=list, max_length=8)


class TopicReviewIssue(AgentOutputModel):
    criterion: TopicCriterion
    severity: Literal["blocking", "suggestion"]
    field: TopicReviewField
    quote: str = Field(min_length=1, max_length=180, description="问题字段中的连续原文，不能改写或跨数组项拼接")
    problem: FindingText
    revision_instruction: FindingText
    note_ids: list[NoteId] = Field(default_factory=list, max_length=8)
    related_idea_ids: list[str] = Field(default_factory=list, max_length=4)


class TopicJudgement(AgentOutputModel):
    idea_id: str = Field(min_length=1, max_length=40, pattern=r"^idea_[a-z0-9_]{1,39}$")
    scores: list[TopicCriterionScore] = Field(min_length=5, max_length=5)
    issues: list[TopicReviewIssue] = Field(max_length=8)
    preserve: list[FindingText] = Field(min_length=1, max_length=5)


class TopicReviewResponse(AgentOutputModel):
    """Model judgement only: pass/fail, totals and selection are program-owned."""

    topic_revision: int = Field(ge=1, strict=True)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1, max_length=1500)
    reviews: list[TopicJudgement] = Field(min_length=1, max_length=5)
    limitations: list[FindingText] = Field(min_length=1, max_length=8)
