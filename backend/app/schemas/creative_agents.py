"""Structured, evidence-linked outputs for the creative agents (M3)."""
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
