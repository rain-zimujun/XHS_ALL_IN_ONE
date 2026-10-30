"""Per-batch response contract for sample analysis."""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from pydantic import ConfigDict, Field, with_config

from backend.app.schemas import creative_graph as responses
from backend.app.schemas.creative_graph import Text, Quote
from backend.app.services.creative_graph.source_schema import literal_values


def _strict(name: str, annotations: dict) -> type:
    return with_config(ConfigDict(extra="forbid", str_strip_whitespace=True))(
        TypedDict(name, annotations, total=True)
    )


def sample_response_type(payload: dict) -> type:
    """Bind observation/evidence note IDs to this sample batch."""
    # Diagnostics and transport tests may exercise the agent without a real
    # batch. Keep the base contract available for that low-level path; graph
    # execution always supplies sources and receives the bound contract below.
    if not payload.get("sources"):
        return responses.SampleAnalysisResponse
    note_ids = [source["note_id"] for source in payload["sources"]]
    note_id = literal_values(note_ids, "样本 note_id")
    observation = _strict("BoundSampleObservation", {
        "note_id": note_id,
        "content_type": literal_values(
            ["作品展示", "灵感分享", "行业观察", "工具体验", "教程", "其他"], "样本内容类型",
        ),
        "topic": Annotated[str, Field(min_length=1, max_length=100)],
        "reader_value": Text,
        "quote": Quote,
    })
    evidence = _strict("BoundSampleEvidence", {
        "note_id": note_id,
        "field": literal_values(["title", "body"], "样本字段"),
        "quote": Quote,
    })
    finding = _strict("BoundSampleFinding", {
        "finding": Text,
        "evidence": Annotated[list[evidence], Field(min_length=0, max_length=8)],
    })
    style_finding = _strict("BoundSampleStyleFinding", {
        "finding": Text,
        "evidence": Annotated[list[evidence], Field(min_length=0, max_length=8)],
        "dimension": literal_values(
            ["title", "opening", "tone", "paragraphs", "emoji", "tags", "ending"], "样本风格维度",
        ),
    })
    count = len(note_ids)
    return _strict("SampleAnalysisResponse", {
        "summary": Annotated[str, Field(min_length=1, max_length=1500)],
        "observations": Annotated[list[observation], Field(min_length=count, max_length=count)],
        "topic_patterns": Annotated[list[finding], Field(min_length=1, max_length=8)],
        "style_summary": Annotated[str, Field(min_length=1, max_length=1500)],
        "style_patterns": Annotated[list[style_finding], Field(min_length=1, max_length=7)],
        "value_patterns": Annotated[list[finding], Field(min_length=1, max_length=6)],
        "avoid_patterns": Annotated[list[finding], Field(max_length=6)],
        "limitations": Annotated[list[Text], Field(min_length=1, max_length=8)],
    })
