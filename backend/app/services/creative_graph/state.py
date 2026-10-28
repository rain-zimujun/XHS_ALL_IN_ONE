"""Global state and the explicitly allowed partial update of every graph node."""
from typing import Any
from typing_extensions import TypedDict

from backend.app.schemas.creative_graph import (
    AudienceResponse, CopyResponse, ImageResponse, SampleAnalysisResponse,
    TitleResponse, TopicResponse,
)


class RevisionRequest(TypedDict):
    target_ids: list[str]
    direct: bool
    feedback: dict[str, Any]


class TitleArtifact(TitleResponse):
    topic_revision: int
    input_fingerprint: str


class CopyArtifact(CopyResponse):
    input_fingerprint: str


class ImageArtifact(ImageResponse):
    input_fingerprint: str


class ApprovedSampleClaim(TypedDict):
    claim_id: str
    section: str
    text: str
    note_ids: list[str]
    review_reason: str
    evidence: list[dict[str, str]]


class SampleReviewSummary(TypedDict):
    status: str
    batch_count: int
    claim_count: int
    input_fingerprints: list[str]
    approved_claims: list[ApprovedSampleClaim]
    approved_analysis: dict[str, Any]
    approved_note_ids: list[str]
    rejected_claim_count: int
    rejected_claims: list[dict[str, str]]


class CreativeGraphState(TypedDict):
    search: dict[str, Any]
    sources: list[dict[str, Any]]
    options: dict[str, Any]
    sample_analysis: SampleAnalysisResponse | None
    sample_review: SampleReviewSummary | None
    audience: AudienceResponse | None
    topic: TopicResponse | None
    topic_review: dict[str, Any] | None
    title: TitleArtifact | None
    title_review: dict[str, Any] | None
    copy: CopyArtifact | None
    copy_review: dict[str, Any] | None
    image: ImageArtifact | None
    final_review: dict[str, Any] | None
    revision_plan: dict[str, RevisionRequest]
    pipeline: dict[str, Any]
    result: dict[str, Any] | None
    metadata: dict[str, Any]


class NodeUpdate(TypedDict, total=False):
    pipeline: dict[str, Any]
    revision_plan: dict[str, RevisionRequest]


class SampleUpdate(NodeUpdate, total=False):
    sample_analysis: SampleAnalysisResponse


class SampleReviewUpdate(NodeUpdate, total=False):
    sample_review: SampleReviewSummary


class AudienceUpdate(NodeUpdate, total=False):
    audience: AudienceResponse


class TopicUpdate(NodeUpdate, total=False):
    topic: TopicResponse


class TopicReviewUpdate(NodeUpdate, total=False):
    topic_review: dict[str, Any]


class TitleUpdate(NodeUpdate, total=False):
    title: TitleArtifact


class TitleReviewUpdate(NodeUpdate, total=False):
    title_review: dict[str, Any]


class CopyUpdate(NodeUpdate, total=False):
    copy: CopyArtifact


class CopyReviewUpdate(NodeUpdate, total=False):
    copy_review: dict[str, Any]


class ImageUpdate(NodeUpdate, total=False):
    image: ImageArtifact


class FinalReviewUpdate(NodeUpdate, total=False):
    final_review: dict[str, Any]


class RevisionPlanUpdate(NodeUpdate, total=False):
    # Only this control node can invalidate several downstream review fields.
    topic_review: None
    title_review: None
    copy_review: None
    final_review: None


class ResultUpdate(NodeUpdate, total=False):
    result: dict[str, Any]
    metadata: dict[str, Any]
