"""Serializable state for the staged creative-agent pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.creative_quality import QualityReview


PipelineStage = Literal[
    "sample_analysis", "audience", "topic", "title", "copy", "image", "final_review"
]
StageStatus = Literal["pending", "running", "completed", "failed", "skipped", "revision_required", "invalidated", "cancelled"]
PIPELINE_SCHEMA_VERSION = 2
PIPELINE_STAGE_ORDER: tuple[PipelineStage, ...] = (
    "sample_analysis", "audience", "topic", "title", "copy", "image", "final_review"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    error: str | None = None


class CreativeStageState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: PipelineStage
    status: StageStatus = "pending"
    attempts: list[AgentAttempt] = Field(default_factory=list, max_length=36)
    output: dict[str, Any] | None = None
    review: QualityReview | None = None
    error: str | None = None
    revision: int = 0
    quality_revisions: int = 0
    review_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=36)


class CreativePipelineState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PIPELINE_SCHEMA_VERSION
    status: Literal["pending", "running", "completed", "failed", "cancelled", "revision_required"] = "pending"
    current_stage: PipelineStage | None = None
    stages: dict[str, CreativeStageState]
    request_count: int = 0
    max_requests: int = 36
    max_quality_revisions: int = 2
    activity: str = "等待生成"
    error_code: str | None = None
    updated_at: str | None = None


def new_pipeline_state() -> dict[str, Any]:
    return CreativePipelineState(
        stages={stage: CreativeStageState(stage=stage) for stage in PIPELINE_STAGE_ORDER}
    ).model_dump(mode="json")


def validate_pipeline_state(value: dict[str, Any]) -> CreativePipelineState:
    return CreativePipelineState.model_validate(value)


def begin_stage(value: dict[str, Any], stage: PipelineStage) -> dict[str, Any]:
    state = validate_pipeline_state(value)
    item = state.stages[stage]
    attempt = AgentAttempt(attempt=len(item.attempts) + 1, started_at=utc_now())
    item.status = "running"
    item.error = None
    item.attempts.append(attempt)
    state.status = "running"
    state.current_stage = stage
    return state.model_dump(mode="json")


def complete_stage(value: dict[str, Any], stage: PipelineStage, output: dict[str, Any], review: QualityReview | None = None) -> dict[str, Any]:
    state = validate_pipeline_state(value)
    item = state.stages[stage]
    if not item.attempts:
        raise ValueError(f"Stage {stage} has not started")
    item.status = "completed"
    item.output = output
    item.review = review
    item.attempts[-1].status = "completed"
    item.attempts[-1].finished_at = utc_now()
    state.current_stage = None
    if all(state.stages[name].status in ("completed", "skipped") for name in PIPELINE_STAGE_ORDER):
        state.status = "completed"
    return state.model_dump(mode="json")


def fail_stage(value: dict[str, Any], stage: PipelineStage, error: str) -> dict[str, Any]:
    state = validate_pipeline_state(value)
    item = state.stages[stage]
    item.status = "failed"
    item.error = error
    if item.attempts:
        item.attempts[-1].status = "failed"
        item.attempts[-1].finished_at = utc_now()
        item.attempts[-1].error = error
    state.status = "failed"
    state.current_stage = stage
    return state.model_dump(mode="json")
