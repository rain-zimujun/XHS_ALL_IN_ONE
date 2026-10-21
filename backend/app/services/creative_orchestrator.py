"""Generic structured Agent runner used by the staged creative pipeline.

This module intentionally does not decide what a stage means. Each Agent owns
its prompt and Pydantic output schema; the runner owns JSON parsing, retry
instructions, progress callbacks and a compact execution trace.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from backend.app.services.creative_pipeline import PipelineStage
from backend.app.services.structured_output import json_schema_response_format


class AgentValidationError(Exception):
    pass


class AgentRunError(Exception):
    def __init__(self, message: str, *, stage: PipelineStage, attempts: list["AgentAttemptTrace"]):
        super().__init__(message)
        self.stage = stage
        self.attempts = attempts


@dataclass(frozen=True)
class AgentSpec:
    stage: PipelineStage
    name: str
    system_prompt: str
    max_attempts: int = 2
    temperature: float = 0.4


@dataclass(frozen=True)
class AgentAttemptTrace:
    attempt: int
    duration_ms: int
    status: str
    error: str | None = None


@dataclass(frozen=True)
class AgentRun:
    stage: PipelineStage
    name: str
    output: BaseModel
    attempts: tuple[AgentAttemptTrace, ...]


def parse_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise AgentValidationError("Agent output must be text containing a JSON object")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) < 3 or lines[0].lower() not in ("```", "```json") or lines[-1] != "```":
            raise AgentValidationError("Agent output contains an incomplete JSON code block")
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AgentValidationError("Agent output is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AgentValidationError("Agent output must be a JSON object")
    return value


def run_agent(
    spec: AgentSpec,
    *,
    input_payload: dict[str, Any],
    output_model: type[BaseModel],
    complete: Callable[..., str],
    on_progress: Callable[[str], None] | None = None,
    validate_output: Callable[[BaseModel], None] | None = None,
) -> AgentRun:
    if spec.max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    response_format = json_schema_response_format(output_model)
    schema = response_format["json_schema"]["schema"]
    user_prompt = json.dumps({"input": input_payload, "output_schema": schema}, ensure_ascii=False)
    repair = ""
    traces: list[AgentAttemptTrace] = []
    for attempt in range(1, spec.max_attempts + 1):
        if on_progress:
            on_progress(f"{spec.name}（第 {attempt}/{spec.max_attempts} 次）")
        started = monotonic()
        try:
            raw = complete(
                system_prompt=spec.system_prompt,
                user_prompt=user_prompt + repair,
                temperature=spec.temperature,
                response_format=response_format,
            )
            result = output_model.model_validate(parse_json_object(raw))
            if validate_output:
                # The stage supplies deterministic checks, not another model call.
                # Raise AgentValidationError for an invalid reference/coverage;
                # programming, cancellation and transport errors must propagate.
                validate_output(result)
            traces.append(AgentAttemptTrace(attempt, round((monotonic() - started) * 1000), "completed"))
            return AgentRun(spec.stage, spec.name, result, tuple(traces))
        except (AgentValidationError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                # Do not put raw model output, source text or input values in logs.
                # Unknown JSON keys can themselves contain arbitrary private
                # text. Do not echo them as paths in extra-field errors.
                issues = exc.errors(include_input=False, include_context=False, include_url=False)
                message = "; ".join(
                    f"{'output' if issue['type'] == 'extra_forbidden' else '.'.join(map(str, issue['loc'])) or 'output'}: {issue['type']}"
                    for issue in issues
                )[:500]
            else:
                message = str(exc)[:500] or exc.__class__.__name__
            traces.append(AgentAttemptTrace(attempt, round((monotonic() - started) * 1000), "failed", message))
            repair = (
                "\n上一次输出未通过 JSON、字段或证据校验。请只返回完整 JSON 对象，严格遵守 output_schema；"
                f"修正原因：{message}\n"
            )
    raise AgentRunError(f"{spec.name} 连续 {spec.max_attempts} 次输出未通过校验", stage=spec.stage, attempts=traces)
