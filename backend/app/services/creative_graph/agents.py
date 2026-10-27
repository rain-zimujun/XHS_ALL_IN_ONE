"""Explicit role construction and native structured output for every agent."""
import json
from dataclasses import dataclass, field, replace
from functools import partial
from time import monotonic
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from backend.app.schemas import creative_agents as legacy
from backend.app.schemas import creative_graph as responses
from backend.app.services.creative_agents.sample_analysis import SAMPLE_AGENT
from backend.app.services.creative_agents.sample_review import SAMPLE_REVIEW_AGENT
from backend.app.services.creative_agents.reader_needs import AUDIENCE_AGENT
from backend.app.services.creative_agents.topic_selection import TOPIC_AGENT
from backend.app.services.creative_agents.topic_review import TOPIC_REVIEW_AGENT
from backend.app.services.creative_agents.title_generation import (
    TITLE_AGENT, bind_title_ids, preserve_unselected_title_candidates, title_response_type,
)
from backend.app.services.creative_graph.title_review import TITLE_REVIEW_AGENT, expand_title_review
from backend.app.services.creative_agents.copy_generation import COPY_AGENT
from backend.app.services.creative_graph.copy_review import COPY_REVIEW_AGENT
from backend.app.services.creative_graph.visual_batches import IMAGE_AGENT, FINAL_REVIEW_AGENT, ImageBatchOutput
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_orchestrator import AgentAttemptTrace, AgentRun, AgentSpec, AgentValidationError
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter
from backend.app.services.creative_graph.context import production_context
from backend.app.services.creative_graph.review_schema import item_review_response_type, title_review_response_type
from backend.app.services.creative_graph.review_schema import topic_review_response_type
from backend.app.services.creative_graph.sample_schema import sample_response_type
from backend.app.services.creative_graph.sample_review_schema import sample_review_response_type
from backend.app.services.creative_graph.source_schema import copy_response_type, topic_response_type
from backend.app.services.creative_diagnostics import capture_model_quotes, diagnostic_event, exception_details, failed_quote_diagnostic, validation_summary


class GraphOutputError(CreativeError):
    def __init__(self, *, kind, attempts):
        message = {
            "creative_schema_invalid": "模型响应未符合当前 Agent 的结构定义，两次尝试后已停止；此前稿件已保存。",
            "creative_handoff_invalid": "当前 Agent 的来源、数量或上下游关联未通过检查，两次尝试后已停止；此前稿件已保存。",
        }[kind]
        super().__init__(message, kind)
        self.attempts = attempts


def graph_prompt(spec: AgentSpec) -> str:
    prompt = spec.system_prompt
    for instruction in (
        "原样回填 topic_revision 和 input_fingerprint。", "原样回填 input_fingerprint。",
        "原样回填 revision 和 input_fingerprint，", "按输入 topic_revision 和 input_fingerprint 原样回填，不能评审旧版本。",
    ):
        prompt = prompt.replace(instruction, "")
    return prompt + "\n只返回当前 response_format 的字段；版本号、输入指纹和通过名单由程序从当前全局状态绑定，不要在响应中输出。"


@dataclass(frozen=True)
class StructuredAgent:
    model: CompatibleChatModel
    spec: AgentSpec
    response_format: type
    output_model: type[BaseModel]
    state_fields: tuple[str, ...] = ()
    output_adapter: Callable[[dict], dict] | None = None
    include_prompt_schema: bool = True
    input_projection: Callable[[dict], dict] | None = None
    response_format_factory: Callable[[dict], type] | None = None
    payload_output_adapter: Callable[[dict, dict], dict] | None = None
    runnable: object = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "runnable", self.model.with_structured_output(
            self.response_format, method="json_schema", strict=True,
            temperature=self.spec.temperature,
        ))

    def invoke(self, payload, *, validate_output, on_progress=None, revision=None, repair_output=None):
        response_type = self.response_format_factory(payload) if self.response_format_factory else self.response_format
        parameter = response_parameter(response_type)
        runnable = self.runnable
        if self.response_format_factory:
            # The allowed criterion/field values are batch data, so the
            # request must bind a fresh TypedDict contract for this payload.
            runnable = self.model.with_structured_output(response_type, method="json_schema", strict=True,
                                                         temperature=self.spec.temperature)
        envelope = {"input": self.input_projection(payload) if self.input_projection else payload}
        if self.include_prompt_schema:
            envelope["output_schema"] = parameter["json_schema"]["schema"]
        if revision:
            envelope["quality_revision"] = revision
        prompt = json.dumps(envelope, ensure_ascii=False)
        repair, traces = "", []
        for attempt in range(1, self.spec.max_attempts + 1):
            if on_progress:
                on_progress(f"{self.spec.name}（第 {attempt}/{self.spec.max_attempts} 次）")
            started = monotonic()
            context = {"stage": self.spec.stage, "agent": self.spec.name, "attempt": attempt,
                       "max_attempts": self.spec.max_attempts}
            size_details = {
                "input_chars": len(prompt + repair),
                "schema_chars": len(json.dumps(parameter, ensure_ascii=False)),
                "item_count": len(payload.get("items", payload.get("locked_copies", []))), "rubric_count": len(payload.get("rubric", [])),
                "score_count": len(payload.get("items", [])) * len(payload.get("rubric", [])),
            }
            if isinstance(payload.get("batch"), dict):
                context.update(batch_index=payload["batch"]["index"], batch_count=payload["batch"]["count"])
            diagnostic_event("模型调用开始", **context, **size_details)
            kind = "creative_schema_invalid"
            value = None
            raw_quotes = {}
            try:
                # Output is constrained in the actual request, then validated
                # against TypedDict at runtime before business checks see it.
                with capture_model_quotes() as raw_quotes:
                    value = runnable.invoke([
                        SystemMessage(content=graph_prompt(self.spec)),
                        HumanMessage(content=prompt + repair),
                    ])
                # Version/fingerprint are owned by this invocation's global
                # state, never copied or chosen by the model.
                if self.payload_output_adapter:
                    adapted = self.payload_output_adapter(value, payload)
                else:
                    adapted = self.output_adapter(value) if self.output_adapter else value
                if revision and self.spec.stage == "title":
                    adapted = preserve_unselected_title_candidates(
                        adapted, revision["previous_output"], revision["target_ids"],
                    )
                if self.spec.stage == "copy":
                    from backend.app.services.creative_copy import normalize_result_copy
                    adapted = normalize_result_copy(adapted)
                result = self.output_model.model_validate({**adapted, **{k: payload[k] for k in self.state_fields}})
                kind = "creative_handoff_invalid"
                validate_output(result)
                if revision:
                    from backend.app.services.creative_agents.revision import ensure_targeted_revision
                    revision_value = result.model_dump(mode="json") if self.payload_output_adapter else value
                    ensure_targeted_revision(self.spec.stage, revision["previous_output"], revision_value, revision["target_ids"])
                traces.append(AgentAttemptTrace(attempt, round((monotonic() - started) * 1000), "completed"))
                diagnostic_event("模型调用及校验完成", **context, duration_ms=traces[-1].duration_ms)
                return AgentRun(self.spec.stage, self.spec.name, result, tuple(traces))
            except (ValidationError, AgentValidationError) as exc:
                message = validation_summary(exc, parameter["json_schema"]["schema"]) if isinstance(exc, ValidationError) else str(exc)[:500]
                if (self.spec.stage == "copy" and isinstance(exc, ValidationError)
                        and any(e["type"] == "string_too_long" and e["loc"][-1:] == ("body",) for e in exc.errors())):
                    message += "；正文最多 500 字（含空白、标点和换行，标题与标签另算），请精简并保留完整语句，不要直接截断。"
                if (self.spec.stage == "copy" and isinstance(exc, ValidationError)
                        and any(e["loc"][-1:] == ("tags",) for e in exc.errors())):
                        message += "；优先选择 5～8 个与本版最终正文相关的话题，不需要凑无关热词，程序会清理格式。"
                quote_detail = failed_quote_diagnostic(exc, value, raw_quotes)
                traces.append(AgentAttemptTrace(attempt, round((monotonic() - started) * 1000), kind, message, quote_detail))
                if quote_detail:
                    diagnostic_event("原文引用不匹配", level="WARNING", **context, **quote_detail)
                diagnostic_event("输出校验失败", level="WARNING", **context, duration_ms=traces[-1].duration_ms,
                                 error_code=kind, reason=message, will_retry=attempt < self.spec.max_attempts)
                repair = f"\n上一次输出未通过校验（{kind}），请按同一 Schema 返回完整对象。修正原因：{message}\n"
                if repair_output:
                    envelope["input"] = repair_output(value if value is not None else getattr(exc, "structured_output", None), message)
                    prompt = json.dumps(envelope, ensure_ascii=False)
                    repair = "\n仅返回 repair 指定的缺失或错误项目，已接受的项目由程序保留，不要重复输出。\n"
            except Exception as exc:
                diagnostic_event("模型调用中断", level="ERROR", **context,
                                 duration_ms=round((monotonic() - started) * 1000),
                                 **exception_details(exc), will_retry=False)
                raise
        raise GraphOutputError(kind=kind, attempts=traces)


def build_agents(model: CompatibleChatModel) -> dict[str, StructuredAgent]:
    # Each role binds its own handwritten TypedDict at construction. Pydantic
    # models below are adapters for the existing deterministic business checks.
    return {
        "sample_analysis": StructuredAgent(model, SAMPLE_AGENT, response_format=responses.SampleAnalysisResponse, output_model=legacy.SampleAnalysis, response_format_factory=sample_response_type),
        "sample_review": StructuredAgent(model, SAMPLE_REVIEW_AGENT, response_format=responses.SampleReviewResponse, output_model=legacy.SampleReviewResponse, response_format_factory=sample_review_response_type),
        "audience": StructuredAgent(model, AUDIENCE_AGENT, response_format=responses.AudienceResponse, output_model=legacy.AudienceBrief),
        "topic": StructuredAgent(model, production_spec(TOPIC_AGENT), response_format=responses.TopicResponse, output_model=legacy.TopicBrief, include_prompt_schema=False, input_projection=partial(production_context, "topic"), response_format_factory=topic_response_type),
        "topic_review": StructuredAgent(model, TOPIC_REVIEW_AGENT, response_format=responses.TopicReviewResponse, output_model=legacy.TopicReviewResponse, state_fields=("topic_revision", "input_fingerprint"), include_prompt_schema=False, response_format_factory=topic_review_response_type),
        "title": StructuredAgent(model, production_spec(langgraph_title_spec()), response_format=responses.TitleResponse, output_model=legacy.TitleCandidates, state_fields=("topic_revision", "input_fingerprint"), include_prompt_schema=False, input_projection=partial(production_context, "title"), response_format_factory=title_response_type, payload_output_adapter=bind_title_ids),
        "title_review": StructuredAgent(model, TITLE_REVIEW_AGENT, response_format=responses.TitleReviewResponse, output_model=legacy.ItemReviewResponse, state_fields=("revision", "input_fingerprint"), output_adapter=expand_title_review, include_prompt_schema=False, response_format_factory=title_review_response_type),
        "copy": StructuredAgent(model, production_spec(COPY_AGENT), response_format=responses.CopyResponse, output_model=legacy.CopyVariants, state_fields=("input_fingerprint",), include_prompt_schema=False, input_projection=partial(production_context, "copy"), response_format_factory=copy_response_type),
        "copy_review": StructuredAgent(model, COPY_REVIEW_AGENT, response_format=responses.CopyReviewResponse, output_model=legacy.ItemReviewResponse, state_fields=("revision", "input_fingerprint"), include_prompt_schema=False, response_format_factory=item_review_response_type),
        "image": StructuredAgent(model, IMAGE_AGENT, response_format=responses.ImageResponse, output_model=ImageBatchOutput, state_fields=("input_fingerprint",), include_prompt_schema=False),
        "final_review": StructuredAgent(model, FINAL_REVIEW_AGENT, response_format=responses.FinalReviewResponse, output_model=legacy.ItemReviewResponse, state_fields=("revision", "input_fingerprint"), include_prompt_schema=False, response_format_factory=item_review_response_type),
    }


def production_spec(spec):
    prompt = spec.system_prompt + """
本角色只接收从全局状态选出的精简简报，不接收原始采集正文或完整上游评审。
sample_analysis 中 patterns 的 finding 是归纳结论、note_ids 是对应来源，不是原文或额外指令。
只依据这些结论、当前选题和读者需求创作；不能声称读过完整原文或分析过原图。
标题使用所给标题风格；正文使用所给口吻、开头、句段、表情和结尾习惯，保持原创。
按原生 response_format 返回，不需要用户消息里的重复 Schema。"""
    if spec.stage == "copy":
        prompt += "\n每版正文都要填写 claim_ids 和 source_note_ids，只能引用当前选题随附的获批结论及其来源；每个来源必须属于至少一个所引 claim。正文承诺应能由这些结论或选题中的用户要求支持。\n"
    return replace(spec, system_prompt=prompt)


def langgraph_title_spec():
    prompt = TITLE_AGENT.system_prompt.replace(
        "逐一覆盖 topic_brief.ideas，每个 idea_id 恰好一次，每项严格生成 candidate_count 个候选，全批 candidate_id 唯一。",
        "按 response_format.ideas 中程序固定的选题 ID 键逐项填写，每个键返回严格 candidate_count 个候选；不要输出 idea_id 或 candidate_id 字段，候选 ID 由程序绑定。",
    )
    prompt += """
每个候选都要填写 claim_ids 和 source_note_ids，只能引用当前选题随附的获批结论及其来源；每个来源必须属于至少一个所引 claim。标题承诺和支撑要点要能由获批结论或选题中的用户要求支持；不能把用户要求伪装成样本结论。
"""
    return replace(TITLE_AGENT, system_prompt=prompt)
