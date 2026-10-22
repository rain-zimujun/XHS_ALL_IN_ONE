"""Explicit role construction and native structured output for every agent."""
import json
from dataclasses import dataclass, field
from time import monotonic

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from backend.app.schemas import creative_agents as legacy
from backend.app.schemas import creative_graph as responses
from backend.app.services.creative_agents.sample_analysis import SAMPLE_AGENT
from backend.app.services.creative_agents.reader_needs import AUDIENCE_AGENT
from backend.app.services.creative_agents.topic_selection import TOPIC_AGENT
from backend.app.services.creative_agents.topic_review import TOPIC_REVIEW_AGENT
from backend.app.services.creative_agents.title_generation import TITLE_AGENT
from backend.app.services.creative_agents.title_review import TITLE_REVIEW_AGENT
from backend.app.services.creative_agents.copy_generation import COPY_AGENT
from backend.app.services.creative_agents.copy_review import COPY_REVIEW_AGENT
from backend.app.services.creative_agents.image_planning import IMAGE_AGENT
from backend.app.services.creative_agents.final_review import FINAL_REVIEW_AGENT
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_orchestrator import AgentAttemptTrace, AgentRun, AgentSpec, AgentValidationError
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter


class GraphOutputError(CreativeError):
    def __init__(self, *, kind, attempts):
        message = {
            "creative_schema_invalid": "模型响应未符合当前 Agent 的结构定义，两次尝试后已停止；此前稿件已保存。",
            "creative_handoff_invalid": "当前 Agent 的来源、数量或上下游关联未通过检查，两次尝试后已停止；此前稿件已保存。",
        }[kind]
        super().__init__(message, kind)
        self.attempts = attempts


def validation_summary(exc: ValidationError) -> str:
    # Paths may contain model-generated object keys. Report only error codes.
    return "; ".join(dict.fromkeys(e["type"] for e in exc.errors(
        include_input=False, include_context=False, include_url=False,
    )))[:300]


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
    runnable: object = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "runnable", self.model.with_structured_output(
            self.response_format, method="json_schema", strict=True,
            temperature=self.spec.temperature,
        ))

    def invoke(self, payload, *, validate_output, on_progress=None, revision=None):
        parameter = response_parameter(self.response_format)
        envelope = {"input": payload, "output_schema": parameter["json_schema"]["schema"]}
        if revision:
            envelope["quality_revision"] = revision
        prompt = json.dumps(envelope, ensure_ascii=False)
        repair, traces = "", []
        for attempt in range(1, self.spec.max_attempts + 1):
            if on_progress:
                on_progress(f"{self.spec.name}（第 {attempt}/{self.spec.max_attempts} 次）")
            started = monotonic()
            kind = "creative_schema_invalid"
            try:
                # Output is constrained in the actual request, then validated
                # against TypedDict at runtime before business checks see it.
                value = self.runnable.invoke([
                    SystemMessage(content=graph_prompt(self.spec)),
                    HumanMessage(content=prompt + repair),
                ])
                # Version/fingerprint are owned by this invocation's global
                # state, never copied or chosen by the model.
                result = self.output_model.model_validate({**value, **{k: payload[k] for k in self.state_fields}})
                kind = "creative_handoff_invalid"
                validate_output(result)
                if revision:
                    from backend.app.services.creative_agents.revision import ensure_targeted_revision
                    ensure_targeted_revision(self.spec.stage, revision["previous_output"], value, revision["target_ids"])
                traces.append(AgentAttemptTrace(attempt, round((monotonic() - started) * 1000), "completed"))
                return AgentRun(self.spec.stage, self.spec.name, result, tuple(traces))
            except (ValidationError, AgentValidationError) as exc:
                message = validation_summary(exc) if isinstance(exc, ValidationError) else str(exc)[:500]
                traces.append(AgentAttemptTrace(attempt, round((monotonic() - started) * 1000), kind, message))
                repair = f"\n上一次输出未通过校验（{kind}），请按同一 Schema 返回完整对象。修正原因：{message}\n"
        raise GraphOutputError(kind=kind, attempts=traces)


def build_agents(model: CompatibleChatModel) -> dict[str, StructuredAgent]:
    # Each role binds its own handwritten TypedDict at construction. Pydantic
    # models below are adapters for the existing deterministic business checks.
    return {
        "sample_analysis": StructuredAgent(model, SAMPLE_AGENT, response_format=responses.SampleAnalysisResponse, output_model=legacy.SampleAnalysis),
        "audience": StructuredAgent(model, AUDIENCE_AGENT, response_format=responses.AudienceResponse, output_model=legacy.AudienceBrief),
        "topic": StructuredAgent(model, TOPIC_AGENT, response_format=responses.TopicResponse, output_model=legacy.TopicBrief),
        "topic_review": StructuredAgent(model, TOPIC_REVIEW_AGENT, response_format=responses.TopicReviewResponse, output_model=legacy.TopicReviewResponse, state_fields=("topic_revision", "input_fingerprint")),
        "title": StructuredAgent(model, TITLE_AGENT, response_format=responses.TitleResponse, output_model=legacy.TitleCandidates, state_fields=("topic_revision", "input_fingerprint")),
        "title_review": StructuredAgent(model, TITLE_REVIEW_AGENT, response_format=responses.TitleReviewResponse, output_model=legacy.ItemReviewResponse, state_fields=("revision", "input_fingerprint")),
        "copy": StructuredAgent(model, COPY_AGENT, response_format=responses.CopyResponse, output_model=legacy.CopyVariants, state_fields=("input_fingerprint",)),
        "copy_review": StructuredAgent(model, COPY_REVIEW_AGENT, response_format=responses.CopyReviewResponse, output_model=legacy.ItemReviewResponse, state_fields=("revision", "input_fingerprint")),
        "image": StructuredAgent(model, IMAGE_AGENT, response_format=responses.ImageResponse, output_model=legacy.ImagePlans, state_fields=("input_fingerprint",)),
        "final_review": StructuredAgent(model, FINAL_REVIEW_AGENT, response_format=responses.FinalReviewResponse, output_model=legacy.ItemReviewResponse, state_fields=("revision", "input_fingerprint")),
    }
