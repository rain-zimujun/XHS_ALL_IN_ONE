"""M10: evaluate every body variant, preserving the user's chosen quantity."""
from __future__ import annotations

from functools import partial

from pydantic import ValidationError

from backend.app.core.security import decrypt_text
from backend.app.schemas.creative import CreativeAnalysisOptions
from backend.app.schemas.creative_agents import CopyVariants, ItemReviewResponse
from backend.app.services.creative_agents.copy_generation import prepare_copy_input, validate_copy_variants
from backend.app.services.creative_agents.reviewing import REVIEW_INSTRUCTIONS, decide_review, prepare_review_payload
from backend.app.services.creative_analysis import CreativeError, owned_text_model, request_analysis
from backend.app.services.creative_orchestrator import AgentRun, AgentSpec, AgentValidationError, run_agent


COPY_REVIEW_AGENT = AgentSpec(stage="copy", name="文案评委 Agent", temperature=0.2, system_prompt=REVIEW_INSTRUCTIONS + """
本阶段评正文的前两句吸引力、具体信息与价值、目标读者匹配、小红书语气与可读性、收藏/转发价值、直接发布可用度。
对照锁定标题的 promise/support_points、选题 must_cover/boundaries 及样本风格证据，检查正文实际讲了什么，不能只看 covered_points 自报覆盖。
短文有价值不等于教程或长文。作品/灵感分享可提供具体审美观察，不强求赚钱收益，不把冗长当充实。
body 必须读起来是给小红书读者看的成稿。编辑指导、免责声明、配图制作说明、助手口吻、占位符都影响直接使用。
比较同标题各版本是否真的改变开头、视角、组织或节奏；只是替换同义词应给出具体定位和关联版本。
只借鉴多篇共同表达，不复制样本特色句子或作者亲历；虚构亲测、效果和数据是阻断问题。
标签独立，逐个对照本版正文检查话题相关性：主题、品类、风格、技术或场景应有正文依据；不因搜索词、目标受众或热词出现就认为相关，不要求不同版本强行使用不同标签。无关或误导标签定位到 tags 并给出修改意见；合理的不同排序只作为建议，不单独阻断。来源归属必须与选题一致。所有用户要求的版本都必须评审，不以删除失败版本代替返工。""")


def prepare_copy_review_input(*, copy_variants: CopyVariants | dict, copy_revision: int = 1, **copy_inputs) -> dict:
    context = prepare_copy_input(**copy_inputs)
    try:
        copies = CopyVariants.model_validate(copy_variants.model_dump() if isinstance(copy_variants, CopyVariants) else copy_variants)
        validate_copy_variants(copies, context)
    except (ValidationError, AgentValidationError) as exc:
        raise CreativeError("正文集合与当前标题不一致或未通过校验。", "copy_review_input_invalid") from exc
    items = [{
        "item_id": v.variant_id, "idea_id": idea.idea_id, "candidate_id": idea.candidate_id,
        "fields": v.model_dump(mode="json", exclude={"variant_id"}),
    } for idea in copies.ideas for v in idea.variants]
    return prepare_review_payload(kind="copy", revision=copy_revision, context=context, items=items, sample_note_ids=context["sample_note_ids"])


def _run_prepared(payload, complete, on_progress):
    run = run_agent(COPY_REVIEW_AGENT, input_payload=payload, output_model=ItemReviewResponse, complete=complete, on_progress=on_progress,
                    validate_output=lambda raw: decide_review(raw, payload))
    result = decide_review(run.output, payload)
    return AgentRun(stage=run.stage, name=run.name, output=result, attempts=run.attempts), {
        "prompt_version": "copy-review-v1", "request_count": len(run.attempts),
        "quality_review_status": result.status, "downstream_ready": result.status == "passed",
        "approved_count": len(result.approved_ids), "revision_required_count": len(result.revision_required_ids),
    }


def run_copy_review(*, complete, on_progress=None, **inputs):
    return _run_prepared(prepare_copy_review_input(**inputs), complete, on_progress)


def analyze_copy_review(db, user_id: int, *, options: dict, client_factory, on_progress=None, **inputs):
    payload = prepare_copy_review_input(**inputs)
    config = CreativeAnalysisOptions.model_validate(options)
    for option, expected in (("idea_count", payload["context"]["requested_idea_count"]), ("variant_count", payload["context"]["variant_count"])):
        if option in config.model_fields_set and (type(options[option]) is not int or getattr(config, option) != expected):
            raise CreativeError("正文评审数量与请求选项不一致。", "copy_review_input_invalid")
    model = owned_text_model(db, user_id, config.model_id)
    complete = partial(request_analysis, client_factory(), model_config=model, api_key=decrypt_text(model.encrypted_api_key))
    return _run_prepared(payload, complete, on_progress)
