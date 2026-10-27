"""M12: evaluate complete title/copy/image bundles without rewriting them."""
from __future__ import annotations

from functools import partial

from pydantic import ValidationError

from backend.app.core.security import decrypt_text
from backend.app.schemas.creative import CreativeAnalysisOptions
from backend.app.schemas.creative_agents import ImagePlans, ItemReviewResponse, SampleAnalysis
from backend.app.services.creative_agents.image_planning import prepare_image_input, validate_image_plans
from backend.app.services.creative_agents.reviewing import (
    FINAL_FIELD_OWNERS, REVIEW_INSTRUCTIONS, decide_review, prepare_review_payload,
)
from backend.app.services.creative_analysis import CreativeError, owned_text_model, request_analysis
from backend.app.services.creative_orchestrator import AgentRun, AgentSpec, AgentValidationError, run_agent


FINAL_REVIEW_AGENT = AgentSpec(stage="final_review", name="最终评委 Agent", temperature=0.2, system_prompt=REVIEW_INSTRUCTIONS + """
本阶段逐版检查完整组合：标题与正文一致性、标题承诺兑现、配图匹配、标签质量、事实/原创/素材边界。
逐一对照当前标题、正文、标签、封面文案、构图、找图说明、生图提示词与选题证据，不因上游通过而跳过核对。若 item.evidence 声明获批 claim/source，必须按对应获批结论检查正文承诺，不得仅凭 ID 合法或上游评审通过就认定语义有依据。
标签以本版最终正文为依据，检查每个话题的主题相关性；正文未涉及的热词、行业或引流标签不能因受众或关键词包含就通过。合理的不同排序只作为建议，不单独阻断。发现问题交正文阶段修改，不让配图阶段改标签。
配图要具体可执行，正负向提示词应独立可用；封面文字交用户后期排版，不要求生图准确生成汉字。
概念服装图不能伪装成真实商品细节/案例或效果证据；没有提供原图，不能声称核对了原图或视觉趋势。
原创文案只借鉴多篇样本共同习惯，不照抄特色句、继承作者身份或虚构亲测/数字/效果。
每个问题必须填写 responsible_stage：topic 字段归 topic，title 归 title，body/tags/source_note_ids 归 copy，cover_text/image_plan/material_notes 归 image。
定位需要修改的字段并逐字引用；负责阶段由程序核验。不要建议配图 Agent 改正文或正文 Agent 偷改标题，不直接给出替换稿。
不能用“其他部分不错”抵消事实或承诺问题。高分仅作辅助筛选，真实发布前仍需用户核实素材和事实。""")


def prepare_final_review_input(*, image_plans: ImagePlans | dict, final_revision: int = 1, **image_inputs) -> dict:
    context = prepare_image_input(**image_inputs)
    try:
        images = ImagePlans.model_validate(image_plans.model_dump() if isinstance(image_plans, ImagePlans) else image_plans)
        validate_image_plans(images, context)
    except (ValidationError, AgentValidationError) as exc:
        raise CreativeError("配图集合与当前正文不一致或未通过校验。", "final_review_input_invalid") from exc
    sample = image_inputs["sample_analysis"]
    context = {**context, "sample_analysis": SampleAnalysis.model_validate(sample.model_dump() if isinstance(sample, SampleAnalysis) else sample).model_dump(mode="json"),
               "field_owners": FINAL_FIELD_OWNERS}
    titles = {t["idea_id"]: t for t in context["locked_titles"]}
    topics = {t["idea_id"]: t for t in context["topic_brief"]["ideas"]}
    pictures = {v.variant_id: v for v in images.variants}
    items = [{
        "item_id": v["variant_id"], "idea_id": v["idea_id"], "candidate_id": v["candidate_id"],
        "evidence": {
            "claim_ids": list(v.get("claim_ids", [])),
            "source_note_ids": list(v["source_note_ids"]),
            "approved_claims": context.get("approved_claims_by_idea", {}).get(v["idea_id"], []),
        },
        "fields": {
            "topic": topics[v["idea_id"]], "title": titles[v["idea_id"]]["title"],
            "body": v["body"], "tags": v["tags"], "source_note_ids": v["source_note_ids"],
            **pictures[v["variant_id"]].model_dump(mode="json", exclude={"idea_id", "candidate_id", "variant_id"}),
        },
    } for v in context["locked_copies"]]
    return prepare_review_payload(kind="final", revision=final_revision, context=context, items=items, sample_note_ids=context["sample_note_ids"])


def _run_prepared(payload, complete, on_progress):
    run = run_agent(FINAL_REVIEW_AGENT, input_payload=payload, output_model=ItemReviewResponse, complete=complete, on_progress=on_progress,
                    validate_output=lambda raw: decide_review(raw, payload))
    result = decide_review(run.output, payload)
    return AgentRun(stage=run.stage, name=run.name, output=result, attempts=run.attempts), {
        "prompt_version": "final-review-v1", "request_count": len(run.attempts),
        "quality_review_status": result.status, "downstream_ready": result.status == "passed",
        "approved_count": len(result.approved_ids), "revision_required_count": len(result.revision_required_ids),
    }


def run_final_review(*, complete, on_progress=None, **inputs):
    return _run_prepared(prepare_final_review_input(**inputs), complete, on_progress)


def analyze_final_review(db, user_id: int, *, options: dict, client_factory, on_progress=None, **inputs):
    payload = prepare_final_review_input(**inputs)
    config = CreativeAnalysisOptions.model_validate(options)
    for option, expected in (("idea_count", payload["context"]["requested_idea_count"]), ("variant_count", payload["context"]["variant_count"])):
        if option in config.model_fields_set and (type(options[option]) is not int or getattr(config, option) != expected):
            raise CreativeError("最终评审数量与请求选项不一致。", "final_review_input_invalid")
    model = owned_text_model(db, user_id, config.model_id)
    complete = partial(request_analysis, client_factory(), model_config=model, api_key=decrypt_text(model.encrypted_api_key))
    return _run_prepared(payload, complete, on_progress)
