"""M8: compare title candidates and lock one qualified title per idea."""
from __future__ import annotations

from functools import partial
from typing import Literal

from pydantic import ValidationError

from backend.app.core.security import decrypt_text
from backend.app.schemas.creative import CreativeAnalysisOptions
from backend.app.schemas.creative_agents import AgentOutputModel, ItemReviewResponse, TitleCandidates
from backend.app.services.creative_agents.reviewing import (
    REVIEW_INSTRUCTIONS, ItemReviewResult, decide_review, prepare_review_payload, verify_review,
)
from backend.app.services.creative_agents.title_generation import prepare_title_input, validate_title_candidates
from backend.app.services.creative_analysis import CreativeError, owned_text_model, request_analysis
from backend.app.services.creative_orchestrator import AgentRun, AgentSpec, AgentValidationError, run_agent


TITLE_REVIEW_AGENT = AgentSpec(
    stage="title", name="标题评委 Agent", temperature=0.2,
    system_prompt=REVIEW_INSTRUCTIONS + """
本阶段评价标题：点击吸引力、具体程度、读者收益、关键词和受众匹配、候选差异化、自然度。
吸引力不等于夸张或焦虑。将候选承诺与已通过选题的要点和事实边界对照；理由写得好不能替代标题本身好用。
同组选项若只是同义改写，给出差异化低分及关联候选。不要为了保证有获选项而抬高分数。
程序会按固定加权分数从合格项中锁定每组选题的一个标题；你只提供评价及比较，不输出获选 ID 或新标题。""",
)


class TitleReviewResult(AgentOutputModel):
    review: ItemReviewResult
    selected_candidate_ids: dict[str, str]
    unresolved_idea_ids: list[str]
    status: Literal["passed", "revision_required"]


def prepare_title_review_input(*, title_candidates: TitleCandidates | dict, title_revision: int = 1, **title_inputs) -> dict:
    context = prepare_title_input(**title_inputs)
    try:
        candidates = TitleCandidates.model_validate(
            title_candidates.model_dump() if isinstance(title_candidates, TitleCandidates) else title_candidates,
        )
        validate_title_candidates(candidates, context)
    except (ValidationError, AgentValidationError) as exc:
        raise CreativeError("标题集合与当前选题不一致或未通过校验。", "title_review_input_invalid") from exc
    items = [{
        "item_id": candidate.candidate_id, "idea_id": idea.idea_id,
        "fields": candidate.model_dump(mode="json", exclude={"candidate_id"}),
    } for idea in candidates.ideas for candidate in idea.candidates]
    return prepare_review_payload(kind="title", revision=title_revision, context=context, items=items, sample_note_ids=context["sample_note_ids"])


def select_titles(review: ItemReviewResult, payload: dict) -> TitleReviewResult:
    scores = {d.judgement.item_id: d.quality for d in review.reviews}
    selected, unresolved = {}, []
    for idea in payload["context"]["topic_brief"]["ideas"]:
        candidates = [item["item_id"] for item in payload["items"] if item["idea_id"] == idea["idea_id"] and scores[item["item_id"]].passed]
        if candidates:
            # Stable original order resolves equal scores; the model cannot pick an unreviewed ID.
            selected[idea["idea_id"]] = max(candidates, key=lambda item_id: scores[item_id].overall_score)
        else:
            unresolved.append(idea["idea_id"])
    return TitleReviewResult(review=review, selected_candidate_ids=selected, unresolved_idea_ids=unresolved, status="revision_required" if unresolved else "passed")


def verified_title_review(value: TitleReviewResult | dict, payload: dict) -> TitleReviewResult:
    try:
        stored = TitleReviewResult.model_validate(value.model_dump() if isinstance(value, TitleReviewResult) else value)
        review = verify_review(stored.review, payload, require_passed=False)
        computed = select_titles(review, payload)
        if stored.model_dump(mode="json") != computed.model_dump(mode="json") or computed.status != "passed":
            raise ValueError("no qualified selection")
        return computed
    except (ValidationError, ValueError, CreativeError) as exc:
        raise CreativeError("部分选题没有合格标题，或标题评审与当前内容不一致。", "title_review_not_approved") from exc


def _run_prepared(payload, complete, on_progress):
    run = run_agent(
        TITLE_REVIEW_AGENT, input_payload=payload, output_model=ItemReviewResponse, complete=complete,
        on_progress=on_progress, validate_output=lambda raw: decide_review(raw, payload),
    )
    result = select_titles(decide_review(run.output, payload), payload)
    return AgentRun(stage=run.stage, name=run.name, output=result, attempts=run.attempts), {
        "prompt_version": "title-review-v1", "request_count": len(run.attempts),
        "quality_review_status": result.status, "downstream_ready": result.status == "passed",
        "selected_count": len(result.selected_candidate_ids), "reviewed_count": len(result.review.reviews),
    }


def run_title_review(*, complete, on_progress=None, **inputs):
    return _run_prepared(prepare_title_review_input(**inputs), complete, on_progress)


def analyze_title_review(db, user_id: int, *, options: dict, client_factory, on_progress=None, **inputs):
    payload = prepare_title_review_input(**inputs)
    config = CreativeAnalysisOptions.model_validate(options)
    if "idea_count" in config.model_fields_set and (
        type(options["idea_count"]) is not int or config.idea_count != payload["context"]["requested_idea_count"]
    ):
        raise CreativeError("标题评审数量与请求选项不一致。", "title_review_input_invalid")
    model = owned_text_model(db, user_id, config.model_id)
    complete = partial(request_analysis, client_factory(), model_config=model, api_key=decrypt_text(model.encrypted_api_key))
    return _run_prepared(payload, complete, on_progress)
