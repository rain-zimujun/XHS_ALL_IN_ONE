"""Restore server-saved LangGraph snapshots after model or execution-window failures."""
from copy import deepcopy

from backend.app.schemas.creative import CreativeAnalysisOptions
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph.state import CreativeGraphState
from backend.app.services.creative_pipeline import validate_pipeline_state


MODEL_ERROR_CODES = {"ai_connect_timeout", "ai_read_timeout", "ai_network_error", "ai_http_error", "ai_response_invalid"}
TIME_BUDGET_CODES = {"creative_time_exhausted", "creative_budget_exhausted"}
INTERRUPTED_CODE = "creative_interrupted"
LEGACY_TAG_SCHEMA_MARKERS = (
    "tags 必须是与本版最终正文最相关的 6 个不同话题",
    "tags 必须是与本版最终正文最相关的6个不同话题",
)
TARGETED_REVISION_MARKERS = (
    "未列入返工范围的已通过内容必须原样保留",
)
SAMPLE_REVIEW_MARKERS = (
    "未获原文语义支持",
    "样本评审必须逐条覆盖",
    "评审来源必须来自该结论已绑定",
    "判为有依据时必须至少给出一个有效来源",
    "判为无依据时 note_ids 必须为空",
)
SAMPLE_QUOTE_FAILURE_MARKERS = (
    "observations.quote 必须来自对应样本标题或已提供正文片段",
    "evidence.quote 必须来自对应标题或正文片段",
)
AUDIENCE_COVERAGE_MARKER = "segments 分组校验失败；缺少受众词="
AUDIENCE_REJECTED_TERMS_MARKER = "未知受众词="
AUDIENCE_CLAIM_BINDING_MARKER = "note_id 必须属于本条 evidence 引用的获批 claim_id"
TOPIC_DUPLICATE_SOURCE_MARKERS = (
    "同一条 evidence.note_ids 不可重复",
    "同一条 evidence.claim_ids 不可重复",
    "source_note_ids 不可重复",
)
TOPIC_SOURCE_BINDING_MARKERS = (
    "note_id 必须属于本条 evidence 引用的获批 claim_id",
    "source_note_ids 必须与 evidence 的样本来源合集一致",
)
TOPIC_REVISION_NO_PROGRESS_MARKERS = (
    "未落实评审意见：目标内容字段没有任何变化",
    "未落实评审意见：必改字段未变化=",
)
TOPIC_ISSUE_QUOTE_MARKER = "问题 quote 必须逐字出现在当前选题的对应 field 中"
SINGLE_ROLES = {"sample_analysis", "sample_review", "audience", "image", "final_review"}
ALL_ROLES = SINGLE_ROLES | {"topic", "topic_review", "title", "title_review", "copy", "copy_review"}


def prepare_resume(task):
    """Never accepts client-supplied artifacts, sources, role or options."""
    try:
        payload = task.payload or {}
        if task.status != "failed":
            raise ValueError("not a resumable failure")
        if payload.get("analysis", {}).get("workflow") != "langgraph":
            raise ValueError("not LangGraph")
        state = deepcopy(payload["graph_state"])
        expected_fields = set(CreativeGraphState.__annotations__)
        legacy_fields = expected_fields - {"sample_review"}
        if set(state) == legacy_fields:
            state["sample_review"] = None
        if set(state) != expected_fields:
            raise ValueError("incomplete state")
        pipeline = validate_pipeline_state(state["pipeline"]).model_dump(mode="json")
        stage = pipeline["current_stage"]
        if pipeline["status"] != "failed" or stage not in pipeline["stages"] or pipeline["stages"][stage]["status"] != "failed":
            raise ValueError("not failed at a model stage")
        if state["result"] is not None or state["search"] != payload["search"] or state["sources"] != payload["sources"]:
            raise ValueError("mismatched snapshot")
        options = CreativeAnalysisOptions.model_validate(payload["analysis"]).model_dump(exclude_none=True)
        saved_options = CreativeAnalysisOptions.model_validate(state["options"]).model_dump(exclude_none=True)
        if saved_options != options:
            raise ValueError("changed options")
        item = pipeline["stages"][stage]
        if task.error_type == INTERRUPTED_CODE:
            # A process restart is not a model/business failure. It is safe to
            # continue only when startup recovery marked both persisted copies
            # consistently and the original cumulative budget still has room.
            if pipeline["error_code"] != INTERRUPTED_CODE or payload.get("pipeline") != state["pipeline"]:
                raise ValueError("inconsistent interrupted snapshot")
        elif task.error_type not in MODEL_ERROR_CODES | TIME_BUDGET_CODES:
            attempts = item.get("metadata", {}).get("failed_attempts", [])
            legacy_tag_failure = any(
                any(marker in str(attempt.get("error", "")) for marker in LEGACY_TAG_SCHEMA_MARKERS)
                for attempt in attempts
            )
            targeted_revision_failure = any(
                any(marker in str(attempt.get("error", "")) for marker in TARGETED_REVISION_MARKERS)
                for attempt in attempts
            )
            sample_review_failure = item["metadata"].get("failed_role") == "sample_review" and any(
                any(marker in str(attempt.get("error", "")) for marker in SAMPLE_REVIEW_MARKERS)
                for attempt in attempts
            )
            sample_failures = item["metadata"].get("failed_attempts", [])
            sample_quote_failure = (
                stage == "sample_analysis"
                and item["metadata"].get("failed_role") == "sample_analysis"
                and isinstance(sample_failures, list) and bool(sample_failures)
                and all(any(marker in str(attempt.get("error", ""))
                            for marker in SAMPLE_QUOTE_FAILURE_MARKERS)
                        for attempt in sample_failures)
            )
            audience_failures = item["metadata"].get("failed_attempts", [])
            audience_coverage_failure = (
                item["metadata"].get("failed_role") == "audience"
                and isinstance(audience_failures, list) and len(audience_failures) >= 2
                and all(
                    AUDIENCE_COVERAGE_MARKER in str(attempt.get("error", ""))
                    and ("；错误=[]；最多8组" in str(attempt.get("error", ""))
                         or AUDIENCE_REJECTED_TERMS_MARKER in str(attempt.get("error", "")))
                    for attempt in audience_failures[-2:]
                )
            )
            audience_claim_binding_failure = (
                stage == "audience"
                and item["metadata"].get("failed_role") == "audience"
                and isinstance(audience_failures, list) and len(audience_failures) >= 2
                and all(
                    AUDIENCE_CLAIM_BINDING_MARKER in str(attempt.get("error", ""))
                    for attempt in audience_failures[-2:]
                )
            )
            topic_failures = item["metadata"].get("failed_attempts", [])
            topic_duplicate_source_failure = (
                item["metadata"].get("failed_role") == "topic"
                and isinstance(topic_failures, list) and len(topic_failures) >= 2
                and all(str(attempt.get("error", "")) in TOPIC_DUPLICATE_SOURCE_MARKERS
                        for attempt in topic_failures[-2:])
            )
            topic_source_binding_failure = (
                item["metadata"].get("failed_role") == "topic"
                and isinstance(topic_failures, list) and len(topic_failures) >= 2
                and all(str(attempt.get("error", "")) in TOPIC_SOURCE_BINDING_MARKERS
                        for attempt in topic_failures[-2:])
            )
            topic_revision_no_progress_failure = (
                item["metadata"].get("failed_role") == "topic"
                and stage == "topic" and bool(state.get("revision_plan", {}).get("topic"))
                and isinstance(topic_failures, list) and len(topic_failures) >= 2
                and all(any(marker in str(attempt.get("error", "")) for marker in TOPIC_REVISION_NO_PROGRESS_MARKERS)
                        for attempt in topic_failures[-2:])
            )
            topic_issue_quote_failure = (
                item["metadata"].get("failed_role") == "topic_review"
                and isinstance(topic_failures, list) and len(topic_failures) >= 2
                and all(str(attempt.get("error", "")) == TOPIC_ISSUE_QUOTE_MARKER
                        for attempt in topic_failures[-2:])
            )
            compatible_handoff = (legacy_tag_failure or targeted_revision_failure
                                  or sample_review_failure or sample_quote_failure or audience_coverage_failure
                                  or audience_claim_binding_failure
                                  or topic_duplicate_source_failure or topic_issue_quote_failure)
            compatible_handoff = compatible_handoff or topic_source_binding_failure
            compatible_handoff = compatible_handoff or topic_revision_no_progress_failure
            compatible_legacy_topic_revision = (
                topic_revision_no_progress_failure and task.error_type == "creative_schema_invalid"
            )
            if (task.error_type != "creative_handoff_invalid" and not compatible_legacy_topic_revision) or not compatible_handoff:
                raise ValueError("not a compatible model failure")
        role = item["metadata"].get("failed_role")
        if role is None and stage == "sample_analysis":
            role = "sample_analysis" if state["sample_analysis"] is None else "sample_review"
        if role is None:
            # Older snapshots grouped producers and reviewers into one stage.
            # A committed draft with no pending revision is awaiting its judge.
            if stage in SINGLE_ROLES:
                role = stage
            else:
                role = f"{stage}_review" if state[stage] is not None and stage not in state["revision_plan"] else stage
        role_stage = "sample_analysis" if role in {"sample_analysis", "sample_review"} else (
            role.removesuffix("_review") if role != "final_review" else role
        )
        if role not in ALL_ROLES or role_stage != stage:
            raise ValueError("invalid role")
        if state["sample_review"] is None and state["sample_analysis"] is not None and stage != "sample_analysis":
            sample_metadata = pipeline["stages"]["sample_analysis"]["metadata"]
            relevance = sample_metadata.get("sample_relevance", {})
            selected_count = relevance.get("selected_count", len(state["sources"]))
            batch_cache = sample_metadata.get("sample_analysis_batches", {})
            if selected_count > 3 and not batch_cache.get("results"):
                role = "sample_analysis"
            else:
                role = "sample_review"
        if pipeline["request_count"] >= pipeline["max_requests"]:
            if task.error_type != INTERRUPTED_CODE:
                raise ValueError("request budget exhausted")
        if task.error_type == INTERRUPTED_CODE:
            from backend.app.services.creative_graph.workflow import estimate_graph_request_budget

            required_budget = estimate_graph_request_budget(len(state["sources"]), options)
            if required_budget > pipeline["max_requests"]:
                pipeline["max_requests"] = required_budget
            if pipeline["request_count"] >= pipeline["max_requests"]:
                raise ValueError("request budget exhausted after adaptive estimate")
        # Old versions used one error code for both limits. With requests still
        # available, a matching budget failure was an expired execution window.
        if task.error_type in TIME_BUDGET_CODES:
            if pipeline["error_code"] != task.error_type:
                raise ValueError("mismatched budget failure")
            detail = item["metadata"].get("budget_exhaustion", {})
            if detail and detail.get("reason") != "time":
                raise ValueError("not a time budget failure")
        item.update(status="pending", error=None)
        pipeline.update(status="pending", error_code=None, activity="等待从失败阶段继续")
        state.update(pipeline=pipeline, result=None, metadata={})
        return state, role
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise CreativeError("该任务没有可继续的 LangGraph 模型失败或执行超时快照，或累计请求次数已用尽；请检查任务状态，或使用已采集正文重新生成。", "creative_resume_invalid") from exc


def can_resume(task):
    try:
        prepare_resume(task)
        return True
    except CreativeError:
        return False
