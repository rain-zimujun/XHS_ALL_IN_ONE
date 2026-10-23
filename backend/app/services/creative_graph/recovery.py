"""Restore server-saved LangGraph snapshots after model or execution-window failures."""
from copy import deepcopy

from backend.app.schemas.creative import CreativeAnalysisOptions
from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_graph.state import CreativeGraphState
from backend.app.services.creative_pipeline import validate_pipeline_state


MODEL_ERROR_CODES = {"ai_connect_timeout", "ai_read_timeout", "ai_network_error", "ai_http_error", "ai_response_invalid"}
TIME_BUDGET_CODES = {"creative_time_exhausted", "creative_budget_exhausted"}
SINGLE_ROLES = {"sample_analysis", "audience", "image", "final_review"}
ALL_ROLES = SINGLE_ROLES | {"topic", "topic_review", "title", "title_review", "copy", "copy_review"}


def prepare_resume(task):
    """Never accepts client-supplied artifacts, sources, role or options."""
    try:
        payload = task.payload or {}
        if task.status != "failed" or task.error_type not in MODEL_ERROR_CODES | TIME_BUDGET_CODES:
            raise ValueError("not a resumable failure")
        if payload.get("analysis", {}).get("workflow") != "langgraph":
            raise ValueError("not LangGraph")
        state = deepcopy(payload["graph_state"])
        if set(state) != set(CreativeGraphState.__annotations__):
            raise ValueError("incomplete state")
        pipeline = validate_pipeline_state(state["pipeline"]).model_dump(mode="json")
        stage = pipeline["current_stage"]
        if pipeline["status"] != "failed" or stage not in pipeline["stages"] or pipeline["stages"][stage]["status"] != "failed":
            raise ValueError("not failed at a model stage")
        if state["result"] is not None or state["search"] != payload["search"] or state["sources"] != payload["sources"]:
            raise ValueError("mismatched snapshot")
        options = CreativeAnalysisOptions.model_validate(payload["analysis"]).model_dump()
        if state["options"] != options:
            raise ValueError("changed options")
        item = pipeline["stages"][stage]
        role = item["metadata"].get("failed_role")
        if role is None:
            # Older snapshots grouped producers and reviewers into one stage.
            # A committed draft with no pending revision is awaiting its judge.
            if stage in SINGLE_ROLES:
                role = stage
            else:
                role = f"{stage}_review" if state[stage] is not None and stage not in state["revision_plan"] else stage
        if role not in ALL_ROLES or (role.removesuffix("_review") if role != "final_review" else role) != stage:
            raise ValueError("invalid role")
        if pipeline["request_count"] >= pipeline["max_requests"]:
            raise ValueError("request budget exhausted")
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
