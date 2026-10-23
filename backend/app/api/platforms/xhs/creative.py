from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.platforms.xhs.pc import _get_owned_pc_account_cookies, get_xhs_pc_api_adapter_factory
from backend.app.api.tasks import serialize_task
from backend.app.core.database import get_db
from backend.app.core.deps import get_current_user
from backend.app.core.time import shanghai_now
from backend.app.models import Note, Task, User
from backend.app.schemas.creative import CreativeAnalyzeRequest, CreativeCollectRequest, CreativeSearchRequest
from backend.app.services.creative_analysis import CreativeError, get_creative_text_client_factory, owned_text_model
from backend.app.services.creative_runner import analyze_collection, collect_samples
from backend.app.services.creative_report import export_report
from backend.app.services.creative_copy import normalize_result_copy
from backend.app.services.creative_service import search_samples
from backend.app.services.creative_graph.recovery import can_resume, prepare_resume

router = APIRouter(prefix="/xhs/creative", tags=["xhs-creative"])


def has_timestamp(value) -> bool:
    try:
        return not isinstance(value, bool) and 0 < float(value) <= 8640000000000000
    except (TypeError, ValueError, OverflowError):
        return False


def serialize_creative_task(db: Session, task: Task) -> dict:
    result = serialize_task(task)
    result["resume_available"] = can_resume(task)
    if result["payload"].get("result"):
        # Read-through formatting for old results; never rewrite saved prose.
        result["payload"] = {**result["payload"], "result": normalize_result_copy(result["payload"]["result"])}
    if task.task_type != "creative_search":
        return result
    items = (task.payload or {}).get("items", [])
    missing = [n["note_id"] for n in items if not has_timestamp(n.get("timestamp"))]
    if not missing:
        return result
    # Read-through enrichment also repairs the display of existing search history.
    # Never mutate the saved search snapshot or read another user's notes.
    saved = dict(db.execute(select(Note.note_id, Note.raw_json).where(
        Note.user_id == task.user_id, Note.platform == "xhs", Note.note_id.in_(missing),
    )).all())
    enriched = []
    for note in items:
        detail = saved.get(note["note_id"]) or {}
        if not has_timestamp(note.get("timestamp")) and has_timestamp(detail.get("timestamp")):
            note = {**note, "timestamp": detail["timestamp"]}
        enriched.append(note)
    result["payload"] = {**result["payload"], "items": enriched}
    return result


def owned_task(db: Session, user: User, task_id: int, task_type: str | None = None) -> Task:
    task = db.scalar(select(Task).where(Task.id == task_id, Task.user_id == user.id))
    if task is None or task.task_type not in ("creative_search", "creative_collect", "creative_analysis"):
        raise HTTPException(404, "任务不存在")
    if task_type and task.task_type != task_type:
        raise HTTPException(400, "任务类型不匹配")
    return task


@router.post("/search")
def search(
    payload: CreativeSearchRequest, current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db), adapter_factory=Depends(get_xhs_pc_api_adapter_factory),
):
    cookies = _get_owned_pc_account_cookies(db, current_user, payload.account_id)
    task = Task(
        user_id=current_user.id, platform="xhs", task_type="creative_search",
        status="running", progress=0, started_at=shanghai_now(),
        payload={"search": payload.model_dump()},
    )
    db.add(task)
    db.commit()
    try:
        items, warnings = search_samples(adapter_factory(cookies), payload)
        db.refresh(task)
        if task.status == "cancelled":
            return serialize_task(task)
        task.payload = {**task.payload, "items": items, "warnings": warnings}
        task.status = "completed"
        task.progress = 100
    except Exception as exc:
        task.status = "failed"
        task.payload = {**task.payload, "error": "搜索失败，请检查账号状态、网络或稍后重试。"}
        task.finished_at = shanghai_now()
        db.commit()
        raise HTTPException(502, task.payload["error"]) from exc
    task.finished_at = shanghai_now()
    db.commit()
    return serialize_creative_task(db, task)


@router.get("/runs")
def history(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    filters = (
        Task.user_id == current_user.id, Task.task_type.in_(("creative_search", "creative_collect", "creative_analysis")),
    )
    total = db.scalar(select(func.count()).select_from(Task).where(*filters)) or 0
    tasks = db.scalars(select(Task).where(*filters).order_by(Task.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [{
        "id": task.id, "task_type": task.task_type, "status": task.status,
        "created_at": task.created_at.isoformat(),
        "keyword": (task.payload or {}).get("search", {}).get("keyword", ""),
    } for task in tasks], "page": page, "page_size": page_size, "total": total,
            "has_more": page * page_size < total}


@router.get("/runs/{task_id}")
def detail(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return serialize_creative_task(db, owned_task(db, current_user, task_id))


@router.post("/collect", status_code=202)
def collect(
    payload: CreativeCollectRequest, background: BackgroundTasks,
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db),
    adapter_factory=Depends(get_xhs_pc_api_adapter_factory),
    client_factory=Depends(get_creative_text_client_factory),
):
    search_task = owned_task(db, current_user, payload.search_task_id, "creative_search")
    if search_task.status != "completed":
        raise HTTPException(400, "请先完成搜索")
    ids = list(dict.fromkeys(payload.note_ids))
    notes = {n["note_id"]: n for n in search_task.payload["items"]}
    if len(ids) < 3 or any(note_id not in notes for note_id in ids):
        raise HTTPException(400, "请选择本次搜索中的至少 3 篇不同笔记")
    _get_owned_pc_account_cookies(db, current_user, search_task.payload["search"]["account_id"])
    if payload.analysis:
        try:
            owned_text_model(db, current_user.id, payload.analysis.model_id)
        except CreativeError as exc:
            raise HTTPException(400, str(exc)) from exc
    task = Task(
        user_id=current_user.id, platform="xhs", task_type="creative_collect", status="pending",
        progress=0, parent_task_id=search_task.id,
        payload={"search": search_task.payload["search"], "selected_notes": [notes[n] for n in ids], "stage": "等待采集",
                 "analysis": payload.analysis.model_dump() if payload.analysis else None},
    )
    db.add(task); db.commit(); db.refresh(task)
    background.add_task(collect_samples, task.id, sessionmaker(bind=db.get_bind()), adapter_factory, client_factory)
    return serialize_task(task)


@router.post("/analyze", status_code=202)
def analyze(
    payload: CreativeAnalyzeRequest, background: BackgroundTasks,
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db),
    client_factory=Depends(get_creative_text_client_factory),
):
    collection = owned_task(db, current_user, payload.collection_task_id)
    sources = collection.payload.get("sources", [])
    if collection.status in ("pending", "running") or len(sources) < 3:
        raise HTTPException(400, "请等待采集结束，并确保至少有 3 篇有效正文。")
    try:
        owned_text_model(db, current_user.id, payload.model_id)
    except CreativeError as exc:
        raise HTTPException(400, str(exc)) from exc
    task = Task(
        user_id=current_user.id, platform="xhs", task_type="creative_analysis", status="pending", progress=0,
        parent_task_id=collection.id,
        payload={"search": collection.payload["search"], "sources": sources, "stage": "等待 AI 分析",
                 "failures": collection.payload.get("failures", []),
                 "analysis": payload.model_dump(exclude={"collection_task_id"})},
    )
    db.add(task); db.commit(); db.refresh(task)
    background.add_task(analyze_collection, task.id, sessionmaker(bind=db.get_bind()), client_factory)
    return serialize_task(task)


@router.post("/runs/{task_id}/cancel")
def cancel(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = owned_task(db, current_user, task_id)
    if task.status in ("pending", "running"):
        task.status = "cancelled"
        task.finished_at = shanghai_now()
        task.payload = {**task.payload, "stage": "已停止"}
        db.commit()
    return serialize_task(task)


@router.post("/runs/{task_id}/resume", status_code=202)
def resume(
    task_id: int, background: BackgroundTasks,
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db),
    client_factory=Depends(get_creative_text_client_factory),
):
    original = owned_task(db, current_user, task_id)
    existing_id = (original.payload or {}).get("resume_task_id")
    if existing_id:
        existing = owned_task(db, current_user, existing_id)
        if existing.status in ("pending", "running", "completed"):
            return serialize_creative_task(db, existing)
        raise HTTPException(409, "已创建过接续任务，请打开最新的接续任务继续操作。")
    try:
        state, role = prepare_resume(original)
        owned_text_model(db, current_user.id, state["options"]["model_id"])
    except CreativeError as exc:
        raise HTTPException(400, str(exc)) from exc
    child = Task(
        user_id=current_user.id, platform="xhs", task_type="creative_analysis", status="pending", progress=0,
        parent_task_id=original.id,
        payload={"search": state["search"], "sources": state["sources"], "analysis": state["options"],
                 "failures": original.payload.get("failures", []), "graph_state": state, "pipeline": state["pipeline"],
                 "stage": "等待从失败阶段继续", "resume": {"from_task_id": original.id, "role": role}},
    )
    db.add(child)
    db.flush()
    # An optimistic compare-and-swap prevents double clicks/concurrent requests
    # from enqueuing two billable children. Keep the original failure for audit.
    claimed = db.execute(update(Task).where(
        Task.id == original.id, Task.user_id == current_user.id, Task.status == "failed",
        Task.retry_count == original.retry_count,
    ).values(retry_count=original.retry_count + 1, payload={**original.payload, "resume_task_id": child.id}))
    if claimed.rowcount != 1:
        db.rollback()
        raise HTTPException(409, "该任务已被接续，请刷新任务历史。")
    db.commit(); db.refresh(child)
    background.add_task(analyze_collection, child.id, sessionmaker(bind=db.get_bind()), client_factory)
    return serialize_creative_task(db, child)


@router.post("/runs/{task_id}/export")
def export(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = owned_task(db, current_user, task_id)
    if task.status in ("pending", "running") or not task.payload.get("result"):
        raise HTTPException(400, "请先完成 AI 分析。")
    try:
        report = export_report(task)
    except Exception as exc:
        raise HTTPException(500, "报告导出失败，请检查本地输出目录是否可写。") from exc
    task.payload = {**task.payload, **report}
    db.commit()
    return serialize_creative_task(db, task)
