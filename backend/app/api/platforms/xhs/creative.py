from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.platforms.xhs.pc import _get_owned_pc_account_cookies, get_xhs_pc_api_adapter_factory
from backend.app.api.tasks import serialize_task
from backend.app.core.database import get_db
from backend.app.core.deps import get_current_user
from backend.app.core.time import shanghai_now
from backend.app.models import Task, User
from backend.app.schemas.creative import CreativeSearchRequest
from backend.app.services.creative_service import search_samples

router = APIRouter(prefix="/xhs/creative", tags=["xhs-creative"])


def owned_task(db: Session, user: User, task_id: int, task_type: str | None = None) -> Task:
    task = db.scalar(select(Task).where(Task.id == task_id, Task.user_id == user.id))
    if task is None or task.task_type not in ("creative_search", "creative_analysis"):
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
    return serialize_task(task)


@router.get("/runs")
def history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = db.scalars(select(Task).where(
        Task.user_id == current_user.id, Task.task_type.in_(("creative_search", "creative_analysis")),
    ).order_by(Task.id.desc()).limit(30)).all()
    return {"items": [{
        "id": task.id, "task_type": task.task_type, "status": task.status,
        "created_at": task.created_at.isoformat(),
        "keyword": (task.payload or {}).get("search", {}).get("keyword", ""),
    } for task in tasks]}


@router.get("/runs/{task_id}")
def detail(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return serialize_task(owned_task(db, current_user, task_id))
