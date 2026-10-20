from __future__ import annotations

import time

from sqlalchemy import select

from backend.app.api.platforms.xhs.pc import _get_owned_pc_account_cookies, _normalize_detail_payload
from backend.app.core.time import shanghai_now
from backend.app.models import Note, Task, User
from backend.app.services.creative_analysis import CreativeError, analyze_sources
from backend.app.services.creative_report import export_report


class RunCancelled(Exception):
    pass


def checkpoint(db, task: Task, **updates):
    db.refresh(task)
    if task.status == "cancelled":
        raise RunCancelled()
    task.payload = {**(task.payload or {}), **updates}
    db.commit()


def generate_for_task(db, task, client_factory):
    def update_stage(stage):
        checkpoint(db, task, stage=stage)
        task.progress = 85
        db.commit()

    result, metadata = analyze_sources(
        db, task.user_id, task.payload["search"], task.payload["sources"],
        task.payload["analysis"], client_factory, update_stage,
    )
    checkpoint(db, task, result=result, analysis_metadata=metadata, stage="创作方案已生成")
    try:
        report = export_report(task)
    except Exception:
        report = {"export_error": "报告文件生成失败，创作方案已保存，可在结果区重新导出。"}
    checkpoint(db, task, **report)


def fail_task(db, task, exc, message):
    db.rollback()
    db.refresh(task)
    if task.status != "cancelled":
        task.status = "failed"
        task.payload = {**task.payload, "error": str(exc) if isinstance(exc, CreativeError) else message}
    task.finished_at = shanghai_now()
    db.commit()


def analyze_collection(task_id: int, session_factory, client_factory):
    with session_factory() as db:
        task = db.get(Task, task_id)
        if task is None or task.status != "pending":
            return
        try:
            task.status = "running"
            task.started_at = shanghai_now()
            db.commit()
            generate_for_task(db, task, client_factory)
            checkpoint(db, task)
            task.status = "completed"
            task.progress = 100
            task.finished_at = shanghai_now()
            db.commit()
        except RunCancelled:
            pass
        except Exception as exc:
            fail_task(db, task, exc, "AI 分析失败，请检查模型配置或网络后重新生成。采集数据已保留。")


def collect_samples(task_id: int, session_factory, adapter_factory, client_factory=None):
    """Persist successful text samples incrementally; no media download or publishing."""
    with session_factory() as db:
        task = db.get(Task, task_id)
        if task is None or task.status != "pending":
            return
        sources: list[dict] = []
        failures: list[dict] = []
        try:
            task.status = "running"
            task.started_at = shanghai_now()
            db.commit()
            user = db.get(User, task.user_id)
            cookies = _get_owned_pc_account_cookies(db, user, task.payload["search"]["account_id"])
            adapter = adapter_factory(cookies)
            candidates = task.payload["selected_notes"]
            for i, candidate in enumerate(candidates):
                checkpoint(db, task, stage=f"采集中 {i + 1}/{len(candidates)}")
                if i:
                    time.sleep(1)
                authentication_failed = False
                try:
                    success, message, raw = adapter.get_note_info(candidate["note_url"])
                    if not success:
                        authentication_failed = any(term in str(message).lower() for term in ("登录", "验证码", "cookie", "login", "captcha", "unauthorized"))
                        raise ValueError("笔记详情获取失败")
                    detail = _normalize_detail_payload(raw or {}, source_url=candidate["note_url"])
                    if detail["note_id"] != candidate["note_id"] or not detail["content"].strip():
                        raise ValueError("笔记内容为空或返回的笔记不匹配")
                    detail["raw"] = {}
                    detail["captured_at"] = shanghai_now().isoformat()
                except Exception:
                    failures.append({"note_id": candidate["note_id"], "title": candidate["title"], "reason": "登录需要重新验证" if authentication_failed else "详情不可用或正文为空"})
                else:
                    # Check cancellation after the network request before saving.
                    checkpoint(db, task)
                    note = db.scalar(select(Note).where(Note.user_id == task.user_id, Note.note_id == detail["note_id"]))
                    if note is None:
                        note = Note(user_id=task.user_id, platform_account_id=task.payload["search"]["account_id"], platform="xhs", note_id=detail["note_id"])
                        db.add(note)
                    note.title, note.content, note.author_name = detail["title"], detail["content"], detail["author_name"]
                    note.raw_json = {k: v for k, v in detail.items() if k != "raw"}
                    db.commit()
                    sources.append(detail)
                checkpoint(db, task, sources=sources, failures=failures, stage=f"已处理 {i + 1}/{len(candidates)}")
                task.progress = round((i + 1) / len(candidates) * (75 if task.payload.get("analysis") else 100))
                db.commit()
                if authentication_failed:
                    raise CreativeError("小红书登录需要重新验证，采集已停止。请重新绑定账号后重试。")
            checkpoint(db, task)
            if len(sources) < 3:
                raise CreativeError(f"仅采集到 {len(sources)} 篇有效正文，至少需要 3 篇才能进行综合分析。")
            if task.payload.get("analysis"):
                generate_for_task(db, task, client_factory)
            checkpoint(db, task)
            task.status = "completed"
            task.progress = 100
            task.payload = {**task.payload, "stage": "创作方案已生成" if task.payload.get("result") else f"采集完成：{len(sources)} 篇有效正文"}
            task.finished_at = shanghai_now()
            db.commit()
        except RunCancelled:
            task.finished_at = shanghai_now()
            db.commit()
        except Exception as exc:
            fail_task(db, task, exc, "任务失败，请检查账号、模型配置或网络后重试；已采集的正文仍保留。")


def recover_interrupted_runs(session_factory):
    with session_factory() as db:
        tasks = db.scalars(select(Task).where(
            Task.task_type.in_(("creative_search", "creative_collect", "creative_analysis")),
            Task.status.in_(("pending", "running")),
        )).all()
        for task in tasks:
            task.status = "failed"
            task.finished_at = shanghai_now()
            task.payload = {**(task.payload or {}), "error": "服务重启中断了任务，已保存的结果仍可查看，请重新执行。"}
        db.commit()
