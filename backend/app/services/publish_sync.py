"""Reconcile scheduled publish jobs with the creator posted list.

Scheduled notes are published by Xiaohongshu itself, so our side only knows they
went live after they appear in the creator's posted list. This module does that
read-only reconciliation; it never re-posts anything.
"""
from __future__ import annotations

from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.platforms.xhs.pc import _cookies_to_string
from backend.app.core.security import decrypt_text
from backend.app.core.time import shanghai_now
from backend.app.models import AccountCookieVersion, PlatformAccount, PublishJob, User


def _candidate_id(note: Any) -> str:
    if not isinstance(note, dict):
        return ""
    for key in ("id", "note_id", "noteId"):
        value = note.get(key)
        if isinstance(value, str) and value:
            return value
    card = note.get("note_card")
    if isinstance(card, dict):
        for key in ("id", "note_id", "noteId"):
            value = card.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _posted_note_ids(raw: Any) -> set[str]:
    items: list[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        items = data.get("notes") or data.get("items") or []
    else:
        items = []
    return {note_id for item in items if (note_id := _candidate_id(item))}


def _latest_cookies(db: Session, account: PlatformAccount) -> str | None:
    version = db.scalars(
        select(AccountCookieVersion)
        .where(AccountCookieVersion.platform_account_id == account.id)
        .order_by(AccountCookieVersion.created_at.desc())
    ).first()
    if version is None:
        return None
    return _cookies_to_string(decrypt_text(version.encrypted_cookies))


def sync_scheduled_jobs(db: Session, user: User, adapter_factory: Callable[[str], Any]) -> dict:
    """Mark scheduled jobs as published once their note shows up on the platform."""
    jobs = db.scalars(select(PublishJob).where(
        PublishJob.user_id == user.id,
        PublishJob.platform == "xhs",
        PublishJob.publish_mode == "scheduled",
        PublishJob.status == "scheduled",
    )).all()
    summary = {"checked": len(jobs), "published": 0, "still_scheduled": 0, "errors": []}
    if not jobs:
        return summary

    by_account: dict[int, list[PublishJob]] = {}
    for job in jobs:
        if job.platform_account_id is None:
            summary["errors"].append(f"任务 #{job.id} 没有绑定发布账号")
            continue
        by_account.setdefault(job.platform_account_id, []).append(job)

    for account_id, account_jobs in by_account.items():
        account = db.get(PlatformAccount, account_id)
        if account is None or account.user_id != user.id:
            summary["errors"].append(f"账号 #{account_id} 不存在")
            continue
        cookies = _latest_cookies(db, account)
        if not cookies:
            summary["errors"].append(f"账号「{account.nickname or account_id}」没有可用 Cookie")
            continue
        try:
            success, message, raw = adapter_factory(cookies).get_published_notes()
        except Exception as exc:  # network / adapter errors must not break the page
            summary["errors"].append(f"账号「{account.nickname or account_id}」同步失败：{exc}"[:200])
            continue
        if not success:
            summary["errors"].append(f"账号「{account.nickname or account_id}」同步失败：{message}"[:200])
            continue
        posted = _posted_note_ids(raw)
        for job in account_jobs:
            if job.external_note_id and job.external_note_id in posted:
                job.status = "published"
                job.published_at = job.published_at or job.submitted_at or shanghai_now()
                job.publish_error = ""
                summary["published"] += 1
            else:
                summary["still_scheduled"] += 1

    db.commit()
    return summary
