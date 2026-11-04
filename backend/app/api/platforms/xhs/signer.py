"""Signer endpoints: drive a logged-in browser for signatures and topic lookup."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.core.deps import get_current_user
from backend.app.models import PlatformAccount, User
from backend.app.services.browser_bridge import BrowserBridge, BrowserUnavailable, CREATOR_ORIGIN
from backend.app.services.topic_lookup import store_topics

router = APIRouter(prefix="/xhs/signer", tags=["xhs-signer"])


class TopicSearchRequest(BaseModel):
    keyword: str = Field(min_length=1, max_length=64)
    limit: int = Field(default=10, ge=1, le=50)
    save: bool = True


def _data_root() -> Path:
    return Path(get_settings().data_dir or "data")


def _creator_account(db: Session, user: User) -> PlatformAccount | None:
    return db.scalars(select(PlatformAccount).where(
        PlatformAccount.user_id == user.id,
        PlatformAccount.platform == "xhs",
        PlatformAccount.sub_type == "creator",
    ).order_by(PlatformAccount.id.desc())).first()


def profile_dir_for(db: Session, user: User) -> Path:
    account = _creator_account(db, user)
    name = f"account-{account.id}" if account is not None else "default"
    return _data_root() / "browser" / name


def _bridge_for(profile_dir: Path) -> BrowserBridge:
    """Reuse a running browser for the profile, else start a new one on demand."""
    bridge = BrowserBridge(profile_dir)
    bridge.attach()
    return bridge


def _ensure_browser(db: Session, user: User, *, start: bool) -> BrowserBridge:
    bridge = _bridge_for(profile_dir_for(db, user))
    if start and not bridge.profile_in_use():
        bridge.launch(visible=False)
    return bridge


@router.post("/import-account")
def signer_import_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create/refresh a Creator account from the signing browser's session.

    The creator CAS login endpoints reject our anonymous device sessions with a
    probabilistic 406 gate, but the browser is already logged in (the user scanned
    once), so importing its cookie jar skips that gate entirely.
    """
    bridge = _bridge_for(profile_dir_for(db, current_user))
    if not bridge.attach():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="签名浏览器未启动，请先在签名浏览器里登录")
    if not bridge.is_logged_in():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="签名浏览器未登录小红书，请先扫码登录")
    flat = {
        cookie["name"]: cookie["value"]
        for cookie in bridge.cookies()
        if "xiaohongshu.com" in str(cookie.get("domain") or "") and cookie.get("name") and cookie.get("value")
    }
    if not flat.get("a1"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="签名浏览器没有可用的登录 Cookie")

    from backend.app.adapters.xhs.creator_login_adapter import XhsCreatorLoginAdapter
    from backend.app.api.login_sessions import _create_account_from_login

    adapter = XhsCreatorLoginAdapter()
    try:
        user_info = adapter.get_user_info(flat)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"读取创作者信息失败：{exc}") from exc
    account, action = _create_account_from_login(
        db=db, user_id=current_user.id, sub_type="creator", user_info=user_info, cookies=flat,
    )
    db.commit()
    from backend.app.api.accounts import serialize_account

    return serialize_account(account, action)



@router.get("/status")
def signer_status(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bridge = _bridge_for(profile_dir_for(db, current_user))
    info = bridge.status()
    info["profile"] = bridge.profile_dir.name
    info["available"] = True
    return info


@router.post("/login")
def signer_login(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Open a visible browser window so the user can scan the QR code."""
    bridge = _bridge_for(profile_dir_for(db, current_user))
    if not bridge.profile_in_use():
        bridge.launch(visible=True, url=f"{CREATOR_ORIGIN}/login")
    else:
        bridge.visible_window()
        bridge.navigate(f"{CREATOR_ORIGIN}/login", wait=1.5)
    return {"opened": True, "port": bridge.port, "logged_in": bridge.is_logged_in()}


@router.post("/topics")
def signer_topics(
    payload: TopicSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search topics through the signing browser (the page signs the request)."""
    bridge = _ensure_browser(db, current_user, start=True)
    try:
        topics = bridge.search_topics(payload.keyword, timeout=30.0)
    except BrowserUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    finally:
        pass
    if payload.save and topics:
        store_topics(db, current_user.id, topics)
    return {"items": topics[: payload.limit], "total": len(topics)}
