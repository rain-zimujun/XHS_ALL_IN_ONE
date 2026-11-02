"""Search and collect evidence for user-reviewed content planning."""
from __future__ import annotations

import time
from urllib.parse import urlparse

from backend.app.api.platforms.xhs.crawl import _data_items
from backend.app.api.platforms.xhs.pc import _normalize_search_item
from backend.app.schemas.creative import CreativeSearchRequest
from xhs_utils.xhs_pc.params import generate_search_id


def valid_note_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "www.xiaohongshu.com"
        and parsed.path.startswith(("/explore/", "/discovery/item/"))
    )


def _user_id_from_link(link: str) -> str:
    """``xhsdiscover://1/user/user.<id>`` carries the id when ``id`` is absent."""
    marker = "/user/user."
    return link.split(marker, 1)[1].split("?", 1)[0] if marker in link else ""


def normalize_user_candidates(raw: dict | None) -> list[dict]:
    """Public fields only; the profile token stays out of the response."""
    users = ((raw or {}).get("data") or {}).get("users") or []
    candidates = []
    for user in users:
        user_id = str(user.get("id") or _user_id_from_link(str(user.get("link") or "")) or "")
        if not user_id:
            continue
        candidates.append({
            "user_id": user_id,
            "name": str(user.get("name") or ""),
            "red_id": str(user.get("red_id") or ""),
            "avatar": str(user.get("image") or ""),
            "fans": str(user.get("fans") or ""),
            "note_count": user.get("note_count") or 0,
            "description": str(user.get("profession") or user.get("desc") or ""),
        })
    return candidates


def search_account_notes(adapter, user_id: str, *, scan: int = 300, sort: str = "latest"):
    """Latest ``scan`` public notes of one account, normalized like keyword hits.

    The posted-notes endpoint returns only the note card (title, cover, metrics);
    body text and tags need a detail fetch, so account mode never filters here.
    """
    items: list[dict] = []
    seen: set[str] = set()
    warnings: list[str] = []
    cursor = ""
    scanned = 0
    while scanned < scan:
        if cursor:
            time.sleep(1)
        try:
            success, message, raw = adapter.user_notes(user_id, cursor)
        except Exception:
            if not items:
                raise
            warnings.append("账号笔记翻页中断，已保留成功获取的部分；可稍后重试。")
            break
        if not success:
            if not items:
                raise ValueError("获取账号笔记失败，请检查小红书账号状态或稍后重试。")
            warnings.append("账号笔记后续页获取失败，已保留成功获取的部分；可稍后重试。")
            break
        data = (raw or {}).get("data") or {}
        notes = data.get("notes") or []
        for note in notes:
            if scanned >= scan:
                break
            scanned += 1
            if isinstance(note, dict):
                # Profile links must carry a profile token for the later detail fetch.
                note.setdefault("xsec_source", "pc_user")
            normalized = _normalize_search_item(note if isinstance(note, dict) else {})
            note_id = str(normalized["note_id"] or "")
            if not note_id or note_id in seen or not valid_note_url(normalized["note_url"]):
                continue
            seen.add(note_id)
            normalized["raw"] = {}
            normalized["note_id"] = note_id
            items.append(normalized)
        next_cursor = str(data.get("cursor") or "")
        if not data.get("has_more") or not notes or next_cursor == cursor:
            break
        cursor = next_cursor
    if sort == "likes":
        items.sort(key=lambda note: note.get("likes") or 0, reverse=True)
    else:
        items.sort(key=lambda note: note.get("timestamp") or 0, reverse=True)
    return items, warnings, scanned


def search_samples(adapter, request: CreativeSearchRequest) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    seen: set[str] = set()
    warnings: list[str] = []
    search_id = generate_search_id()
    for page in range(1, 6):
        if page > 1:
            time.sleep(1)
        try:
            success, _, raw = adapter.search_note(
                request.keyword, page=page, sort_type_choice=request.sort_type_choice,
                note_type=request.note_type, note_time=request.note_time,
                search_id=search_id,
            )
        except Exception:
            if not items:
                raise
            warnings.append("后续搜索页请求异常，已保留成功获得的结果；可稍后重新搜索。")
            break
        if not success:
            if not items:
                raise ValueError("搜索失败，请检查小红书账号状态或稍后重试。")
            warnings.append("后续搜索页获取失败，已保留成功获得的结果；可稍后重新搜索。")
            break
        previous_count = len(items)
        for raw_item in _data_items(raw):
            # Search may also return suggestions or user cards rather than notes.
            if raw_item.get("model_type") in ("hot_query", "query", "user", "ads"):
                continue
            note = _normalize_search_item(raw_item)
            note_id = str(note["note_id"])
            if not note_id or note_id in seen or not valid_note_url(note["note_url"]):
                continue
            seen.add(note_id)
            # Only normalized public fields are persisted and sent to the UI.
            note["raw"] = {}
            note["note_id"] = note_id
            items.append(note)
            if len(items) >= request.limit:
                return items, warnings
        data = (raw or {}).get("data") or {}
        if not data.get("has_more") or len(items) == previous_count:
            break
    if len(items) < request.limit:
        warnings.append(f"目标 {request.limit} 篇，实际获得 {len(items)} 篇去重结果；未用其他内容补足。")
    return items, warnings
