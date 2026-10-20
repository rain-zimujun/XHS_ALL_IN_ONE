"""Search and collect evidence for user-reviewed content planning."""
from __future__ import annotations

import time
from urllib.parse import urlparse

from backend.app.api.platforms.xhs.crawl import _data_items
from backend.app.api.platforms.xhs.pc import _normalize_search_item
from backend.app.schemas.creative import CreativeSearchRequest


def valid_note_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "www.xiaohongshu.com"
        and parsed.path.startswith(("/explore/", "/discovery/item/"))
    )


def search_samples(adapter, request: CreativeSearchRequest) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    seen: set[str] = set()
    warnings: list[str] = []
    for page in range(1, 6):
        if page > 1:
            time.sleep(1)
        success, _, raw = adapter.search_note(
            request.keyword, page=page, sort_type_choice=request.sort_type_choice,
            note_type=request.note_type, note_time=request.note_time,
        )
        if not success:
            if not items:
                raise ValueError("搜索失败，请检查小红书账号状态或稍后重试。")
            warnings.append("后续搜索页获取失败，已保留成功获得的结果；可稍后重新搜索。")
            break
        previous_count = len(items)
        for raw_item in _data_items(raw):
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
