"""Resolve topic names to platform topic entities.

The creator topic-search API rejects our own signatures, so entities are either
harvested through the signing browser (see ``browser_bridge``) or replayed from
this cache. Publishing therefore never depends on the search API being reachable.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.time import shanghai_now
from backend.app.models import TopicCache


def cached_topic_map(db: Session, user_id: int, names: list[str]) -> dict[str, dict]:
    wanted = [name.strip() for name in names if isinstance(name, str) and name.strip()]
    if not wanted:
        return {}
    rows = db.scalars(select(TopicCache).where(
        TopicCache.user_id == user_id,
        TopicCache.name.in_(wanted),
    )).all()
    return {
        row.name: {"id": row.topic_id, "name": row.name, "link": row.link}
        for row in rows
        if row.topic_id
    }


def store_topics(db: Session, user_id: int, topics: list[dict]) -> int:
    saved = 0
    for item in topics:
        name = str(item.get("name") or "").strip()
        topic_id = str(item.get("id") or "").strip()
        if not name or not topic_id:
            continue
        row = db.scalar(select(TopicCache).where(TopicCache.user_id == user_id, TopicCache.name == name))
        if row is None:
            db.add(TopicCache(user_id=user_id, name=name, topic_id=topic_id, link=str(item.get("link") or "")))
            saved += 1
        else:
            changed = row.topic_id != topic_id or (item.get("link") and row.link != item["link"])
            if changed:
                row.topic_id = topic_id
                row.link = str(item.get("link") or row.link)
                row.updated_at = shanghai_now()
                saved += 1
    db.commit()
    return saved


def entities_for_topics(db: Session, user_id: int, topics: list) -> list:
    """Replace cached names with topic entities; unknown names stay as strings."""
    if not topics:
        return []
    names = [topic for topic in topics if isinstance(topic, str)]
    mapping = cached_topic_map(db, user_id, names)
    resolved: list = []
    for topic in topics:
        if not isinstance(topic, str):
            resolved.append(topic)
            continue
        hit = mapping.get(topic.strip())
        if hit:
            resolved.append({"id": hit["id"], "name": hit["name"], "link": hit["link"], "type": "topic"})
        else:
            resolved.append(topic)
    return resolved
