from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base
from backend.app.core.time import shanghai_now


DEFAULT_TEXT_MODEL_NAME = "gpt-5.4"


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    model_type: Mapped[str] = mapped_column(String(32), index=True)
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(128), default="")
    base_url: Mapped[str] = mapped_column(Text, default="")
    encrypted_api_key: Mapped[str] = mapped_column(Text, default="")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)


class AiDraft(Base):
    __tablename__ = "ai_drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    source_note_id: Mapped[Optional[int]] = mapped_column(ForeignKey("notes.id"), nullable=True)
    # Where the draft came from (manual | creative | note | benchmark) and which
    # creative variant it mirrors; the pair is the idempotency key for re-saving.
    source_type: Mapped[str] = mapped_column(String(32), default="manual")
    source_task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    source_variant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=shanghai_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=shanghai_now, onupdate=shanghai_now)

    __table_args__ = (
        UniqueConstraint("user_id", "source_task_id", "source_variant_id", name="uq_draft_source_variant"),
    )


class AiGeneratedAsset(Base):
    __tablename__ = "ai_generated_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    draft_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ai_drafts.id"), nullable=True)
    prompt: Mapped[str] = mapped_column(Text, default="")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    file_path: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=shanghai_now)


class TopicCache(Base):
    """Topic name -> platform topic entity, so publishing never needs the search API."""

    __tablename__ = "topic_cache"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_topic_cache_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    topic_id: Mapped[str] = mapped_column(String(64), default="")
    link: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=shanghai_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=shanghai_now)


class DraftAsset(Base):
    __tablename__ = "draft_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("ai_drafts.id"), index=True)
    asset_type: Mapped[str] = mapped_column(String(32))
    url: Mapped[str] = mapped_column(Text, default="")
    local_path: Mapped[str] = mapped_column(Text, default="")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    # upload | generated：区分手动上传与内置生图，生图保留提示词便于复现
    source: Mapped[str] = mapped_column(String(32), default="upload")
    prompt: Mapped[str] = mapped_column(Text, default="")
