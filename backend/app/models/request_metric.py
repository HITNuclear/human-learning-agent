from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class RequestMetric(Base):
    __tablename__ = "request_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event: Mapped[str] = mapped_column(String(64), index=True)
    node_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    tree_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)

    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_build_ms: Mapped[float] = mapped_column(Float, default=0.0)
    generation_ms: Mapped[float] = mapped_column(Float, default=0.0)
    ttft_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)

    input_tokens_est: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens_est: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd_est: Mapped[float] = mapped_column(Float, default=0.0)

    rag_total_hits: Mapped[int] = mapped_column(Integer, default=0)
    rag_global_hits: Mapped[int] = mapped_column(Integer, default=0)
    rag_tree_hits: Mapped[int] = mapped_column(Integer, default=0)
    rag_other_hits: Mapped[int] = mapped_column(Integer, default=0)
    rag_unique_sources: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
