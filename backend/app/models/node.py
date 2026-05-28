from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tree_id: Mapped[str] = mapped_column(String(36), ForeignKey("trees.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("nodes.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(300), default="")
    selected_text: Mapped[str] = mapped_column(Text, default="")
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    depth: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tree: Mapped["Tree"] = relationship("Tree", back_populates="nodes")  # noqa: F821
    children: Mapped[list["Node"]] = relationship("Node", foreign_keys=[parent_id])
