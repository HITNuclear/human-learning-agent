from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tree_id: Mapped[str] = mapped_column(String(36), ForeignKey("trees.id"), index=True)
    type: Mapped[str] = mapped_column(String(50), default="concept")
    title: Mapped[str] = mapped_column(String(300), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    canonical_text: Mapped[str] = mapped_column(Text, default="")
    difficulty: Mapped[int] = mapped_column(Integer, default=1)
    importance: Mapped[int] = mapped_column(Integer, default=3)
    source_node_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("nodes.id"), nullable=True)
    source_span: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeEdge(Base):
    __tablename__ = "knowledge_edges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tree_id: Mapped[str] = mapped_column(String(36), ForeignKey("trees.id"), index=True)
    source_item_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_items.id"), index=True)
    target_item_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_items.id"), index=True)
    relation_type: Mapped[str] = mapped_column(String(50), default="explains")
    evidence_node_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("nodes.id"), nullable=True)
    strength: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LearningState(Base):
    __tablename__ = "learning_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tree_id: Mapped[str] = mapped_column(String(36), ForeignKey("trees.id"), index=True)
    knowledge_item_id: Mapped[str] = mapped_column(String(36), ForeignKey("knowledge_items.id"), index=True)
    mastery: Mapped[float] = mapped_column(Float, default=0.25)
    confidence: Mapped[float] = mapped_column(Float, default=0.2)
    wrong_count: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    fuzzy_count: Mapped[int] = mapped_column(Integer, default=0)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_review_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    weakness_tags: Mapped[list[str]] = mapped_column(JSON, default=list)


class GeneratedQuestion(Base):
    __tablename__ = "generated_questions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tree_id: Mapped[str] = mapped_column(String(36), ForeignKey("trees.id"), index=True)
    question_type: Mapped[str] = mapped_column(String(50), default="recall")
    difficulty: Mapped[int] = mapped_column(Integer, default=1)
    prompt: Mapped[str] = mapped_column(Text)
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    related_item_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_node_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tree_id: Mapped[str] = mapped_column(String(36), ForeignKey("trees.id"), index=True)
    question_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("generated_questions.id"), nullable=True)
    knowledge_item_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("knowledge_items.id"), nullable=True)
    result: Mapped[str] = mapped_column(String(40))
    user_answer: Mapped[str] = mapped_column(Text, default="")
    ai_feedback: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
