from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


StudyMode = Literal["review", "deepen", "connect"]
ReviewResult = Literal["knew", "fuzzy", "didnt_know", "ai_graded_correct", "ai_graded_wrong"]


class KnowledgeItemRead(BaseModel):
    id: str
    tree_id: str
    type: str
    title: str
    summary: str
    canonical_text: str
    difficulty: int
    importance: int
    source_node_id: str | None
    source_span: str
    created_at: datetime
    mastery: float | None = None
    wrong_count: int = 0
    correct_count: int = 0
    fuzzy_count: int = 0

    model_config = {"from_attributes": True}


class StudyQuestionRead(BaseModel):
    id: str
    tree_id: str
    question_type: str
    difficulty: int
    prompt: str
    expected_answer: str
    related_item_ids: list[str] = Field(default_factory=list)
    source_node_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    related_items: list[KnowledgeItemRead] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ReviewEventCreate(BaseModel):
    tree_id: str
    question_id: str
    result: ReviewResult
    user_answer: str = ""
    ai_feedback: str = ""


class ReviewEventRead(BaseModel):
    id: str
    tree_id: str
    question_id: str | None
    knowledge_item_id: str | None
    result: str
    user_answer: str
    ai_feedback: str
    created_at: datetime

    model_config = {"from_attributes": True}


class WeaknessRead(BaseModel):
    item: KnowledgeItemRead
    reason: str
    priority: float
