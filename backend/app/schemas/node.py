from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class NodeCreate(BaseModel):
    parent_id: str
    selected_text: str = ""
    question: str


class NodeRead(BaseModel):
    id: str
    tree_id: str
    parent_id: str | None
    title: str
    selected_text: str
    question: str
    answer: str
    depth: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PanelRect(BaseModel):
    top: float
    left: float
    right: float
    bottom: float
    width: float
    height: float


class NodePanelItem(BaseModel):
    id: str
    node_id: str | None = None
    question: str = ""
    rect: PanelRect
    rects: list[PanelRect] = Field(default_factory=list)
    content_rect: PanelRect
    content_rects: list[PanelRect] = Field(default_factory=list)
    text: str


class NodePanelsRead(BaseModel):
    node_id: str
    panels: list[NodePanelItem] = Field(default_factory=list)


class NodePanelsUpdate(BaseModel):
    panels: list[NodePanelItem] = Field(default_factory=list)
