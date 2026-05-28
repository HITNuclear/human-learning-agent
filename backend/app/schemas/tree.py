from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class TreeCreate(BaseModel):
    title: str
    question: str


class TreeRead(BaseModel):
    id: str
    title: str
    created_at: datetime
    node_count: int = 0

    model_config = {"from_attributes": True}


class TreeListItem(BaseModel):
    id: str
    title: str
    created_at: datetime
    node_count: int = 0

    model_config = {"from_attributes": True}
