from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.tree import Tree
from app.schemas.study import (
    KnowledgeItemRead,
    ReviewEventCreate,
    ReviewEventRead,
    StudyMode,
    StudyQuestionRead,
    WeaknessRead,
)
from app.services import question_planner

router = APIRouter(tags=["study"])


@router.get("/api/trees/{tree_id}/study/next", response_model=StudyQuestionRead)
async def next_study_question(
    tree_id: str,
    mode: StudyMode = Query("review"),
    db: AsyncSession = Depends(get_db),
):
    tree = await db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    question = await question_planner.plan_next_question(tree_id, db, mode)
    if not question:
        raise HTTPException(status_code=404, detail="No answered nodes to study yet")
    return await question_planner.question_to_dict(question, db)


@router.post("/api/study/events", response_model=ReviewEventRead, status_code=201)
async def create_review_event(
    payload: ReviewEventCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await question_planner.record_review_event(payload, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/api/trees/{tree_id}/knowledge", response_model=list[KnowledgeItemRead])
async def list_tree_knowledge(tree_id: str, db: AsyncSession = Depends(get_db)):
    tree = await db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    return await question_planner.list_knowledge(tree_id, db)


@router.get("/api/trees/{tree_id}/weaknesses", response_model=list[WeaknessRead])
async def list_tree_weaknesses(tree_id: str, db: AsyncSession = Depends(get_db)):
    tree = await db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    return await question_planner.list_weaknesses(tree_id, db)
