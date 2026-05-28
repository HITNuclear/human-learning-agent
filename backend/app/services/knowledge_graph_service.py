from __future__ import annotations
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.knowledge import KnowledgeItem, LearningState
from app.models.node import Node


def normalize_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def compact_text(text: str, limit: int = 700) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


async def get_learning_state(
    db: AsyncSession,
    tree_id: str,
    item_id: str,
) -> LearningState:
    result = await db.execute(
        select(LearningState).where(
            LearningState.tree_id == tree_id,
            LearningState.knowledge_item_id == item_id,
        )
    )
    state = result.scalar_one_or_none()
    if state:
        return state
    state = LearningState(tree_id=tree_id, knowledge_item_id=item_id)
    db.add(state)
    await db.flush()
    return state


async def list_items_with_state(tree_id: str, db: AsyncSession) -> list[tuple[KnowledgeItem, LearningState]]:
    result = await db.execute(
        select(KnowledgeItem)
        .where(KnowledgeItem.tree_id == tree_id)
        .order_by(KnowledgeItem.importance.desc(), KnowledgeItem.created_at.desc())
    )
    items = list(result.scalars().all())
    pairs: list[tuple[KnowledgeItem, LearningState]] = []
    changed = False
    for item in items:
        if item.type == "conversation" and item.source_node_id:
            node = await db.get(Node, item.source_node_id)
            if node and node.title and item.title != node.title:
                item.title = node.title
                item.summary = item.summary or compact_text(node.answer, 500)
                item.canonical_text = item.canonical_text or compact_text(node.answer, 1400)
                changed = True
        pairs.append((item, await get_learning_state(db, tree_id, item.id)))
    if changed:
        await db.commit()
    return pairs


async def bootstrap_tree_knowledge(tree_id: str, db: AsyncSession, limit: int = 30) -> int:
    """Create coarse knowledge items from answered nodes when no extraction exists yet."""
    existing = await db.execute(select(KnowledgeItem.id).where(KnowledgeItem.tree_id == tree_id).limit(1))
    if existing.scalar_one_or_none():
        return 0

    result = await db.execute(
        select(Node)
        .where(Node.tree_id == tree_id, Node.answer != "")
        .order_by(Node.depth, Node.created_at)
        .limit(limit)
    )
    nodes = list(result.scalars().all())
    created = 0
    seen: set[str] = set()
    for node in nodes:
        title = compact_text(node.title or node.question, 120)
        key = normalize_title(title)
        if not title or key in seen:
            continue
        seen.add(key)
        item = KnowledgeItem(
            tree_id=tree_id,
            type="conversation",
            title=title,
            summary=compact_text(node.answer, 500),
            canonical_text=compact_text(node.answer, 1400),
            difficulty=max(1, min(5, node.depth + 1)),
            importance=3,
            source_node_id=node.id,
            source_span=compact_text(node.selected_text or node.question, 300),
        )
        db.add(item)
        await db.flush()
        db.add(
            LearningState(
                tree_id=tree_id,
                knowledge_item_id=item.id,
                mastery=0.25,
                confidence=0.15,
                weakness_tags=["needs_review"],
            )
        )
        created += 1
    if created:
        await db.commit()
    return created


def item_read_dict(item: KnowledgeItem, state: LearningState | None = None) -> dict:
    return {
        "id": item.id,
        "tree_id": item.tree_id,
        "type": item.type,
        "title": item.title,
        "summary": item.summary,
        "canonical_text": item.canonical_text,
        "difficulty": item.difficulty,
        "importance": item.importance,
        "source_node_id": item.source_node_id,
        "source_span": item.source_span,
        "created_at": item.created_at,
        "mastery": state.mastery if state else None,
        "wrong_count": state.wrong_count if state else 0,
        "correct_count": state.correct_count if state else 0,
        "fuzzy_count": state.fuzzy_count if state else 0,
    }


def next_review_reason(state: LearningState, now: datetime | None = None) -> str:
    now = now or datetime.utcnow()
    if state.wrong_count > 0 and state.wrong_count >= state.correct_count:
        return "最近答错较多"
    if state.mastery < 0.45:
        return "掌握度偏低"
    if state.next_review_at and state.next_review_at <= now:
        return "到了复习时间"
    if state.fuzzy_count > 0:
        return "曾标记为模糊"
    return "值得巩固"
