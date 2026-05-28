from __future__ import annotations
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.knowledge import GeneratedQuestion, KnowledgeEdge, KnowledgeItem, LearningState, ReviewEvent
from app.schemas.study import ReviewEventCreate
from app.services.knowledge_graph_service import (
    bootstrap_tree_knowledge,
    compact_text,
    get_learning_state,
    item_read_dict,
    list_items_with_state,
    next_review_reason,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _candidate_score(item: KnowledgeItem, state: LearningState, mode: str) -> float:
    now = datetime.utcnow()
    due_bonus = 0.0
    if state.next_review_at is None:
        due_bonus = 0.3
    elif state.next_review_at <= now:
        due_bonus = 0.55

    score = (
        (1.0 - state.mastery) * 1.7
        + item.importance * 0.18
        + item.difficulty * 0.08
        + state.wrong_count * 0.35
        + state.fuzzy_count * 0.18
        + due_bonus
        - state.correct_count * 0.04
    )
    if mode == "deepen":
        score += item.difficulty * 0.16 + item.importance * 0.08
    if mode == "connect":
        score += item.importance * 0.15
    return score


def _source_node_ids(items: list[KnowledgeItem]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item.source_node_id and item.source_node_id not in seen:
            seen.append(item.source_node_id)
    return seen


def _answer_for_item(item: KnowledgeItem) -> str:
    return item.canonical_text or item.summary or item.source_span


async def _create_question(
    db: AsyncSession,
    tree_id: str,
    question_type: str,
    difficulty: int,
    prompt: str,
    expected_answer: str,
    items: list[KnowledgeItem],
) -> GeneratedQuestion:
    question = GeneratedQuestion(
        tree_id=tree_id,
        question_type=question_type,
        difficulty=difficulty,
        prompt=prompt,
        expected_answer=expected_answer,
        related_item_ids=[item.id for item in items],
        source_node_ids=_source_node_ids(items),
        last_used_at=datetime.utcnow(),
    )
    db.add(question)
    await db.commit()
    await db.refresh(question)
    return question


async def plan_next_question(tree_id: str, db: AsyncSession, mode: str = "review") -> GeneratedQuestion | None:
    await bootstrap_tree_knowledge(tree_id, db)
    pairs = await list_items_with_state(tree_id, db)
    if not pairs:
        return None

    pairs.sort(key=lambda pair: _candidate_score(pair[0], pair[1], mode), reverse=True)

    if mode == "connect":
        connect_question = await _plan_connect_question(tree_id, db, pairs)
        if connect_question:
            return connect_question

    item, state = pairs[0]
    if mode == "deepen":
        prompt = f"从更深一层解释：{item.title} 背后的机制、适用条件或容易误解的地方是什么？"
        expected = (
            f"{_answer_for_item(item)}\n\n"
            "回答时应能说明它的底层原因、适用边界，以及至少一个容易混淆的点。"
        )
        return await _create_question(db, tree_id, "deepen", min(5, item.difficulty + 1), prompt, expected, [item])

    if state.mastery < 0.35 or state.wrong_count > state.correct_count:
        prompt = f"用你自己的话解释一下：{item.title}。"
        qtype = "explain"
    elif item.type == "procedure":
        prompt = f"如果要实际使用「{item.title}」，关键步骤是什么？"
        qtype = "apply"
    else:
        prompt = f"不看答案回忆一下：{item.title} 的核心意思是什么？"
        qtype = "recall"

    return await _create_question(db, tree_id, qtype, item.difficulty, prompt, _answer_for_item(item), [item])


async def _plan_connect_question(
    tree_id: str,
    db: AsyncSession,
    pairs: list[tuple[KnowledgeItem, LearningState]],
) -> GeneratedQuestion | None:
    edge_result = await db.execute(
        select(KnowledgeEdge)
        .where(KnowledgeEdge.tree_id == tree_id)
        .order_by(KnowledgeEdge.strength.desc(), KnowledgeEdge.created_at.desc())
        .limit(20)
    )
    edges = list(edge_result.scalars().all())
    items_by_id = {item.id: item for item, _ in pairs}
    for edge in edges:
        source = items_by_id.get(edge.source_item_id) or await db.get(KnowledgeItem, edge.source_item_id)
        target = items_by_id.get(edge.target_item_id) or await db.get(KnowledgeItem, edge.target_item_id)
        if not source or not target:
            continue
        prompt = f"把「{source.title}」和「{target.title}」联系起来：它们之间是什么关系？"
        expected = (
            f"关系类型：{edge.relation_type}\n\n"
            f"{source.title}: {_answer_for_item(source)}\n\n"
            f"{target.title}: {_answer_for_item(target)}"
        )
        return await _create_question(
            db,
            tree_id,
            "connect",
            max(source.difficulty, target.difficulty),
            prompt,
            expected,
            [source, target],
        )

    if len(pairs) < 2:
        return None
    first = pairs[0][0]
    second = next((item for item, _ in pairs[1:] if item.id != first.id), None)
    if not second:
        return None
    prompt = f"试着连接两个知识点：「{first.title}」和「{second.title}」有什么共同点、区别或因果关系？"
    expected = (
        f"{first.title}: {_answer_for_item(first)}\n\n"
        f"{second.title}: {_answer_for_item(second)}\n\n"
        "可以从共同问题、前置关系、应用场景、差异和类比这几个角度建立联系。"
    )
    return await _create_question(
        db,
        tree_id,
        "connect",
        max(first.difficulty, second.difficulty),
        prompt,
        expected,
        [first, second],
    )


async def record_review_event(payload: ReviewEventCreate, db: AsyncSession) -> ReviewEvent:
    question = await db.get(GeneratedQuestion, payload.question_id)
    if not question or question.tree_id != payload.tree_id:
        raise ValueError("Question not found")

    item_ids = list(question.related_item_ids or [])
    if not item_ids:
        raise ValueError("Question has no related knowledge item")

    first_event: ReviewEvent | None = None
    for item_id in item_ids:
        state = await get_learning_state(db, payload.tree_id, item_id)
        _apply_result(state, payload.result)
        event = ReviewEvent(
            tree_id=payload.tree_id,
            question_id=payload.question_id,
            knowledge_item_id=item_id,
            result=payload.result,
            user_answer=payload.user_answer,
            ai_feedback=payload.ai_feedback,
        )
        db.add(event)
        if first_event is None:
            first_event = event

    await db.commit()
    if first_event:
        await db.refresh(first_event)
    return first_event


def _apply_result(state: LearningState, result: str) -> None:
    now = datetime.utcnow()
    state.last_reviewed_at = now
    if result in {"knew", "ai_graded_correct"}:
        state.correct_count += 1
        state.mastery = _clamp(state.mastery + 0.18)
        state.confidence = _clamp(state.confidence + 0.16)
        state.next_review_at = now + timedelta(days=3)
    elif result == "fuzzy":
        state.fuzzy_count += 1
        state.mastery = _clamp(state.mastery + 0.03)
        state.confidence = _clamp(state.confidence + 0.02)
        state.next_review_at = now + timedelta(days=1)
    else:
        state.wrong_count += 1
        state.mastery = _clamp(state.mastery - 0.15)
        state.confidence = _clamp(state.confidence - 0.08)
        state.next_review_at = now + timedelta(hours=4)

    tags: set[str] = set(state.weakness_tags or [])
    if state.mastery < 0.45:
        tags.add("low_mastery")
    if state.wrong_count > state.correct_count:
        tags.add("frequent_miss")
    if result == "fuzzy":
        tags.add("fuzzy")
    if result in {"knew", "ai_graded_correct"} and state.mastery >= 0.75:
        tags.discard("low_mastery")
    state.weakness_tags = sorted(tags)


async def list_knowledge(tree_id: str, db: AsyncSession) -> list[dict]:
    await bootstrap_tree_knowledge(tree_id, db)
    pairs = await list_items_with_state(tree_id, db)
    return [item_read_dict(item, state) for item, state in pairs]


async def list_weaknesses(tree_id: str, db: AsyncSession, limit: int = 12) -> list[dict]:
    await bootstrap_tree_knowledge(tree_id, db)
    pairs = await list_items_with_state(tree_id, db)
    scored = [
        (_candidate_score(item, state, "review"), item, state)
        for item, state in pairs
        if state.mastery < 0.7 or state.wrong_count or state.fuzzy_count
    ]
    scored.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "item": item_read_dict(item, state),
            "reason": next_review_reason(state),
            "priority": round(score, 3),
        }
        for score, item, state in scored[:limit]
    ]


async def question_to_dict(question: GeneratedQuestion, db: AsyncSession) -> dict:
    related: list[dict] = []
    for item_id in question.related_item_ids or []:
        item = await db.get(KnowledgeItem, item_id)
        if not item:
            continue
        state = await get_learning_state(db, question.tree_id, item.id)
        related.append(item_read_dict(item, state))
    return {
        "id": question.id,
        "tree_id": question.tree_id,
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "prompt": question.prompt,
        "expected_answer": question.expected_answer,
        "related_item_ids": question.related_item_ids or [],
        "source_node_ids": question.source_node_ids or [],
        "created_at": question.created_at,
        "related_items": related,
    }
