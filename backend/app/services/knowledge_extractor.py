from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import AsyncSessionLocal
from app.models.knowledge import KnowledgeEdge, KnowledgeItem, LearningState
from app.models.node import Node
from app.services import llm_service
from app.services.knowledge_graph_service import compact_text, normalize_title

logger = logging.getLogger(__name__)

ALLOWED_ITEM_TYPES = {"concept", "proposition", "procedure", "example", "misconception"}
ALLOWED_EDGE_TYPES = {"prerequisite", "explains", "example_of", "contrast", "cause", "analogy", "applies_to"}


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _safe_int(value: Any, default: int, min_value: int = 1, max_value: int = 5) -> int:
    try:
        return max(min_value, min(max_value, int(value)))
    except Exception:
        return default


def _safe_float(value: Any, default: float, min_value: float = 0.1, max_value: float = 1.0) -> float:
    try:
        return max(min_value, min(max_value, float(value)))
    except Exception:
        return default


def fallback_items_from_node(node: Node) -> list[dict]:
    answer = node.answer or ""
    headings = re.findall(r"^#{1,3}\s+(.+)$", answer, flags=re.MULTILINE)
    items: list[dict] = []

    for heading in headings[:5]:
        items.append(
            {
                "type": "concept",
                "title": compact_text(heading, 120),
                "summary": compact_text(answer, 500),
                "canonical_text": compact_text(answer, 1400),
                "difficulty": max(1, min(5, node.depth + 1)),
                "importance": 3,
                "source_span": compact_text(heading, 300),
            }
        )

    if not items:
        items.append(
            {
                "type": "concept",
                "title": compact_text(node.title or node.question, 120),
                "summary": compact_text(answer, 500),
                "canonical_text": compact_text(answer, 1400),
                "difficulty": max(1, min(5, node.depth + 1)),
                "importance": 3,
                "source_span": compact_text(node.selected_text or node.question, 300),
            }
        )
    return items


async def extract_with_llm(node: Node) -> dict:
    prompt = f"""
Question:
{node.question}

Node title:
{node.title}

Selected parent text:
{node.selected_text}

Answer:
{node.answer}
""".strip()
    messages = [
        {
            "role": "system",
            "content": (
                "Extract study memory from a learning Q&A. Return only valid JSON. "
                "Use the same language as the content. Keep items concise and useful for future quizzes. "
                "Schema: {\"items\":[{\"type\":\"concept|proposition|procedure|example|misconception\","
                "\"title\":\"short title\",\"summary\":\"1-3 sentence summary\","
                "\"canonical_text\":\"self-contained answer snippet\",\"difficulty\":1-5,"
                "\"importance\":1-5,\"source_span\":\"short evidence text\"}],"
                "\"edges\":[{\"source_title\":\"title from items\","
                "\"target_title\":\"title from items\",\"relation_type\":\"prerequisite|explains|example_of|contrast|cause|analogy|applies_to\","
                "\"strength\":0.1}]}. Limit to 8 items and 8 edges."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    raw = await llm_service.complete_text(messages, max_tokens=1800)
    return json.loads(_strip_json_fence(raw))


async def extract_node_knowledge(node_id: str, db: AsyncSession) -> int:
    node = await db.get(Node, node_id)
    if not node or not node.answer.strip():
        return 0

    existing = await db.execute(
        select(KnowledgeItem.id)
        .where(KnowledgeItem.source_node_id == node_id, KnowledgeItem.type != "conversation")
        .limit(1)
    )
    if existing.scalar_one_or_none():
        return 0

    payload: dict[str, Any]
    try:
        payload = await extract_with_llm(node)
        items_payload = payload.get("items") or []
        if not items_payload:
            raise ValueError("no items extracted")
    except Exception as exc:
        logger.warning("LLM knowledge extraction failed for node %s: %s", node_id, exc)
        payload = {"items": fallback_items_from_node(node), "edges": []}

    existing_items = await db.execute(select(KnowledgeItem).where(KnowledgeItem.tree_id == node.tree_id))
    by_title = {normalize_title(item.title): item for item in existing_items.scalars().all()}
    created_by_title: dict[str, KnowledgeItem] = {}
    created = 0

    for raw_item in (payload.get("items") or [])[:8]:
        title = compact_text(str(raw_item.get("title") or "").strip(), 120)
        if not title:
            continue
        key = normalize_title(title)
        item = by_title.get(key)
        item_type = str(raw_item.get("type") or "concept")
        if item_type not in ALLOWED_ITEM_TYPES:
            item_type = "concept"
        if item is None:
            item = KnowledgeItem(
                tree_id=node.tree_id,
                type=item_type,
                title=title,
                summary=compact_text(str(raw_item.get("summary") or ""), 700),
                canonical_text=compact_text(str(raw_item.get("canonical_text") or raw_item.get("summary") or ""), 1600),
                difficulty=_safe_int(raw_item.get("difficulty"), max(1, min(5, node.depth + 1))),
                importance=_safe_int(raw_item.get("importance"), 3),
                source_node_id=node.id,
                source_span=compact_text(str(raw_item.get("source_span") or node.selected_text or ""), 400),
            )
            db.add(item)
            await db.flush()
            db.add(
                LearningState(
                    tree_id=node.tree_id,
                    knowledge_item_id=item.id,
                    mastery=0.25,
                    confidence=0.15,
                    weakness_tags=["new"],
                )
            )
            by_title[key] = item
            created += 1
        elif item.type == "conversation":
            item.type = item_type
            item.summary = compact_text(str(raw_item.get("summary") or item.summary), 700)
            item.canonical_text = compact_text(
                str(raw_item.get("canonical_text") or raw_item.get("summary") or item.canonical_text),
                1600,
            )
            item.difficulty = _safe_int(raw_item.get("difficulty"), item.difficulty)
            item.importance = _safe_int(raw_item.get("importance"), item.importance)
            item.source_span = compact_text(str(raw_item.get("source_span") or item.source_span), 400)
        created_by_title[key] = item

    for raw_edge in (payload.get("edges") or [])[:8]:
        source = created_by_title.get(normalize_title(str(raw_edge.get("source_title") or "")))
        target = created_by_title.get(normalize_title(str(raw_edge.get("target_title") or "")))
        if not source or not target or source.id == target.id:
            continue
        relation_type = str(raw_edge.get("relation_type") or "explains")
        if relation_type not in ALLOWED_EDGE_TYPES:
            relation_type = "explains"
        db.add(
            KnowledgeEdge(
                tree_id=node.tree_id,
                source_item_id=source.id,
                target_item_id=target.id,
                relation_type=relation_type,
                evidence_node_id=node.id,
                strength=_safe_float(raw_edge.get("strength"), 0.5),
            )
        )

    await db.commit()
    return created


async def extract_node_knowledge_by_id(node_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await extract_node_knowledge(node_id, db)


def schedule_node_extraction(node_id: str) -> None:
    try:
        asyncio.create_task(extract_node_knowledge_by_id(node_id))
    except RuntimeError:
        logger.exception("Could not schedule knowledge extraction for node %s", node_id)
