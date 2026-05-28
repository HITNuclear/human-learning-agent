from __future__ import annotations
import logging
import re
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.node import Node
from app.services import llm_service

logger = logging.getLogger(__name__)


def _compact(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def sanitize_title(title: str) -> str:
    title = title.strip()
    title = re.sub(r"^#+\s*", "", title)
    title = title.strip(" \t\r\n\"'“”‘’`[]（）()：:。.!！?")
    title = re.sub(r"\s+", " ", title)
    return _compact(title, 80)


def fallback_node_title(question: str, selected_text: str = "", answer: str = "") -> str:
    question = _compact(question, 80)
    selected = _compact(selected_text, 36)
    if selected and question in {"是什么", "为什么", "怎么理解", "什么意思", "what is it", "why"}:
        return sanitize_title(selected)
    if selected and len(question) <= 12:
        return sanitize_title(f"{selected}：{question}")
    if question:
        return sanitize_title(question)
    return sanitize_title(answer) or "未命名节点"


def _normalize_title_candidate(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[\s\"'“”‘’`\[\]（）()：:。.!！?？]+", "", text)


def _looks_too_generic(title: str, question: str) -> bool:
    generic = {
        "是什么",
        "这是什么",
        "为什么",
        "怎么理解",
        "如何理解",
        "什么意思",
        "含义",
        "原因",
        "区别",
        "作用",
        "解释",
        "举例",
        "例子",
    }
    normalized = _normalize_title_candidate(title)
    normalized_question = _normalize_title_candidate(question)
    if re.search(r"[：:]\s*(是什么|为什么|怎么理解|如何理解|什么意思)\s*$", title.strip(), re.I):
        return True
    if normalized in generic:
        return True
    if len(normalized) <= 2:
        return True
    if normalized.endswith(("是什么", "为什么", "怎么理解", "如何理解", "什么意思")) and len(normalized) <= 8:
        return True
    return normalized == normalized_question and len(normalized) <= 20


def should_regenerate_title(node: Node) -> bool:
    title = node.title or ""
    if not title.strip():
        return True
    if _looks_too_generic(title, node.question):
        return True

    normalized_title = _normalize_title_candidate(title)
    normalized_question = _normalize_title_candidate(node.question)
    if normalized_title == normalized_question and node.selected_text and len(normalized_question) <= 40:
        return True
    return False


async def generate_node_title(node: Node) -> str:
    fallback = fallback_node_title(node.question, node.selected_text, node.answer)
    if not node.answer.strip():
        return fallback

    messages = [
        {
            "role": "system",
            "content": (
                "你是学习笔记标题生成器。请为一个学习树节点生成标题。"
                "标题必须点明具体知识对象，不能只写“是什么”“为什么”“区别”“作用”。"
                "输出同用户语言一致，8到24个中文字或等长短语，不要引号，不要编号，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{node.question}\n\n"
                f"选中的上下文：{_compact(node.selected_text, 500)}\n\n"
                f"AI回答摘要来源：{_compact(node.answer, 1400)}"
            ),
        },
    ]

    try:
        title = sanitize_title(await llm_service.complete_text(messages, max_tokens=80))
    except Exception as exc:
        logger.warning("Failed to generate title for node %s: %s", node.id, exc)
        return fallback

    if not title or _looks_too_generic(title, node.question):
        return fallback
    return title


async def update_node_title(node: Node, db: AsyncSession, force: bool = False) -> str:
    if not force and not should_regenerate_title(node):
        return node.title
    node.title = await generate_node_title(node)
    await db.commit()
    return node.title
