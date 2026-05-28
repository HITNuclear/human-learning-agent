from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.node import Node


async def get_ancestor_path(node_id: str, db: AsyncSession) -> list[Node]:
    """Returns [root, ..., node] in root-first order."""
    path: list[Node] = []
    current_id: str | None = node_id
    while current_id is not None:
        result = await db.execute(select(Node).where(Node.id == current_id))
        node = result.scalar_one_or_none()
        if node is None:
            break
        path.append(node)
        current_id = node.parent_id
    path.reverse()
    return path


async def build_messages(
    node: Node,
    db: AsyncSession,
    rag_chunks: list[str],
) -> list[dict]:
    path = await get_ancestor_path(node.id, db)
    # path[-1] is the current node (answer="" at this point)
    ancestors = path[:-1]

    system_content = (
        "You are a learning assistant helping the user deeply understand topics step by step. "
        "When answering, explain clearly and make connections to the broader context. "
        "Use Markdown formatting with headers, bullet points, and code blocks where appropriate. "
        "Respond in the same language the user uses."
    )
    if rag_chunks:
        rag_context = "\n\n---\n\n".join(rag_chunks)
        system_content += f"\n\nRelevant knowledge from the user's documents:\n\n{rag_context}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    for i, ancestor in enumerate(ancestors):
        if i == 0:
            messages.append({"role": "user", "content": ancestor.question})
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"Based on this part: \"{ancestor.selected_text}\", "
                    f"I want to understand: {ancestor.question}"
                ),
            })
        messages.append({"role": "assistant", "content": ancestor.answer})

    # Current node question (no answer yet)
    if node.selected_text:
        messages.append({
            "role": "user",
            "content": (
                f"Based on this part: \"{node.selected_text}\", "
                f"I want to understand: {node.question}"
            ),
        })
    else:
        messages.append({"role": "user", "content": node.question})

    return messages
