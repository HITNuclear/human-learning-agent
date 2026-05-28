from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models.tree import Tree
from app.models.node import Node
from app.schemas.tree import TreeCreate, TreeRead, TreeListItem
from app.schemas.node import NodeRead
from app.services.node_title_service import fallback_node_title, should_regenerate_title, update_node_title

router = APIRouter(prefix="/api/trees", tags=["trees"])


@router.get("", response_model=list[TreeListItem])
async def list_trees(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Tree, func.count(Node.id).label("node_count"))
        .outerjoin(Node, Node.tree_id == Tree.id)
        .group_by(Tree.id)
        .order_by(Tree.created_at.desc())
    )
    rows = result.all()
    return [
        TreeListItem(
            id=row.Tree.id,
            title=row.Tree.title,
            created_at=row.Tree.created_at,
            node_count=row.node_count,
        )
        for row in rows
    ]


@router.post("", response_model=NodeRead, status_code=201)
async def create_tree(payload: TreeCreate, db: AsyncSession = Depends(get_db)):
    tree = Tree(id=str(uuid.uuid4()), title=payload.title)
    db.add(tree)
    await db.flush()

    root_node = Node(
        id=str(uuid.uuid4()),
        tree_id=tree.id,
        parent_id=None,
        title=fallback_node_title(payload.title or payload.question),
        selected_text="",
        question=payload.question,
        answer="",
        depth=0,
    )
    db.add(root_node)
    await db.commit()
    await db.refresh(root_node)
    return root_node


@router.get("/{tree_id}/nodes", response_model=list[NodeRead])
async def get_tree_nodes(tree_id: str, db: AsyncSession = Depends(get_db)):
    tree = await db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    result = await db.execute(
        select(Node).where(Node.tree_id == tree_id).order_by(Node.depth, Node.created_at)
    )
    nodes = list(result.scalars().all())
    ai_updates = 0
    changed = False
    for node in nodes:
        if node.answer and should_regenerate_title(node) and ai_updates < 8:
            await update_node_title(node, db)
            ai_updates += 1
        elif should_regenerate_title(node):
            node.title = fallback_node_title(node.question, node.selected_text, node.answer)
            changed = True
    if changed:
        await db.commit()
    return nodes


@router.delete("/{tree_id}", status_code=204)
async def delete_tree(tree_id: str, db: AsyncSession = Depends(get_db)):
    tree = await db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    await db.delete(tree)
    await db.commit()
