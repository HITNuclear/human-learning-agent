from __future__ import annotations
import uuid
import json
import logging
from time import perf_counter
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from app.database import get_db, AsyncSessionLocal
from app.models.node import Node
from app.models.panel_state import NodePanelState
from app.models.request_metric import RequestMetric
from app.schemas.node import NodeCreate, NodeRead, NodePanelsRead, NodePanelsUpdate
from app.services import context_builder, llm_service
from app.services.knowledge_extractor import schedule_node_extraction
from app.services.node_title_service import fallback_node_title, should_regenerate_title, update_node_title
from app.services.rag_service import get_rag_service

router = APIRouter(prefix="/api/nodes", tags=["nodes"])
logger = logging.getLogger(__name__)


def metric_row_from_payload(payload: dict) -> RequestMetric:
    rag = payload.get("rag") or {}
    return RequestMetric(
        event=str(payload.get("event") or "node_stream_metrics"),
        node_id=str(payload.get("node_id") or "") or None,
        tree_id=str(payload.get("tree_id") or "") or None,
        success=payload.get("event") != "node_stream_failed",
        cache_hit=bool(payload.get("cache_hit", False)),
        retrieval_ms=float(payload.get("retrieval_ms") or 0),
        prompt_build_ms=float(payload.get("prompt_build_ms") or 0),
        generation_ms=float(payload.get("generation_ms") or 0),
        ttft_ms=float(payload.get("ttft_ms") or 0),
        total_ms=float(payload.get("total_ms") or 0),
        input_tokens_est=int(payload.get("input_tokens_est") or 0),
        output_tokens_est=int(payload.get("output_tokens_est") or 0),
        cost_usd_est=float(payload.get("cost_usd_est") or 0),
        rag_total_hits=int(rag.get("total_hits") or 0),
        rag_global_hits=int(rag.get("global_hits") or 0),
        rag_tree_hits=int(rag.get("tree_hits") or 0),
        rag_other_hits=int(rag.get("other_hits") or 0),
        rag_unique_sources=int(rag.get("unique_sources") or 0),
        error=str(payload.get("error") or ""),
    )


async def persist_metric(payload: dict) -> None:
    async with AsyncSessionLocal() as metric_db:
        metric_db.add(metric_row_from_payload(payload))
        await metric_db.commit()


@router.get("/{node_id}", response_model=NodeRead)
async def get_node(node_id: str, db: AsyncSession = Depends(get_db)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    if node.answer and should_regenerate_title(node):
        await update_node_title(node, db)
    elif should_regenerate_title(node):
        node.title = fallback_node_title(node.question, node.selected_text)
        await db.commit()
    return node


@router.get("/{node_id}/panels", response_model=NodePanelsRead)
async def get_node_panels(node_id: str, db: AsyncSession = Depends(get_db)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    result = await db.execute(select(NodePanelState).where(NodePanelState.node_id == node_id))
    state = result.scalar_one_or_none()
    if not state:
        return NodePanelsRead(node_id=node_id, panels=[])
    return NodePanelsRead(node_id=node_id, panels=state.panels or [])


@router.put("/{node_id}/panels", response_model=NodePanelsRead)
async def put_node_panels(node_id: str, payload: NodePanelsUpdate, db: AsyncSession = Depends(get_db)):
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    result = await db.execute(select(NodePanelState).where(NodePanelState.node_id == node_id))
    state = result.scalar_one_or_none()
    panels = [panel.model_dump(mode="json") for panel in payload.panels]
    if state is None:
        state = NodePanelState(node_id=node_id, panels=panels)
        db.add(state)
    else:
        state.panels = panels
    await db.commit()

    return NodePanelsRead(node_id=node_id, panels=panels)


@router.post("", response_model=NodeRead, status_code=201)
async def create_node(payload: NodeCreate, db: AsyncSession = Depends(get_db)):
    parent = await db.get(Node, payload.parent_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Parent node not found")

    node = Node(
        id=str(uuid.uuid4()),
        tree_id=parent.tree_id,
        parent_id=parent.id,
        title=fallback_node_title(payload.question, payload.selected_text),
        selected_text=payload.selected_text,
        question=payload.question,
        answer="",
        depth=parent.depth + 1,
    )
    db.add(node)
    await db.commit()
    await db.refresh(node)
    return node


@router.get("/{node_id}/stream")
async def stream_node_answer(node_id: str, db: AsyncSession = Depends(get_db)):
    request_started_at = perf_counter()
    node = await db.get(Node, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    if node.answer:
        output_tokens = llm_service.estimate_tokens(node.answer)
        metrics = {
            "event": "node_stream_metrics",
            "node_id": node_id,
            "tree_id": node.tree_id,
            "cache_hit": True,
            "retrieval_ms": 0,
            "prompt_build_ms": 0,
            "generation_ms": 0,
            "ttft_ms": 0,
            "total_ms": round((perf_counter() - request_started_at) * 1000, 2),
            "input_tokens_est": 0,
            "output_tokens_est": output_tokens,
            "cost_usd_est": llm_service.estimate_cost_usd(0, output_tokens),
            "rag": {
                "total_hits": 0,
                "global_hits": 0,
                "tree_hits": 0,
                "other_hits": 0,
                "unique_sources": 0,
            },
        }
        logger.info(json.dumps(metrics, ensure_ascii=False))
        await persist_metric(metrics)

        async def already_done():
            yield ServerSentEvent(data=node.answer, event="full")
            yield ServerSentEvent(data="", event="done")
        return EventSourceResponse(already_done())

    rag = get_rag_service()
    retrieval_started_at = perf_counter()
    rag_chunks, rag_stats = rag.retrieve_with_stats(node.question, node.tree_id)
    retrieval_ms = (perf_counter() - retrieval_started_at) * 1000

    prompt_started_at = perf_counter()
    messages = await context_builder.build_messages(node, db, rag_chunks)
    prompt_build_ms = (perf_counter() - prompt_started_at) * 1000
    input_tokens_est = llm_service.estimate_message_tokens(messages)

    # db session closes after this function returns — use a fresh session in the generator

    async def event_generator():
        full_answer: list[str] = []
        generation_started_at = perf_counter()
        first_token_at: float | None = None
        try:
            async for token in llm_service.stream_completion(messages):
                if first_token_at is None:
                    first_token_at = perf_counter()
                full_answer.append(token)
                yield ServerSentEvent(data=token, event="token")
        except Exception as e:
            metrics = {
                "event": "node_stream_failed",
                "node_id": node_id,
                "tree_id": node.tree_id,
                "error": str(e),
                "retrieval_ms": round(retrieval_ms, 2),
                "prompt_build_ms": round(prompt_build_ms, 2),
                "generation_ms": round((perf_counter() - generation_started_at) * 1000, 2),
                "ttft_ms": round(((first_token_at - generation_started_at) * 1000), 2) if first_token_at else 0,
                "total_ms": round((perf_counter() - request_started_at) * 1000, 2),
                "input_tokens_est": input_tokens_est,
                "output_tokens_est": llm_service.estimate_tokens("".join(full_answer)),
                "cost_usd_est": llm_service.estimate_cost_usd(
                    input_tokens_est,
                    llm_service.estimate_tokens("".join(full_answer)),
                ),
                "rag": rag_stats,
            }
            logger.error(json.dumps(metrics, ensure_ascii=False))
            await persist_metric(metrics)
            yield ServerSentEvent(data=str(e), event="error")
            return

        generation_ms = (perf_counter() - generation_started_at) * 1000
        output_text = "".join(full_answer)
        output_tokens_est = llm_service.estimate_tokens(output_text)
        total_ms = (perf_counter() - request_started_at) * 1000

        metrics = {
            "event": "node_stream_metrics",
            "node_id": node_id,
            "tree_id": node.tree_id,
            "cache_hit": False,
            "retrieval_ms": round(retrieval_ms, 2),
            "prompt_build_ms": round(prompt_build_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "ttft_ms": round(((first_token_at - generation_started_at) * 1000), 2) if first_token_at else 0,
            "total_ms": round(total_ms, 2),
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": output_tokens_est,
            "cost_usd_est": llm_service.estimate_cost_usd(input_tokens_est, output_tokens_est),
            "rag": rag_stats,
        }
        logger.info(json.dumps(metrics, ensure_ascii=False))
        await persist_metric(metrics)

        async with AsyncSessionLocal() as save_db:
            saved_node = await save_db.get(Node, node_id)
            if saved_node:
                saved_node.answer = output_text
                await save_db.commit()
                await update_node_title(saved_node, save_db, force=True)
                schedule_node_extraction(node_id)
        yield ServerSentEvent(data="", event="done")

    return EventSourceResponse(event_generator())


@router.get("/{node_id}/path", response_model=list[NodeRead])
async def get_node_path(node_id: str, db: AsyncSession = Depends(get_db)):
    """Returns the ancestor path from root to this node (inclusive)."""
    path = await context_builder.get_ancestor_path(node_id, db)
    if not path:
        raise HTTPException(status_code=404, detail="Node not found")
    return path
