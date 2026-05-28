from __future__ import annotations
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.request_metric import RequestMetric
from app.schemas.metrics import MetricsSummaryRead

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * q)
    return round(sorted_values[index], 2)


@router.get("/summary", response_model=MetricsSummaryRead)
async def metrics_summary(
    window_hours: int = Query(24, ge=1, le=24 * 30),
    tree_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(hours=window_hours)
    conditions = [RequestMetric.created_at >= since, RequestMetric.event == "node_stream_metrics"]
    if tree_id:
        conditions.append(RequestMetric.tree_id == tree_id)

    result = await db.execute(select(RequestMetric).where(and_(*conditions)).order_by(RequestMetric.created_at.desc()))
    rows = list(result.scalars().all())

    total_requests = len(rows)
    success_requests = sum(1 for row in rows if row.success)
    failed_requests = total_requests - success_requests
    cache_hit_requests = sum(1 for row in rows if row.cache_hit)

    total_ms_values = [float(row.total_ms or 0) for row in rows]
    ttft_values = [float(row.ttft_ms or 0) for row in rows if float(row.ttft_ms or 0) > 0]

    total_input_tokens_est = int(sum(int(row.input_tokens_est or 0) for row in rows))
    total_output_tokens_est = int(sum(int(row.output_tokens_est or 0) for row in rows))
    total_cost_usd_est = float(sum(float(row.cost_usd_est or 0) for row in rows))

    rag_total_hits = int(sum(int(row.rag_total_hits or 0) for row in rows))
    rag_global_hits = int(sum(int(row.rag_global_hits or 0) for row in rows))
    rag_tree_hits = int(sum(int(row.rag_tree_hits or 0) for row in rows))
    rag_other_hits = int(sum(int(row.rag_other_hits or 0) for row in rows))
    rag_unique_sources_sum = int(sum(int(row.rag_unique_sources or 0) for row in rows))

    avg_total_ms = round(sum(total_ms_values) / total_requests, 2) if total_requests else 0.0
    avg_ttft_ms = round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else 0.0
    avg_cost_usd_est = round(total_cost_usd_est / total_requests, 8) if total_requests else 0.0

    return MetricsSummaryRead(
        window_hours=window_hours,
        tree_id=tree_id,
        total_requests=total_requests,
        success_requests=success_requests,
        failed_requests=failed_requests,
        cache_hit_requests=cache_hit_requests,
        avg_total_ms=avg_total_ms,
        p50_total_ms=_percentile(total_ms_values, 0.5),
        p95_total_ms=_percentile(total_ms_values, 0.95),
        avg_ttft_ms=avg_ttft_ms,
        total_input_tokens_est=total_input_tokens_est,
        total_output_tokens_est=total_output_tokens_est,
        total_cost_usd_est=round(total_cost_usd_est, 8),
        avg_cost_usd_est=avg_cost_usd_est,
        rag_total_hits=rag_total_hits,
        rag_global_hits=rag_global_hits,
        rag_tree_hits=rag_tree_hits,
        rag_other_hits=rag_other_hits,
        rag_unique_sources_sum=rag_unique_sources_sum,
    )
