from __future__ import annotations
from pydantic import BaseModel


class MetricsSummaryRead(BaseModel):
    window_hours: int
    tree_id: str | None

    total_requests: int
    success_requests: int
    failed_requests: int
    cache_hit_requests: int

    avg_total_ms: float
    p50_total_ms: float
    p95_total_ms: float
    avg_ttft_ms: float

    total_input_tokens_est: int
    total_output_tokens_est: int
    total_cost_usd_est: float
    avg_cost_usd_est: float

    rag_total_hits: int
    rag_global_hits: int
    rag_tree_hits: int
    rag_other_hits: int
    rag_unique_sources_sum: int
