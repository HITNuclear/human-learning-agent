from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import ir_datasets
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    dataset: str
    embedding_model: str
    mode: str
    top_k: int
    dense_candidate_k: int
    bm25_candidate_k: int
    rrf_k: int
    rerank_candidate_k: int
    reranker_model: str
    max_docs: int | None
    max_queries: int | None
    output_dir: Path
    reset_index: bool


def get_doc_id(doc: object) -> str:
    for field in ("doc_id", "id"):
        value = getattr(doc, field, None)
        if value:
            return str(value)
    raise ValueError("Document has no id/doc_id field")


def get_query_id(query: object) -> str:
    for field in ("query_id", "id"):
        value = getattr(query, field, None)
        if value:
            return str(value)
    raise ValueError("Query has no query_id/id field")


def get_query_text(query: object) -> str:
    for field in ("text", "query"):
        value = getattr(query, field, None)
        if value:
            return str(value)
    raise ValueError("Query has no text/query field")


def get_doc_text(doc: object) -> str:
    parts: list[str] = []
    for field in ("title", "text", "body"):
        value = getattr(doc, field, None)
        if value:
            parts.append(str(value).strip())
    text = "\n\n".join(part for part in parts if part)
    if not text:
        raise ValueError("Document has no title/text/body fields")
    return text


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def dcg(relevances: list[float]) -> float:
    total = 0.0
    for rank, rel in enumerate(relevances, start=1):
        total += (2**rel - 1) / math.log2(rank + 1)
    return total


def evaluate_ranking(
    ranked_doc_ids: list[str],
    qrels_for_query: dict[str, int],
    top_k: int,
) -> dict[str, float]:
    ranked_top = ranked_doc_ids[:top_k]
    if not qrels_for_query:
        return {
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
        }

    relevant_doc_ids = {doc_id for doc_id, rel in qrels_for_query.items() if rel > 0}

    def recall_at(k: int) -> float:
        if not relevant_doc_ids:
            return 0.0
        hit = len(set(ranked_doc_ids[:k]).intersection(relevant_doc_ids))
        return hit / len(relevant_doc_ids)

    mrr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / rank
            break

    ranked_rels = [float(qrels_for_query.get(doc_id, 0)) for doc_id in ranked_doc_ids[:10]]
    ideal_rels = sorted((float(v) for v in qrels_for_query.values()), reverse=True)[:10]
    actual_dcg = dcg(ranked_rels)
    ideal_dcg = dcg(ideal_rels)
    ndcg = (actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    return {
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "mrr@10": mrr,
        "ndcg@10": ndcg,
    }


def build_qrels(dataset: ir_datasets.Dataset) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for item in dataset.qrels_iter():
        query_id = str(item.query_id)
        doc_id = str(item.doc_id)
        qrels.setdefault(query_id, {})[doc_id] = int(item.relevance)
    return qrels


def tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if tok]


def rrf_fuse(rankings: list[list[str]], rrf_k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for idx, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + idx + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def rerank_ids(
    query: str,
    ranked_ids: list[str],
    docs_by_id: dict[str, str],
    top_n: int,
    reranker: CrossEncoder,
) -> list[str]:
    if not ranked_ids:
        return []
    rerank_n = min(max(top_n, 1), len(ranked_ids))
    candidates = ranked_ids[:rerank_n]
    pairs = [[query, docs_by_id.get(doc_id, "")] for doc_id in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
    reranked = [doc_id for doc_id, _score in scored]
    if rerank_n < len(ranked_ids):
        reranked.extend(ranked_ids[rerank_n:])
    return reranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark with common resume-friendly datasets.")
    parser.add_argument("--dataset", default="beir/scifact/test", help="ir_datasets id, e.g. beir/scifact/test or miracl/zh/dev")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "hybrid_rerank"], default="dense")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-candidate-k", type=int, default=50)
    parser.add_argument("--bm25-candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--max-docs", type=int, default=15000, help="Limit corpus size for quick local runs")
    parser.add_argument("--max-queries", type=int, default=500, help="Limit query count for quick local runs")
    parser.add_argument("--output-dir", default="./eval/results")
    parser.add_argument("--reset-index", action="store_true", help="Delete existing benchmark index before running")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        dataset=args.dataset,
        embedding_model=args.embedding_model,
        mode=args.mode,
        top_k=args.top_k,
        dense_candidate_k=max(args.top_k, args.dense_candidate_k),
        bm25_candidate_k=max(args.top_k, args.bm25_candidate_k),
        rrf_k=args.rrf_k,
        rerank_candidate_k=max(args.top_k, args.rerank_candidate_k),
        reranker_model=args.reranker_model,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        output_dir=Path(args.output_dir),
        reset_index=args.reset_index,
    )

    dataset = ir_datasets.load(cfg.dataset)
    qrels = build_qrels(dataset)

    safe_dataset_name = cfg.dataset.replace("/", "_")
    index_path = cfg.output_dir / f"chroma_{safe_dataset_name}"
    output_path = cfg.output_dir / f"metrics_{safe_dataset_name}_{cfg.mode}.json"

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.reset_index and index_path.exists():
        shutil.rmtree(index_path)

    model = SentenceTransformer(cfg.embedding_model)
    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_or_create_collection(
        name="benchmark",
        metadata={"hnsw:space": "cosine"},
    )

    docs_added = 0
    batch_texts: list[str] = []
    batch_doc_ids: list[str] = []
    batch_metas: list[dict] = []
    bm25_doc_ids: list[str] = []
    bm25_tokenized_corpus: list[list[str]] = []
    docs_by_id: dict[str, str] = {}

    print(f"[INFO] Building index for dataset: {cfg.dataset}")
    for doc in tqdm(dataset.docs_iter(), desc="Indexing docs"):
        if cfg.max_docs is not None and docs_added >= cfg.max_docs:
            break
        try:
            doc_id = get_doc_id(doc)
            text = get_doc_text(doc)
        except ValueError:
            continue

        batch_texts.append(text)
        batch_doc_ids.append(doc_id)
        batch_metas.append({"doc_id": doc_id})
        bm25_doc_ids.append(doc_id)
        bm25_tokenized_corpus.append(tokenize(text))
        docs_by_id[doc_id] = text
        docs_added += 1

        if len(batch_texts) >= 128:
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(
                ids=batch_doc_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            batch_texts = []
            batch_doc_ids = []
            batch_metas = []

    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_doc_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metas,
        )

    print(f"[INFO] Indexed docs: {docs_added}")

    bm25: BM25Okapi | None = None
    if cfg.mode in {"bm25", "hybrid", "hybrid_rerank"}:
        bm25 = BM25Okapi(bm25_tokenized_corpus)

    reranker: CrossEncoder | None = None
    if cfg.mode == "hybrid_rerank":
        reranker = CrossEncoder(cfg.reranker_model)

    per_query_scores: list[dict[str, float]] = []
    evaluated_queries = 0

    for query in tqdm(dataset.queries_iter(), desc="Evaluating queries"):
        if cfg.max_queries is not None and evaluated_queries >= cfg.max_queries:
            break

        query_id = get_query_id(query)
        if query_id not in qrels:
            continue

        query_text = get_query_text(query)

        dense_ids: list[str] = []
        if cfg.mode in {"dense", "hybrid", "hybrid_rerank"}:
            embedding = model.encode([query_text], show_progress_bar=False).tolist()[0]
            dense_results = collection.query(
                query_embeddings=[embedding],
                n_results=cfg.dense_candidate_k if cfg.mode == "hybrid" else cfg.top_k,
            )
            dense_ids = [str(doc_id) for doc_id in (dense_results.get("ids") or [[]])[0]]

        bm25_ids: list[str] = []
        if cfg.mode in {"bm25", "hybrid"} and bm25 is not None:
            query_tokens = tokenize(query_text)
            bm25_scores = bm25.get_scores(query_tokens)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[: cfg.bm25_candidate_k if cfg.mode == "hybrid" else cfg.top_k]
            bm25_ids = [bm25_doc_ids[i] for i in top_indices]

        if cfg.mode == "dense":
            retrieved_ids = dense_ids[: cfg.top_k]
        elif cfg.mode == "bm25":
            retrieved_ids = bm25_ids[: cfg.top_k]
        else:
            fused = rrf_fuse([dense_ids, bm25_ids], cfg.rrf_k)
            if cfg.mode == "hybrid_rerank" and reranker is not None:
                fused = rerank_ids(
                    query_text,
                    fused,
                    docs_by_id,
                    cfg.rerank_candidate_k,
                    reranker,
                )
            retrieved_ids = fused[: cfg.top_k]

        per_query_scores.append(evaluate_ranking(retrieved_ids, qrels[query_id], cfg.top_k))
        evaluated_queries += 1

    metrics = {
        "dataset": cfg.dataset,
        "embedding_model": cfg.embedding_model,
        "mode": cfg.mode,
        "reranker_model": cfg.reranker_model if cfg.mode == "hybrid_rerank" else None,
        "top_k": cfg.top_k,
        "indexed_docs": docs_added,
        "evaluated_queries": evaluated_queries,
        "recall@5": round(mean(item["recall@5"] for item in per_query_scores), 6),
        "recall@10": round(mean(item["recall@10"] for item in per_query_scores), 6),
        "mrr@10": round(mean(item["mrr@10"] for item in per_query_scores), 6),
        "ndcg@10": round(mean(item["ndcg@10"] for item in per_query_scores), 6),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Metrics saved to: {output_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
