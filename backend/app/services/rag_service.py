from __future__ import annotations
import re
from typing import Any
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from app.config import settings


class RagService:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        self._model = SentenceTransformer(settings.embedding_model)
        self._reranker: CrossEncoder | None = None
        self._sparse_cache: dict[str, dict[str, Any]] = {}

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.rag_reranker_model)
        return self._reranker

    def _apply_rerank(self, query: str, ranked_ids: list[str], docs_map: dict[str, str], top_n: int) -> list[str]:
        if not ranked_ids:
            return []
        rerank_n = min(max(top_n, 1), len(ranked_ids))
        candidates = ranked_ids[:rerank_n]
        pairs = [[query, docs_map.get(doc_id, "")] for doc_id in candidates]
        reranker = self._get_reranker()
        scores = reranker.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        reranked = [doc_id for doc_id, _score in scored]
        if rerank_n < len(ranked_ids):
            reranked.extend(ranked_ids[rerank_n:])
        return reranked

    def _clear_sparse_cache(self) -> None:
        self._sparse_cache.clear()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _hit_stats_from_metas(metadatas: list[dict] | None, tree_id: str) -> dict[str, int]:
        metadatas = metadatas or []
        global_hits = 0
        tree_hits = 0
        other_hits = 0
        source_ids: set[str] = set()
        for meta in metadatas:
            if not meta:
                other_hits += 1
                continue
            meta_tree_id = meta.get("tree_id")
            if meta_tree_id == "global":
                global_hits += 1
            elif meta_tree_id == tree_id:
                tree_hits += 1
            else:
                other_hits += 1
            source_id = meta.get("source_id")
            if source_id:
                source_ids.add(str(source_id))
        return {
            "total_hits": len(metadatas),
            "global_hits": global_hits,
            "tree_hits": tree_hits,
            "other_hits": other_hits,
            "unique_sources": len(source_ids),
        }

    def _ensure_sparse_index(self, tree_id: str) -> dict[str, Any]:
        total_count = self._collection.count()
        cached = self._sparse_cache.get(tree_id)
        if cached and cached.get("total_count") == total_count:
            return cached

        records = self._collection.get(
            where={"tree_id": {"$in": [tree_id, "global"]}},
            include=["documents", "metadatas"],
        )

        ids = [str(item) for item in (records.get("ids") or [])]
        docs = [str(item or "") for item in (records.get("documents") or [])]
        metadatas = list(records.get("metadatas") or [])
        tokenized = [self._tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized) if tokenized else None

        entry = {
            "total_count": total_count,
            "ids": ids,
            "docs": docs,
            "metadatas": metadatas,
            "bm25": bm25,
        }
        self._sparse_cache[tree_id] = entry
        return entry

    @staticmethod
    def _rrf_fuse(rank_lists: list[list[str]], rrf_k: int) -> list[str]:
        scores: dict[str, float] = {}
        for rank_list in rank_lists:
            for rank, doc_id in enumerate(rank_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def add_chunks(
        self,
        chunks: list[str],
        source_id: str,
        source_name: str,
        tree_id: str = "global",
    ) -> None:
        if not chunks:
            return
        embeddings = self._model.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source_id": source_id, "source_name": source_name, "tree_id": tree_id}
            for _ in chunks
        ]
        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        self._clear_sparse_cache()

    def retrieve(self, query: str, tree_id: str, top_k: int | None = None) -> list[str]:
        docs, _stats = self.retrieve_with_stats(query, tree_id, top_k)
        return docs

    def retrieve_with_stats(self, query: str, tree_id: str, top_k: int | None = None) -> tuple[list[str], dict]:
        k = top_k or settings.rag_top_k
        mode = (settings.rag_retrieval_mode or "dense").strip().lower()
        if mode not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            mode = "dense"

        count = self._collection.count()
        if count == 0:
            return [], {
                "retrieval_mode": mode,
                "total_hits": 0,
                "global_hits": 0,
                "tree_hits": 0,
                "other_hits": 0,
                "unique_sources": 0,
                "dense_candidates": 0,
                "bm25_candidates": 0,
                "hybrid_overlap": 0,
                "rerank_applied": False,
                "rerank_candidates": 0,
            }
        k = min(k, count)

        dense_candidate_k = min(max(settings.rag_dense_candidate_k, k), count)
        bm25_candidate_k = min(max(settings.rag_bm25_candidate_k, k), count)

        dense_ranked_ids: list[str] = []
        dense_docs_map: dict[str, str] = {}
        dense_meta_map: dict[str, dict] = {}
        if mode in {"dense", "hybrid", "hybrid_rerank"}:
            embedding = self._model.encode([query], show_progress_bar=False).tolist()[0]
            dense_results = self._collection.query(
                query_embeddings=[embedding],
                n_results=dense_candidate_k,
                where={"tree_id": {"$in": [tree_id, "global"]}},
                include=["documents", "metadatas"],
            )
            dense_docs = dense_results.get("documents", [[]])[0] if dense_results.get("documents") else []
            dense_metas = dense_results.get("metadatas", [[]])[0] if dense_results.get("metadatas") else []
            dense_ranked_ids = [str(item) for item in (dense_results.get("ids", [[]])[0] if dense_results.get("ids") else [])]
            for idx, doc_id in enumerate(dense_ranked_ids):
                dense_docs_map[doc_id] = str(dense_docs[idx]) if idx < len(dense_docs) else ""
                dense_meta_map[doc_id] = dense_metas[idx] if idx < len(dense_metas) else {}

        bm25_ranked_ids: list[str] = []
        bm25_docs_map: dict[str, str] = {}
        bm25_meta_map: dict[str, dict] = {}
        if mode in {"bm25", "hybrid", "hybrid_rerank"}:
            sparse = self._ensure_sparse_index(tree_id)
            bm25 = sparse.get("bm25")
            if bm25 is not None:
                query_tokens = self._tokenize(query)
                bm25_scores = bm25.get_scores(query_tokens)
                ranked_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: float(bm25_scores[i]),
                    reverse=True,
                )[:bm25_candidate_k]
                ids = sparse.get("ids") or []
                docs = sparse.get("docs") or []
                metas = sparse.get("metadatas") or []
                for idx in ranked_indices:
                    if idx >= len(ids):
                        continue
                    doc_id = str(ids[idx])
                    bm25_ranked_ids.append(doc_id)
                    bm25_docs_map[doc_id] = str(docs[idx]) if idx < len(docs) else ""
                    bm25_meta_map[doc_id] = metas[idx] if idx < len(metas) else {}

        final_ids: list[str]
        rerank_applied = False
        rerank_candidates = 0
        if mode == "dense":
            final_ids = dense_ranked_ids[:k]
        elif mode == "bm25":
            final_ids = bm25_ranked_ids[:k]
        else:
            fused = self._rrf_fuse([dense_ranked_ids, bm25_ranked_ids], settings.rag_rrf_k)
            if mode == "hybrid_rerank":
                docs_map_for_rerank = {**bm25_docs_map, **dense_docs_map}
                rerank_candidates = min(max(settings.rag_rerank_candidate_k, k), len(fused))
                fused = self._apply_rerank(query, fused, docs_map_for_rerank, rerank_candidates)
                rerank_applied = True
            final_ids = fused[:k]

        docs_map = {**bm25_docs_map, **dense_docs_map}
        metas_map = {**bm25_meta_map, **dense_meta_map}
        final_docs = [docs_map.get(doc_id, "") for doc_id in final_ids]
        final_metas = [metas_map.get(doc_id, {}) for doc_id in final_ids]

        hit_stats = self._hit_stats_from_metas(final_metas, tree_id)
        stats = {
            "retrieval_mode": mode,
            **hit_stats,
            "dense_candidates": len(dense_ranked_ids),
            "bm25_candidates": len(bm25_ranked_ids),
            "hybrid_overlap": len(set(dense_ranked_ids).intersection(set(bm25_ranked_ids))),
            "rerank_applied": rerank_applied,
            "rerank_candidates": rerank_candidates,
        }
        return final_docs, stats

    def list_sources(self) -> list[dict]:
        """Return unique sources with metadata."""
        result = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in result["metadatas"] or []:
            sid = meta["source_id"]
            if sid not in seen:
                seen[sid] = {
                    "source_id": sid,
                    "source_name": meta.get("source_name", sid),
                    "tree_id": meta.get("tree_id", "global"),
                }
        return list(seen.values())

    def delete_source(self, source_id: str) -> None:
        self._collection.delete(where={"source_id": source_id})
        self._clear_sparse_cache()


# Singleton initialized at startup
rag_service: RagService | None = None


def get_rag_service() -> RagService:
    global rag_service
    if rag_service is None:
        rag_service = RagService()
    return rag_service
