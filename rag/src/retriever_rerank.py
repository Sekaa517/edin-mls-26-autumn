"""Dense Top-k retrieval followed by local cross-encoder reranking."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .retriever import ExactDenseRetriever


class RerankRetriever:
    """Retrieve dense candidates, rerank their record text, and return Top-k_final."""

    def __init__(
        self,
        embeddings_path: str | Path,
        ids_path: str | Path,
        data,
        reranker,
        *,
        k: int = 40,
        k_final: int = 5,
        dense_retriever=None,
    ):
        if k < 1 or k_final < 1:
            raise ValueError("k and k_final must be positive")
        if k_final > k:
            raise ValueError("k_final cannot exceed k")
        self.data = data
        self.reranker = reranker
        self.k = int(k)
        self.k_final = int(k_final)
        self.dense = dense_retriever or ExactDenseRetriever(
            embeddings_path,
            ids_path,
        )

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        k: int | None = None,
        k_final: int | None = None,
    ) -> dict:
        """Return dense/rerank timings as top-level fields.

        Timings start after the dense embedding matrix and reranker model have
        already been loaded. The current course corpus has one text record per
        chunk ID, so each dense record is treated as one reranking chunk.
        """
        retrieval_k = self.k if k is None else int(k)
        final_k = self.k_final if k_final is None else int(k_final)
        if retrieval_k < 1 or final_k < 1:
            raise ValueError("k and k_final must be positive")
        if final_k > retrieval_k:
            raise ValueError("k_final cannot exceed k")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")

        total_start = time.perf_counter()
        dense_start = time.perf_counter()
        dense = self.dense.retrieve(query_vector, retrieval_k)
        dense_ms = (time.perf_counter() - dense_start) * 1000.0

        candidate_ids = dense["record_ids"]
        candidate_scores = dense["scores"]
        candidate_texts = [self.data.record_text(record_id) for record_id in candidate_ids]

        rerank_start = time.perf_counter()
        rerank_scores = self.reranker.score_pairs(query_text, candidate_texts)
        ranked = sorted(
            zip(candidate_ids, candidate_scores, rerank_scores),
            key=lambda item: (-item[2], item[0]),
        )
        selected = ranked[:min(final_k, len(ranked))]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        retrieval_total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "record_ids": [record_id for record_id, _, _ in selected],
            "scores": [float(score) for _, _, score in selected],
            "dense_scores": [float(score) for _, score, _ in selected],
            "candidate_record_ids": candidate_ids,
            "candidate_dense_scores": [float(score) for score in candidate_scores],
            "candidate_rerank_scores": [float(score) for score in rerank_scores],
            "dense_ms": float(dense_ms),
            "rerank_ms": float(rerank_ms),
            "retrieval_total_ms": float(retrieval_total_ms),
            "k": retrieval_k,
            "k_final": final_k,
            "reranker_model": getattr(
                self.reranker,
                "model_name",
                type(self.reranker).__name__,
            ),
        }
