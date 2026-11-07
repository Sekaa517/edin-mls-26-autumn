"""Persistent BM25 plus exact dense retrieval fused with reciprocal rank fusion."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import numpy as np

from .retriever import ExactDenseRetriever


TOKEN_RE = re.compile(r"[a-z0-9_]+")
TOKENIZER_VERSION = "ascii_lower_word_stopwords_v2"
BM25_INDEX_VERSION = 2
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such t than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself
yourselves s re ve ll d m
""".split())


def tokenize(text: str) -> list[str]:
    """Tokenize the English course corpus for lexical BM25 matching."""
    return [
        token for token in TOKEN_RE.findall(str(text).casefold())
        if token not in STOPWORDS
    ]


def estimate_bm25_memory_bytes(
    record_ids: list[str],
    doc_lengths: np.ndarray,
    postings: dict[str, list[list[int]]],
) -> int:
    """Approximate the persistent inverted-index object footprint in Python."""
    size = (
        sys.getsizeof(record_ids)
        + sum(sys.getsizeof(record_id) for record_id in record_ids)
        + int(doc_lengths.nbytes)
        + sys.getsizeof(postings)
    )
    entry_bytes = sys.getsizeof([0, 0]) + 2 * sys.getsizeof(0)
    for term, entries in postings.items():
        size += sys.getsizeof(term) + sys.getsizeof(entries)
        size += len(entries) * entry_bytes
    return int(size)


def process_rss_bytes() -> int | None:
    """Return current process RSS without making psutil a hard dependency."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        status = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def gpu_allocated_bytes() -> int | None:
    """Read PyTorch's allocator only when torch is already loaded in this process."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


def approx_object_bytes(value, seen: set[int] | None = None) -> int:
    """Approximate Python object footprint for small candidate/result structures."""
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(approx_object_bytes(k, seen) + approx_object_bytes(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(approx_object_bytes(item, seen) for item in value)
    return int(size)


class BM25Index:
    """A compact in-memory BM25 inverted index with a JSON persistence format."""

    def __init__(
        self,
        record_ids: list[str],
        doc_lengths: list[int],
        postings: dict[str, list[list[int]]],
        *,
        k1: float,
        b: float,
        index_file_bytes: int | None = None,
        estimated_memory_bytes: int | None = None,
    ):
        if len(record_ids) != len(doc_lengths):
            raise ValueError("record_ids and doc_lengths must have equal length")
        if not 0.0 <= b <= 1.0:
            raise ValueError("BM25 b must be between 0 and 1")
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        self.record_ids = [str(record_id) for record_id in record_ids]
        self.doc_lengths = np.asarray(doc_lengths, dtype=np.int32)
        self.postings = postings
        self.k1 = float(k1)
        self.b = float(b)
        self.index_file_bytes = index_file_bytes
        self.avgdl = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.estimated_memory_bytes = estimated_memory_bytes
        self.document_frequencies = {
            term: len(entries) for term, entries in self.postings.items()
        }
        if self.estimated_memory_bytes is None:
            self.estimated_memory_bytes = estimate_bm25_memory_bytes(
                self.record_ids,
                self.doc_lengths,
                self.postings,
            )

    @classmethod
    def from_corpus(
        cls,
        rows: Iterable[dict],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        record_ids: list[str] = []
        doc_lengths: list[int] = []
        postings: dict[str, list[list[int]]] = defaultdict(list)
        seen: set[str] = set()

        for doc_index, row in enumerate(rows):
            record_id = str(row["record_id"])
            if record_id in seen:
                raise ValueError(f"duplicate record_id: {record_id}")
            seen.add(record_id)
            tokens = tokenize(row["text"])
            record_ids.append(record_id)
            doc_lengths.append(len(tokens))
            for term, term_frequency in Counter(tokens).items():
                postings[term].append([doc_index, int(term_frequency)])

        if not record_ids:
            raise ValueError("cannot build BM25 index from an empty corpus")
        return cls(
            record_ids,
            doc_lengths,
            dict(postings),
            k1=k1,
            b=b,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("version") != BM25_INDEX_VERSION:
            raise ValueError(f"unsupported BM25 index version: {payload.get('version')}")
        if payload.get("tokenizer") != TOKENIZER_VERSION:
            raise ValueError(
                f"BM25 index tokenizer {payload.get('tokenizer')} does not match "
                f"{TOKENIZER_VERSION}"
            )
        index = cls(
            payload["record_ids"],
            payload["doc_lengths"],
            payload["postings"],
            k1=payload["k1"],
            b=payload["b"],
            index_file_bytes=path.stat().st_size,
            estimated_memory_bytes=payload.get("estimated_memory_bytes"),
        )
        if payload.get("doc_count") != len(index.record_ids):
            raise ValueError("BM25 index doc_count does not match record_ids")
        return index

    def save(self, path: str | Path) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": BM25_INDEX_VERSION,
            "tokenizer": TOKENIZER_VERSION,
            "k1": self.k1,
            "b": self.b,
            "doc_count": len(self.record_ids),
            "vocabulary_size": len(self.postings),
            "posting_count": int(sum(len(v) for v in self.postings.values())),
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "average_document_length": self.avgdl,
            "record_ids": self.record_ids,
            "doc_lengths": self.doc_lengths.tolist(),
            "postings": self.postings,
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
        self.index_file_bytes = path.stat().st_size
        return {
            "path": str(path),
            "file_bytes": self.index_file_bytes,
            "doc_count": len(self.record_ids),
            "vocabulary_size": len(self.postings),
            "posting_count": payload["posting_count"],
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "average_document_length": self.avgdl,
        }

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def posting_count(self) -> int:
        return int(sum(len(entries) for entries in self.postings.values()))

    def search(self, query: str, k: int = 20) -> list[dict]:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        doc_count = len(self.record_ids)
        for term, query_frequency in query_terms.items():
            entries = self.postings.get(term)
            if not entries:
                continue
            document_frequency = len(entries)
            inverse_document_frequency = math.log(
                1.0
                + (doc_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for doc_index, term_frequency in entries:
                doc_length = int(self.doc_lengths[doc_index])
                normalization = self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(self.avgdl, 1e-12)
                )
                scores[doc_index] += (
                    query_frequency
                    * inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / (term_frequency + normalization)
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.record_ids[item[0]]),
        )[:min(k, len(scores))]
        return [
            {
                "record_id": self.record_ids[doc_index],
                "score": float(score),
                "rank": rank,
            }
            for rank, (doc_index, score) in enumerate(ranked, start=1)
        ]


class HybridRetriever:
    """Cosine dense retrieval plus BM25, fused by reciprocal rank fusion."""

    def __init__(
        self,
        embeddings_path: str | Path,
        ids_path: str | Path,
        bm25_index_path: str | Path,
        *,
        candidate_depth: int = 20,
        rrf_k: int = 60,
        final_k: int = 5,
    ):
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if final_k < 1:
            raise ValueError("final_k must be positive")
        self.dense = ExactDenseRetriever(embeddings_path, ids_path)
        self.bm25 = BM25Index.load(bm25_index_path)
        self.candidate_depth = int(candidate_depth)
        self.rrf_k = int(rrf_k)
        self.final_k = int(final_k)
        self.bm25_index_path = Path(bm25_index_path)

        if set(self.bm25.record_ids) != set(self.dense.record_ids):
            raise ValueError("BM25 and dense record ID sets do not match")

    def _rrf_merge(self, bm25_results: list[dict], dense_results: list[dict]) -> dict[str, dict]:
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, dict] = {}
        for ranking_name, results in (("bm25", bm25_results), ("dense", dense_results)):
            for rank, result in enumerate(results, start=1):
                record_id = result["record_id"]
                scores[record_id] += 1.0 / (self.rrf_k + rank)
                source = sources.setdefault(record_id, {})
                source[ranking_name] = {
                    "rank": rank,
                    "score": float(result["score"]),
                }
        return {
            record_id: {
                "rrf_score": float(score),
                **sources[record_id],
            }
            for record_id, score in scores.items()
        }

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        k: int | None = None,
    ) -> dict:
        """Return Top-k IDs and detailed stage timings and memory observations.

        Timings start after the index and embedding matrix have been loaded.
        `process_rss_*` values include Python allocator behaviour, while the
        logical memory fields make the BM25 file and dense matrix sizes explicit.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string for BM25")
        final_k = self.final_k if k is None else int(k)
        if final_k < 1:
            raise ValueError("k must be positive")

        total_start = time.perf_counter()
        rss_start = process_rss_bytes()
        gpu_start = gpu_allocated_bytes()

        bm25_start = time.perf_counter()
        bm25_results = self.bm25.search(query_text, self.candidate_depth)
        bm25_ms = (time.perf_counter() - bm25_start) * 1000.0
        rss_after_bm25 = process_rss_bytes()

        dense_start = time.perf_counter()
        dense_result = self.dense.retrieve(query_vector, self.candidate_depth)
        dense_ms = (time.perf_counter() - dense_start) * 1000.0
        rss_after_dense = process_rss_bytes()
        dense_results = [
            {
                "record_id": record_id,
                "score": float(score),
                "rank": rank,
            }
            for rank, (record_id, score) in enumerate(
                zip(dense_result["record_ids"], dense_result["scores"]),
                start=1,
            )
        ]

        rrf_start = time.perf_counter()
        fused = self._rrf_merge(bm25_results, dense_results)
        rrf_ms = (time.perf_counter() - rrf_start) * 1000.0
        rss_after_rrf = process_rss_bytes()

        topk_start = time.perf_counter()
        ranked = sorted(fused.items(), key=lambda item: (-item[1]["rrf_score"], item[0]))
        selected = ranked[:min(final_k, len(ranked))]
        topk_ms = (time.perf_counter() - topk_start) * 1000.0
        rss_after_topk = process_rss_bytes()

        gpu_end = gpu_allocated_bytes()
        total_ms = (time.perf_counter() - total_start) * 1000.0
        query_array = np.asarray(query_vector, dtype=np.float32)
        gpu_delta = (
            gpu_end - gpu_start
            if gpu_start is not None and gpu_end is not None
            else None
        )
        rss_delta = (
            rss_after_topk - rss_start
            if rss_start is not None and rss_after_topk is not None
            else None
        )

        return {
            "record_ids": [record_id for record_id, _ in selected],
            "scores": [details["rrf_score"] for _, details in selected],
            "fused_record_ids": [record_id for record_id, _ in ranked],
            "rrf_fusion": [
                {"record_id": record_id, **details}
                for record_id, details in ranked
            ],
            "candidates": {
                "bm25": bm25_results,
                "dense": dense_results,
                "candidate_depth": self.candidate_depth,
                "union_count": len(ranked),
            },
            "timings": {
                "bm25_ms": bm25_ms,
                "dense_ms": dense_ms,
                "dense_search_ms": float(dense_result["latency_ms"]),
                "rrf_ms": rrf_ms,
                "topk_ms": topk_ms,
                "retrieval_total_ms": total_ms,
            },
            "latency_ms": total_ms,
            "memory": {
                "bm25_index_file_bytes": self.bm25.index_file_bytes,
                "bm25_index_estimated_memory_bytes": self.bm25.estimated_memory_bytes,
                "dense_corpus_bytes": int(self.dense.embeddings.nbytes),
                "query_vector_bytes": int(query_array.nbytes),
                "bm25_candidates_estimated_bytes": approx_object_bytes(bm25_results),
                "dense_candidates_estimated_bytes": approx_object_bytes(dense_results),
                "rrf_fusion_estimated_bytes": approx_object_bytes(fused),
                "final_results_estimated_bytes": approx_object_bytes(selected),
                "process_rss_start_bytes": rss_start,
                "process_rss_after_bm25_bytes": rss_after_bm25,
                "process_rss_after_dense_bytes": rss_after_dense,
                "process_rss_after_rrf_bytes": rss_after_rrf,
                "process_rss_after_topk_bytes": rss_after_topk,
                "process_rss_delta_bytes": rss_delta,
                "gpu_allocated_start_bytes": gpu_start,
                "gpu_allocated_end_bytes": gpu_end,
                "gpu_allocated_delta_bytes": gpu_delta,
                "bm25_doc_count": len(self.bm25.record_ids),
                "bm25_vocabulary_size": self.bm25.vocabulary_size,
                "bm25_posting_count": self.bm25.posting_count,
            },
            "configuration": {
                "bm25_k1": self.bm25.k1,
                "bm25_b": self.bm25.b,
                "candidate_depth": self.candidate_depth,
                "rrf_k": self.rrf_k,
                "final_k": final_k,
            },
        }

    def describe(self) -> str:
        return (
            f"hybrid BM25+{self.dense.embeddings.shape[0]} dense documents; "
            f"candidate_depth={self.candidate_depth}, "
            f"rrf_k={self.rrf_k}, final_k={self.final_k}"
        )
