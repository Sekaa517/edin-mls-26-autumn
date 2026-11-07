"""Evaluate BM25+dense RRF retrieval with stage latency and memory details."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.hybrid_retriever import HybridRetriever
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.transcript_cache import append_transcript_cache, load_transcript_cache


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def summarize(values: list[float]) -> dict:
    return {
        "mean": float(statistics.mean(values)) if values else 0.0,
        "median": float(statistics.median(values)) if values else 0.0,
        "p95": percentile(values, 0.95),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], default="practice")
    parser.add_argument("--transcript-cache", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.candidate_depth < 1 or args.rrf_k < 0 or args.final_k < 1:
        parser.error("candidate depth and final k must be positive; rrf k must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.final_k > args.candidate_depth * 2:
        parser.error("--final-k cannot exceed twice --candidate-depth")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    qrels = data.load_qrels(args.split)
    cache = load_transcript_cache(args.transcript_cache)

    asr = ASRAdapter()
    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    retriever = HybridRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
        data_dir / "embeddings/bm25_index.json",
        candidate_depth=args.candidate_depth,
        rrf_k=args.rrf_k,
        final_k=args.final_k,
    )
    warmup_vector = np.asarray(retriever.dense.embeddings[0], dtype=np.float32)
    for _ in range(args.warmup):
        retriever.retrieve(warmup_vector, "warmup tax fees", k=args.final_k)

    rows = []
    for i, item in enumerate(items, start=1):
        query_id = item["query_id"]
        audio = data_dir / item["audio_file"]
        if query_id in cache:
            transcript = cache[query_id]
        else:
            transcript = asr.transcribe(audio)
            if args.transcript_cache:
                append_transcript_cache(args.transcript_cache, query_id, transcript)

        query_vector = encoder.encode(transcript)
        retrieval = retriever.retrieve(query_vector, transcript, k=args.final_k)
        ranked = retrieval["record_ids"]
        related = qrels[query_id]
        candidate_ids = retrieval["fused_record_ids"]
        row = {
            "query_id": query_id,
            f"Recall@{args.final_k}": recall_at_k(ranked, related, args.final_k),
            f"MRR@{args.final_k}": mrr_at_k(ranked, related, args.final_k),
            f"NDCG@{args.final_k}": ndcg_at_k(ranked, related, args.final_k),
            "CandidateRecall": (
                len(set(candidate_ids).intersection(related)) / len(related)
                if related else 0.0
            ),
            "CandidateCount": len(candidate_ids),
            "timings": retrieval["timings"],
            "memory": retrieval["memory"],
            "retrieved_record_ids": ranked,
        }
        rows.append(row)
        print(
            f"[{i:>3}/{len(items)}] {query_id} "
            f"R@{args.final_k}={row[f'Recall@{args.final_k}']:.3f} "
            f"total={row['timings']['retrieval_total_ms']:.2f} ms"
        )

    stage_names = ["bm25_ms", "dense_ms", "rrf_ms", "topk_ms", "retrieval_total_ms"]
    metric_names = [
        f"Recall@{args.final_k}",
        f"MRR@{args.final_k}",
        f"NDCG@{args.final_k}",
        "CandidateRecall",
    ]
    summary = {
        "script": "evaluate_rag_hybrid.py",
        "split": args.split,
        "queries": len(rows),
        "candidate_depth": args.candidate_depth,
        "rrf_k": args.rrf_k,
        "final_k": args.final_k,
        "warmup": args.warmup,
        "is_subset": args.limit is not None,
        "metrics": {
            name: float(statistics.mean([row[name] for row in rows]))
            for name in metric_names
        },
        "latency_ms": {
            name: summarize([row["timings"][name] for row in rows])
            for name in stage_names
        },
        "memory": rows[0]["memory"] if rows else {},
    }
    print("\nAggregate:")
    for name, value in summary["metrics"].items():
        print(f"  {name:<18} {value:.6f}")
    for name, values in summary["latency_ms"].items():
        print(
            f"  {name:<18} mean={values['mean']:.3f} "
            f"median={values['median']:.3f} p95={values['p95']:.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(
                {"summary": summary, "per_query": rows},
                stream,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
