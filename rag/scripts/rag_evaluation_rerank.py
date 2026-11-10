"""Evaluate dense Top-k plus local BGE reranking and save retrieval results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.rerank.reranker import (
    DEFAULT_MODEL_DIR,
    LocalBGEReranker,
)
from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.retriever_rerank import RerankRetriever
from src.transcript_cache import append_transcript_cache, load_transcript_cache


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], default="practice")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--k-final", type=int, default=5)
    parser.add_argument("--transcript-cache", type=Path, default=None)
    parser.add_argument("--device", default=None, help="enc_a device")
    parser.add_argument("--rerank-device", default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 1 or args.k_final < 1:
        parser.error("--k and --k-final must be positive")
    if args.k_final > args.k:
        parser.error("--k-final cannot exceed --k")
    if args.rerank_batch_size < 1 or args.rerank_max_length < 1:
        parser.error("rerank batch size and max length must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "per_query.jsonl"
    summary_path = args.output_dir / "summary.json"
    if per_query_path.exists() or summary_path.exists():
        parser.error(
            f"{args.output_dir} already contains retrieval results; "
            "choose a new output directory"
        )

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    qrels = data.load_qrels(args.split)
    transcript_cache = load_transcript_cache(args.transcript_cache)

    asr = ASRAdapter()
    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.device)
    reranker = LocalBGEReranker(
        args.model_dir,
        device=args.rerank_device,
        batch_size=args.rerank_batch_size,
        max_length=args.rerank_max_length,
    )
    retriever = RerankRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
        data,
        reranker,
        k=args.k,
        k_final=args.k_final,
    )

    rows = []
    recall_key = f"R@{args.k_final}"
    with per_query_path.open("x", encoding="utf-8") as stream:
        for index, item in enumerate(items, start=1):
            query_id = item["query_id"]
            audio_path = data_dir / item["audio_file"]
            if query_id in transcript_cache:
                transcript = transcript_cache[query_id]
            else:
                transcript = asr.transcribe(audio_path)
                if args.transcript_cache:
                    append_transcript_cache(
                        args.transcript_cache,
                        query_id,
                        transcript,
                    )

            query_vector = encoder.encode(transcript)
            retrieval = retriever.retrieve(
                query_vector,
                transcript,
                k=args.k,
                k_final=args.k_final,
            )
            relevant = qrels[query_id]
            row = {
                "query_id": query_id,
                recall_key: recall_at_k(
                    retrieval["record_ids"],
                    relevant,
                    args.k_final,
                ),
                f"MRR@{args.k_final}": mrr_at_k(
                    retrieval["record_ids"],
                    relevant,
                    args.k_final,
                ),
                f"NDCG@{args.k_final}": ndcg_at_k(
                    retrieval["record_ids"],
                    relevant,
                    args.k_final,
                ),
                "dense_ms": retrieval["dense_ms"],
                "dense_wall_ms": retrieval["dense_wall_ms"],
                "rerank_ms": retrieval["rerank_ms"],
                "retrieval_total_ms": retrieval["retrieval_total_ms"],
                "record_ids": retrieval["record_ids"],
                "scores": retrieval["scores"],
                "dense_scores": retrieval["dense_scores"],
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"[{index:>3}/{len(items)}] {query_id} "
                f"{recall_key}={row[recall_key]:.3f} "
                f"retrieval={row['retrieval_total_ms']:.3f} ms"
            )

    metric_names = [
        recall_key,
        f"MRR@{args.k_final}",
        f"NDCG@{args.k_final}",
    ]
    metric_values = {
        name: float(statistics.mean([row[name] for row in rows])) if rows else 0.0
        for name in metric_names
    }
    summary = {
        "script": "rag_evaluation_rerank.py",
        "split": args.split,
        "queries": len(rows),
        "is_subset": args.limit is not None,
        "k": args.k,
        "k_final": args.k_final,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "rerank_batch_size": args.rerank_batch_size,
        "rerank_max_length": args.rerank_max_length,
        **metric_values,
        "mean_dense_ms": float(
            statistics.mean([row["dense_ms"] for row in rows])
        ) if rows else 0.0,
        "mean_dense_wall_ms": float(
            statistics.mean([row["dense_wall_ms"] for row in rows])
        ) if rows else 0.0,
        "mean_rerank_ms": float(
            statistics.mean([row["rerank_ms"] for row in rows])
        ) if rows else 0.0,
        "mean_retrieval_total_ms": float(
            statistics.mean([row["retrieval_total_ms"] for row in rows])
        ) if rows else 0.0,
        "latency": {
            name: summarize([row[name] for row in rows])
            for name in (
                "dense_ms",
                "dense_wall_ms",
                "rerank_ms",
                "retrieval_total_ms",
            )
        },
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print("\nAggregate:")
    for name in metric_names:
        print(f"  {name:<14} {statistics.mean([row[name] for row in rows]):.6f}")
    print(f"  mean_dense_ms  {summary['mean_dense_ms']:.3f}")
    print(f"  mean_dense_wall_ms {summary['mean_dense_wall_ms']:.3f}")
    print(f"  mean_rerank_ms {summary['mean_rerank_ms']:.3f}")
    print(f"  mean_total_ms  {summary['mean_retrieval_total_ms']:.3f}")
    print(f"\nSaved: {args.output_dir}")


if __name__ == "__main__":
    main()
