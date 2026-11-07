"""Generate answer candidates with BM25+dense hybrid retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.hybrid_retriever import HybridRetriever
from src.rag_hybrid import RAGPipelineHybrid
from src.transcript_cache import append_transcript_cache, load_transcript_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcript-cache", type=Path, default=None)
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.candidate_depth < 1 or args.rrf_k < 0 or args.k < 1:
        parser.error("candidate depth and k must be positive; rrf k non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.k > args.candidate_depth * 2:
        parser.error("--k cannot exceed twice --candidate-depth")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.output.exists():
        parser.error("Output exists; choose a new filename to preserve prior results")

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    cache = load_transcript_cache(args.transcript_cache)
    asr = ASRAdapter()

    pipeline = RAGPipelineHybrid(
        asr,
        LocalEncoder(ROOT / "models/enc_a", device=args.device),
        HybridRetriever(
            data_dir / "embeddings/corpus_enc_a.npy",
            data_dir / "embeddings/corpus_enc_a_ids.json",
            data_dir / "embeddings/bm25_index.json",
            candidate_depth=args.candidate_depth,
            rrf_k=args.rrf_k,
            final_k=args.k,
        ),
        data,
        SimpleContextBuilder(),
        ELMGenerator(),
    )
    warmup_vector = np.asarray(pipeline.retriever.dense.embeddings[0], dtype=np.float32)
    for _ in range(args.warmup):
        pipeline.retriever.retrieve(warmup_vector, "warmup tax fees", k=args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for item in items:
            query_id = item["query_id"]
            audio = data_dir / item["audio_file"]
            if query_id in cache:
                transcript = cache[query_id]
                asr_ms = 0.0
            else:
                asr_start = time.perf_counter()
                transcript = asr.transcribe(audio)
                asr_ms = (time.perf_counter() - asr_start) * 1000.0
                if args.transcript_cache:
                    append_transcript_cache(args.transcript_cache, query_id, transcript)

            result = pipeline.answer_transcript(
                transcript,
                k=args.k,
                asr_ms=asr_ms,
            )
            row = {
                "query_id": query_id,
                "split": args.split,
                "audio_file": item["audio_file"],
                "transcript": result["transcript"],
                "answer": result["answer"],
                "retrieved_record_ids": result["retrieval"]["record_ids"],
                "retrieval": result["retrieval"],
                "context_count": len(result["contexts"]),
                "context_chars": sum(len(text) for text in result["contexts"]),
                "timing": result["timing"],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(query_id)


if __name__ == "__main__":
    main()
