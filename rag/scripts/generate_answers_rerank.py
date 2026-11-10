"""Generate answer candidates with dense retrieval plus local BGE reranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.rerank.reranker import (
    DEFAULT_MODEL_DIR,
    LocalBGEReranker,
)
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag_rerank import RAGPipelineRerank
from src.retriever_rerank import RerankRetriever
from src.transcript_cache import append_transcript_cache, load_transcript_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["practice", "dev"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--k-final", type=int, default=5)
    parser.add_argument("--transcript-cache", type=Path, default=None)
    parser.add_argument("--device", default=None, help="enc_a device")
    parser.add_argument("--rerank-device", default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=512)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.k < 1 or args.k_final < 1:
        parser.error("--k and --k-final must be positive")
    if args.k_final > args.k:
        parser.error("--k-final cannot exceed --k")
    if args.rerank_batch_size < 1 or args.rerank_max_length < 1:
        parser.error("rerank batch size and max length must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new filename to preserve prior results")

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    transcript_cache = load_transcript_cache(args.transcript_cache)

    asr = ASRAdapter()
    reranker = LocalBGEReranker(
        args.model_dir,
        device=args.rerank_device,
        batch_size=args.rerank_batch_size,
        max_length=args.rerank_max_length,
    )
    pipeline = RAGPipelineRerank(
        asr,
        LocalEncoder(ROOT / "models/enc_a", device=args.device),
        RerankRetriever(
            data_dir / "embeddings/corpus_enc_a.npy",
            data_dir / "embeddings/corpus_enc_a_ids.json",
            data,
            reranker,
            k=args.k,
            k_final=args.k_final,
        ),
        data,
        SimpleContextBuilder(),
        ELMGenerator(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for item in items:
            query_id = item["query_id"]
            audio_path = data_dir / item["audio_file"]
            if query_id in transcript_cache:
                transcript = transcript_cache[query_id]
                asr_ms = 0.0
            else:
                asr_start = time.perf_counter()
                transcript = asr.transcribe(audio_path)
                asr_ms = (time.perf_counter() - asr_start) * 1000.0
                if args.transcript_cache:
                    append_transcript_cache(
                        args.transcript_cache,
                        query_id,
                        transcript,
                    )

            result = pipeline.answer_transcript(
                transcript,
                k=args.k,
                k_final=args.k_final,
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
