"""Run one audio query with BM25+dense hybrid retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--audio", help="Path to a 16 kHz mono PCM16 WAV")
    selection.add_argument("--query-id", help="Select a query from the course manifest")
    parser.add_argument("--split", choices=["practice", "dev"], default="practice")
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.candidate_depth < 1 or args.rrf_k < 0 or args.k < 1:
        parser.error("candidate depth and k must be positive; rrf k non-negative")
    if args.k > args.candidate_depth * 2:
        parser.error("--k cannot exceed twice --candidate-depth")

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    query_id = None
    if args.audio:
        audio = Path(args.audio)
    else:
        items = data.audio_items(args.split)
        if args.query_id:
            items = [item for item in items if item["query_id"] == args.query_id]
        if not items:
            parser.error("No matching query in the selected split")
        query_id = items[0]["query_id"]
        audio = data_dir / items[0]["audio_file"]

    asr = ASRAdapter()
    asr.prepare()
    pipeline = RAGPipelineHybrid(
        asr=asr,
        encoder=LocalEncoder(ROOT / "models/enc_a", device=args.device),
        retriever=HybridRetriever(
            data_dir / "embeddings/corpus_enc_a.npy",
            data_dir / "embeddings/corpus_enc_a_ids.json",
            data_dir / "embeddings/bm25_index.json",
            candidate_depth=args.candidate_depth,
            rrf_k=args.rrf_k,
            final_k=args.k,
        ),
        data=data,
        context_builder=SimpleContextBuilder(),
        generator=ELMGenerator(),
    )

    result = pipeline.answer_audio(audio, k=args.k)
    result["query_id"] = query_id
    result["audio_file"] = str(audio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
