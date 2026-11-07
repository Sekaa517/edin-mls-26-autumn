"""Build the persistent BM25 inverted index used by the hybrid retriever."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_retriever import BM25Index, process_rss_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data/rag_corpus/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/rag_corpus/embeddings/bm25_index.json",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing BM25 index.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to replace it")

    rss_start = process_rss_bytes()
    start = time.perf_counter()
    with args.corpus.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    load_ms = (time.perf_counter() - start) * 1000.0

    build_start = time.perf_counter()
    index = BM25Index.from_corpus(rows, k1=args.k1, b=args.b)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    save_start = time.perf_counter()
    stats = index.save(args.output)
    save_ms = (time.perf_counter() - save_start) * 1000.0
    rss_end = process_rss_bytes()

    print(json.dumps({
        **stats,
        "k1": args.k1,
        "b": args.b,
        "corpus_load_ms": load_ms,
        "index_build_ms": build_ms,
        "index_save_ms": save_ms,
        "process_rss_start_bytes": rss_start,
        "process_rss_end_bytes": rss_end,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
