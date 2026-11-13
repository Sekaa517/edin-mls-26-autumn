"""Summarize latency and context-size statistics from rerank answer JSONL files."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


LATENCY_FIELDS = (
    "asr_ms",
    "embedding_ms",
    "dense_ms",
    "rerank_ms",
    "retrieval_ms",
    "generation_ms",
    "end_to_end_ms",
)
CONTEXT_FIELDS = ("context_count", "context_chars")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
    return float(ordered[index])


def describe(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)),
    }


def numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def timing_value(row: dict, field: str) -> float | None:
    timing = row.get("timing")
    if isinstance(timing, dict):
        value = numeric(timing.get(field))
        if value is not None:
            return value

    retrieval = row.get("retrieval")
    if isinstance(retrieval, dict):
        fallback = {
            "dense_ms": "dense_ms",
            "rerank_ms": "rerank_ms",
            "retrieval_ms": "retrieval_total_ms",
        }.get(field)
        if fallback:
            value = numeric(retrieval.get(fallback))
            if value is not None:
                return value
    return None


def iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def summarize_file(path: Path) -> dict:
    values = {name: [] for name in LATENCY_FIELDS + CONTEXT_FIELDS}
    query_ids: list[str] = []
    split_counts: Counter[str] = Counter()

    for row in iter_rows(path):
        query_id = row.get("query_id")
        if isinstance(query_id, str):
            query_ids.append(query_id)
        split = row.get("split")
        if isinstance(split, str):
            split_counts[split] += 1

        for field in LATENCY_FIELDS:
            value = timing_value(row, field)
            if value is not None:
                values[field].append(value)

        context_count = numeric(row.get("context_count"))
        if context_count is None and isinstance(row.get("retrieved_record_ids"), list):
            context_count = float(len(row["retrieved_record_ids"]))
        if context_count is not None:
            values["context_count"].append(context_count)

        context_chars = numeric(row.get("context_chars"))
        if context_chars is not None:
            values["context_chars"].append(context_chars)

    duplicate_ids = sorted(
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    )
    missing_fields = {
        field: len(query_ids) - len(values[field])
        for field in LATENCY_FIELDS + CONTEXT_FIELDS
    }
    return {
        "input": str(path.resolve()),
        "queries": len(query_ids),
        "unique_queries": len(set(query_ids)),
        "duplicate_query_ids": duplicate_ids,
        "split_counts": dict(sorted(split_counts.items())),
        "latency_ms": {
            field: describe(values[field])
            for field in LATENCY_FIELDS
        },
        "context": {
            field: describe(values[field])
            for field in CONTEXT_FIELDS
        },
        "missing_counts": missing_fields,
    }


def print_summary(summary: dict) -> None:
    print(f"\nInput: {summary['input']}")
    print(
        f"Queries: {summary['queries']} "
        f"(unique={summary['unique_queries']})"
    )
    print(f"Splits: {summary['split_counts']}")
    if summary["duplicate_query_ids"]:
        print(f"Duplicate query IDs: {summary['duplicate_query_ids']}")

    print("\nLatency (ms):")
    print(f"{'field':<18} {'n':>5} {'mean':>10} {'median':>10} {'p95':>10} {'max':>10}")
    for field, stats in summary["latency_ms"].items():
        if stats["count"] == 0:
            print(f"{field:<18} {0:>5} {'N/A':>10}")
            continue
        print(
            f"{field:<18} {stats['count']:>5} "
            f"{stats['mean']:>10.3f} {stats['median']:>10.3f} "
            f"{stats['p95']:>10.3f} {stats['max']:>10.3f}"
        )

    print("\nContext size:")
    print(f"{'field':<18} {'n':>5} {'mean':>10} {'median':>10} {'p95':>10} {'max':>10}")
    for field, stats in summary["context"].items():
        if stats["count"] == 0:
            print(f"{field:<18} {0:>5} {'N/A':>10}")
            continue
        print(
            f"{field:<18} {stats['count']:>5} "
            f"{stats['mean']:>10.2f} {stats['median']:>10.2f} "
            f"{stats['p95']:>10.2f} {stats['max']:>10.2f}"
        )

    missing = {
        field: count for field, count in summary["missing_counts"].items() if count
    }
    if missing:
        print(f"\nMissing fields: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One or more answer-candidate JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write one <stem>-summary.json per input into this directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in args.inputs:
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        summary = summarize_file(path)
        print_summary(summary)
        if args.output_dir:
            output = args.output_dir / f"{path.stem}-summary.json"
            if output.exists() and not args.overwrite:
                parser.error(
                    f"Summary exists: {output}; pass --overwrite to replace it"
                )
            output.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")


if __name__ == "__main__":
    main()
