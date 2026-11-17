"""Average the first N answer-judge results from one or more experiment paths."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PER_QUERY_FILE = "answer_judge_per_query.jsonl"
SUMMARY_FILE = "answer_judge_summary.json"
METRICS = (
    "normalised_answer_score",
    "correctness",
    "grounding",
    "completeness",
)


def resolve_per_query_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / PER_QUERY_FILE
    return path


def resolve_judge_model(path: str | Path) -> str | None:
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    summary_path = directory / SUMMARY_FILE
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model = summary.get("judge_model")
    return str(model) if model else None


def load_prefix(per_query_path: str | Path, n: int) -> list[dict]:
    if n < 1:
        raise ValueError("n must be positive")
    path = resolve_per_query_path(per_query_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            query_id = row.get("query_id")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{path}:{line_number}: missing query_id")
            if query_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id}")
            seen.add(query_id)
            rows.append(row)
            if len(rows) == n:
                break

    if len(rows) < n:
        raise ValueError(
            f"{path}: requested first {n} queries, but only {len(rows)} are available"
        )
    return rows


def summarize_prefix(input_path: str | Path, n: int) -> dict:
    per_query_path = resolve_per_query_path(input_path)
    rows = load_prefix(per_query_path, n)
    values = {metric: [] for metric in METRICS}
    for index, row in enumerate(rows, start=1):
        components = []
        for metric in ("correctness", "grounding", "completeness"):
            value = row.get(metric)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{per_query_path}: row {index} has no numeric {metric}"
                )
            values[metric].append(float(value))
            components.append(float(value))

        score = row.get("normalised_answer_score")
        if not isinstance(score, (int, float)):
            score = sum(components) / 12.0
        values["normalised_answer_score"].append(float(score))

    return {
        "input": str(Path(input_path).resolve()),
        "per_query_file": str(per_query_path.resolve()),
        "judge_model": resolve_judge_model(input_path),
        "n": n,
        "first_query_id": rows[0]["query_id"],
        "last_query_id": rows[-1]["query_id"],
        "query_ids": [row["query_id"] for row in rows],
        "mean_scores": {
            metric: float(statistics.mean(values[metric]))
            for metric in METRICS
        },
        "zero_normalised_score_count": sum(
            score == 0.0 for score in values["normalised_answer_score"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "Judge output directories or answer_judge_per_query.jsonl files."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Use the first N records in file order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for all summaries.",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")

    summaries = []
    for input_path in args.inputs:
        summary = summarize_prefix(input_path, args.n)
        summaries.append(summary)

    print(f"First {args.n} judged queries, in file order:\n")
    print(
        f"{'score':>9} {'correct':>9} {'ground':>8} "
        f"{'complete':>9} {'model':<42} input"
    )
    for summary in summaries:
        scores = summary["mean_scores"]
        print(
            f"{scores['normalised_answer_score']:>9.6f} "
            f"{scores['correctness']:>9.3f} "
            f"{scores['grounding']:>8.3f} "
            f"{scores['completeness']:>9.3f} "
            f"{(summary['judge_model'] or '-'):<42} "
            f"{summary['input']}"
        )

    if len(summaries) > 1:
        query_sets = [tuple(item["query_ids"]) for item in summaries]
        print(
            "\nSame query_id order across all inputs:",
            all(items == query_sets[0] for items in query_sets[1:]),
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"Output exists: {args.output}")
        args.output.write_text(
            json.dumps(
                {
                    "n": args.n,
                    "inputs": summaries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
