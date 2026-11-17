"""Evaluate public answers against evidence reconstructed from public qrels."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import csv
import os
from src.evaluation.answer_judge import (
    Judge, JudgeOutputError, RUBRIC_VERSION, SCRIPT_VERSION, build_public_gold, load_jsonl,
    load_cache, cache_key, append_cache, mean, write_json_atomic, write_jsonl_atomic,
)

def validate_candidates(rows, gold, split, allow_subset=False):
    if not rows:
        raise ValueError("Candidate file is empty")
    seen = set()
    for row in rows:
        qid = row.get("query_id")
        if not isinstance(qid, str) or qid not in gold:
            raise ValueError(f"Unknown query ID for {split}: {qid}")
        if qid in seen:
            raise ValueError(f"Duplicate query ID: {qid}")
        seen.add(qid)
        if row.get("split", split) != split:
            raise ValueError(f"{qid}: split mismatch")
        if not isinstance(row.get("transcript"), str) or not row["transcript"].strip():
            raise ValueError(f"{qid}: nonempty ASR transcript required")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"{qid}: answer must be a string (empty answers are permitted)")
    if not allow_subset and seen != set(gold):
        raise ValueError(f"Missing {len(set(gold)-seen)} queries; use --allow-subset for a labelled diagnostic subset")
    return seen


def evaluate_rows(rows, gold, judge, model, cache_path, cache, repeats, split,
                  skip_invalid_judge=False):
    results = []
    skipped = []
    for row in rows:
        qid = row["query_id"]
        docs = gold[qid]["gold_documents"]
        repeated = []
        try:
            for rep in range(repeats):
                key = cache_key(
                    model + "@" + str(judge.client.base_url),
                    row["transcript"], row["answer"], docs, rep,
                )
                if key in cache:
                    score = Judge.parse(__import__("json").dumps(cache[key]["result"]))
                else:
                    score = judge.evaluate(row["transcript"], row["answer"], docs)
                    entry = {"cache_key": key, "query_id": qid, "result": score}
                    append_cache(cache_path, entry)
                    cache[key] = entry
                repeated.append(score)
        except JudgeOutputError as error:
            if not skip_invalid_judge:
                raise
            skipped.append({
                "query_id": qid,
                "repeat_index": rep,
                "error": str(error),
            })
            print(f"{qid}: SKIPPED invalid judge output after retries")
            continue

        result = {"query_id": qid, "split": split, "question_mode": "transcript"}
        for dim in ("correctness", "grounding", "completeness"):
            result[dim] = mean([r[dim] for r in repeated])
        result["normalised_answer_score"] = sum(
            result[d] for d in ("correctness", "grounding", "completeness")
        ) / 12
        result["judge_explanations"] = [r["explanation"] for r in repeated]
        results.append(result)
        print(f"{qid}: {result['normalised_answer_score']:.3f}")
    return results, skipped


def write_skipped_tsv(path, skipped):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("query_id", "repeat_index", "error"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(skipped)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/rag_corpus")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/answer_evaluation")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--api-retries", type=int, default=3)
    ap.add_argument("--allow-subset", action="store_true")
    ap.add_argument(
        "--skip-invalid-judge",
        action="store_true",
        help=(
            "Skip a query after judge JSON/schema retries are exhausted and "
            "continue with later queries."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs without API calls or output writes")
    args = ap.parse_args()
    if args.repeats < 1 or args.api_retries < 1:
        ap.error("--repeats and --api-retries must be positive")
    gold = build_public_gold(args.data_dir, args.split)
    rows = load_jsonl(args.candidates)
    seen = validate_candidates(rows, gold, args.split, args.allow_subset)
    if args.dry_run:
        print(f"Validated {len(seen)}/{len(gold)} {args.split} candidates; no API calls")
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    model = args.model or os.getenv("ELM_JUDGE_MODEL") or os.getenv("ELM_MODEL", "utter-project/EuroLLM-22B-Instruct-2512")
    judge = Judge(model, retries=args.api_retries)
    cache_path = args.out_dir / "judge_cache.jsonl"
    cache = load_cache(cache_path)
    results, skipped = evaluate_rows(
        rows,
        gold,
        judge,
        model,
        cache_path,
        cache,
        args.repeats,
        args.split,
        args.skip_invalid_judge,
    )
    write_jsonl_atomic(args.out_dir / "answer_judge_per_query.jsonl", results)
    write_skipped_tsv(args.out_dir / "skipped_queries.tsv", skipped)
    summary = {"script_version": SCRIPT_VERSION, "rubric_version": RUBRIC_VERSION,
               "judge_model": model, "question_mode": "transcript", "split": args.split,
               "queries": len(results), "candidate_queries": len(rows),
               "evaluated_queries": len(results), "skipped_queries": len(skipped),
               "expected_queries": len(gold), "coverage": len(seen)/len(gold),
               "evaluation_coverage": len(results)/len(gold),
               "is_subset": seen != set(gold), "repeats": args.repeats,
               "skip_invalid_judge": args.skip_invalid_judge}
    for dim in ("correctness", "grounding", "completeness", "normalised_answer_score"):
        summary["mean_" + dim] = mean([r[dim] for r in results])
    write_json_atomic(args.out_dir / "answer_judge_summary.json", summary)

if __name__ == "__main__":
    main()
