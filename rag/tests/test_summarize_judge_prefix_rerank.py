"""Tests for averaging the first N answer-judge results."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.rerank.summarize_judge_prefix_rerank import summarize_prefix


class JudgePrefixSummaryTests(unittest.TestCase):
    def test_first_n_average_and_model_name(self):
        rows = [
            {
                "query_id": "q1",
                "correctness": 4,
                "grounding": 4,
                "completeness": 4,
                "normalised_answer_score": 1.0,
            },
            {
                "query_id": "q2",
                "correctness": 2,
                "grounding": 1,
                "completeness": 1,
                "normalised_answer_score": 4 / 12,
            },
            {
                "query_id": "q3",
                "correctness": 0,
                "grounding": 0,
                "completeness": 0,
                "normalised_answer_score": 0.0,
            },
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            directory = Path(temp)
            (directory / "answer_judge_per_query.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            (directory / "answer_judge_summary.json").write_text(
                json.dumps({"judge_model": "test-model"}),
                encoding="utf-8",
            )
            summary = summarize_prefix(directory, 2)

        self.assertEqual(summary["judge_model"], "test-model")
        self.assertEqual(summary["query_ids"], ["q1", "q2"])
        self.assertAlmostEqual(
            summary["mean_scores"]["normalised_answer_score"],
            (1.0 + 4 / 12) / 2,
        )
        self.assertEqual(summary["mean_scores"]["correctness"], 3.0)
        self.assertEqual(summary["zero_normalised_score_count"], 0)

    def test_rejects_short_input(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "answer_judge_per_query.jsonl"
            path.write_text(
                json.dumps({
                    "query_id": "q1",
                    "correctness": 1,
                    "grounding": 1,
                    "completeness": 1,
                    "normalised_answer_score": 0.25,
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only 1 are available"):
                summarize_prefix(path, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
