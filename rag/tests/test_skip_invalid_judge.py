"""Tests for continuing past malformed judge output."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.answer_judge import Judge, JudgeOutputError
from scripts.evaluate_answers import evaluate_rows


class FakeJudge:
    def __init__(self):
        self.client = SimpleNamespace(base_url="fake://endpoint")

    def evaluate(self, question, answer, gold_documents):
        if question == "bad":
            raise JudgeOutputError("judge output could not be parsed: bad JSON")
        return {
            "correctness": 4,
            "grounding": 3,
            "completeness": 2,
            "explanation": "test",
        }


class SkipInvalidJudgeTests(unittest.TestCase):
    def test_skips_query_and_continues(self):
        rows = [
            {"query_id": "q1", "transcript": "good one", "answer": "a1"},
            {"query_id": "q2", "transcript": "bad", "answer": "a2"},
            {"query_id": "q3", "transcript": "good two", "answer": "a3"},
        ]
        gold = {
            "q1": {"gold_documents": []},
            "q2": {"gold_documents": []},
            "q3": {"gold_documents": []},
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            cache_path = Path(temp) / "judge_cache.jsonl"
            results, skipped = evaluate_rows(
                rows,
                gold,
                FakeJudge(),
                "fake-model",
                cache_path,
                {},
                2,
                "practice",
                skip_invalid_judge=True,
            )
            cache_lines = cache_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual([row["query_id"] for row in results], ["q1", "q3"])
        self.assertEqual([row["query_id"] for row in skipped], ["q2"])
        self.assertEqual(len(cache_lines), 4)
        self.assertEqual(
            skipped[0]["repeat_index"],
            0,
        )

    def test_invalid_output_still_fails_without_flag(self):
        rows = [{"query_id": "q1", "transcript": "bad", "answer": "a1"}]
        gold = {"q1": {"gold_documents": []}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            with self.assertRaises(JudgeOutputError):
                evaluate_rows(
                    rows,
                    gold,
                    FakeJudge(),
                    "fake-model",
                    Path(temp) / "judge_cache.jsonl",
                    {},
                    1,
                    "practice",
                    skip_invalid_judge=False,
                )

    def test_judge_raises_output_error_after_retries(self):
        class Completions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="not valid JSON")
                        )
                    ]
                )

        judge = Judge.__new__(Judge)
        judge.model = "fake"
        judge.retries = 2
        judge.client = SimpleNamespace(
            base_url="fake://endpoint",
            chat=SimpleNamespace(
                completions=Completions(),
            ),
        )
        with patch("src.evaluation.answer_judge.time.sleep"):
            with self.assertRaises(JudgeOutputError):
                judge.evaluate("q", "a", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
