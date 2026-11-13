"""Tests for the rerank answer-candidate summary utility."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.rerank.summarize_answers_rerank import summarize_file


class SummaryTests(unittest.TestCase):
    def test_latency_and_context_statistics(self):
        rows = [
            {
                "query_id": "q1",
                "split": "practice",
                "context_count": 5,
                "context_chars": 100,
                "timing": {
                    "asr_ms": 0.0,
                    "embedding_ms": 1.0,
                    "dense_ms": 2.0,
                    "rerank_ms": 10.0,
                    "retrieval_ms": 12.0,
                    "generation_ms": 20.0,
                    "end_to_end_ms": 33.0,
                },
            },
            {
                "query_id": "q2",
                "split": "practice",
                "context_count": 5,
                "context_chars": 200,
                "timing": {
                    "asr_ms": 0.0,
                    "embedding_ms": 3.0,
                    "dense_ms": 4.0,
                    "rerank_ms": 20.0,
                    "retrieval_ms": 24.0,
                    "generation_ms": 30.0,
                    "end_to_end_ms": 57.0,
                },
            },
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "answers-k40-f5.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = summarize_file(path)
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["unique_queries"], 2)
        self.assertEqual(summary["latency_ms"]["rerank_ms"]["mean"], 15.0)
        self.assertEqual(summary["latency_ms"]["end_to_end_ms"]["median"], 45.0)
        self.assertEqual(summary["context"]["context_count"]["mean"], 5.0)
        self.assertEqual(summary["context"]["context_chars"]["mean"], 150.0)
        self.assertEqual(
            summary["missing_counts"]["generation_ms"],
            0,
        )

    def test_retrieval_fallback_and_missing_fields(self):
        row = {
            "query_id": "q1",
            "retrieval": {
                "dense_ms": 2.5,
                "rerank_ms": 8.0,
                "retrieval_total_ms": 10.5,
            },
            "timing": {"generation_ms": 1.0},
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "answers.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary = summarize_file(path)
        self.assertEqual(summary["latency_ms"]["dense_ms"]["mean"], 2.5)
        self.assertEqual(summary["latency_ms"]["rerank_ms"]["mean"], 8.0)
        self.assertEqual(summary["latency_ms"]["retrieval_ms"]["mean"], 10.5)
        self.assertIsNone(summary["context"]["context_chars"]["mean"])
        self.assertEqual(summary["missing_counts"]["context_chars"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
