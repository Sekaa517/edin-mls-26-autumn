"""Tests for dense retrieval followed by cross-encoder reranking."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever_rerank import RerankRetriever
from src.rag_rerank import RAGPipelineRerank


class RerankRetrieverTests(unittest.TestCase):
    def test_rag_pipeline_threads_both_k_values_and_timings(self):
        class Encoder:
            def encode(self, text):
                return np.asarray([1.0, 0.0], dtype=np.float32)

        class Retriever:
            def retrieve(self, query_vector, query_text, k, k_final):
                self.called_with = (query_text, k, k_final)
                return {
                    "record_ids": ["a", "b"],
                    "scores": [0.9, 0.8],
                    "dense_ms": 1.0,
                    "rerank_ms": 2.0,
                    "retrieval_total_ms": 3.0,
                }

        class Data:
            def record_text(self, record_id):
                return f"text {record_id}"

        class Context:
            def build(self, transcript, record_ids, data):
                return {
                    "contexts": ["context"],
                    "prompt": transcript,
                }

        class Generator:
            def generate(self, prompt):
                return {"answer": "answer", "latency_ms": 4.0}

        retriever = Retriever()
        pipeline = RAGPipelineRerank(
            None,
            Encoder(),
            retriever,
            Data(),
            Context(),
            Generator(),
        )
        result = pipeline.answer_transcript(
            "cached transcript",
            k=40,
            k_final=5,
            asr_ms=10.0,
        )
        self.assertEqual(retriever.called_with, ("cached transcript", 40, 5))
        self.assertEqual(result["timing"]["dense_ms"], 1.0)
        self.assertEqual(result["timing"]["rerank_ms"], 2.0)
        self.assertEqual(result["timing"]["retrieval_ms"], 3.0)
        self.assertGreaterEqual(result["timing"]["end_to_end_ms"], 10.0)

    def test_rerank_pipeline_returns_top_level_timings(self):
        class Dense:
            def retrieve(self, query_vector, k):
                self.k = k
                return {
                    "record_ids": ["a", "b", "c"],
                    "scores": [0.8, 0.7, 0.6],
                    "latency_ms": 1.25,
                }

        class Data:
            texts = {
                "a": "alpha passage",
                "b": "best passage",
                "c": "middle passage",
            }

            def record_text(self, record_id):
                return self.texts[record_id]

        class Reranker:
            model_name = "fake-reranker"

            def score_pairs(self, query, texts):
                self.query = query
                self.texts = texts
                return [0.1, 0.9, 0.5]

        dense = Dense()
        reranker = Reranker()
        retriever = RerankRetriever(
            "unused.npy",
            "unused.json",
            Data(),
            reranker,
            k=3,
            k_final=2,
            dense_retriever=dense,
        )
        result = retriever.retrieve(
            np.asarray([1.0, 0.0], dtype=np.float32),
            "test query",
        )
        self.assertEqual(dense.k, 3)
        self.assertEqual(reranker.query, "test query")
        self.assertEqual(reranker.texts, [
            "alpha passage",
            "best passage",
            "middle passage",
        ])
        self.assertEqual(result["record_ids"], ["b", "c"])
        self.assertEqual(result["scores"], [0.9, 0.5])
        self.assertIn("dense_ms", result)
        self.assertEqual(result["dense_ms"], 1.25)
        self.assertIn("dense_wall_ms", result)
        self.assertIn("rerank_ms", result)
        self.assertIn("retrieval_total_ms", result)
        self.assertNotIn("latency_ms", result)
        self.assertGreaterEqual(result["retrieval_total_ms"], result["rerank_ms"])

    def test_k_final_cannot_exceed_k(self):
        with self.assertRaises(ValueError):
            RerankRetriever(
                "unused.npy",
                "unused.json",
                object(),
                object(),
                k=5,
                k_final=6,
                dense_retriever=object(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
