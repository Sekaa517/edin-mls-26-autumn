"""Tests for persistent BM25 indexing and BM25+dense RRF retrieval."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_retriever import BM25Index, HybridRetriever
from src.rag_hybrid import RAGPipelineHybrid


class HybridRetrieverTests(unittest.TestCase):
    def test_hybrid_pipeline_accepts_cached_transcript_timing(self):
        class Encoder:
            def encode(self, text):
                self.text = text
                return np.asarray([1.0, 0.0], dtype=np.float32)

        class Retriever:
            def retrieve(self, query_vector, query_text, k):
                return {
                    "record_ids": ["a"],
                    "timings": {
                        "bm25_ms": 2.0,
                        "dense_ms": 1.0,
                        "rrf_ms": 0.1,
                        "topk_ms": 0.1,
                        "retrieval_total_ms": 3.2,
                    },
                }

        class Data:
            def record_text(self, record_id):
                return f"text {record_id}"

        class Context:
            def build(self, transcript, record_ids, data):
                return {
                    "contexts": [data.record_text(record_ids[0])],
                    "prompt": transcript,
                }

        class Generator:
            def generate(self, prompt):
                return {"answer": "answer", "latency_ms": 4.0}

        pipeline = RAGPipelineHybrid(
            None,
            Encoder(),
            Retriever(),
            Data(),
            Context(),
            Generator(),
        )
        result = pipeline.answer_transcript(
            "cached transcript",
            k=5,
            asr_ms=12.5,
        )
        self.assertEqual(result["answer"], "answer")
        self.assertEqual(result["timing"]["asr_ms"], 12.5)
        self.assertEqual(result["timing"]["retrieval_ms"], 3.2)
        self.assertEqual(
            result["timing"]["retrieval_stages"]["bm25_ms"],
            2.0,
        )
        self.assertGreaterEqual(result["timing"]["end_to_end_ms"], 12.5)

    def test_bm25_index_round_trip_and_keyword_ranking(self):
        rows = [
            {"record_id": "a", "text": "alpha beta beta"},
            {"record_id": "b", "text": "gamma delta"},
            {"record_id": "c", "text": "alpha gamma omega"},
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "bm25.json"
            BM25Index.from_corpus(rows).save(path)
            index = BM25Index.load(path)
            self.assertEqual(index.search("alpha beta", 2)[0]["record_id"], "a")
            self.assertEqual(index.vocabulary_size, 5)
            self.assertGreater(path.stat().st_size, 0)

    def test_hybrid_retriever_reports_stages_and_preserves_union(self):
        rows = [
            {"record_id": "a", "text": "alpha beta"},
            {"record_id": "b", "text": "gamma delta"},
            {"record_id": "c", "text": "alpha gamma omega"},
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            bm25_path = root / "bm25.json"
            BM25Index.from_corpus(rows).save(bm25_path)

            embeddings = np.asarray([
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
            ], dtype=np.float32)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_path = root / "dense.npy"
            np.save(embeddings_path, embeddings)
            ids_path = root / "ids.json"
            ids_path.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")

            retriever = HybridRetriever(
                embeddings_path,
                ids_path,
                bm25_path,
                candidate_depth=2,
                rrf_k=60,
                final_k=2,
            )
            result = retriever.retrieve(
                np.asarray([1.0, 0.0], dtype=np.float32),
                "alpha beta",
                k=2,
            )
            self.assertEqual(result["candidates"]["union_count"], 3)
            self.assertEqual(len(result["record_ids"]), 2)
            self.assertEqual(
                set(result["timings"]),
                {
                    "bm25_ms",
                    "dense_ms",
                    "dense_search_ms",
                    "rrf_ms",
                    "topk_ms",
                    "retrieval_total_ms",
                },
            )
            self.assertEqual(result["memory"]["bm25_doc_count"], 3)
            self.assertGreater(result["memory"]["dense_corpus_bytes"], 0)
            self.assertGreater(
                result["timings"]["retrieval_total_ms"],
                result["timings"]["bm25_ms"],
            )

    def test_rrf_prefers_documents_ranked_by_both_retrievers(self):
        rows = [
            {"record_id": "a", "text": "alpha"},
            {"record_id": "b", "text": "alpha beta"},
            {"record_id": "c", "text": "alpha gamma"},
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            bm25_path = root / "bm25.json"
            BM25Index.from_corpus(rows).save(bm25_path)
            embeddings = np.eye(3, dtype=np.float32)
            np.save(root / "dense.npy", embeddings)
            (root / "ids.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
            retriever = HybridRetriever(
                root / "dense.npy",
                root / "ids.json",
                bm25_path,
                candidate_depth=3,
                final_k=3,
            )
            result = retriever.retrieve(
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                "alpha beta",
                k=3,
            )
            self.assertIn("b", result["record_ids"][:2])
            self.assertEqual(len(result["rrf_fusion"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
