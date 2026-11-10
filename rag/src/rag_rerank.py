"""RAG pipeline for dense retrieval followed by local reranking."""
from __future__ import annotations

import time


class RAGPipelineRerank:
    def __init__(self, asr, encoder, retriever, data, context_builder, generator):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator

    def answer_audio(self, audio_path, k: int = 40, k_final: int = 5):
        total_start = time.perf_counter()
        asr_start = time.perf_counter()
        transcript = self.asr.transcribe(audio_path)
        asr_ms = (time.perf_counter() - asr_start) * 1000.0
        return self.answer_transcript(
            transcript,
            k=k,
            k_final=k_final,
            asr_ms=asr_ms,
            total_start=total_start,
        )

    def answer_transcript(
        self,
        transcript: str,
        k: int = 40,
        k_final: int = 5,
        *,
        asr_ms: float = 0.0,
        total_start: float | None = None,
    ):
        if total_start is None:
            total_start = time.perf_counter() - asr_ms / 1000.0

        embedding_start = time.perf_counter()
        query_vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

        retrieval = self.retriever.retrieve(
            query_vector,
            transcript,
            k=k,
            k_final=k_final,
        )
        built = self.context_builder.build(
            transcript,
            retrieval["record_ids"],
            self.data,
        )
        generation = self.generator.generate(built["prompt"])
        end_to_end_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "transcript": transcript,
            "retrieval": retrieval,
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "dense_ms": retrieval["dense_ms"],
                "rerank_ms": retrieval["rerank_ms"],
                "retrieval_ms": retrieval["retrieval_total_ms"],
                "generation_ms": generation["latency_ms"],
                "end_to_end_ms": end_to_end_ms,
            },
        }
