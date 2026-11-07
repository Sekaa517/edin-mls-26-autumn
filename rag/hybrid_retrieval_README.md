# Hybrid retrieval experiment

This directory keeps the supplied RAG baseline unchanged. The hybrid experiment
adds BM25 plus dense exact retrieval, fused with reciprocal rank fusion (RRF).

## Build the persistent BM25 index

From the repository root:

```bash
python rag/scripts/build_bm25_index.py --force
```

The generated index is:

```text
rag/data/rag_corpus/embeddings/bm25_index.json
```

BM25 uses a sparse inverted index, not a dense embedding, so a `.npy` embedding
matrix is not required. The JSON contains the vocabulary, postings and document
lengths needed to reproduce BM25 scores.

## Evaluate retrieval

```bash
python rag/scripts/evaluate_rag_hybrid.py \
  --split practice \
  --device cpu \
  --candidate-depth 20 \
  --rrf-k 60 \
  --final-k 5 \
  --transcript-cache rag/.cache/practice_hybrid_transcripts.jsonl \
  --output rag/results/hybrid_practice.json
```

The output includes per-query and aggregate timings for BM25, dense retrieval,
RRF, final Top-k selection and total retrieval. Memory fields report the stored
BM25 index and dense corpus matrix sizes, logical candidate footprints, process
RSS observations, and a PyTorch GPU-allocation delta when `torch` is available.

## Generate answers

```bash
python rag/scripts/generate_answers_hybrid.py \
  --split practice \
  --k 5 \
  --device cpu \
  --transcript-cache rag/.cache/practice_hybrid_transcripts.jsonl \
  --output rag/results/hybrid_answers.jsonl
```

The candidate JSONL is compatible with the supplied answer evaluator.
