# Local BGE reranking

The reranker is `BAAI/bge-reranker-v2-m3`, loaded from:

```text
rag/extensions/rerank/models/bge-reranker-v2-m3/
```

Download the model once on a machine with internet access:

```bash
cd <course-repository-root>
source "$HOME/mls-rag-env/bin/activate"
python rag/extensions/rerank/download_model.py
```

Model weights are ignored by Git. The DICE machine must download the model
locally or receive that directory through an approved file-transfer method.

Evaluate dense Top-40 plus reranked Top-5:

```bash
python rag/scripts/rag_evaluation_rerank.py \
  --split practice \
  --k 40 \
  --k-final 5 \
  --device cpu \
  --rerank-device cuda \
  --warmup 2 \
  --output-dir rag/results/rerank-practice
```

Generate compatible answer candidates:

```bash
python rag/scripts/generate_answers_rerank.py \
  --split practice \
  --k 40 \
  --k-final 5 \
  --device cpu \
  --rerank-device cuda \
  --warmup 2 \
  --transcript-cache rag/.cache/practice_shared.jsonl \
  --output rag/results/rerank-practice/answers.jsonl
```
