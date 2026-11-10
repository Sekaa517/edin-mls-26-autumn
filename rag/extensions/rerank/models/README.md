# Local reranker model

Run:

```bash
python rag/extensions/rerank/download_model.py
```

The `BAAI/bge-reranker-v2-m3` files are downloaded into:

```text
rag/extensions/rerank/models/bge-reranker-v2-m3/
```

The model weights are intentionally ignored by Git and must be downloaded
separately on each machine or copied outside Git.
