"""Load and run BAAI/bge-reranker-v2-m3 from a local model directory."""
from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "bge-reranker-v2-m3"
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
)


class RerankerUnavailableError(RuntimeError):
    """The configured local reranker artifact is missing or incomplete."""


class LocalBGEReranker:
    """Minimal local cross-encoder wrapper backed by Hugging Face Transformers."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
        if not self.model_dir.exists():
            raise RerankerUnavailableError(
                f"Reranker model not found at {self.model_dir}. "
                "Run: python rag/extensions/rerank/download_model.py"
            )
        missing = [
            name for name in REQUIRED_MODEL_FILES
            if not (self.model_dir / name).is_file()
        ]
        if missing:
            raise RerankerUnavailableError(
                f"Reranker model is incomplete at {self.model_dir}; missing: "
                + ", ".join(missing)
                + ". Re-run rag/extensions/rerank/download_model.py"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.torch = torch
        self.model_name = "BAAI/bge-reranker-v2-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                dtype=dtype,
            )
        except TypeError:
            # Transformers 4.x uses torch_dtype rather than dtype.
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                torch_dtype=dtype,
            )
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per ``(query, text)`` pair."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not texts:
            return []

        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                encoded = self.tokenizer(
                    [[query, text] for text in batch_texts],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    name: tensor.to(self.device)
                    for name, tensor in encoded.items()
                }
                logits = self.model(**encoded).logits
                if logits.ndim == 2:
                    if logits.shape[1] != 1:
                        raise RuntimeError(
                            "Expected a reranker with one relevance logit per pair, "
                            f"got logits {tuple(logits.shape)}"
                        )
                    logits = logits[:, 0]
                scores.extend(
                    float(score)
                    for score in logits.detach().float().cpu().tolist()
                )

        if len(scores) != len(texts):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {len(texts)} pairs"
            )
        return scores
