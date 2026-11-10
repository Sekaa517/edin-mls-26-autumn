"""Download BAAI/bge-reranker-v2-m3 into the local rerank models directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from reranker import DEFAULT_MODEL_DIR, REQUIRED_MODEL_FILES


MODEL_REPO = "BAAI/bge-reranker-v2-m3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--repo-id", default=MODEL_REPO)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Verify/download only from the existing Hugging Face cache.",
    )
    args = parser.parse_args()
    target = args.model_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(target),
        local_files_only=args.offline,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "*.txt",
        ],
    )
    missing = [
        name for name in REQUIRED_MODEL_FILES
        if not (target / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Downloaded reranker is incomplete; missing: " + ", ".join(missing)
        )
    total_bytes = sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    print(f"Model ready: {target}")
    print(f"Files: {total_bytes} bytes")


if __name__ == "__main__":
    main()
