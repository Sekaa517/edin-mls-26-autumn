"""Small JSONL transcript cache shared by the hybrid experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path


def load_transcript_cache(path: str | Path | None) -> dict[str, str]:
    path = Path(path) if path else None
    cache: dict[str, str] = {}
    if path is None or not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[str(row["query_id"])] = str(row["transcript"])
    return cache


def append_transcript_cache(path: str | Path, query_id: str, transcript: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            {"query_id": query_id, "transcript": transcript},
            ensure_ascii=False,
        ) + "\n")
