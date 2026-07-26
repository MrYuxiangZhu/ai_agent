"""Shared text utilities for Chinese and English RAG content."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    compacted = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = blank
    return "\n".join(compacted).strip()


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
