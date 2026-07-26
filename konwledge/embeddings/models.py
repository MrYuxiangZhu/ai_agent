"""Embedding model implementations."""

from __future__ import annotations

import json
import math
import urllib.request
from typing import List, Sequence

from konwledge.processing.text import stable_hash, tokenize


class HashEmbeddingModel:
    """Deterministic local embedding for demos and tests.

    It is not a replacement for production semantic embeddings, but provides a
    dependency-free vector interface so the whole RAG stack can run locally.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        tokens = tokenize(text)
        for token in tokens:
            bucket = int(stable_hash(token, 8), 16) % self._dimension
            sign = 1.0 if int(stable_hash("sign:" + token, 8), 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


class HttpEmbeddingModel:
    """Generic HTTP embedding client.

    Expected response shape:

    ``{"embeddings": [[...], [...]]}`` or ``{"data": [{"embedding": [...]}]}``
    """

    def __init__(self, endpoint: str, model: str = "", token: str = "", dimension: int = 1536, timeout_seconds: int = 60) -> None:
        self._endpoint = endpoint
        self._model = model
        self._token = token
        self._dimension = dimension
        self._timeout_seconds = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        payload = {"model": self._model, "input": list(texts)}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(self._endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        if "embeddings" in body:
            return [list(map(float, embedding)) for embedding in body["embeddings"]]
        if "data" in body:
            return [list(map(float, item["embedding"])) for item in body["data"]]
        raise ValueError("embedding response must contain 'embeddings' or OpenAI-compatible 'data'")
