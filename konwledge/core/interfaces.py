"""Interfaces for independently replaceable RAG modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Protocol, Sequence

from konwledge.core.models import Chunk, Document, SearchResult


class DocumentLoader(Protocol):
    """Load source data into normalized documents."""

    def load(self) -> List[Document]:
        ...


class DocumentCleaner(Protocol):
    """Clean and normalize document text."""

    def clean(self, document: Document) -> Document:
        ...


class Chunker(Protocol):
    """Split documents into retrieval chunks."""

    def split(self, document: Document) -> List[Chunk]:
        ...


class EmbeddingModel(Protocol):
    """Convert text into dense vectors."""

    @property
    def dimension(self) -> int:
        ...

    def embed_text(self, text: str) -> List[float]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class VectorStore(Protocol):
    """Dense vector index."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        ...

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        metadata_filter: Optional[Mapping[str, object]] = None,
    ) -> List[SearchResult]:
        ...

    def persist(self, path: Path) -> None:
        ...

    def load(self, path: Path) -> None:
        ...


class KeywordIndex(Protocol):
    """Sparse keyword index such as BM25."""

    def add(self, chunks: Sequence[Chunk]) -> None:
        ...

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Mapping[str, object]] = None,
    ) -> List[SearchResult]:
        ...


class Reranker(Protocol):
    """Second-stage ranker."""

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> List[SearchResult]:
        ...


class LLMClient(Protocol):
    """Answer generator."""

    def generate(self, prompt: str) -> str:
        ...


class Evaluator(Protocol):
    """Offline quality evaluator."""

    def evaluate(self, cases: Iterable[object]) -> object:
        ...
