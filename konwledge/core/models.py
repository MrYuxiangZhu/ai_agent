"""Stable data contracts for the RAG knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Metadata = Dict[str, Any]


@dataclass(frozen=True)
class Document:
    """A raw or normalized source document."""

    id: str
    text: str
    source: str
    title: str = ""
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A searchable text unit derived from a document."""

    id: str
    document_id: str
    text: str
    source: str
    title: str = ""
    section_path: List[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk plus its vector representation."""

    chunk: Chunk
    vector: List[float]


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and the scores used to rank it."""

    chunk: Chunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class Citation:
    """A source citation attached to a generated answer."""

    index: int
    chunk_id: str
    document_id: str
    title: str
    source: str
    snippet: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class RAGAnswer:
    """Final RAG answer with retrieval evidence."""

    question: str
    answer: str
    citations: List[Citation]
    retrieved: List[SearchResult]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationCase:
    """A single offline evaluation example."""

    question: str
    expected_answer: str = ""
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_document_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated RAG evaluation metrics."""

    total: int
    recall_at_k: float
    citation_hit_rate: float
    average_answer_length: float
    details: List[Dict[str, Any]] = field(default_factory=list)
