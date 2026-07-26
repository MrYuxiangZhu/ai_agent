"""Configuration objects for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 1200
    overlap_chars: int = 160
    min_chars: int = 120
    split_by_headings: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    vector_top_k: int = 30
    keyword_top_k: int = 30
    final_top_k: int = 8
    vector_weight: float = 0.65
    keyword_weight: float = 0.35
    use_rerank: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    max_context_chars: int = 8000
    language: str = "zh"
    require_citations: bool = True


@dataclass(frozen=True)
class RAGConfig:
    """Top-level config for a complete local RAG system."""

    data_dir: Path = Path("konwledge/data")
    index_dir: Path = Path("konwledge/index")
    supported_extensions: List[str] = field(default_factory=lambda: [".md", ".txt", ".pdf"])
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    default_metadata: Dict[str, object] = field(default_factory=dict)
