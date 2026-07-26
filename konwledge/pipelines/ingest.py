"""Document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from konwledge.core.config import RAGConfig
from konwledge.core.interfaces import Chunker, DocumentCleaner, EmbeddingModel, KeywordIndex, VectorStore
from konwledge.core.models import Chunk, Document
from konwledge.loaders.file_loaders import DirectoryLoader
from konwledge.processing.cleaner import BasicDocumentCleaner, DocumentDeduplicator
from konwledge.processing.chunker import HierarchicalChunker


class IngestionPipeline:
    """Load, clean, deduplicate, chunk, embed and index documents."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        cleaner: DocumentCleaner | None = None,
        chunker: Chunker | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._cleaner = cleaner or BasicDocumentCleaner()
        self._chunker = chunker or HierarchicalChunker((config or RAGConfig()).chunk)
        self._config = config or RAGConfig()
        self._deduplicator = DocumentDeduplicator()

    def ingest_directory(self, directory: Path | None = None) -> List[Chunk]:
        loader = DirectoryLoader(directory or self._config.data_dir, self._config.supported_extensions)
        return self.ingest_documents(loader.load())

    def ingest_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        cleaned = [self._cleaner.clean(document) for document in documents]
        unique_documents = self._deduplicator.deduplicate(cleaned)
        chunks: List[Chunk] = []
        for document in unique_documents:
            chunks.extend(self._chunker.split(document))
        if not chunks:
            return []
        vectors = self._embedding_model.embed_texts([chunk.text for chunk in chunks])
        self._vector_store.add(chunks, vectors)
        self._keyword_index.add(chunks)
        return chunks

    def persist(self, index_dir: Path | None = None) -> None:
        path = (index_dir or self._config.index_dir) / "vectors.json"
        self._vector_store.persist(path)
