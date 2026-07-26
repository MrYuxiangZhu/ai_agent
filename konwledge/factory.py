"""Factories for assembling default RAG systems."""

from __future__ import annotations

from dataclasses import dataclass

from konwledge.core.config import RAGConfig
from konwledge.embeddings.models import HashEmbeddingModel
from konwledge.llm.clients import MockLLMClient
from konwledge.pipelines.ingest import IngestionPipeline
from konwledge.pipelines.qa import RAGPipeline
from konwledge.retrieval.hybrid import HybridRetriever
from konwledge.retrieval.rerank import LexicalReranker
from konwledge.stores.keyword import BM25KeywordIndex
from konwledge.stores.vector import InMemoryVectorStore


@dataclass(frozen=True)
class RAGSystem:
    ingestion: IngestionPipeline
    qa: RAGPipeline
    vector_store: InMemoryVectorStore
    keyword_index: BM25KeywordIndex


def build_local_rag_system(config: RAGConfig | None = None) -> RAGSystem:
    """Build a dependency-free local RAG system."""

    config = config or RAGConfig()
    embedding_model = HashEmbeddingModel()
    vector_store = InMemoryVectorStore()
    keyword_index = BM25KeywordIndex()
    reranker = LexicalReranker()
    retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        keyword_index=keyword_index,
        reranker=reranker,
        config=config.retrieval,
    )
    ingestion = IngestionPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        keyword_index=keyword_index,
        config=config,
    )
    qa = RAGPipeline(retriever=retriever, llm_client=MockLLMClient(), config=config)
    return RAGSystem(ingestion=ingestion, qa=qa, vector_store=vector_store, keyword_index=keyword_index)
