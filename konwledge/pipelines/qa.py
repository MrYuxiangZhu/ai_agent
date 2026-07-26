"""End-to-end RAG question answering pipeline."""

from __future__ import annotations

from typing import Mapping, Optional

from konwledge.core.config import RAGConfig
from konwledge.core.interfaces import LLMClient
from konwledge.core.models import RAGAnswer
from konwledge.llm.prompting import ContextBuilder, RAGPromptBuilder
from konwledge.retrieval.hybrid import HybridRetriever


class RAGPipeline:
    """Retrieve evidence, build prompt, generate answer and attach citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        config: RAGConfig | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        self._config = config or RAGConfig()
        self._retriever = retriever
        self._llm_client = llm_client
        self._context_builder = context_builder or ContextBuilder(self._config.generation.max_context_chars)
        self._prompt_builder = prompt_builder or RAGPromptBuilder(self._config.generation.language)

    def ask(self, question: str, metadata_filter: Optional[Mapping[str, object]] = None) -> RAGAnswer:
        results = self._retriever.retrieve(question, metadata_filter)
        context, citations = self._context_builder.build(results)
        if not context.strip():
            return RAGAnswer(question=question, answer="知识库资料不足，无法确认。", citations=[], retrieved=[])
        prompt = self._prompt_builder.build(question, context)
        answer = self._llm_client.generate(prompt)
        return RAGAnswer(
            question=question,
            answer=answer,
            citations=citations,
            retrieved=results,
            metadata={"context_chars": len(context), "citation_count": len(citations)},
        )
