"""Prompt construction and citation context assembly."""

from __future__ import annotations

from typing import List, Sequence

from konwledge.core.models import Citation, SearchResult
from konwledge.processing.text import truncate


class ContextBuilder:
    """Build compact, source-aware context for answer generation."""

    def __init__(self, max_context_chars: int = 8000) -> None:
        self._max_context_chars = max_context_chars

    def build(self, results: Sequence[SearchResult]) -> tuple[str, List[Citation]]:
        parts: List[str] = []
        citations: List[Citation] = []
        used_chars = 0
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            heading = " / ".join(chunk.section_path) or chunk.title or chunk.source
            snippet = truncate(chunk.text, 1200)
            block = (
                f"[资料 {index}]\n"
                f"标题：{heading}\n"
                f"来源：{chunk.source}\n"
                f"分数：{result.score:.4f}\n"
                f"内容：\n{snippet}\n"
            )
            if used_chars + len(block) > self._max_context_chars:
                break
            used_chars += len(block)
            parts.append(block)
            citations.append(
                Citation(
                    index=index,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=heading,
                    source=chunk.source,
                    snippet=truncate(chunk.text, 300),
                    metadata=chunk.metadata,
                )
            )
        return "\n\n".join(parts), citations


class RAGPromptBuilder:
    """Build a grounded QA prompt."""

    def __init__(self, language: str = "zh") -> None:
        self._language = language

    def build(self, question: str, context: str) -> str:
        if self._language == "zh":
            return (
                "你是一个严谨的 RAG 知识库问答助手。\n"
                "请只基于给定资料回答问题，不要编造资料中没有的信息。\n"
                "如果资料不足，请明确说明“知识库资料不足，无法确认”。\n"
                "回答中需要用 [资料 1]、[资料 2] 这样的格式标注依据。\n\n"
                f"用户问题：\n{question}\n\n"
                f"可用资料：\n{context}\n\n"
                "请给出准确、简洁、可溯源的回答："
            )
        return (
            "You are a careful RAG assistant. Answer only from the provided context. "
            "If the context is insufficient, say so. Cite sources as [Source 1].\n\n"
            f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
        )
