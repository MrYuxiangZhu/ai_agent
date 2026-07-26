"""Offline RAG evaluation helpers."""

from __future__ import annotations

from typing import Iterable, List

from konwledge.core.models import EvaluationCase, EvaluationResult
from konwledge.pipelines.qa import RAGPipeline


class RAGEvaluator:
    """Evaluate retrieval recall and citation grounding for a RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline, k: int = 8) -> None:
        self._pipeline = pipeline
        self._k = k

    def evaluate(self, cases: Iterable[EvaluationCase]) -> EvaluationResult:
        details: List[dict] = []
        total = 0
        recall_hits = 0
        citation_hits = 0
        answer_chars = 0
        for case in cases:
            total += 1
            answer = self._pipeline.ask(case.question)
            retrieved_chunk_ids = [item.chunk.id for item in answer.retrieved[: self._k]]
            retrieved_doc_ids = [item.chunk.document_id for item in answer.retrieved[: self._k]]
            expected_chunks = set(case.expected_chunk_ids)
            expected_docs = set(case.expected_document_ids)
            retrieval_hit = bool(expected_chunks & set(retrieved_chunk_ids)) or bool(expected_docs & set(retrieved_doc_ids))
            citation_doc_ids = {citation.document_id for citation in answer.citations}
            citation_hit = bool(expected_docs & citation_doc_ids) if expected_docs else retrieval_hit
            recall_hits += int(retrieval_hit)
            citation_hits += int(citation_hit)
            answer_chars += len(answer.answer)
            details.append(
                {
                    "question": case.question,
                    "retrieval_hit": retrieval_hit,
                    "citation_hit": citation_hit,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_document_ids": retrieved_doc_ids,
                    "answer": answer.answer,
                    "tags": case.tags,
                }
            )
        if total == 0:
            return EvaluationResult(total=0, recall_at_k=0.0, citation_hit_rate=0.0, average_answer_length=0.0)
        return EvaluationResult(
            total=total,
            recall_at_k=recall_hits / total,
            citation_hit_rate=citation_hits / total,
            average_answer_length=answer_chars / total,
            details=details,
        )
