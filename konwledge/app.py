"""Command-line entry point for the local RAG knowledge base."""

from __future__ import annotations

import argparse
from pathlib import Path

from konwledge.core.config import RAGConfig
from konwledge.factory import build_local_rag_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local modular RAG knowledge base.")
    parser.add_argument("--data-dir", default="konwledge/data", help="Directory containing .md/.txt/.pdf files.")
    parser.add_argument("--question", required=True, help="Question to ask the knowledge base.")
    parser.add_argument("--persist", action="store_true", help="Persist the vector index to index/vectors.json.")
    args = parser.parse_args()

    config = RAGConfig(data_dir=Path(args.data_dir))
    system = build_local_rag_system(config)
    chunks = system.ingestion.ingest_directory(config.data_dir)
    if args.persist:
        system.ingestion.persist(config.index_dir)
    answer = system.qa.ask(args.question)
    print(f"Indexed chunks: {len(chunks)}")
    print("\nAnswer:\n")
    print(answer.answer)
    print("\nCitations:")
    for citation in answer.citations:
        print(f"[资料 {citation.index}] {citation.title} - {citation.source}")


if __name__ == "__main__":
    main()
