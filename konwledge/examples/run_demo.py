"""Run a local RAG demo with the bundled sample document."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from konwledge.core.config import RAGConfig
from konwledge.factory import build_local_rag_system


def main() -> None:
    config = RAGConfig(data_dir=ROOT / "konwledge" / "data")
    system = build_local_rag_system(config)
    chunks = system.ingestion.ingest_directory()
    answer = system.qa.ask("如何提高 RAG 知识库质量？")
    print(f"Indexed chunks: {len(chunks)}")
    print(answer.answer)
    print("\nCitations:")
    for citation in answer.citations:
        print(f"[资料 {citation.index}] {citation.title} - {citation.source}")


if __name__ == "__main__":
    main()
