"""Document cleaning and deduplication."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from konwledge.core.models import Document
from konwledge.processing.text import normalize_whitespace, stable_hash


class BasicDocumentCleaner:
    """Normalize text while preserving paragraph and heading boundaries."""

    def clean(self, document: Document) -> Document:
        text = document.text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\s*第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*\n", "\n", text)
        text = re.sub(r"\n\s*Page\s+\d+\s+of\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
        text = normalize_whitespace(text)
        metadata = dict(document.metadata)
        metadata["content_hash"] = stable_hash(text, 24)
        metadata["char_count"] = len(text)
        return Document(
            id=document.id,
            text=text,
            source=document.source,
            title=document.title,
            metadata=metadata,
        )


class DocumentDeduplicator:
    """Drop documents with identical cleaned content hashes."""

    def deduplicate(self, documents: Iterable[Document]) -> List[Document]:
        seen: Dict[str, Document] = {}
        result: List[Document] = []
        for document in documents:
            key = str(document.metadata.get("content_hash") or stable_hash(document.text, 24))
            if key in seen:
                continue
            seen[key] = document
            result.append(document)
        return result
