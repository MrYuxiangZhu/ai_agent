"""Chunking strategies for retrieval quality and citation precision."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from konwledge.core.config import ChunkConfig
from konwledge.core.models import Chunk, Document
from konwledge.processing.text import stable_hash


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$|^(.{1,80})\n[=-]{3,}$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    title: str
    path: List[str]
    text: str
    start: int
    end: int


class HierarchicalChunker:
    """Split documents by headings first, then by length with overlap."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()

    def split(self, document: Document) -> List[Chunk]:
        sections = self._sections(document)
        chunks: List[Chunk] = []
        for section in sections:
            chunks.extend(self._split_section(document, section))
        return chunks

    def _sections(self, document: Document) -> List[Section]:
        if not self._config.split_by_headings:
            return [Section(document.title, [document.title] if document.title else [], document.text, 0, len(document.text))]

        matches = list(_HEADING_RE.finditer(document.text))
        if not matches:
            return [Section(document.title, [document.title] if document.title else [], document.text, 0, len(document.text))]

        sections: List[Section] = []
        stack: List[Tuple[int, str]] = []
        for idx, match in enumerate(matches):
            title = (match.group(2) or match.group(3) or "").strip()
            level = len(match.group(1) or "#")
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(document.text)
            path = [item[1] for item in stack if item[1]]
            section_text = document.text[start:end].strip()
            if section_text:
                sections.append(Section(title=title, path=path, text=section_text, start=start, end=end))
        return sections or [Section(document.title, [document.title] if document.title else [], document.text, 0, len(document.text))]

    def _split_section(self, document: Document, section: Section) -> List[Chunk]:
        text = section.text
        max_chars = self._config.max_chars
        overlap = min(self._config.overlap_chars, max_chars // 3)
        if len(text) <= max_chars:
            return [self._make_chunk(document, section, text, section.start, section.end, 0)]

        chunks: List[Chunk] = []
        cursor = 0
        ordinal = 0
        while cursor < len(text):
            end = min(len(text), cursor + max_chars)
            if end < len(text):
                boundary = max(
                    text.rfind("\n\n", cursor, end),
                    text.rfind("。", cursor, end),
                    text.rfind(".", cursor, end),
                    text.rfind("\n", cursor, end),
                )
                if boundary > cursor + self._config.min_chars:
                    end = boundary + 1
            part = text[cursor:end].strip()
            if len(part) >= self._config.min_chars or not chunks:
                chunks.append(self._make_chunk(document, section, part, section.start + cursor, section.start + end, ordinal))
                ordinal += 1
            if end >= len(text):
                break
            cursor = max(0, end - overlap)
        return chunks

    @staticmethod
    def _make_chunk(document: Document, section: Section, text: str, start: int, end: int, ordinal: int) -> Chunk:
        fingerprint = stable_hash(f"{document.id}:{start}:{end}:{text}", 20)
        metadata = dict(document.metadata)
        metadata.update({"chunk_ordinal": ordinal, "section_title": section.title})
        return Chunk(
            id=f"chunk_{fingerprint}",
            document_id=document.id,
            text=text,
            source=document.source,
            title=section.title or document.title,
            section_path=section.path,
            start_char=start,
            end_char=end,
            metadata=metadata,
        )
