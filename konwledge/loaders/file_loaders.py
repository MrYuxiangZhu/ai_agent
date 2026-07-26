"""File document loaders for TXT, Markdown and PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from konwledge.core.models import Document
from konwledge.processing.text import stable_hash


class TextFileLoader:
    """Load one text-like file as a document."""

    def __init__(self, path: Path, encoding: str = "utf-8") -> None:
        self._path = Path(path)
        self._encoding = encoding

    def load(self) -> List[Document]:
        text = self._path.read_text(encoding=self._encoding)
        return [
            Document(
                id=f"doc_{stable_hash(str(self._path.resolve()), 20)}",
                text=text,
                source=str(self._path),
                title=self._path.stem,
                metadata={"extension": self._path.suffix.lower(), "loader": self.__class__.__name__},
            )
        ]


class PDFFileLoader:
    """Load PDF text through pypdf when available.

    The class is intentionally optional-dependency friendly. Install ``pypdf``
    to enable PDF parsing; otherwise a clear error is raised.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> List[Document]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF loading requires optional dependency: pip install pypdf") from exc

        reader = PdfReader(str(self._path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"\n\n## Page {index}\n\n{text}")
        return [
            Document(
                id=f"doc_{stable_hash(str(self._path.resolve()), 20)}",
                text="\n".join(pages),
                source=str(self._path),
                title=self._path.stem,
                metadata={"extension": ".pdf", "loader": self.__class__.__name__, "page_count": len(reader.pages)},
            )
        ]


class DirectoryLoader:
    """Load all supported files from a directory recursively."""

    def __init__(self, root: Path, extensions: Sequence[str] | None = None) -> None:
        self._root = Path(root)
        self._extensions = {ext.lower() for ext in (extensions or [".md", ".txt", ".pdf"])}

    def load(self) -> List[Document]:
        documents: List[Document] = []
        for path in self._iter_files():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                documents.extend(PDFFileLoader(path).load())
            else:
                documents.extend(TextFileLoader(path).load())
        return documents

    def _iter_files(self) -> Iterable[Path]:
        if not self._root.exists():
            return []
        return sorted(path for path in self._root.rglob("*") if path.is_file() and path.suffix.lower() in self._extensions)
