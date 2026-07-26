"""Simple web page loader based on the Python standard library."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import List
from urllib.request import Request, urlopen

from konwledge.core.models import Document
from konwledge.processing.text import stable_hash


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: List[str] = []
        self._title = ""
        self._in_title = False

    @property
    def text(self) -> str:
        return "\n".join(part.strip() for part in self._parts if part.strip())

    @property
    def title(self) -> str:
        return self._title.strip()

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "div", "section", "article", "h1", "h2", "h3", "li"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title += data
        self._parts.append(data)


class WebPageLoader:
    """Load one public web page as a document."""

    def __init__(self, url: str, timeout_seconds: int = 20) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds

    def load(self) -> List[Document]:
        request = Request(self._url, headers={"User-Agent": "rag-knowledge-base/1.0"})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            html = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
        parser = _ReadableHTMLParser()
        parser.feed(html)
        return [
            Document(
                id=f"doc_{stable_hash(self._url, 20)}",
                text=parser.text,
                source=self._url,
                title=parser.title or self._url,
                metadata={"loader": self.__class__.__name__, "content_type": "text/html"},
            )
        ]
