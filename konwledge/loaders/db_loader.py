"""Database loaders. SQLite is implemented; other databases can use the same interface."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Mapping, Sequence

from konwledge.core.models import Document
from konwledge.processing.text import stable_hash


class SQLiteQueryLoader:
    """Load rows from SQLite and render each row as a document."""

    def __init__(
        self,
        database_path: Path,
        query: str,
        text_columns: Sequence[str],
        title_column: str | None = None,
        metadata_columns: Sequence[str] | None = None,
    ) -> None:
        self._database_path = Path(database_path)
        self._query = query
        self._text_columns = list(text_columns)
        self._title_column = title_column
        self._metadata_columns = list(metadata_columns or [])

    def load(self) -> List[Document]:
        with sqlite3.connect(str(self._database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(self._query).fetchall()
        return [self._row_to_document(dict(row), index) for index, row in enumerate(rows)]

    def _row_to_document(self, row: Mapping[str, object], index: int) -> Document:
        text = "\n".join(str(row.get(column, "")) for column in self._text_columns if row.get(column) is not None)
        title = str(row.get(self._title_column, "")) if self._title_column else f"row-{index}"
        metadata = {column: row.get(column) for column in self._metadata_columns}
        metadata.update({"loader": self.__class__.__name__, "row_index": index})
        fingerprint = stable_hash(f"{self._database_path}:{self._query}:{index}:{text}", 20)
        return Document(
            id=f"doc_{fingerprint}",
            text=text,
            source=f"sqlite://{self._database_path}",
            title=title,
            metadata=metadata,
        )
