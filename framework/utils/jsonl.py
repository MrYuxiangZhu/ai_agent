"""JSONL persistence utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from framework.core.types import RunEnvelope


def write_envelopes_jsonl(envelopes: Iterable[RunEnvelope], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for envelope in envelopes:
            file_obj.write(json.dumps(_to_jsonable(envelope), ensure_ascii=False) + "\n")
    return output_path


def _to_jsonable(value):
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
