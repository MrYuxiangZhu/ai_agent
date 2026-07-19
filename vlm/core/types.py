"""VLM 核心稳定协议；业务模块和供应商适配器均依赖这里。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class MediaKind(str, Enum):
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    VIDEO = "video"
    TEXT = "text"


@dataclass(frozen=True)
class MediaAsset:
    path: Path
    kind: MediaKind = MediaKind.IMAGE
    description: str = ""


@dataclass(frozen=True)
class PromptSpec:
    system_prompt: str = "你是一个严谨的视觉语言模型业务助手。"
    constraints: Sequence[str] = field(default_factory=tuple)
    output_instruction: str = "请只输出 JSON，字段包括 accepted、label、score、reason。"


@dataclass(frozen=True)
class VlmRequest:
    request_id: str
    task_name: str
    instruction: str
    media_assets: Sequence[MediaAsset] = field(default_factory=tuple)
    context: Mapping[str, Any] = field(default_factory=dict)
    prompt_spec: PromptSpec = field(default_factory=PromptSpec)


@dataclass(frozen=True)
class VlmResponse:
    content: str
    provider: str
    model: str
    latency_ms: int
    success: bool = True
    error: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VlmResult:
    request_id: str
    accepted: bool
    label: str
    score: float
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VlmRun:
    request: VlmRequest
    prompt: str
    response: VlmResponse
    result: VlmResult


class ModelClient(Protocol):
    def infer(self, prompt: str, request: VlmRequest) -> VlmResponse: ...


class ResultParser(Protocol):
    def parse(self, response: VlmResponse, request: VlmRequest) -> VlmResult: ...
