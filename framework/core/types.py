"""Stable data contracts for model-agnostic business inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple


@dataclass(frozen=True)
class PromptExample:
    input_text: str
    output_text: str


@dataclass(frozen=True)
class PromptSpec:
    role: str = "你是一个可替换大模型后端的业务分析助手。"
    output_format: str = "请只输出 JSON，字段包括 accepted(boolean), label(string), score(number), reason(string)。"
    examples: List[PromptExample] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class MediaKind(str, Enum):
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    VIDEO = "video"
    TEXT_ONLY = "text_only"


@dataclass(frozen=True)
class MediaAsset:
    path: Path
    kind: MediaKind = MediaKind.IMAGE
    description: str = ""


@dataclass(frozen=True)
class OutputContract:
    required_fields: Tuple[str, ...] = ("accepted", "label", "score", "reason")
    score_range: Tuple[float, float] = (0.0, 1.0)
    allow_extra_fields: bool = True


@dataclass(frozen=True)
class ModelRequirements:
    modalities: Set[str] = field(default_factory=lambda: {"text"})
    structured_output: bool = True
    max_latency_ms: Optional[int] = None
    max_cost: Optional[float] = None


@dataclass(frozen=True)
class BusinessRequest:
    request_id: str
    task_name: str
    instruction: str
    media_assets: List[MediaAsset] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    prompt_spec: Optional[PromptSpec] = None
    output_contract: OutputContract = field(default_factory=OutputContract)
    model_requirements: ModelRequirements = field(default_factory=ModelRequirements)


@dataclass(frozen=True)
class ModelServiceProfile:
    provider: str
    model: str
    endpoint: str
    timeout_seconds: float = 60.0
    token: Optional[str] = None
    transport: str = "mock"
    options: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=lambda: {"text", "structured_output"})
    priority: int = 100


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    success: bool = True
    error_message: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BusinessResult:
    request_id: str
    accepted: bool
    label: str
    score: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunEvent:
    name: str
    trace_id: str
    request_id: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunEnvelope:
    request: BusinessRequest
    prompt: str
    response: ModelResponse
    result: BusinessResult
    trace_id: str = ""
    attempts: int = 1
    events: Tuple[RunEvent, ...] = ()


EventListener = Callable[[RunEvent], None]


class PromptBuilder(Protocol):
    def build_prompt(self, request: BusinessRequest) -> str:
        """Build model prompt for one business request."""


class ModelClient(Protocol):
    def infer(self, prompt: str, request: BusinessRequest) -> ModelResponse:
        """Call a large-model backend."""


class ResultParser(Protocol):
    def parse(self, response: ModelResponse, request: BusinessRequest) -> BusinessResult:
        """Convert model output into business result."""
