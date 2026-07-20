"""Scene quality business implemented with the common business template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from framework.core.business import BusinessHandler, register_business
from framework.core.types import (
    BusinessRequest,
    BusinessResult,
    MediaAsset,
    ModelRequirements,
    OutputContract,
    PromptSpec,
)


SCENE_QUALITY_PROMPT_SPEC = PromptSpec(
    role="你是视觉数据质量审核专家。",
    output_format="输出 accepted、label、score、reason 字段。",
    constraints=["必须检查媒体是否有效", "仅输出 JSON"],
)


@dataclass(frozen=True)
class SceneQualityInput:
    image_path: Path
    case: str = "valid_media"


@dataclass(frozen=True)
class SceneQualityResult:
    accepted: bool
    label: str
    score: float
    reason: str


@register_business("scene_quality_check")
class SceneQualityBusiness(BusinessHandler[SceneQualityInput, SceneQualityResult]):
    @property
    def task_name(self) -> str:
        return "scene_quality_check"

    def validate_input(self, business_input: SceneQualityInput) -> None:
        if not isinstance(business_input.image_path, Path):
            raise TypeError("image_path must be pathlib.Path")
        if not business_input.case.strip():
            raise ValueError("case cannot be empty")

    def request_id(self, business_input: SceneQualityInput) -> str:
        return f"scene_quality_{business_input.case}"

    def build_instruction(self, business_input: SceneQualityInput) -> str:
        return "判断图片是否可进入后续视觉理解或训练数据处理链路。"

    def build_context(self, business_input: SceneQualityInput) -> Dict[str, Any]:
        return {"business": "scene_quality", "case": business_input.case}

    def media_assets(self, business_input: SceneQualityInput) -> Iterable[MediaAsset]:
        return [MediaAsset(business_input.image_path, description="场景质量审核图片")]

    def prompt_spec(self) -> PromptSpec:
        return SCENE_QUALITY_PROMPT_SPEC

    def output_contract(self) -> OutputContract:
        return OutputContract()

    def model_requirements(self) -> ModelRequirements:
        return ModelRequirements(modalities={"text", "image"}, structured_output=True)

    def map_result(self, result: BusinessResult) -> SceneQualityResult:
        return SceneQualityResult(result.accepted, result.label, result.score, result.reason)


class SceneQualityRequestBuilder:
    """Compatibility facade for callers that still expect request objects."""

    def __init__(self) -> None:
        self._business = SceneQualityBusiness()

    def build_request(self, image_path: Path, case: str = "valid_media") -> BusinessRequest:
        return self._business.build_request(SceneQualityInput(image_path, case))

    def build_demo_requests(self, image_path: Path) -> List[BusinessRequest]:
        missing_path = image_path.parent / "missing_scene.png"
        return [
            self.build_request(image_path, "valid_media"),
            self.build_request(missing_path, "missing_media"),
        ]
