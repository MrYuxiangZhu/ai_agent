"""Example business logic independent from model vendors."""

from __future__ import annotations

from pathlib import Path
from typing import List

from framework.core.types import BusinessRequest, MediaAsset, MediaKind


class SceneQualityRequestBuilder:
    def build_demo_requests(self, image_path: Path) -> List[BusinessRequest]:
        missing_path = image_path.parent / "missing_scene.png"
        return [
            BusinessRequest(
                request_id="scene_quality_valid",
                task_name="scene_quality_check",
                instruction="判断输入图片是否可进入后续视觉理解或训练数据处理链路。",
                media_assets=[MediaAsset(path=image_path, kind=MediaKind.IMAGE, description="自动生成的 demo 图片")],
                context={"business": "scene_quality", "case": "valid_media"},
            ),
            BusinessRequest(
                request_id="scene_quality_missing",
                task_name="scene_quality_check",
                instruction="当图片不存在时，给出稳定的结构化失败结果。",
                media_assets=[MediaAsset(path=missing_path, kind=MediaKind.IMAGE, description="故意缺失的 demo 图片")],
                context={"business": "scene_quality", "case": "missing_media"},
            ),
        ]
