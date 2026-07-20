"""场景质量业务，只负责业务请求，不依赖具体模型厂商。"""

from __future__ import annotations

from pathlib import Path

from vlm.core.types import MediaAsset, PromptSpec, VlmRequest


class SceneQualityRequestBuilder:
    """构造视觉素材质量检查业务请求。"""

    def build_demo_requests(self, image_path: Path) -> list[VlmRequest]:
        """创建正常媒体与缺失媒体两个请求，用于验证成功和异常链路。

        Args:
            image_path: 正常演示图片路径；缺失案例会在同目录构造不存在的路径。

        Returns:
            共享同一质量审核 Prompt 规范的两个统一 VLM 请求。
        """
        missing_path = image_path.parent / "missing_scene.png"
        spec = PromptSpec(
            system_prompt="你是视觉数据质量审核专家。",
            constraints=("必须检查媒体是否有效", "仅输出 JSON"),
            output_instruction="输出 accepted、label、score、reason 字段。",
        )
        return [
            VlmRequest("scene_quality_valid", "scene_quality_check",
                       "判断图片是否可进入后续视觉理解或训练数据处理链路。",
                       (MediaAsset(image_path, description="自动生成的 demo 图片"),),
                       {"business": "scene_quality", "case": "valid_media"}, spec),
            VlmRequest("scene_quality_missing", "scene_quality_check",
                       "图片不存在时给出稳定的结构化失败结果。",
                       (MediaAsset(missing_path, description="故意缺失的 demo 图片"),),
                       {"business": "scene_quality", "case": "missing_media"}, spec),
        ]
