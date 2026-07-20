"""与模型厂商无关的提示词构造。"""

from __future__ import annotations

import json

from vlm.core.types import VlmRequest


class VlmPromptBuilder:
    """将统一请求转换成不依赖具体模型供应商的中文提示词。"""

    def build(self, request: VlmRequest) -> str:
        """按稳定章节组织系统角色、业务任务、媒体、上下文和输出约束。

        Args:
            request: 业务层构造的统一 VLM 请求。

        Returns:
            可直接交给模型适配器的完整提示词。媒体仅写入描述性元数据，真实
            二进制内容由供应商适配器编码，避免 Prompt 中出现大段 Base64。
        """
        media = [
            {"path": str(asset.path), "kind": asset.kind.value, "description": asset.description}
            for asset in request.media_assets
        ]
        parts = [
            request.prompt_spec.system_prompt,
            f"业务任务：{request.task_name}",
            f"任务要求：{request.instruction}",
            f"媒体输入：{json.dumps(media, ensure_ascii=False)}",
            f"业务上下文：{json.dumps(dict(request.context), ensure_ascii=False)}",
        ]
        if request.prompt_spec.constraints:
            parts.append("约束条件：\n- " + "\n- ".join(request.prompt_spec.constraints))
        parts.append(f"输出格式：{request.prompt_spec.output_instruction}")
        return "\n".join(parts)
