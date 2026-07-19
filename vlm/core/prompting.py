"""与模型厂商无关的提示词构造。"""

from __future__ import annotations

import json

from vlm.core.types import VlmRequest


class VlmPromptBuilder:
    def build(self, request: VlmRequest) -> str:
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
