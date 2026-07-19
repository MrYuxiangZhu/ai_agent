"""Generic prompt builder shared by business modules."""

from __future__ import annotations

from framework.core.types import BusinessRequest, PromptSpec


class JsonPromptBuilder:
    def build_prompt(self, request: BusinessRequest) -> str:
        spec = request.prompt_spec or PromptSpec()
        media_lines = []
        for idx, asset in enumerate(request.media_assets, start=1):
            suffix = f"，说明：{asset.description}" if asset.description else ""
            media_lines.append(f"{idx}. {asset.kind.value}: {asset.path}{suffix}")
        context_lines = [f"- {key}: {value}" for key, value in sorted(request.context.items())]
        parts = [
            f"角色：{spec.role}",
            f"业务任务：{request.task_name}",
            f"任务指令：{request.instruction}",
            "媒体输入：",
            "\n".join(media_lines) if media_lines else "无媒体输入",
            "上下文：",
            "\n".join(context_lines) if context_lines else "无额外上下文",
        ]
        if spec.examples:
            parts.append("示例：")
            for idx, example in enumerate(spec.examples, start=1):
                parts.append(f"  输入{idx}：{example.input_text}")
                parts.append(f"  输出{idx}：{example.output_text}")
        if spec.output_format:
            parts.append(f"输出格式：{spec.output_format}")
        if spec.constraints:
            parts.append("约束条件：")
            for constraint in spec.constraints:
                parts.append(f"  - {constraint}")
        return "\n".join(parts)
