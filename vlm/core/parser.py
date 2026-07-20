"""Robust parser for JSON model responses."""

import json
import re
from typing import Any

from vlm.core.types import VlmRequest, VlmResponse, VlmResult


class VlmJsonResultParser:
    """容错解析模型 JSON 输出并生成字段稳定的业务结果。"""

    def parse(self, response: VlmResponse, request: VlmRequest) -> VlmResult:
        """解析一个模型响应，区分调用错误、格式错误和正常业务结果。

        Args:
            response: 供应商适配器返回的标准响应。
            request: 原始请求，用于关联 ``request_id``。

        Returns:
            标准化结果；未知业务字段保留在 ``details``，分数被限制在 0 到 1。
        """
        if not response.success and not response.content:
            return VlmResult(request.request_id, False, "model_error", 0.0,
                             response.error or "model call failed",
                             {"provider": response.provider, "model": response.model})
        payload = self._json_object(response.content)
        if payload is None:
            return VlmResult(request.request_id, False, "parse_error", 0.0,
                             "model response is not a JSON object", {"raw_text": response.content[:500]})
        excluded = {"accepted", "label", "score", "reason", "explanation"}
        return VlmResult(request.request_id, bool(payload.get("accepted", True)),
                         str(payload.get("label", "completed")), self._score(payload.get("score", 1.0)),
                         str(payload.get("reason") or payload.get("explanation") or ""),
                         {key: value for key, value in payload.items() if key not in excluded})

    @staticmethod
    def _json_object(text: str) -> dict[str, Any] | None:
        """提取 JSON 对象，并容忍模型附加 Markdown 或解释性文本。

        首先尝试解析完整字符串；失败后提取首尾大括号包围的内容再次解析。
        最终值不是对象或仍无法解析时返回 ``None``。
        """
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return None
            try:
                value = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _score(value: Any) -> float:
        """将任意数值兼容输入转换为 0 到 1 的浮点置信度。"""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
