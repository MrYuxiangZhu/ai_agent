"""Robust parser for JSON model responses."""

import json
import re
from typing import Any

from vlm.core.types import VlmRequest, VlmResponse, VlmResult


class VlmJsonResultParser:
    def parse(self, response: VlmResponse, request: VlmRequest) -> VlmResult:
        if not response.success and not response.content:
            return VlmResult(request.request_id, False, "model_error", 0.0,
                             response.error or "model call failed",
                             {"provider": response.provider, "model": response.model})
        payload = self._json_object(response.content)
        if payload is None:
            return VlmResult(request.request_id, False, "parse_error", 0.0,
                             "model response is not a JSON object", {"raw_text": response.text[:500]})
        excluded = {"accepted", "label", "score", "reason", "explanation"}
        return VlmResult(request.request_id, bool(payload.get("accepted", True)),
                         str(payload.get("label", "completed")), self._score(payload.get("score", 1.0)),
                         str(payload.get("reason") or payload.get("explanation") or ""),
                         {key: value for key, value in payload.items() if key not in excluded})

    @staticmethod
    def _json_object(text: str) -> dict[str, Any] | None:
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
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
