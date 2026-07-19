"""Default parser for JSON-like model responses."""

from __future__ import annotations

import json

from framework.core.types import BusinessRequest, BusinessResult, ModelResponse


class JsonBusinessResultParser:
    def parse(self, response: ModelResponse, request: BusinessRequest) -> BusinessResult:
        if not response.success and not response.text:
            return BusinessResult(
                request_id=request.request_id,
                accepted=False,
                label="model_error",
                score=0.0,
                reason=response.error_message or "model call failed",
                details={"provider": response.provider, "model": response.model},
            )
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            return BusinessResult(
                request_id=request.request_id,
                accepted=False,
                label="parse_error",
                score=0.0,
                reason="model response is not valid JSON",
                details={"raw_text": response.text[:500]},
            )
        return BusinessResult(
            request_id=request.request_id,
            accepted=bool(payload.get("accepted", False)),
            label=str(payload.get("label", "unknown")),
            score=float(payload.get("score", 0.0)),
            reason=str(payload.get("reason") or payload.get("explanation") or ""),
            details={key: value for key, value in payload.items() if key not in {"accepted", "label", "score", "reason", "explanation"}},
        )
