"""Output contract validation shared by all business tasks."""

from __future__ import annotations

import json
from typing import Any, Dict

from framework.core.types import BusinessRequest, BusinessResult, ModelResponse


class ContractViolation(ValueError):
    """Raised when a model response violates its declared output contract."""


class ContractResultParser:
    def parse(self, response: ModelResponse, request: BusinessRequest) -> BusinessResult:
        if not response.success and not response.text:
            raise ContractViolation(response.error_message or "model call failed")
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ContractViolation("model response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ContractViolation("model response must be a JSON object")
        self._validate(payload, request)
        known = {"accepted", "label", "score", "reason", "explanation"}
        return BusinessResult(
            request_id=request.request_id,
            accepted=payload["accepted"],
            label=payload["label"],
            score=float(payload["score"]),
            reason=str(payload.get("reason") or payload.get("explanation") or ""),
            details={key: value for key, value in payload.items() if key not in known},
        )

    @staticmethod
    def _validate(payload: Dict[str, Any], request: BusinessRequest) -> None:
        contract = request.output_contract
        missing = [field for field in contract.required_fields if field not in payload]
        if missing:
            raise ContractViolation(f"missing required output fields: {', '.join(missing)}")
        if not isinstance(payload.get("accepted"), bool):
            raise ContractViolation("accepted must be boolean")
        if not isinstance(payload.get("label"), str):
            raise ContractViolation("label must be string")
        try:
            score = float(payload.get("score"))
        except (TypeError, ValueError) as exc:
            raise ContractViolation("score must be numeric") from exc
        lower, upper = contract.score_range
        if not lower <= score <= upper:
            raise ContractViolation(f"score must be between {lower} and {upper}")
        if "reason" in contract.required_fields and not isinstance(payload.get("reason"), str):
            raise ContractViolation("reason must be string")
        if not contract.allow_extra_fields:
            allowed = set(contract.required_fields) | {"explanation"}
            extras = sorted(set(payload) - allowed)
            if extras:
                raise ContractViolation(f"unexpected output fields: {', '.join(extras)}")
