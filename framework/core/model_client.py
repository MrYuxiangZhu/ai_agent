"""Model clients and factory.

Only model-related integration code lives in this module. Business packages use
``ModelClient`` through the stable contracts in ``framework.core.types``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict

from framework.core.types import BusinessRequest, ModelResponse, ModelServiceProfile


class MockModelClient:
    def __init__(self, profile: ModelServiceProfile) -> None:
        self._profile = profile

    def infer(self, prompt: str, request: BusinessRequest) -> ModelResponse:
        started = time.perf_counter()
        missing = [str(asset.path) for asset in request.media_assets if not asset.path.exists()]
        accepted = len(missing) == 0
        body = self._build_body(request, accepted, missing)
        return ModelResponse(
            text=json.dumps(body, ensure_ascii=False),
            provider=self._profile.provider,
            model=self._profile.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            success=accepted,
            error_message=";".join(missing) if missing else None,
            raw={"prompt_chars": len(prompt)},
        )

    def _build_body(self, request: BusinessRequest, accepted: bool, missing: list) -> dict:
        if request.task_name == "weather_forecast":
            city = request.context.get("city", "未知城市")
            forecast_days = int(request.context.get("forecast_days", 3))
            return {
                "accepted": True,
                "label": "forecast_ready",
                "score": 0.88,
                "reason": f"已基于示例规则生成 {city} 未来 {forecast_days} 天天气预报。",
                "city": city,
                "forecast_days": forecast_days,
                "daily": [
                    {
                        "date_offset": day,
                        "weather": "晴转多云" if day % 2 else "多云",
                        "temperature_low_c": 18 + day,
                        "temperature_high_c": 26 + day,
                        "wind": "东北风 2 级",
                        "suggestion": "适合通勤和户外活动，注意补水。",
                    }
                    for day in range(1, forecast_days + 1)
                ],
            }
        return {
            "accepted": accepted,
            "label": "ready" if accepted else "missing_media",
            "score": 0.92 if accepted else 0.03,
            "reason": f"request={request.request_id}, media_count={len(request.media_assets)}, missing={missing}",
        }


class HttpJsonModelClient:
    def __init__(self, profile: ModelServiceProfile) -> None:
        self._profile = profile

    def infer(self, prompt: str, request: BusinessRequest) -> ModelResponse:
        started = time.perf_counter()
        payload = {
            "provider": self._profile.provider,
            "model": self._profile.model,
            "prompt": prompt,
            "request_id": request.request_id,
            "task_name": request.task_name,
            "media_assets": [
                {"path": str(asset.path), "kind": asset.kind.value, "description": asset.description}
                for asset in request.media_assets
            ],
            "context": request.context,
            "options": self._profile.options,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._profile.token:
            headers["Authorization"] = f"Bearer {self._profile.token}"
        http_request = urllib.request.Request(self._profile.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(http_request, timeout=self._profile.timeout_seconds) as response:
                text = response.read().decode("utf-8")
            return ModelResponse(
                text=text,
                provider=self._profile.provider,
                model=self._profile.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            return ModelResponse(
                text="",
                provider=self._profile.provider,
                model=self._profile.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                success=False,
                error_message=str(exc),
            )


class ModelClientFactory:
    @staticmethod
    def create(profile: ModelServiceProfile):
        builders = ModelClientFactory._builders()
        if profile.transport not in builders:
            supported = ", ".join(sorted(builders))
            raise ValueError(f"unsupported transport '{profile.transport}', supported: {supported}")
        return builders[profile.transport](profile)

    @staticmethod
    def _builders() -> Dict[str, object]:
        return {
            "mock": MockModelClient,
            "http_json": HttpJsonModelClient,
        }
