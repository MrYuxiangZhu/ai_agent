"""模型供应商适配器；业务层不应直接依赖本模块。"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vlm.core.config import OpenaiApiConfig
from vlm.core.types import MediaKind, VlmRequest, VlmResponse


class MockVlmClient:
    def __init__(self, config: OpenaiApiConfig) -> None:
        self.config = config

    def infer(self, prompt: str, request: VlmRequest) -> VlmResponse:
        started = time.perf_counter()
        missing = [str(asset.path) for asset in request.media_assets if not asset.path.exists()]
        accepted = not missing
        content = json.dumps({
            "accepted": accepted,
            "label": "ready" if accepted else "missing_media",
            "score": 0.92 if accepted else 0.03,
            "reason": f"request={request.request_id}, missing={missing}",
        }, ensure_ascii=False)
        return VlmResponse(content, self.config.provider, self.config.model,
                           int((time.perf_counter() - started) * 1000), accepted,
                           ";".join(missing) or None)


class OpenaiCompatibleVlmClient:
    """支持千问、豆包及其他 OpenAI Chat Completions 兼容服务。"""

    def __init__(self, config: OpenaiApiConfig) -> None:
        self.config = config

    def infer(self, prompt: str, request: VlmRequest) -> VlmResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": self._content(prompt, request)}],
            **dict(self.config.options),
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.config.endpoint, data=data, headers=self.config.headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = self._extract_content(raw)
            return VlmResponse(content, self.config.provider, self.config.model,
                               int((time.perf_counter() - started) * 1000), raw=raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            return VlmResponse("", self.config.provider, self.config.model,
                               int((time.perf_counter() - started) * 1000), False, str(exc))

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)

    @staticmethod
    def _content(prompt: str, request: VlmRequest) -> str | list[dict[str, Any]]:
        images = [asset for asset in request.media_assets if asset.kind == MediaKind.IMAGE]
        if not images:
            return prompt
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for asset in images:
            mime = mimetypes.guess_type(asset.path.name)[0] or "image/jpeg"
            encoded = base64.b64encode(asset.path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return content


class VlmClientFactory:
    @staticmethod
    def create(config: OpenaiApiConfig):
        if config.provider == "mock" or config.base_url.startswith("mock://"):
            return MockVlmClient(config)
        return OpenaiCompatibleVlmClient(config)
