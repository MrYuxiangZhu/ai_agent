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
    """无需网络的确定性模型客户端，用于本地开发和自动化测试。"""

    def __init__(self, config: OpenaiApiConfig) -> None:
        """保存模型元数据，以便模拟响应与真实客户端具有相同结构。"""
        self.config = config

    def infer(self, prompt: str, request: VlmRequest) -> VlmResponse:
        """检查请求中的本地媒体是否存在，并生成确定性的 JSON 响应。

        Args:
            prompt: 已构造的提示词，仅用于模拟统计而不执行推理。
            request: 包含请求标识和媒体列表的统一请求。

        Returns:
            媒体全部存在时返回成功结果，否则返回 ``missing_media``。
        """
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
        """绑定端点、模型、认证头、超时和生成参数等不可变配置。"""
        self.config = config

    def infer(self, prompt: str, request: VlmRequest) -> VlmResponse:
        """调用 OpenAI Chat Completions 兼容端点并标准化响应。

        文本请求发送普通字符串，多模态请求发送 text 与 image_url 内容块。
        网络错误、超时、非预期 JSON 或响应字段缺失均转换成失败的
        ``VlmResponse``，从而让 Runtime 能执行统一重试和供应商降级。
        """
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
        """从 Chat Completions 响应中提取文本，并兼容字符串和内容块列表。"""
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)

    @staticmethod
    def _content(prompt: str, request: VlmRequest) -> str | list[dict[str, Any]]:
        """将统一文本和本地图片转换为 OpenAI 兼容的消息内容。

        没有图片时返回简单字符串以保持最小请求体；有图片时读取文件、推断
        MIME 类型并编码为 Data URL，使千问、豆包等兼容端点可直接消费。
        """
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
    """根据配置选择本地模拟或真实 OpenAI 兼容客户端。"""

    @staticmethod
    def create(config: OpenaiApiConfig):
        """创建符合 ``ModelClient`` 协议的客户端。

        ``mock`` provider 和 ``mock://`` 地址用于测试，其余配置统一走兼容协议，
        因而无需在业务代码中判断千问、豆包或企业代理服务。
        """
        if config.provider == "mock" or config.base_url.startswith("mock://"): 
            return MockVlmClient(config)
        return OpenaiCompatibleVlmClient(config)
