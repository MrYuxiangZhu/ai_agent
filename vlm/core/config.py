"""OpenAI 兼容模型服务配置。"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm_services.json"


@dataclass(frozen=True)
class OpenaiApiConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_base64: str | None = None
    bill_account: str | None = None
    timeout_seconds: float = 300.0
    provider: str = "custom"
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        """返回最终 Chat Completions 地址，避免调用方重复拼接路径。"""
        base_url = self.base_url.rstrip("/")
        return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    def resolved_api_key(self) -> str | None:
        """解析认证密钥，明文字段优先，其次解码 Base64 兼容字段。

        Returns:
            可用于 Bearer 认证的字符串；未配置时返回 ``None``。

        Raises:
            ValueError: Base64 内容无效或解码结果不是 UTF-8 字符串。

        Base64 仅用于接口兼容，不提供加密保护，生产环境应优先使用环境变量。
        """
        if self.api_key:
            return self.api_key
        if not self.api_key_base64:
            return None
        try:
            return base64.b64decode(self.api_key_base64, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("api_key_base64 不是有效的 UTF-8 Base64 字符串") from exc

    def headers(self) -> dict[str, str]:
        """构造 HTTP 请求头，合并认证、计费账号和用户扩展字段。

        扩展字段可覆盖默认 Content-Type；Authorization 只在配置密钥后加入，
        ``X-Bill-Account`` 使用 ``setdefault`` 以尊重用户显式传入的同名 Header。
        """
        headers = {"Content-Type": "application/json", **dict(self.extra_headers)}
        api_key = self.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.bill_account:
            headers.setdefault("X-Bill-Account", self.bill_account)
        return headers

    def safe_dict(self) -> dict[str, Any]:
        """返回适合日志和诊断输出的脱敏配置摘要，不暴露任何密钥值。"""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "bill_account": self.bill_account,
            "timeout_seconds": self.timeout_seconds,
            "has_api_key": bool(self.api_key or self.api_key_base64),
        }


def load_api_configs(path: Path = DEFAULT_CONFIG_PATH) -> tuple[str, dict[str, OpenaiApiConfig]]:
    """加载供应商配置，并用对应环境变量覆盖可部署字段。

    Args:
        path: JSON 配置文件路径。

    Returns:
        默认供应商名称及按 provider 索引的配置字典。配置文件不存在时返回
        可立即运行的本地 mock 配置。

    环境变量格式为 ``VLM_<PROVIDER>_<FIELD>``，用于避免将生产密钥提交到仓库。
    """
    if not path.exists():
        return "mock", {"mock": OpenaiApiConfig("mock://local", "mock-vlm", provider="mock")}
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs: dict[str, OpenaiApiConfig] = {}
    for item in payload.get("providers", []):
        provider = str(item["provider"])
        env_prefix = f"VLM_{provider.upper()}"
        configs[provider] = OpenaiApiConfig(
            provider=provider,
            base_url=os.getenv(f"{env_prefix}_BASE_URL", str(item["base_url"])),
            model=os.getenv(f"{env_prefix}_MODEL", str(item["model"])),
            api_key=os.getenv(f"{env_prefix}_API_KEY", item.get("api_key") or "") or None,
            api_key_base64=os.getenv(f"{env_prefix}_API_KEY_BASE64", item.get("api_key_base64") or "") or None,
            bill_account=os.getenv(f"{env_prefix}_BILL_ACCOUNT", item.get("bill_account") or "") or None,
            timeout_seconds=float(os.getenv(f"{env_prefix}_TIMEOUT", item.get("timeout_seconds", 300.0))),
            extra_headers=item.get("extra_headers", {}),
            options=item.get("options", {}),
        )
    default_provider = os.getenv("VLM_PROVIDER", str(payload.get("default_provider", "mock")))
    if default_provider not in configs:
        raise ValueError(f"默认 provider '{default_provider}' 未配置")
    return default_provider, configs


def pick_api_config(provider: str = "", path: Path = DEFAULT_CONFIG_PATH) -> OpenaiApiConfig:
    """选择指定供应商配置；名称为空时使用配置文件的默认供应商。

    Raises:
        ValueError: 目标供应商未配置，并在异常消息中列出所有可选值。
    """
    default_provider, configs = load_api_configs(path)
    selected = provider or default_provider
    if selected not in configs:
        raise ValueError(f"未知 provider '{selected}'，可选值：{', '.join(sorted(configs))}")
    return configs[selected]
