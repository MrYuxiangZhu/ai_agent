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
        base_url = self.base_url.rstrip("/")
        return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if not self.api_key_base64:
            return None
        try:
            return base64.b64decode(self.api_key_base64, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("api_key_base64 不是有效的 UTF-8 Base64 字符串") from exc

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **dict(self.extra_headers)}
        api_key = self.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.bill_account:
            headers.setdefault("X-Bill-Account", self.bill_account)
        return headers

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "bill_account": self.bill_account,
            "timeout_seconds": self.timeout_seconds,
            "has_api_key": bool(self.api_key or self.api_key_base64),
        }


def load_api_configs(path: Path = DEFAULT_CONFIG_PATH) -> tuple[str, dict[str, OpenaiApiConfig]]:
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
    default_provider, configs = load_api_configs(path)
    selected = provider or default_provider
    if selected not in configs:
        raise ValueError(f"未知 provider '{selected}'，可选值：{', '.join(sorted(configs))}")
    return configs[selected]
