"""Configuration loader for model service profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from framework.core.types import ModelServiceProfile


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_services.json"


def load_model_profiles(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, ModelServiceProfile]:
    if not config_path.exists():
        return _default_profiles()
    with config_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    profiles = {}
    for item in payload.get("profiles", []):
        profile = ModelServiceProfile(
            provider=item["provider"],
            model=os.getenv(f"EXAMPLE_{item['provider'].upper()}_MODEL", item.get("model", "")),
            endpoint=os.getenv(f"EXAMPLE_{item['provider'].upper()}_ENDPOINT", item.get("endpoint", "")),
            timeout_seconds=float(os.getenv(f"EXAMPLE_{item['provider'].upper()}_TIMEOUT", item.get("timeout_seconds", 60))),
            token=os.getenv(f"EXAMPLE_{item['provider'].upper()}_TOKEN", item.get("token") or "") or None,
            transport=item.get("transport", "http_json"),
            options=item.get("options", {}),
        )
        profiles[profile.provider] = profile
    return profiles or _default_profiles()


def pick_model_profile(provider: str = "mock", config_path: Path = DEFAULT_CONFIG_PATH) -> ModelServiceProfile:
    profiles = load_model_profiles(config_path)
    selected = provider or os.getenv("EXAMPLE_MODEL_PROVIDER", "mock")
    if selected not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"unknown provider '{selected}', available: {available}")
    return profiles[selected]


def _default_profiles() -> Dict[str, ModelServiceProfile]:
    return {
        "mock": ModelServiceProfile(
            provider="mock",
            model="local-deterministic-model",
            endpoint="local://mock",
            timeout_seconds=1,
            transport="mock",
        )
    }
