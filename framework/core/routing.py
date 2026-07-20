"""Capability-based model routing with deterministic fallback ordering."""

from __future__ import annotations

from typing import Iterable, List

from framework.core.types import BusinessRequest, ModelServiceProfile


class NoCompatibleModelError(LookupError):
    pass


class CapabilityRouter:
    def __init__(self, profiles: Iterable[ModelServiceProfile]) -> None:
        self._profiles = tuple(profiles)

    def select(self, request: BusinessRequest) -> List[ModelServiceProfile]:
        requirements = set(request.model_requirements.modalities)
        if request.model_requirements.structured_output:
            requirements.add("structured_output")
        compatible = [profile for profile in self._profiles if requirements <= profile.capabilities]
        compatible.sort(key=lambda profile: profile.priority)
        if not compatible:
            needed = ", ".join(sorted(requirements))
            raise NoCompatibleModelError(f"no model supports required capabilities: {needed}")
        return compatible
