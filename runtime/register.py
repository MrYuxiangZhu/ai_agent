"""Runtime composition and decorator-driven business discovery."""

from __future__ import annotations

import business.scene_quality.service
import business.weather_forecast.service
from framework.core.business import BusinessManager
from framework.core.config_loader import load_model_profiles
from framework.core.prompt import JsonPromptBuilder
from framework.core.routing import CapabilityRouter
from framework.core.workflow import WorkflowRuntime

_profiles = load_model_profiles()
workflow_runtime = WorkflowRuntime(
    prompt_builder=JsonPromptBuilder(),
    router=CapabilityRouter(_profiles.values()),
)
business_manager = BusinessManager(workflow_runtime)

__all__ = ["business_manager", "workflow_runtime"]
