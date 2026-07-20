"""Weather forecast business implemented as a framework handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from business.weather_forecast.prompt_spec import WEATHER_FORECAST_PROMPT_SPEC
from framework.core.business import BusinessHandler, register_business
from framework.core.types import BusinessResult, ModelRequirements, OutputContract, PromptSpec


@dataclass(frozen=True)
class WeatherForecastInput:
    city: str
    forecast_days: int = 3
    concern: str = "通勤、穿衣和户外活动"


@dataclass(frozen=True)
class WeatherForecastResult:
    city: str
    forecast_days: int
    daily: List[Dict[str, Any]]
    score: float
    reason: str


@register_business("weather_forecast")
class WeatherForecastBusiness(BusinessHandler[WeatherForecastInput, WeatherForecastResult]):
    @property
    def task_name(self) -> str:
        return "weather_forecast"

    def validate_input(self, business_input: WeatherForecastInput) -> None:
        if not business_input.city.strip():
            raise ValueError("city cannot be empty")
        if not 1 <= business_input.forecast_days <= 7:
            raise ValueError("forecast_days must be between 1 and 7")

    def request_id(self, business_input: WeatherForecastInput) -> str:
        return f"weather_forecast_{business_input.city}_{business_input.forecast_days}d"

    def build_instruction(self, business_input: WeatherForecastInput) -> str:
        return f"为 {business_input.city} 生成未来 {business_input.forecast_days} 天的天气预报，并给出生活建议。"

    def build_context(self, business_input: WeatherForecastInput) -> Dict[str, Any]:
        return {"business": self.task_name, "city": business_input.city,
                "forecast_days": business_input.forecast_days, "concern": business_input.concern,
                "data_policy": "demo_only_not_realtime_weather"}

    def prompt_spec(self) -> PromptSpec:
        return WEATHER_FORECAST_PROMPT_SPEC

    def output_contract(self) -> OutputContract:
        return OutputContract(required_fields=("accepted", "label", "score", "reason", "city", "forecast_days", "daily"))

    def model_requirements(self) -> ModelRequirements:
        return ModelRequirements(modalities={"text"}, structured_output=True)

    def map_result(self, result: BusinessResult) -> WeatherForecastResult:
        return WeatherForecastResult(str(result.details["city"]), int(result.details["forecast_days"]),
                                     list(result.details["daily"]), result.score, result.reason)
