"""Business request builder for the weather forecast demo."""

from __future__ import annotations

from typing import List

from business.weather_forecast.prompt_spec import WEATHER_FORECAST_PROMPT_SPEC
from framework.core.types import BusinessRequest


class WeatherForecastRequestBuilder:
    def build_request(self, city: str, forecast_days: int = 3, concern: str = "通勤、穿衣和户外活动") -> BusinessRequest:
        normalized_days = max(1, min(forecast_days, 7))
        return BusinessRequest(
            request_id=f"weather_forecast_{city}_{normalized_days}d",
            task_name="weather_forecast",
            instruction=f"为 {city} 生成未来 {normalized_days} 天的天气预报，并给出生活建议。",
            context={
                "business": "weather_forecast",
                "city": city,
                "forecast_days": normalized_days,
                "concern": concern,
                "data_policy": "demo_only_not_realtime_weather",
            },
            prompt_spec=WEATHER_FORECAST_PROMPT_SPEC,
        )

    def build_demo_requests(self) -> List[BusinessRequest]:
        return [
            self.build_request("杭州", 3, "通勤、穿衣和晨跑"),
            self.build_request("北京", 2, "商务出行和户外活动"),
        ]
