"""Business request builder for the weather forecast demo."""

from __future__ import annotations

from typing import List

from business.weather_forecast.prompt_spec import WEATHER_FORECAST_PROMPT_SPEC
from framework.core.types import BusinessRequest


class WeatherForecastRequestBuilder:
    """把天气业务参数转换成与模型供应商无关的请求。"""

    def build_request(self, city: str, forecast_days: int = 3, concern: str = "通勤、穿衣和户外活动") -> BusinessRequest:
        """构造单个天气预报请求，并将预报天数约束在 1 到 7 天。

        Args:
            city: 需要生成示例预报的城市。
            forecast_days: 用户期望天数，越界值会自动截断。
            concern: 需要模型重点生成建议的生活场景。

        Returns:
            携带天气 Prompt 规范和业务上下文的旧业务请求对象。
        """
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
        """返回覆盖不同城市、天数和用户关注点的演示请求集合。"""
        return [
            self.build_request("杭州", 3, "通勤、穿衣和晨跑"),
            self.build_request("北京", 2, "商务出行和户外活动"),
        ]
