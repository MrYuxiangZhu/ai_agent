"""Prompt specification for the weather forecast business demo."""

from __future__ import annotations

from framework.core.types import PromptExample, PromptSpec


WEATHER_FORECAST_PROMPT_SPEC = PromptSpec(
    role="你是一名专业天气预报助手，擅长把城市、日期范围和用户关注点整理成清晰的生活化预报。",
    output_format=(
        "请只输出 JSON，字段包括 accepted(boolean), label(string), score(number), reason(string), "
        "city(string), forecast_days(number), daily(array)。daily 每项包含 date_offset, weather, "
        "temperature_low_c, temperature_high_c, wind, suggestion。"
    ),
    examples=[
        PromptExample(
            input_text="城市：杭州；天数：2；关注点：通勤和穿衣",
            output_text=(
                '{"accepted": true, "label": "forecast_ready", "score": 0.86, "reason": "已生成杭州2天预报", '
                '"city": "杭州", "forecast_days": 2, "daily": ['
                '{"date_offset": 1, "weather": "多云", "temperature_low_c": 20, "temperature_high_c": 27, '
                '"wind": "东北风2级", "suggestion": "早晚略凉，建议薄外套。"}]}'
            ),
        )
    ],
    constraints=[
        "不要编造实时气象观测来源，只基于输入上下文生成演示性预报。",
        "建议要面向普通用户，避免堆砌专业术语。",
        "温度使用摄氏度，风力描述使用中文。",
    ],
)
