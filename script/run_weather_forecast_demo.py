"""天气预报演示；直接修改下方配置后运行本文件。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business.weather_forecast.service import WeatherForecastRequestBuilder
from vlm import OpenaiApiConfig, RequestContext, create_vlm
from vlm.core import PromptSpec, VlmRequest


# 在这里选择和配置模型。将 ACTIVE_MODEL 改成 QWEN_CONFIG 或 DOUBAO_CONFIG。
QWEN_CONFIG = OpenaiApiConfig(
    base_url="http://llmapi.hobot.cc/v1",
    model="qwen3-vl-plus",
    bill_account="yuxiang.zhu",
    provider="qwen",
)
DOUBAO_CONFIG = OpenaiApiConfig(
    api_key="your-doubao-api-key",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    model="your-doubao-endpoint-id",
    provider="doubao",
)
ACTIVE_MODEL = QWEN_CONFIG

CITY = "杭州"
FORECAST_DAYS = 3
CONCERN = "通勤、穿衣和户外活动"
WORK_DIR = "/tmp/example_weather_forecast_work"


def main() -> int:
    """读取文件顶部业务和模型配置，执行天气建议生成并保存原始响应。

    业务 Prompt 会从旧天气请求规范转换为统一 ``VlmRequest``；调用方只需在
    本文件顶部切换 ``ACTIVE_MODEL``，无需传递命令行参数。
    """
    work_dir = Path(WORK_DIR)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    business_request = WeatherForecastRequestBuilder().build_request(CITY, FORECAST_DAYS, CONCERN)
    spec = business_request.prompt_spec
    request = VlmRequest(
        request_id=business_request.request_id,
        task_name=business_request.task_name,
        instruction=business_request.instruction,
        context=business_request.context,
        prompt_spec=PromptSpec(
            system_prompt=spec.role if spec else "你是一名专业天气预报助手。",
            constraints=tuple(spec.constraints) if spec else (),
            output_instruction=spec.output_format if spec else "请只输出 JSON。",
        ),
    )
    run = create_vlm(ACTIVE_MODEL, max_concurrency=100, max_retries=2).run(
        request, RequestContext(user_id="weather-demo", tenant_id="examples")
    )
    result = run.result
    result_path = work_dir / "weather_forecast_result.json"
    result_path.write_text(run.response.content, encoding="utf-8")

    print(
        f"request_id={result.request_id}, provider={run.response.provider}, model={run.response.model}, "
        f"accepted={result.accepted}, label={result.label}, score={result.score:.2f}, reason={result.reason}"
    )
    for item in result.details.get("daily", []):
        print(
            f"day+{item.get('date_offset')}: {item.get('weather')}, "
            f"{item.get('temperature_low_c')}~{item.get('temperature_high_c')}℃, "
            f"{item.get('wind')}, suggestion={item.get('suggestion')}"
        )
    print(f"result_file={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
