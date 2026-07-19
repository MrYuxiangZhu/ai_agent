import unittest

from business.weather_forecast.service import WeatherForecastRequestBuilder
from framework.core.config_loader import pick_model_profile
from framework.core.model_client import ModelClientFactory
from framework.core.parser import JsonBusinessResultParser
from framework.core.prompt import JsonPromptBuilder
from framework.core.runner import InferenceRunner


class WeatherForecastDemoTest(unittest.TestCase):
    def test_weather_forecast_mock_runner(self):
        request = WeatherForecastRequestBuilder().build_request("杭州", 3, "通勤和晨跑")
        profile = pick_model_profile("mock")
        runner = InferenceRunner(JsonPromptBuilder(), ModelClientFactory.create(profile), JsonBusinessResultParser())

        envelope = runner.run_one(request)

        self.assertTrue(envelope.result.accepted)
        self.assertEqual(envelope.result.label, "forecast_ready")
        self.assertEqual(envelope.result.details["city"], "杭州")
        self.assertEqual(envelope.result.details["forecast_days"], 3)
        self.assertEqual(len(envelope.result.details["daily"]), 3)

    def test_weather_forecast_prompt_uses_business_spec(self):
        request = WeatherForecastRequestBuilder().build_request("北京", 2, "商务出行")
        prompt = JsonPromptBuilder().build_prompt(request)

        self.assertIn("角色：你是一名专业天气预报助手", prompt)
        self.assertIn("业务任务：weather_forecast", prompt)
        self.assertIn("输出格式：请只输出 JSON", prompt)
        self.assertIn("约束条件：", prompt)
        self.assertIn("城市：杭州；天数：2；关注点：通勤和穿衣", prompt)


if __name__ == "__main__":
    unittest.main()
