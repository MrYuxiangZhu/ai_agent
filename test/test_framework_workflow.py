import unittest

from business.weather_forecast.service import WeatherForecastBusiness, WeatherForecastInput
from framework.core.prompt import JsonPromptBuilder
from framework.core.routing import CapabilityRouter, NoCompatibleModelError
from framework.core.types import BusinessRequest, ModelRequirements, ModelServiceProfile
from framework.core.workflow import WorkflowRuntime


class FrameworkWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.profile = ModelServiceProfile(
            provider="mock", model="deterministic", endpoint="local://mock",
            transport="mock", capabilities={"text", "structured_output"}, priority=1,
        )

    def test_weather_contract_and_trace(self):
        runtime = WorkflowRuntime(JsonPromptBuilder(), router=CapabilityRouter([self.profile]))
        request = WeatherForecastBusiness().build_request(WeatherForecastInput("杭州", 3))
        run = runtime.run_one(request)
        self.assertTrue(run.result.accepted)
        self.assertEqual(run.result.details["city"], "杭州")
        self.assertEqual(len(run.result.details["daily"]), 3)
        self.assertTrue(run.trace_id)
        self.assertEqual([event.name for event in run.events], [
            "run.started", "prompt.built", "model.started", "run.completed"
        ])

    def test_router_rejects_missing_capability(self):
        request = BusinessRequest(
            "video", "video", "analyze",
            model_requirements=ModelRequirements(modalities={"video"}),
        )
        with self.assertRaises(NoCompatibleModelError):
            CapabilityRouter([self.profile]).select(request)


if __name__ == "__main__":
    unittest.main()
