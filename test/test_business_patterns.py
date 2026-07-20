import tempfile
import unittest
from pathlib import Path

from business.scene_quality.service import SceneQualityBusiness, SceneQualityInput
from business.weather_forecast.service import WeatherForecastInput
from runtime.register import business_manager
from framework.core.business import BusinessManager
from framework.core.prompt import JsonPromptBuilder
from framework.core.routing import CapabilityRouter
from framework.core.types import ModelServiceProfile
from framework.core.workflow import WorkflowRuntime


class BusinessPatternTest(unittest.TestCase):
    def setUp(self):
        profile = ModelServiceProfile(
            provider="mock", model="local", endpoint="local://mock",
            transport="mock", capabilities={"text", "image", "structured_output"},
        )
        self.runtime = WorkflowRuntime(
            prompt_builder=JsonPromptBuilder(),
            router=CapabilityRouter([profile]),
        )

    def test_business_manager_and_registry(self):
        self.assertIsInstance(business_manager, BusinessManager)
        self.assertEqual(business_manager.tasks(), ("scene_quality_check", "weather_forecast"))
        result = business_manager.execute("weather_forecast", WeatherForecastInput("杭州", 2))
        self.assertEqual(result.city, "杭州")
        self.assertEqual(len(result.daily), 2)

    def test_scene_quality_derived_business(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.png"
            path.write_bytes(b"demo")
            result = SceneQualityBusiness().execute(SceneQualityInput(path), self.runtime)
            self.assertTrue(result.accepted)
            self.assertEqual(result.label, "ready")


if __name__ == "__main__":
    unittest.main()
