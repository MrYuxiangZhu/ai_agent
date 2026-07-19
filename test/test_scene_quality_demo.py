from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business.scene_quality.service import SceneQualityRequestBuilder
from framework.core.config_loader import pick_model_profile
from framework.core.model_client import ModelClientFactory
from framework.core.parser import JsonBusinessResultParser
from framework.core.prompt import JsonPromptBuilder
from framework.core.runner import InferenceRunner


class SceneQualityDemoTest(unittest.TestCase):
    def test_scene_quality_mock_runner(self):
        tmp_dir = Path("/tmp/example_scene_quality_test")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        image_path = tmp_dir / "sample.png"
        image_path.write_bytes(b"fake image bytes")
        requests = SceneQualityRequestBuilder().build_demo_requests(image_path)
        profile = pick_model_profile("mock")
        runner = InferenceRunner(JsonPromptBuilder(), ModelClientFactory.create(profile), JsonBusinessResultParser())

        envelopes = runner.run_many(requests)

        self.assertEqual(len(envelopes), 2)
        self.assertTrue(envelopes[0].result.accepted)
        self.assertEqual(envelopes[0].result.label, "ready")
        self.assertFalse(envelopes[1].result.accepted)
        self.assertEqual(envelopes[1].result.label, "missing_media")


if __name__ == "__main__":
    unittest.main()
