"""Run the scene-quality business demo.

Examples:
    python3 script/run_scene_quality_demo.py
    python3 script/run_scene_quality_demo.py --provider qwen
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business.scene_quality.service import SceneQualityRequestBuilder
from framework.core.config_loader import load_model_profiles, pick_model_profile
from framework.core.model_client import ModelClientFactory
from framework.core.parser import JsonBusinessResultParser
from framework.core.prompt import JsonPromptBuilder
from framework.core.runner import InferenceRunner
from framework.utils.jsonl import write_envelopes_jsonl

_SAMPLE_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-agnostic scene quality demo.")
    parser.add_argument("--provider", default="mock", help="Provider key in config/model_services.json")
    parser.add_argument("--work-dir", default="/tmp/example_scene_quality_work", help="Directory for generated demo files")
    parser.add_argument("--list-providers", action="store_true", help="Print configured providers and exit")
    return parser.parse_args()


def prepare_demo_image(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_path = work_dir / "scene_quality_sample.png"
    if not image_path.exists():
        image_path.write_bytes(base64.b64decode(_SAMPLE_PNG_BASE64))
    return image_path


def main() -> int:
    args = parse_args()
    if args.list_providers:
        for key, profile in sorted(load_model_profiles().items()):
            print(f"provider={key}, model={profile.model}, transport={profile.transport}, endpoint={profile.endpoint}")
        return 0

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir
    image_path = prepare_demo_image(work_dir)
    requests = SceneQualityRequestBuilder().build_demo_requests(image_path)
    profile = pick_model_profile(args.provider)
    runner = InferenceRunner(
        prompt_builder=JsonPromptBuilder(),
        model_client=ModelClientFactory.create(profile),
        result_parser=JsonBusinessResultParser(),
    )
    envelopes = runner.run_many(requests)
    result_path = write_envelopes_jsonl(envelopes, work_dir / "scene_quality_result.jsonl")
    for envelope in envelopes:
        result = envelope.result
        print(
            f"request_id={result.request_id}, provider={envelope.response.provider}, "
            f"accepted={result.accepted}, label={result.label}, score={result.score:.2f}, reason={result.reason}"
        )
    print(f"result_file={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
