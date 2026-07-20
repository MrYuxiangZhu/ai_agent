"""场景质量演示；直接在文件中选择模型。"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from business.scene_quality.service import SceneQualityRequestBuilder
from vlm import OpenaiApiConfig, RequestContext, create_vlm

_SAMPLE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
MODEL_CONFIG = OpenaiApiConfig("mock://local", "mock-vlm", provider="mock")
WORK_DIR = Path("/tmp/example_scene_quality_work")


def main() -> int:
    """准备最小 PNG，批量运行成功和缺失媒体案例并打印审核结果。

    两个请求共享用户和租户上下文，以演示批量并发处理及链路日志记录。
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    image = WORK_DIR / "scene_quality_sample.png"
    image.write_bytes(base64.b64decode(_SAMPLE))
    runner = create_vlm(MODEL_CONFIG, max_concurrency=50)
    runs = runner.run_many(SceneQualityRequestBuilder().build_demo_requests(image),
                           RequestContext(user_id="scene-demo", tenant_id="examples"))
    for run in runs:
        print(f"request_id={run.result.request_id}, provider={run.response.provider}, "
              f"accepted={run.result.accepted}, label={run.result.label}, score={run.result.score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
