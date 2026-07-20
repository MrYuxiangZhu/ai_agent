"""统一 VLM 演示：业务构造请求，模型核心负责调用与解析。"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from vlm import OpenaiApiConfig, create_vlm
from vlm.core import MediaAsset, PromptSpec, VlmRequest, load_api_configs, pick_api_config


def parse_args() -> argparse.Namespace:
    """解析通用演示的供应商、图片、任务指令和配置查询参数。"""
    parser = argparse.ArgumentParser(description="运行统一 VLM 示例")
    parser.add_argument("--provider", default="mock", help="mock、qwen、doubao 或 custom")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction", default="判断图片是否可用于后续业务处理。")
    parser.add_argument("--list-providers", action="store_true")
    return parser.parse_args()


def main() -> int:
    """组装配置与请求，执行一次推理并将结构化结果输出到终端。

    ``--list-providers`` 分支只展示脱敏配置，不发起模型请求。正常分支允许
    可选图片输入，并由统一 Runtime 负责适配模型、解析输出和记录日志。
    """
    args = parse_args()
    if args.list_providers:
        _, configs = load_api_configs()
        for name, config in configs.items():
            print(name, config.safe_dict())
        return 0

    config = pick_api_config(args.provider)
    runner = create_vlm(config)
    assets = (MediaAsset(args.image, description="用户输入图片"),) if args.image else ()
    request = VlmRequest(
        request_id="vlm_demo",
        task_name="image_check",
        instruction=args.instruction,
        media_assets=assets,
        prompt_spec=PromptSpec(constraints=("结论必须清晰",)),
    )
    run = runner.run(request)
    print(f"provider={run.response.provider}, model={run.response.model}")
    print(f"accepted={run.result.accepted}, label={run.result.label}, score={run.result.score:.2f}")
    print(f"reason={run.result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# 用户也可以绕过配置文件，直接在业务启动代码中自定义：
API_CONFIG_EXAMPLE = OpenaiApiConfig(
    base_url="http://llmapi.hobot.cc/v1",
    model="qwen3-vl-plus",
    bill_account="yuxiang.zhu",
)
