# Model-Agnostic Example Framework

这个目录是一套独立的模型无关示例框架。框架目标是：业务可以按需调用不同大模型完成自己的任务，但业务代码不绑定某一个具体模型，底层模型替换不会影响基础链路。

## 分层约定

```text
example/
├── framework/               # 通用模型调用框架、稳定协议和工具
│   ├── core/                # 模型配置、Prompt、Client、Parser、Runner
│   └── utils/               # 通用工具依赖
├── business/                # 业务模块，每个业务一个子目录
│   ├── scene_quality/        # 示例业务：图片链路质量判断
│   └── weather_forecast/     # 示例业务：天气预报结构化生成
├── mini_vlm/                # 独立 Mini VLM 示例模块
│   ├── core/                # Mini VLM 的配置、适配器、协议、运行时、解析和 prompt
│   └── io/work/             # Mini VLM 示例输出目录
├── script/                  # 可执行脚本
├── config/                  # 配置文件
├── test/                    # 测试用例
└── README.md
```

## 各层职责

### `framework/core/`

放通用模型相关和框架核心能力：

- `types.py`：统一数据协议，包括 `BusinessRequest`、`PromptSpec`、`ModelServiceProfile`、`ModelResponse`、`BusinessResult`。
- `config_loader.py`：加载 `config/model_services.json`，选择模型 provider。
- `model_client.py`：模型调用客户端和工厂，目前支持 `mock` 和 `http_json`。
- `prompt.py`：通用结构化 Prompt 构造器，支持业务自定义角色、示例、输出格式和约束。
- `parser.py`：通用 JSON 结果解析器。
- `runner.py`：运行编排，不依赖具体业务和模型供应商。

### `business/`

放业务逻辑。示例业务包括 `business/scene_quality/` 和 `business/weather_forecast/`。业务模块负责定义任务、上下文和业务 Prompt 规格，不直接关心模型是 Qwen、InternVL 还是自研服务。

### `mini_vlm/`

放 Mini VLM 独立示例模块，避免 `mini_vlm_*.py` 散落在项目根目录：

- `core/types.py`：Mini VLM 独立数据协议。
- `core/config.py`：Mini VLM 客户配置文件加载与 provider 选择。
- `core/adapters.py`：Mini VLM 本地 mock 与 HTTP JSON 适配器。
- `core/prompting.py`：Mini VLM Prompt 组装。
- `core/parser.py`：Mini VLM JSON/free-text 结果解析。
- `core/runtime.py`：Mini VLM 运行编排和 JSONL 输出。
- `io/work/`：Mini VLM demo 默认输入输出目录。

### `script/`

放可执行脚本。当前入口：

```bash
python3 script/run_scene_quality_demo.py
python3 script/run_weather_forecast_demo.py
python3 script/run_mini_vlm_demo.py
```

### `utils/`

放通用工具。当前包含 JSONL 结果落盘工具。

### `config/`

放模型服务客户配置。当前配置文件：

```text
config/model_services.json      # 通用框架模型服务配置
config/mini_vlm_services.json   # Mini VLM 客户模型服务配置
```

默认包含：

- `mock`：本地确定性模型，无 GPU、无网络依赖。
- `qwen`：千问/Qwen 兼容 HTTP 服务。
- `internvl`：InternVL 兼容 HTTP 服务。
- `custom`：自定义大模型服务。

### `test/`

放测试用例。当前测试验证 mock 链路能跑通。

## 快速运行

进入 `/home/yuxiangzhu/volume/example`：

```bash
cd /home/yuxiangzhu/volume/example
python3 script/run_scene_quality_demo.py
```

默认使用 `mock` provider，输出类似：

```text
request_id=scene_quality_valid, provider=mock, accepted=True, label=ready, score=0.92, reason=request=scene_quality_valid, media_count=1, missing=[]
request_id=scene_quality_missing, provider=mock, accepted=False, label=missing_media, score=0.03, reason=request=scene_quality_missing, media_count=1, missing=['.../missing_scene.png']
result_file=/tmp/example_scene_quality_work/scene_quality_result.jsonl
```

运行天气预报示例：

```bash
python3 script/run_weather_forecast_demo.py --city 杭州 --days 3
```

运行 Mini VLM 示例：

```bash
python3 script/run_mini_vlm_demo.py
```

## 查看 provider

```bash
python3 script/run_scene_quality_demo.py --list-providers
python3 script/run_weather_forecast_demo.py --list-providers
python3 script/run_mini_vlm_demo.py --show-providers
```

## 切换模型服务

使用千问/Qwen：

```bash
python3 script/run_scene_quality_demo.py --provider qwen
python3 script/run_weather_forecast_demo.py --provider qwen
python3 script/run_mini_vlm_demo.py --provider qwen
```

使用 InternVL：

```bash
python3 script/run_scene_quality_demo.py --provider internvl
python3 script/run_weather_forecast_demo.py --provider internvl
python3 script/run_mini_vlm_demo.py --provider internvl
```

使用自定义服务：

```bash
python3 script/run_scene_quality_demo.py --provider custom
python3 script/run_weather_forecast_demo.py --provider custom
python3 script/run_mini_vlm_demo.py --provider custom
```

## 环境变量覆盖配置

`config_loader.py` 支持用环境变量覆盖模型、endpoint、timeout、token。

以 qwen 为例：

```bash
export EXAMPLE_QWEN_MODEL=Qwen2.5-VL-7B-Instruct
export EXAMPLE_QWEN_ENDPOINT=http://127.0.0.1:23333/infer
export EXAMPLE_QWEN_TIMEOUT=300
export EXAMPLE_QWEN_TOKEN=your_token
python3 script/run_scene_quality_demo.py --provider qwen
```

Mini VLM 不使用 `export MINI_VLM_*` 环境变量配置模型服务，而是读取客户配置文件 `config/mini_vlm_services.json`。例如配置 qwen：

```json
{
  "default_provider": "mock",
  "providers": [
    {
      "provider_key": "qwen",
      "display_name": "Qwen compatible VLM service",
      "model_identifier": "Qwen2.5-VL-7B-Instruct",
      "endpoint_url": "http://127.0.0.1:23333/infer",
      "timeout_seconds": 300,
      "bearer_token": null,
      "model_path": null,
      "request_style": "generic_json"
    }
  ]
}
```

运行时通过配置中的 provider key 选择服务：

```bash
python3 script/run_mini_vlm_demo.py --provider qwen
```

## HTTP 服务请求格式

`HttpJsonModelClient` 会发送：

```json
{
  "provider": "qwen",
  "model": "Qwen2.5-VL-7B-Instruct",
  "prompt": "完整 Prompt 文本",
  "request_id": "scene_quality_valid",
  "task_name": "scene_quality_check",
  "media_assets": [
    {
      "path": "/path/to/image.png",
      "kind": "image",
      "description": "自动生成的 demo 图片"
    }
  ],
  "context": {
    "business": "scene_quality"
  },
  "options": {
    "temperature": 0,
    "max_tokens": 512
  }
}
```

建议模型服务返回：

```json
{
  "accepted": true,
  "label": "ready",
  "score": 0.95,
  "reason": "图片可读，满足业务要求。"
}
```

## 新增业务

在 `business/` 下新增目录，例如：

```text
business/construction_check/
```

业务层只需要构造 `BusinessRequest`，然后复用 `core.runner.InferenceRunner`。

## 新增模型

在 `config/model_services.json` 里新增 profile。如果是通用 HTTP JSON 协议，不需要改核心代码；如果协议不同，在 `core/model_client.py` 新增 client，并在 `ModelClientFactory` 注册新的 `transport`。

## 运行测试

```bash
cd /home/yuxiangzhu/volume/example
python3 -m unittest discover -s test
```

也可以直接运行 demo 验证主链路：

```bash
python3 script/run_scene_quality_demo.py
python3 script/run_mini_vlm_demo.py
```
