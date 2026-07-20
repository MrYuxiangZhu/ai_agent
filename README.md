# Model-Agnostic AI Business Framework

这是一个面向 AI 业务的模型无关运行框架。业务代码通过统一的抽象基类描述输入、Prompt、输出契约和模型能力要求，底层通过 Workflow Runtime 完成模型路由、重试、契约校验和运行事件记录。

框架的核心目标是：

- 业务与模型供应商解耦
- 业务输入输出强类型化
- 统一业务生命周期
- 支持模型能力路由和失败降级
- 支持结构化输出契约校验
- 支持 Trace 和运行事件观测
- 支持新增业务插件化注册

## 目录结构

```text
ai_agent/
├── framework/
│   ├── core/
│   │   ├── business.py       # 业务抽象基类、模板方法、注册中心
│   │   ├── contracts.py      # 模型输出契约校验
│   │   ├── model_client.py   # Mock/HTTP 模型客户端
│   │   ├── prompt.py         # 通用 Prompt 构造器
│   │   ├── routing.py        # 基于能力的模型路由
│   │   ├── runner.py         # 旧版简单执行器，仅供历史模块使用
│   │   ├── types.py          # 稳定数据契约
│   │   └── workflow.py       # Workflow Runtime
│   └── utils/
│       └── jsonl.py          # JSONL 工具
├── business/
│   ├── weather_forecast/
│   │   ├── prompt_spec.py
│   │   └── service.py        # WeatherForecastBusiness
│   └── scene_quality/
│       └── service.py        # SceneQualityBusiness
├── runtime/
│   ├── __init__.py
│   └── register.py            # 业务发现和共享 Runtime 装配
├── config/
│   └── model_services.json
├── script/
├── test/
└── README.md
```

## 设计模式

### Template Method

所有业务继承 `BusinessHandler[InputT, OutputT]`。基类的 `build_request()` 固定业务请求构建流程：

```text
输入校验
  → 构建任务指令
  → 构建上下文
  → 组装媒体资源
  → 声明 Prompt 规范
  → 声明输出契约
  → 声明模型能力
  → 创建 BusinessRequest
```

派生业务只需要重载领域相关的纯虚函数。

### Strategy

业务可以分别定义自己的：

- Prompt 策略
- 输出契约策略
- 模型能力策略
- 结果映射策略

### Registry / Manager

`BusinessRegistry` 保存通过 `@register_business` 注册的业务类，并按任务名完成延迟实例化。`BusinessManager` 统一协调业务 Handler 与 `WorkflowRuntime`，负责业务请求构建和执行。模型客户端仍由 `ModelClientFactory` 创建。

运行时装配位于 `runtime/register.py`，它显式导入业务模块以触发装饰器注册，然后创建共享 Runtime 和 BusinessManager。Framework 核心不主动依赖具体业务。

### Adapter

当前不再为天气业务保留旧代码兼容 Builder。天气业务只使用新的 `WeatherForecastBusiness` API。场景质量业务仍保留请求构建器，以便现有场景 Demo 继续运行。

### Chain of Responsibility

模型路由器按优先级生成兼容模型列表。当前模型重试失败后，Runtime 会切换到下一个兼容模型。

### Observer

Workflow Runtime 发布以下运行事件：

```text
run.started
prompt.built
model.started
output.invalid
run.completed
run.failed
```

可以通过监听器接入日志、指标、审计、计费或 Trace 存储。

## 业务抽象基类

文件：`framework/core/business.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

from framework.core.types import (
    BusinessRequest,
    BusinessResult,
    ModelRequirements,
    OutputContract,
    PromptSpec,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BusinessHandler(ABC, Generic[InputT, OutputT]):
    @property
    @abstractmethod
    def task_name(self) -> str:
        ...

    @abstractmethod
    def validate_input(self, business_input: InputT) -> None:
        ...

    @abstractmethod
    def build_instruction(self, business_input: InputT) -> str:
        ...

    @abstractmethod
    def build_context(self, business_input: InputT) -> Dict[str, Any]:
        ...

    @abstractmethod
    def prompt_spec(self) -> PromptSpec:
        ...

    @abstractmethod
    def output_contract(self) -> OutputContract:
        ...

    @abstractmethod
    def model_requirements(self) -> ModelRequirements:
        ...

    @abstractmethod
    def map_result(self, result: BusinessResult) -> OutputT:
        ...

    def build_request(self, business_input: InputT) -> BusinessRequest:
        self.validate_input(business_input)
        return BusinessRequest(
            request_id=self.request_id(business_input),
            task_name=self.task_name,
            instruction=self.build_instruction(business_input),
            media_assets=list(self.media_assets(business_input)),
            context=self.build_context(business_input),
            prompt_spec=self.prompt_spec(),
            output_contract=self.output_contract(),
            model_requirements=self.model_requirements(),
        )
```

## 天气业务

天气业务使用新的原生业务 API，不再提供 `WeatherForecastRequestBuilder` 兼容入口。

### 输入和输出

```python
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class WeatherForecastInput:
    city: str
    forecast_days: int = 3
    concern: str = "通勤、穿衣和户外活动"


@dataclass(frozen=True)
class WeatherForecastResult:
    city: str
    forecast_days: int
    daily: List[Dict[str, Any]]
    score: float
    reason: str
```

### 执行天气业务

```python
from business.weather_forecast.service import WeatherForecastInput
from runtime.register import business_manager

result = business_manager.execute(
    "weather_forecast",
    WeatherForecastInput(
        city="杭州",
        forecast_days=3,
        concern="通勤、穿衣和晨跑",
    ),
)

print(result.city)
print(result.forecast_days)
print(result.daily)
print(result.score)
print(result.reason)
```

### 创建天气请求但暂不执行

```python
from runtime.register import business_manager, workflow_runtime

request = business_manager.build_request(
    "weather_forecast",
    WeatherForecastInput(
        city="北京",
        forecast_days=2,
        concern="商务出行和户外活动",
    ),
)

run = workflow_runtime.run_one(request)
```

天气输出契约要求以下字段：

```text
accepted
label
score
reason
city
forecast_days
daily
```

## 场景质量业务

```python
from pathlib import Path

from business.scene_quality.service import SceneQualityInput
from runtime.register import business_manager

result = business_manager.execute(
    "scene_quality_check",
    SceneQualityInput(Path("/tmp/example.png")),
)

print(result.accepted)
print(result.label)
print(result.score)
print(result.reason)
```

场景质量任务声明需要以下模型能力：

```text
text
image
structured_output
```

因此 Runtime 会自动排除不支持图片或结构化输出的模型。

## Workflow Runtime

```python
from runtime.register import workflow_runtime
```

`runtime/register.py` 已完成模型配置加载、能力路由器和 Workflow Runtime 的统一装配。

Runtime 执行过程：

```text
生成 Trace ID
  → 构建 Prompt
  → 根据能力要求选择模型
  → 调用模型
  → 校验 JSON 输出契约
  ├── 校验失败：重试当前模型
  ├── 仍失败：切换备用模型
  └── 校验成功：返回 RunEnvelope
```

`RunEnvelope` 包含：

- 原始 `BusinessRequest`
- 最终 Prompt
- `ModelResponse`
- `BusinessResult`
- `trace_id`
- `attempts`
- 完整事件列表

## 业务注册与统一管理

业务类定义时通过 `@register_business` 装饰器注册到 Framework Registry，应用层只负责导入业务模块并创建共享 Runtime。Framework 核心不主动导入具体业务，业务模块导入是应用层的显式发现步骤。

```python
from runtime.register import business_manager
from business.weather_forecast.service import WeatherForecastInput

print(business_manager.tasks())

result = business_manager.execute(
    "weather_forecast",
    WeatherForecastInput(
        city="杭州",
        forecast_days=3,
        concern="通勤和晨跑",
    ),
)
```

职责划分：

```text
BusinessRegistry
  保存、查找和防止重复注册

BusinessManager
  根据 task_name 查找业务，并统一注入 WorkflowRuntime 执行

runtime/register.py
  导入业务模块、触发装饰器注册并创建共享 Runtime

ModelClientFactory
  只负责创建模型客户端，不负责创建业务
```

业务通过装饰器注册，运行时统一从 `runtime.register.business_manager` 获取。

## 模型配置

配置文件：`config/model_services.json`

```json
{
  "profiles": [
    {
      "provider": "mock",
      "model": "local-deterministic-model",
      "endpoint": "local://mock",
      "timeout_seconds": 1,
      "transport": "mock",
      "capabilities": [
        "text",
        "structured_output"
      ],
      "priority": 1,
      "options": {}
    }
  ]
}
```

常用配置字段：

- `provider`：模型供应商标识
- `model`：模型名称
- `endpoint`：HTTP 服务地址
- `transport`：传输类型，目前支持 `mock` 和 `http_json`
- `capabilities`：模型能力集合
- `priority`：路由优先级，数值越小优先级越高
- `timeout_seconds`：请求超时时间
- `options`：供应商扩展参数

当前示例配置包含：

- `mock`：本地确定性模型，无网络和 GPU 依赖
- `qwen`：Qwen 兼容 HTTP 服务
- `internvl`：InternVL 兼容 HTTP 服务
- `custom`：自定义 HTTP 服务

## HTTP 请求格式

`HttpJsonModelClient` 会发送类似请求：

```json
{
  "provider": "qwen",
  "model": "Qwen2.5-VL-7B-Instruct",
  "prompt": "完整 Prompt 文本",
  "request_id": "scene_quality_valid",
  "task_name": "scene_quality_check",
  "media_assets": [],
  "context": {
    "business": "scene_quality"
  },
  "options": {
    "temperature": 0,
    "max_tokens": 512
  }
}
```

模型建议返回：

```json
{
  "accepted": true,
  "label": "ready",
  "score": 0.95,
  "reason": "图片可读，满足业务要求。"
}
```

如果业务定义了扩展字段，例如天气业务的 `city`、`forecast_days` 和 `daily`，这些字段也必须出现在输出 JSON 中。

## 新增业务

新增业务时必须继承 `BusinessHandler`，不能直接在业务代码中调用具体模型客户端。

例如：

```text
business/document_review/
└── service.py
```

基本结构：

```python
from dataclasses import dataclass
from typing import Any, Dict

from framework.core.business import BusinessHandler
from framework.core.types import (
    BusinessResult,
    ModelRequirements,
    OutputContract,
    PromptSpec,
)


@dataclass(frozen=True)
class DocumentReviewInput:
    text: str


@dataclass(frozen=True)
class DocumentReviewResult:
    accepted: bool
    label: str
    score: float
    reason: str


class DocumentReviewBusiness(
    BusinessHandler[DocumentReviewInput, DocumentReviewResult]
):
    @property
    def task_name(self) -> str:
        return "document_review"

    def validate_input(self, business_input: DocumentReviewInput) -> None:
        if not business_input.text.strip():
            raise ValueError("text cannot be empty")

    def build_instruction(self, business_input: DocumentReviewInput) -> str:
        return "审核输入文档。"

    def build_context(self, business_input: DocumentReviewInput) -> Dict[str, Any]:
        return {"text": business_input.text}

    def prompt_spec(self) -> PromptSpec:
        return PromptSpec()

    def output_contract(self) -> OutputContract:
        return OutputContract()

    def model_requirements(self) -> ModelRequirements:
        return ModelRequirements(modalities={"text"})

    def map_result(self, result: BusinessResult) -> DocumentReviewResult:
        return DocumentReviewResult(
            accepted=result.accepted,
            label=result.label,
            score=result.score,
            reason=result.reason,
        )
```

新增业务后，在应用启动阶段注册：

```python
registry.register(DocumentReviewBusiness())
```

## 新增模型

如果新模型兼容当前 HTTP JSON 协议，只需在 `config/model_services.json` 中增加 profile。

如果协议不同：

1. 在 `framework/core/model_client.py` 新增 `ModelClient` 实现。
2. 在 `ModelClientFactory` 注册新的 `transport`。
3. 为模型填写准确的 `capabilities`。
4. 增加模型客户端测试。

## 运行 Demo

进入工程目录：

```bash
cd /home/yuxiangzhu/volume/ai_agent
```

运行场景质量 Demo：

```bash
python3 script/run_scene_quality_demo.py
```

运行天气预报 Demo：

```bash
python3 script/run_weather_forecast_demo.py --city 杭州 --days 3
```

查看可用 provider：

```bash
python3 script/run_scene_quality_demo.py --list-providers
python3 script/run_weather_forecast_demo.py --list-providers
```

切换模型：

```bash
python3 script/run_weather_forecast_demo.py --provider qwen
python3 script/run_weather_forecast_demo.py --provider internvl
python3 script/run_weather_forecast_demo.py --provider custom
```

## 环境变量

`config_loader.py` 支持使用环境变量覆盖模型配置：

```bash
export EXAMPLE_QWEN_MODEL=Qwen2.5-VL-7B-Instruct
export EXAMPLE_QWEN_ENDPOINT=http://127.0.0.1:23333/infer
export EXAMPLE_QWEN_TIMEOUT=300
export EXAMPLE_QWEN_TOKEN=your_token

python3 script/run_weather_forecast_demo.py --provider qwen
```

不要将真实 Token 写入 `config/model_services.json`，优先使用环境变量或外部密钥管理系统。

## 运行测试

```bash
cd /home/yuxiangzhu/volume/ai_agent
python3 -m unittest discover -s test -v
```

测试覆盖：

- 业务抽象基类和模板方法
- 业务注册中心
- 天气业务契约
- 场景质量业务派生类
- 能力模型路由
- Workflow Trace 事件
- Mock 模型链路
- 并发运行隔离
