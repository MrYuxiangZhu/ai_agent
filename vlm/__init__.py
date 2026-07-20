"""类型安全、多供应商、多用户并发的统一 AI Runtime。"""

from collections.abc import Sequence

from vlm.core import OpenaiApiConfig, VlmClientFactory, VlmJsonResultParser, VlmRunner
from vlm.core.context import RequestContext, request_context
from vlm.core.logging import TraceLogger


def create_vlm(config: OpenaiApiConfig | Sequence[OpenaiApiConfig], *, max_concurrency: int = 100,
               max_retries: int = 2, trace_logger: TraceLogger | None = None) -> VlmRunner:
    """根据一个或多个供应商配置创建生产可复用的统一运行器。

    Args:
        config: 单配置表示固定模型；配置序列按顺序构成主模型和降级模型。
        max_concurrency: 运行器允许同时进入模型调用阶段的请求上限。
        max_retries: 每个供应商失败后的额外重试次数。
        trace_logger: 可选日志写入器，便于宿主应用指定日志目录。

    Returns:
        已装配客户端、解析器、并发控制和日志机制的 ``VlmRunner``。
    """
    configs = [config] if isinstance(config, OpenaiApiConfig) else list(config)
    clients = [VlmClientFactory.create(item) for item in configs]
    return VlmRunner(clients, VlmJsonResultParser(), max_concurrency=max_concurrency,
                     max_retries=max_retries, trace_logger=trace_logger)


__all__ = ["OpenaiApiConfig", "RequestContext", "TraceLogger", "VlmClientFactory",
           "VlmJsonResultParser", "VlmRunner", "create_vlm", "request_context"]
