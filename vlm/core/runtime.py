"""支持多用户并发、重试和供应商降级的统一 AI Runtime。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence

from vlm.core.context import RequestContext, current_context
from vlm.core.logging import TraceLogger
from vlm.core.prompting import VlmPromptBuilder
from vlm.core.types import ModelClient, ResultParser, VlmRequest, VlmResponse, VlmRun


class VlmRunner:
    """统一编排提示词构造、并发控制、重试、模型降级、解析和追踪。"""

    def __init__(self, model_clients: ModelClient | Sequence[ModelClient], result_parser: ResultParser,
                 prompt_builder: VlmPromptBuilder | None = None, max_concurrency: int = 100,
                 max_retries: int = 2, trace_logger: TraceLogger | None = None) -> None:
        """创建可被多个用户复用的模型运行时。

        Args:
            model_clients: 单个客户端或按优先级排列的客户端序列；后者用于降级。
            result_parser: 将模型原始响应转换成统一业务结果的解析器。
            prompt_builder: 可替换提示词构造器，默认使用通用构造器。
            max_concurrency: 同时占用模型调用槽位的最大请求数。
            max_retries: 每个供应商首次调用失败后的额外重试次数。
            trace_logger: 自定义链路日志组件；省略时写入工程 ``logs`` 目录。

        Raises:
            ValueError: 客户端列表为空时抛出。
        """
        self._clients = [model_clients] if not isinstance(model_clients, Sequence) else list(model_clients)
        if not self._clients:
            raise ValueError("至少需要一个模型客户端")
        self._parser = result_parser
        self._prompt_builder = prompt_builder or VlmPromptBuilder()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max(0, max_retries)
        self._logger = trace_logger or TraceLogger()

    def run(self, request: VlmRequest, context: RequestContext | None = None) -> VlmRun:
        """在同步程序中执行一个请求并返回完整运行记录。

        该方法通过 ``asyncio.run`` 驱动异步实现，适合脚本和普通同步服务。
        已运行事件循环的 Web/Notebook 环境应改用 ``await arun(...)``。
        """
        return asyncio.run(self.arun(request, context))

    async def arun(self, request: VlmRequest, context: RequestContext | None = None) -> VlmRun:
        """异步执行单个请求，并记录请求开始、模型尝试和完成事件。

        Args:
            request: 与供应商无关的模型请求。
            context: 可选用户上下文；省略时读取当前 ``ContextVar``。

        Returns:
            包含请求、最终提示词、模型响应和解析结果的 ``VlmRun``。
        """
        context = (context or current_context()).normalized()
        prompt = self._prompt_builder.build(request)
        started = time.perf_counter()
        self._logger.event("request.started", context, request_id=request.request_id, task=request.task_name)
        response = await self._infer_with_fallback(request, prompt, context)
        result = self._parser.parse(response, request)
        self._logger.event("request.completed", context, request_id=request.request_id,
                           provider=response.provider, model=response.model, success=response.success,
                           label=result.label, latency_ms=int((time.perf_counter() - started) * 1000))
        return VlmRun(request, prompt, response, result)

    async def _infer_with_fallback(self, request: VlmRequest, prompt: str,
                                   context: RequestContext) -> VlmResponse:
        """在并发限制内按顺序重试客户端，并在失败后切换备用供应商。

        同步供应商客户端通过 ``asyncio.to_thread`` 执行，避免阻塞事件循环。
        每个客户端最多调用 ``max_retries + 1`` 次；失败重试之间执行有上限的
        指数退避。全部失败时返回最后一次响应，由解析器生成标准错误结果。
        """
        last_response: VlmResponse | None = None
        async with self._semaphore:
            for client_index, client in enumerate(self._clients):
                for attempt in range(self._max_retries + 1):
                    response = await asyncio.to_thread(client.infer, prompt, request)
                    last_response = response
                    self._logger.event("model.attempt", context, request_id=request.request_id,
                                       provider=response.provider, model=response.model, attempt=attempt + 1,
                                       fallback_index=client_index, success=response.success,
                                       latency_ms=response.latency_ms, error=response.error)
                    if response.success:
                        return response
                    if attempt < self._max_retries:
                        await asyncio.sleep(min(0.25 * (2 ** attempt), 2.0))
        assert last_response is not None
        return last_response

    def run_many(self, requests: Iterable[VlmRequest], context: RequestContext | None = None) -> list[VlmRun]:
        """在同步入口中并发执行一批请求，并保持输入顺序返回结果。"""
        return asyncio.run(self.arun_many(requests, context))

    async def arun_many(self, requests: Iterable[VlmRequest],
                        context: RequestContext | None = None) -> list[VlmRun]:
        """使用 ``asyncio.gather`` 并发执行请求集合。

        所有任务共享运行时信号量，因此即使调用方一次提交大量请求，实际模型
        调用数也不会超过 ``max_concurrency``。返回列表顺序与输入迭代顺序一致。
        """
        return list(await asyncio.gather(*(self.arun(request, context) for request in requests)))
