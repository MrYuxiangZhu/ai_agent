"""模型调用编排，不包含任何具体用户业务。"""

from __future__ import annotations

from collections.abc import Iterable

from vlm.core.prompting import VlmPromptBuilder
from vlm.core.types import ModelClient, ResultParser, VlmRequest, VlmRun


class VlmRunner:
    def __init__(self, model_client: ModelClient, result_parser: ResultParser, prompt_builder: VlmPromptBuilder | None = None) -> None:
        self._client = model_client
        self._parser = result_parser
        self._prompt_builder = prompt_builder or VlmPromptBuilder()

    def run(self, request: VlmRequest) -> VlmRun:
        prompt = self._prompt_builder.build(request)
        response = self._client.infer(prompt, request)
        return VlmRun(request, prompt, response, self._parser.parse(response, request))

    def run_many(self, requests: Iterable[VlmRequest]) -> list[VlmRun]:
        return [self.run(request) for request in requests]
