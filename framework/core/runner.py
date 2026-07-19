"""Runtime orchestration independent from concrete business and model vendors."""

from __future__ import annotations

from typing import Iterable, List

from framework.core.types import BusinessRequest, ModelClient, PromptBuilder, ResultParser, RunEnvelope


class InferenceRunner:
    def __init__(self, prompt_builder: PromptBuilder, model_client: ModelClient, result_parser: ResultParser) -> None:
        self._prompt_builder = prompt_builder
        self._model_client = model_client
        self._result_parser = result_parser

    def run_one(self, request: BusinessRequest) -> RunEnvelope:
        prompt = self._prompt_builder.build_prompt(request)
        response = self._model_client.infer(prompt, request)
        result = self._result_parser.parse(response, request)
        return RunEnvelope(request=request, prompt=prompt, response=response, result=result)

    def run_many(self, requests: Iterable[BusinessRequest]) -> List[RunEnvelope]:
        return [self.run_one(request) for request in requests]
