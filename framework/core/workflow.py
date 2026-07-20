"""Resilient workflow runtime with routing, retries and lifecycle events."""

from __future__ import annotations

import time
import uuid
from typing import Iterable, List, Optional

from framework.core.contracts import ContractResultParser, ContractViolation
from framework.core.model_client import ModelClientFactory
from framework.core.routing import CapabilityRouter
from framework.core.types import BusinessRequest, EventListener, ModelClient, RunEnvelope, RunEvent


class WorkflowRuntime:
    def __init__(self, prompt_builder, clients: Optional[dict[str, ModelClient]] = None,
                 router: Optional[CapabilityRouter] = None, max_attempts: int = 2,
                 listeners: Optional[Iterable[EventListener]] = None) -> None:
        self._prompt_builder = prompt_builder
        self._clients = clients or {}
        self._router = router
        self._max_attempts = max(1, max_attempts)
        self._listeners = tuple(listeners or ())
        self._parser = ContractResultParser()

    def run_one(self, request: BusinessRequest) -> RunEnvelope:
        trace_id = uuid.uuid4().hex
        events: List[RunEvent] = []
        self._emit(events, "run.started", trace_id, request)
        prompt = self._prompt_builder.build_prompt(request)
        self._emit(events, "prompt.built", trace_id, request, prompt_chars=len(prompt))
        profiles = self._router.select(request) if self._router else []
        if not profiles:
            raise ValueError("WorkflowRuntime requires a CapabilityRouter")
        last_error: Optional[Exception] = None
        attempts = 0
        for profile in profiles:
            client = self._clients.get(profile.provider) or ModelClientFactory.create(profile)
            for _ in range(self._max_attempts):
                attempts += 1
                self._emit(events, "model.started", trace_id, request, provider=profile.provider, attempt=attempts)
                response = client.infer(prompt, request)
                try:
                    result = self._parser.parse(response, request)
                    self._emit(events, "run.completed", trace_id, request, provider=profile.provider, attempts=attempts)
                    return RunEnvelope(request, prompt, response, result, trace_id, attempts, tuple(events))
                except ContractViolation as exc:
                    last_error = exc
                    self._emit(events, "output.invalid", trace_id, request, error=str(exc), attempt=attempts)
        self._emit(events, "run.failed", trace_id, request, error=str(last_error))
        raise ContractViolation(str(last_error or "workflow failed"))

    def run_many(self, requests: Iterable[BusinessRequest]) -> List[RunEnvelope]:
        return [self.run_one(request) for request in requests]

    def _emit(self, events: List[RunEvent], name: str, trace_id: str, request: BusinessRequest, **attributes):
        event = RunEvent(name, trace_id, request.request_id, time.time(), attributes)
        events.append(event)
        for listener in self._listeners:
            listener(event)
        return ""
