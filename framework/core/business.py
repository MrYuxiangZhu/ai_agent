"""Business extension points using Template Method, Strategy and Registry patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterable, Protocol, TypeVar

from framework.core.types import (
    BusinessRequest,
    BusinessResult,
    MediaAsset,
    ModelRequirements,
    OutputContract,
    PromptSpec,
    RunEnvelope,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BusinessHandler(ABC, Generic[InputT, OutputT]):
    """业务模板基类；派生类只实现领域差异，公共流程由基类固定。"""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Return the globally unique task name."""

    @abstractmethod
    def validate_input(self, business_input: InputT) -> None:
        """Raise ValueError when domain input is invalid."""

    @abstractmethod
    def build_instruction(self, business_input: InputT) -> str:
        """Build the task instruction."""

    @abstractmethod
    def build_context(self, business_input: InputT) -> Dict[str, Any]:
        """Convert domain input to serializable model context."""

    @abstractmethod
    def prompt_spec(self) -> PromptSpec:
        """Return the business-owned prompt strategy."""

    @abstractmethod
    def output_contract(self) -> OutputContract:
        """Return the business output contract."""

    @abstractmethod
    def model_requirements(self) -> ModelRequirements:
        """Declare model capabilities required by this business."""

    @abstractmethod
    def map_result(self, result: BusinessResult) -> OutputT:
        """Map generic model result to a strongly typed domain result."""

    def request_id(self, business_input: InputT) -> str:
        return f"{self.task_name}_{id(business_input)}"

    def media_assets(self, business_input: InputT) -> Iterable[MediaAsset]:
        return ()

    def build_request(self, business_input: InputT) -> BusinessRequest:
        """Template Method: validation and request assembly cannot be bypassed."""
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

    def execute(self, business_input: InputT, runtime) -> OutputT:
        envelope: RunEnvelope = runtime.run_one(self.build_request(business_input))
        return self.map_result(envelope.result)


class BusinessRuntime(Protocol):
    def run_one(self, request: BusinessRequest) -> RunEnvelope:
        """Execute one framework request."""


class BusinessRegistry:
    """Stores decorated business classes and lazily creates handler instances."""

    def __init__(self) -> None:
        self._classes: Dict[str, type[BusinessHandler[Any, Any]]] = {}
        self._instances: Dict[str, BusinessHandler[Any, Any]] = {}

    def register(self, task_name: str, handler_class: type[BusinessHandler[Any, Any]]) -> None:
        if task_name in self._classes:
            raise ValueError(f"business handler already registered: {task_name}")
        self._classes[task_name] = handler_class

    def get(self, task_name: str) -> BusinessHandler[Any, Any]:
        try:
            handler_class = self._classes[task_name]
        except KeyError as exc:
            raise LookupError(f"unknown business task: {task_name}") from exc
        if task_name not in self._instances:
            handler = handler_class()
            if handler.task_name != task_name:
                raise ValueError(
                    f"decorated task name '{task_name}' does not match handler task name '{handler.task_name}'"
                )
            self._instances[task_name] = handler
        return self._instances[task_name]

    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))


business_registry = BusinessRegistry()


def register_business(task_name: str):
    """Class decorator registering a stateless business handler at import time."""
    def decorator(handler_class: type[BusinessHandler[Any, Any]]):
        if not issubclass(handler_class, BusinessHandler):
            raise TypeError("registered business must inherit BusinessHandler")
        business_registry.register(task_name, handler_class)
        return handler_class

    return decorator


class BusinessManager:
    """Facade coordinating registered businesses with one shared runtime."""

    def __init__(self, runtime: BusinessRuntime, registry: BusinessRegistry = business_registry) -> None:
        self._registry = registry
        self._runtime = runtime

    def execute(self, task_name: str, business_input: Any) -> Any:
        handler = self._registry.get(task_name)
        envelope = self._runtime.run_one(handler.build_request(business_input))
        return handler.map_result(envelope.result)

    def build_request(self, task_name: str, business_input: Any) -> BusinessRequest:
        return self._registry.get(task_name).build_request(business_input)

    def tasks(self) -> tuple[str, ...]:
        return self._registry.tasks()
