"""多用户请求上下文，使用 contextvars 保证异步任务间隔离。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator
from uuid import uuid4


@dataclass(frozen=True)
class RequestContext:
    user_id: str = "anonymous"
    tenant_id: str = "default"
    trace_id: str = ""

    def normalized(self) -> "RequestContext":
        """返回具有完整链路标识的不可变上下文。

        调用方可以省略 ``trace_id``；此时方法生成随机 UUID 十六进制值。
        已携带标识时直接返回当前对象，避免一次请求在不同组件间产生多个标识。

        Returns:
            当前上下文，或补全 ``trace_id`` 后的新上下文。
        """
        return self if self.trace_id else RequestContext(self.user_id, self.tenant_id, uuid4().hex)


_current_context: ContextVar[RequestContext] = ContextVar("vlm_request_context", default=RequestContext())


def current_context() -> RequestContext:
    """读取当前异步执行上下文，并确保其具有可追踪的 ``trace_id``。

    ``ContextVar`` 会为协程维护独立值，因此并发用户不会互相覆盖身份信息。
    未进入 ``request_context`` 时返回匿名默认上下文。
    """
    return _current_context.get().normalized()


@contextmanager
def request_context(user_id: str, tenant_id: str = "default", trace_id: str = "") -> Iterator[RequestContext]:
    """在代码块内绑定用户、租户和链路信息，并在退出时安全恢复。

    Args:
        user_id: 发起调用的用户唯一标识。
        tenant_id: 用户所属租户，默认使用 ``default``。
        trace_id: 可由上游网关传入；为空时自动生成。

    Yields:
        已补全链路标识的请求上下文。

    即使代码块抛出异常，``finally`` 也会恢复之前的上下文，防止线程或
    协程复用时发生用户信息泄漏。
    """
    context = RequestContext(user_id, tenant_id, trace_id).normalized()
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)
