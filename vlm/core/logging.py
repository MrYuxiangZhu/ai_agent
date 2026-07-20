"""线程安全的 JSON Lines 链路日志，按天写入项目 logs 目录。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from vlm.core.context import RequestContext


class TraceLogger:
    """将运行时事件持久化为便于采集和检索的 JSON Lines 日志。"""

    def __init__(self, log_dir: Path | str | None = None) -> None:
        """初始化链路日志写入器。

        Args:
            log_dir: 自定义日志目录；未提供时使用工程根目录下的 ``logs``。

        每个实例持有独立互斥锁，保证同一 Runtime 中多个线程写入时每条 JSON
        不会交错。Logger 禁止向根节点传播，避免宿主应用重复输出。
        """
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parents[2] / "logs"
        self._lock = Lock()
        self._logger = logging.getLogger(f"vlm.trace.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def event(self, name: str, context: RequestContext, **fields: Any) -> None:
        """写入一条包含用户和链路元数据的结构化事件。

        Args:
            name: 稳定的事件名称，例如 ``request.started`` 或 ``model.attempt``。
            context: 当前用户、租户及 trace 标识。
            **fields: 与事件相关的扩展字段，值无法直接序列化时转为字符串。

        日志按 UTC 日期写入 ``runtime-YYYY-MM-DD.jsonl``。方法不会记录模型
        密钥或请求正文；调用方也不应通过扩展字段传入敏感凭证。
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"runtime-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        payload: Mapping[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": name,
            "trace_id": context.trace_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            **fields,
        }
        with self._lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
