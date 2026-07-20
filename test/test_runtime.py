import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from vlm import OpenaiApiConfig, RequestContext, TraceLogger, create_vlm
from vlm.core.types import VlmRequest


class RuntimeTest(unittest.TestCase):
    """验证运行时在并发用户场景下的结果数量和链路隔离。"""

    def test_concurrent_users_have_trace_logs(self):
        """并发提交十个用户请求，并断言每个用户均有独立完整日志。"""
        with tempfile.TemporaryDirectory() as directory:
            runner = create_vlm(OpenaiApiConfig("mock://local", "mock-vlm", provider="mock"),
                                max_concurrency=4, trace_logger=TraceLogger(directory))

            async def execute():
                """并行执行请求，模拟同一租户下十个不同用户。"""
                return await asyncio.gather(*(
                    runner.arun(VlmRequest(f"request-{index}", "demo", "test"),
                                RequestContext(f"user-{index}", "tenant-a"))
                    for index in range(10)
                ))

            runs = asyncio.run(execute())
            self.assertEqual(len(runs), 10)
            events = [json.loads(line) for path in Path(directory).glob("*.jsonl")
                      for line in path.read_text(encoding="utf-8").splitlines()]
            completed = [event for event in events if event["event"] == "request.completed"]
            self.assertEqual(len(completed), 10)
            self.assertEqual({event["user_id"] for event in completed}, {f"user-{i}" for i in range(10)})
            self.assertTrue(all(event["trace_id"] for event in completed))


if __name__ == "__main__":
    unittest.main()
