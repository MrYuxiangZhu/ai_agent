"""LLM clients for answer generation."""

from __future__ import annotations

import json
import re
import urllib.request


class MockLLMClient:
    """Local deterministic generator used when no real LLM is configured."""

    def generate(self, prompt: str) -> str:
        question = self._extract(prompt, "用户问题：", "可用资料：") or self._extract(prompt, "Question:", "Context:")
        sources = re.findall(r"\[资料\s+(\d+)\].*?内容：\n(.+?)(?=\n\n\[资料|\n\n请给出|$)", prompt, flags=re.S)
        if not sources:
            return "知识库资料不足，无法确认。"
        top = sources[:3]
        bullets = []
        for source_id, content in top:
            sentence = self._first_sentence(content.strip())
            if sentence:
                bullets.append(f"{sentence} [资料 {source_id}]")
        if not bullets:
            return "知识库资料不足，无法确认。"
        prefix = f"针对问题“{question.strip()}”，根据知识库可得：\n" if question else "根据知识库可得：\n"
        return prefix + "\n".join(f"- {bullet}" for bullet in bullets)

    @staticmethod
    def _extract(text: str, start: str, end: str) -> str:
        if start not in text or end not in text:
            return ""
        return text.split(start, 1)[1].split(end, 1)[0].strip()

    @staticmethod
    def _first_sentence(text: str) -> str:
        match = re.search(r"(.{20,220}?[。！？.!?])", text)
        if match:
            return match.group(1).strip()
        return text[:220].strip()


class HttpJsonLLMClient:
    """Generic HTTP JSON LLM client.

    Request body:
    ``{"model": "...", "prompt": "...", "options": {...}}``

    Accepted response fields:
    ``text``, ``answer``, or OpenAI-compatible ``choices[0].message.content``.
    """

    def __init__(self, endpoint: str, model: str = "", token: str = "", timeout_seconds: int = 120, **options: object) -> None:
        self._endpoint = endpoint
        self._model = model
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._options = options

    def generate(self, prompt: str) -> str:
        payload = {"model": self._model, "prompt": prompt, "options": self._options}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(self._endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        if "text" in body:
            return str(body["text"])
        if "answer" in body:
            return str(body["answer"])
        if "choices" in body:
            return str(body["choices"][0]["message"]["content"])
        raise ValueError("LLM response must contain text, answer, or choices[0].message.content")
