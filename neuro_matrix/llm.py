"""Tiny OpenAI-compatible chat client (stdlib only).

Used ONLY for optional consolidation/extraction.  Every call is wrapped so a
missing key, bad URL or network failure degrades to ``None`` — the memory
engine must keep working offline (extractive mode) at all times.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 45.0,
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, messages: list[dict[str, str]]) -> Optional[Any]:
        """POST chat completion, parse response JSON, return the message content
        parsed as JSON (or the raw string when it is not JSON).  None on any
        failure — callers must treat None as 'skip LLM step'."""
        if not self.available():
            return None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except (ValueError, TypeError):
                return content
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            return None
