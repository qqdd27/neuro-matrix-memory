"""Tiny OpenAI-compatible chat client (stdlib only).

Used ONLY for optional consolidation/extraction.  Every call is wrapped so a
missing key, bad URL or network failure degrades to ``None`` — the memory
engine must keep working offline (extractive mode) at all times.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# Provider name -> (default base_url, default model, env var for the API key).
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash", "GEMINI_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash", "GEMINI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "grok-3-mini", "XAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOGETHER_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
}


def resolve_hermes_llm_config() -> Optional[dict[str, str]]:
    """Reuse the LLM provider Hermes is already configured with.

    Order of preference: the model block of the Hermes config decides provider
    and (base_url, model); the API key comes from the matching ``*_API_KEY``
    env var (Hermes exports ``.env`` into its processes) or the ``.env`` file
    via ``load_env``.  Returns ``None`` when nothing resolvable — the caller
    stays in extractive mode.  Never raises.
    """
    provider = ""
    base_url = ""
    model = ""
    try:
        from hermes_cli.config import load_config, load_env

        cfg = load_config() or {}
        m = cfg.get("model")
        if isinstance(m, dict):
            provider = str(m.get("provider") or "").strip().lower()
            base_url = str(m.get("base_url") or "").strip()
            model = str(m.get("default") or "").strip()
        elif isinstance(m, str) and "/" in m:
            provider = m.split("/", 1)[0].strip().lower()
        env_map: dict[str, str] = {}
        try:
            env_map = {k: str(v or "") for k, v in dict(load_env() or {}).items()}
        except Exception:
            pass
        defaults = _PROVIDER_DEFAULTS.get(provider) or _PROVIDER_DEFAULTS.get("deepseek")
        env_key = defaults[2]
        api_key = os.environ.get(env_key) or env_map.get(env_key) or ""
        if not api_key:
            # Also try the default provider env var when no explicit provider is set.
            for alt in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
                api_key = os.environ.get(alt) or env_map.get(alt) or ""
                if api_key:
                    defaults = _PROVIDER_DEFAULTS.get(
                        alt.replace("_API_KEY", "").lower()) or defaults
                    break
        if not api_key:
            return None
        return {
            "api_key": api_key,
            "base_url": (base_url or defaults[0]),
            "model": (model or defaults[1]),
            "provider": provider or "deepseek",
        }
    except Exception:
        return None


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
