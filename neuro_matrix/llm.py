"""Tiny OpenAI-compatible chat client (stdlib only).

Used ONLY for optional consolidation/extraction.  Every call is wrapped so a
missing key, bad URL or network failure degrades to ``None`` — the memory
engine must keep working offline (extractive mode) at all times.

Two wire formats are supported, picked in ``build_llm_client``:
OpenAI-compatible (`LLMClient`, the default), Anthropic's native Messages API
(`AnthropicLLMClient`), and a local Ollama server (`OllamaLLMClient`, native
``/api/chat`` — see that class for why the OpenAI-compatible endpoint is the
wrong choice for local reasoning models).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# Provider name -> (default base_url, default model, env var for the API key).
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    # Local models via Ollama.  No default model name on purpose: whatever the
    # user pulled decides, and build_llm_client()/OllamaLLMClient resolve the
    # first installed one from /api/tags when the caller names none.  Routing
    # here is about correctness, not style — see OllamaLLMClient for the
    # measured reason the OpenAI-compatible endpoint is wrong for local
    # reasoning models (empty content, ~17x slower).
    "ollama": ("http://127.0.0.1:11434", "", "OLLAMA_API_KEY"),
    # NOTE: Anthropic's real API is NOT OpenAI-compatible (different
    # endpoint path, auth header, request/response shape) -- LLMClient
    # cannot talk to it directly. build_llm_client() below routes provider
    # "anthropic" to AnthropicLLMClient instead; this entry only supplies
    # the default base_url/model/env-var for that routing. Defaults to a
    # cheap, fast model (Haiku) since consolidation/extraction/rerank are
    # frequent, low-stakes, budget-gated background calls, not the kind of
    # task that needs the strongest available model.
    "anthropic": ("https://api.anthropic.com", "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
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
            # Also try the default provider env var when no explicit provider
            # is set.  ANTHROPIC_API_KEY is intentionally absent from this
            # blind-probe list: an Anthropic key must only be used when the
            # provider is explicitly "anthropic", otherwise a stray key in the
            # environment silently routes the whole memory layer (and its
            # costs) to Claude.
            for alt in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
                api_key = os.environ.get(alt) or env_map.get(alt) or ""
                if api_key:
                    defaults = _PROVIDER_DEFAULTS.get(
                        alt.replace("_API_KEY", "").lower()) or defaults
                    break
        if not api_key and (provider == "ollama" or _is_local_ollama_url(base_url)):
            # A local server needs no key: do not report "no LLM configured"
            # merely because there is nothing to authenticate with.  Without
            # this, a Hermes configured against Ollama silently runs the whole
            # memory layer in extractive mode.
            return {
                "api_key": "",
                "base_url": (base_url or _PROVIDER_DEFAULTS["ollama"][0]),
                "model": model,
                "provider": "ollama",
            }
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


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_MD_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class AnthropicLLMClient:
    """Adapter for Anthropic's native Messages API — deliberately NOT the
    same wire format as ``LLMClient`` (different endpoint path, auth header,
    request/response shape; no OpenAI-style ``response_format`` JSON mode),
    but the SAME duck-typed interface (``available()``, ``chat_json``) every
    call site in this project already expects, so it drops in anywhere
    ``self.llm`` is used without any other code change.

    Prompt caching: every call here repeats the exact same, often-long
    system instruction across many calls in one session (sweep_decisions,
    sweep_traits, rerank, consolidate all call this in a loop/batch) — the
    system block is marked ``cache_control: {"type": "ephemeral"}`` so
    Anthropic serves the cached-prefix discount instead of billing the full
    system prompt on every single call.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-haiku-4-5-20251001",
        timeout_s: float = 45.0,
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.model = model or "claude-haiku-4-5-20251001"
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, messages: list[dict[str, str]]) -> Optional[Any]:
        if not self.available():
            return None
        system_text = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system")
        system_text += ("\n\nRespond with ONLY a single valid JSON object — "
                        "no markdown code fences, no commentary before or after it.")
        convo = [{"role": m.get("role") or "user", "content": str(m.get("content", ""))}
                for m in messages if m.get("role") != "system"]
        if not convo:
            convo = [{"role": "user", "content": ""}]
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system_text,
                       "cache_control": {"type": "ephemeral"}}],
            "messages": convo,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = "".join(
                p.get("text", "") for p in (body.get("content") or [])
                if p.get("type") == "text").strip()
            text = _MD_FENCE_RE.sub("", text).strip()
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                m = _JSON_BLOCK_RE.search(text)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except (ValueError, TypeError):
                        pass
                return text or None
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            return None


def _is_local_ollama_url(base_url: str) -> bool:
    """True for a URL that can only be a local Ollama server.

    Guards the user-facing failure mode: someone follows Ollama's own docs and
    pastes ``http://127.0.0.1:11434/v1`` into the provider settings while
    leaving the provider name empty.  The OpenAI-compatible path then works —
    badly (see ``OllamaLLMClient``) — so it fails silently instead of loudly.
    """
    u = (base_url or "").strip().lower()
    return bool(u) and (":11434" in u or "ollama" in u)


class OllamaLLMClient:
    """Adapter for a LOCAL Ollama server (native ``/api/chat``, not ``/v1``).

    Pointing ``LLMClient`` at ``http://127.0.0.1:11434/v1`` does work, but for
    reasoning-capable local models (``qwen3.5:*``, deepseek-r1, ...) the hidden
    "thinking" tokens are generated on every call and are billed against the
    same budget as the answer.  Measured on qwen3.5:9b (RTX 5060 Laptop,
    Q4_K_M), same one-line QA prompt, same output ("7 May 2023"):

    ==================  ==============  ============  ==================
    endpoint            latency         completion    content
    ==================  ==============  ============  ==================
    ``/v1`` max_tokens 800  6.7 s       330 tokens     ok
    ``/v1`` max_tokens 256  4.8 s       256 tokens     **empty**
    ``/v1`` max_tokens 64   1.3 s       64 tokens      **empty**
    ``/api/chat`` think=false  0.33 s   13 tokens      ok
    ==================  ==============  ============  ==================

    Two failure modes follow, and both are silent: ~20x wasted latency/tokens,
    and -- whenever ``max_tokens`` runs out before the hidden reasoning does --
    an EMPTY ``content``, which this project correctly-but-unhelpfully reads as
    "LLM unavailable" and degrades to extractive mode.  Local inference is
    exactly the case where memory is meant to be free and private, so this
    client also sends no auth (Ollama needs none) and asks for ``format: json``
    natively instead of OpenAI's ``response_format``.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        timeout_s: float = 120.0,
        max_tokens: int = 800,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        if self.base_url.endswith("/v1"):  # accept a pasted OpenAI-style URL
            self.base_url = self.base_url[:-3].rstrip("/")
        self.model = (model or "").strip() or self._first_installed_model()
        # Local generation is slower than a hosted API and the main use here
        # is reading a long context, so the default timeout is 120s rather
        # than LLMClient's 45s.
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._probe_ok = False

    def _get_json(self, path: str, timeout: float = 5.0) -> Optional[Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _first_installed_model(self) -> str:
        """Whatever the user pulled, so a bare ``--llm-provider ollama`` works
        without naming a model explicitly."""
        body = self._get_json("/api/tags")
        models = (body or {}).get("models") or []
        return str((models[0] or {}).get("name") or "") if models else ""

    def available(self) -> bool:
        """A model is known AND the server answers.  Probed once and cached:
        this is consulted on every ``chat_json`` call, and a probe per call
        would double the request count against the local server."""
        if not self.model:
            return False
        if self._probe_ok:
            return True
        self._probe_ok = self._get_json("/api/tags") is not None
        return self._probe_ok

    def chat_json(self, messages: list[dict[str, str]]) -> Optional[Any]:
        if not self.available():
            return None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,      # never spend the budget on hidden reasoning
            "format": "json",    # native structured output
            "options": {"temperature": 0.2, "num_predict": self.max_tokens},
        }
        body = self._chat(payload)
        if body is None:
            # Some models/servers reject the 'think' field outright; retry
            # without it rather than degrading to extractive mode.
            payload.pop("think", None)
            body = self._chat(payload)
        if not isinstance(body, dict):
            return None
        text = str((body.get("message") or {}).get("content") or "").strip()
        text = _MD_FENCE_RE.sub("", text).strip()
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            m = _JSON_BLOCK_RE.search(text)
            if m:
                try:
                    return json.loads(m.group(0))
                except (ValueError, TypeError):
                    pass
            return text or None

    def _chat(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body if isinstance(body, dict) else None
        except Exception:
            return None


def build_llm_client(
    api_key: str, *, provider: str = "", base_url: str = "", model: str = "",
    timeout_s: float = 0.0, max_tokens: int = 800,
) -> "LLMClient | AnthropicLLMClient | OllamaLLMClient":
    """Construct the right client for `provider` — the single place that
    decides between the OpenAI-compatible wire format (``LLMClient``,
    default), Anthropic's native one (``AnthropicLLMClient``) and a local
    Ollama server (``OllamaLLMClient``).  Every caller that used to build
    ``LLMClient`` directly from a resolved config should go through this
    instead, or an ``anthropic`` provider silently gets the wrong wire format
    (404s the whole session, degrading to extractive mode with no visible
    error) — a local Ollama endpoint has exactly the same silent-failure
    shape.

    ``timeout_s=0`` means "the chosen provider's own default" (45s hosted,
    120s local) so that a caller cannot accidentally give a local run a
    hosted provider's shorter budget.
    """
    p = (provider or "").strip().lower()
    if p == "anthropic":
        defaults = _PROVIDER_DEFAULTS["anthropic"]
        return AnthropicLLMClient(
            api_key, base_url=base_url or defaults[0], model=model or defaults[1],
            timeout_s=timeout_s or 45.0, max_tokens=max_tokens)
    if p == "ollama" or (not p and _is_local_ollama_url(base_url)):
        defaults = _PROVIDER_DEFAULTS["ollama"]
        return OllamaLLMClient(
            api_key, base_url=base_url or defaults[0], model=model,
            timeout_s=timeout_s or 120.0, max_tokens=max_tokens)
    return LLMClient(
        api_key, base_url=base_url or DEFAULT_BASE_URL, model=model or DEFAULT_MODEL,
        timeout_s=timeout_s or 45.0, max_tokens=max_tokens)
