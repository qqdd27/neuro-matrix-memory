"""OpenAI-compatible shim in front of Ollama for local benchmark runs.

Why this exists: the local reader/generator is a reasoning model, and over
Ollama's OpenAI-compatible endpoint it spends the whole token budget on hidden
"thinking" and returns an empty `content` (measured: finish_reason=length with
content=""). Third-party memory frameworks (Mem0, A-Mem, ...) call the model
through that interface to extract facts, so with an empty response their
write path silently produces nothing and a comparison would measure our shim,
not their architecture.

Ollama's own /api/chat accepts `think: false`, which suppresses the reasoning
phase entirely (measured: 0.33 s, correct answer). This shim accepts OpenAI
requests and forwards them to /api/chat with that flag, translating the
response back.

Endpoints:
  POST /v1/chat/completions   -> Ollama /api/chat   (think=false, json mode passthrough)
  POST /v1/embeddings         -> Ollama /v1/embeddings (pass-through)
  GET  /v1/models             -> model list

Run:  python openai_ollama_shim.py [--port 11435]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

OLLAMA = "http://127.0.0.1:11434"
CALLS = {"chat": 0, "embed": 0, "chat_seconds": 0.0}


def _post(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost only
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost only
        return json.loads(resp.read().decode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - keep the console readable
        return

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.rstrip("/").endswith("/v1/models"):
            try:
                models = _get(f"{OLLAMA}/v1/models")
            except Exception as e:  # noqa: BLE001
                self._send(502, {"error": {"message": str(e)}})
                return
            self._send(200, models)
            return
        if self.path.rstrip("/").endswith("/stats"):
            self._send(200, CALLS)
            return
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send(400, {"error": {"message": f"bad json: {e}"}})
            return

        path = self.path.rstrip("/")
        if path.endswith("/chat/completions"):
            self._chat(body)
        elif path.endswith("/embeddings"):
            self._embeddings(body)
        else:
            self._send(404, {"error": {"message": "not found"}})

    def _chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages") or []
        payload: dict[str, Any] = {
            "model": body.get("model"),
            "messages": messages,
            "stream": False,
            # The whole point of the shim: no hidden reasoning phase.
            "think": False,
        }
        options: dict[str, Any] = {}
        if body.get("max_tokens"):
            options["num_predict"] = int(body["max_tokens"])
        if body.get("temperature") is not None:
            options["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            options["top_p"] = body["top_p"]
        if options:
            payload["options"] = options
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            payload["format"] = "json"
        if body.get("tools"):
            payload["tools"] = body["tools"]

        t0 = time.time()
        try:
            resp = _post(f"{OLLAMA}/api/chat", payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            self._send(e.code, {"error": {"message": detail}})
            return
        except Exception as e:  # noqa: BLE001
            self._send(502, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return
        CALLS["chat"] += 1
        CALLS["chat_seconds"] += time.time() - t0

        msg = resp.get("message") or {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        out_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        pt = int(resp.get("prompt_eval_count") or 0)
        ct = int(resp.get("eval_count") or 0)
        self._send(200, {
            "id": "chatcmpl-shim",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{
                "index": 0,
                "message": out_msg,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        })

    def _embeddings(self, body: dict[str, Any]) -> None:
        try:
            resp = _post(f"{OLLAMA}/v1/embeddings", body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            self._send(e.code, {"error": {"message": detail}})
            return
        except Exception as e:  # noqa: BLE001
            self._send(502, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return
        CALLS["embed"] += 1
        self._send(200, resp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"shim listening on http://127.0.0.1:{args.port}/v1 (ollama at {OLLAMA})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
