"""NeuroMatrix MCP surface — the same engine behind a standard MCP stdio server.

Speaks the Model Context Protocol (JSON-RPC 2.0 over newline-delimited stdio):
``initialize`` -> ``notifications/initialized`` -> ``tools/list`` /
``tools/call`` -> ``ping``.  Zero dependencies; the tool bodies call the
``NeuroMatrixStore`` directly, so every capability (facts, graph, decisions,
foresight, artifacts) is available to any MCP client (Claude Desktop, Hermes
MCP registry, custom tools).

Run:  python -m neuro_matrix.mcp --db /path/neuromatrix.db
Env:  NEUROMATRIX_DB (default $HERMES_HOME/neuromatrix.db)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

from .store import NeuroMatrixStore

TOOLS: list[dict[str, Any]] = [
    {"name": "memory_search", "description": "Associative recall over the graph",
     "inputSchema": {"type": "object",
                     "properties": {"query": {"type": "string"},
                                    "limit": {"type": "integer"}},
                     "required": ["query"]}},
    {"name": "memory_remember", "description": "Store a durable/episodic fact",
     "inputSchema": {"type": "object",
                     "properties": {"content": {"type": "string"}},
                     "required": ["content"]}},
    {"name": "memory_probe", "description": "Entity dossier + aliases + decisions",
     "inputSchema": {"type": "object",
                     "properties": {"entity": {"type": "string"}},
                     "required": ["entity"]}},
    {"name": "memory_decide", "description": "Record a decision with structured criteria",
     "inputSchema": {"type": "object",
                     "properties": {"concept": {"type": "string"}, "choice": {"type": "string"},
                                    "criteria": {"type": "array", "items": {"type": "string"}},
                                    "reason": {"type": "string"}},
                     "required": ["concept", "choice"]}},
    {"name": "memory_decisions", "description": "Decision trail for a concept",
     "inputSchema": {"type": "object",
                     "properties": {"concept": {"type": "string"}},
                     "required": ["concept"]}},
    {"name": "memory_foresight", "description": "Plan a time-bounded reminder",
     "inputSchema": {"type": "object",
                     "properties": {"content": {"type": "string"},
                                    "trigger_at": {"type": "string"},
                                    "entity": {"type": "string"}},
                     "required": ["content", "trigger_at"]}},
    {"name": "memory_reminders", "description": "List due foresights",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_stats", "description": "Store statistics",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _call(store: Optional[NeuroMatrixStore], name: str, args: dict[str, Any]) -> dict[str, Any]:
    if store is None:
        return _err("store unavailable")
    try:
        if name == "memory_search":
            return _ok(json.dumps(store.search(args.get("query", ""),
                                               limit=int(args.get("limit", 6))),
                                  ensure_ascii=False))
        if name == "memory_remember":
            return _ok(json.dumps({"fact_id": store.remember(
                args["content"], source="mcp")}, ensure_ascii=False))
        if name == "memory_probe":
            return _ok(json.dumps(store.entity(args.get("entity", "")), ensure_ascii=False))
        if name == "memory_decide":
            crit = [str(c) for c in (args.get("criteria") or [])]
            fid = store.decide(args["concept"], args["choice"], criteria=crit,
                               reason=args.get("reason", ""))
            return _ok(json.dumps({"decision_id": fid}, ensure_ascii=False))
        if name == "memory_decisions":
            return _ok(json.dumps(store.decisions(args["concept"]), ensure_ascii=False))
        if name == "memory_foresight":
            try:
                trig = float(args["trigger_at"])
            except (TypeError, ValueError):
                trig = args["trigger_at"]  # ISO handled by plan_foresight meta parser
            fid = store.plan_foresight(args["content"], trig,
                                       entity=args.get("entity"))
            return _ok(json.dumps({"foresight_id": fid}, ensure_ascii=False))
        if name == "memory_reminders":
            return _ok(json.dumps({"due": store.foresights_due()}, ensure_ascii=False))
        if name == "memory_stats":
            return _ok(json.dumps(store.stats(), ensure_ascii=False))
        return _err(f"unknown tool: {name}")
    except Exception as e:  # noqa: BLE001
        return _err(f"{type(e).__name__}: {e}")


def handle_message(msg: dict[str, Any], store: Optional[NeuroMatrixStore]) -> Optional[dict[str, Any]]:
    """Process one JSON-RPC message (unit-testable, no I/O)."""
    mid = msg.get("id")
    method = msg.get("method", "")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid,
                "result": {"protocolVersion": "2025-06-18",
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "neuromatrix", "version": "0.3.0"}}}
    if method == "notifications/initialized" or method.startswith("notifications/"):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        result = _call(store, name, args)
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    db = None
    if "--db" in argv:
        db = argv[argv.index("--db") + 1]
    if not db:
        db = os.environ.get("NEUROMATRIX_DB", "")
    if not db or db.startswith("$"):
        home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        db = os.path.join(home, "neuromatrix.db")
    store: Optional[NeuroMatrixStore] = None
    try:
        store = NeuroMatrixStore(db)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"neuromatrix mcp: cannot open {db}: {e}\n")
        store = None
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            reply = handle_message(msg, store)
            if reply is not None:
                sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        if store is not None:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
