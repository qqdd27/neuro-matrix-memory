"""Load all 10 LoCoMo conversations into Mem0 locally — the comparison ingest.

Everything runs on this machine: Ollama for extraction, bge-m3 for embeddings,
embedded Qdrant for storage. No API keys, no cloud.

Cost measured before writing this (why batches matter):
  one message per add()      16.2 s
  ten messages per add()      2.7 s per message

Progress is written to ingest_state.json after every conversation, so a crash or
a manual stop does not throw away hours of work: rerun and it resumes.

Run:  python ingest_mem0.py [--data <locomo10.json>] [--batch 20]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from mem0 import Memory

BENCH = Path("D:/WEB/nm-bench")
STATE = BENCH / "ingest_state.json"
DEFAULT_DATA = Path("D:/WEB/neuro-matrix-memory/scripts/bench_data/locomo10.json")

CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3.5:9b",
            "openai_base_url": "http://127.0.0.1:11435/v1",
            "api_key": "local",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "bge-m3",
            "openai_base_url": "http://127.0.0.1:11435/v1",
            "api_key": "local",
            "embedding_dims": 1024,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {"path": str(BENCH / "qdrant_locomo"), "embedding_model_dims": 1024},
    },
    "history_db_path": str(BENCH / "mem0_locomo_history.db"),
}


def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"done": [], "messages": 0, "add_calls": 0, "seconds": 0.0}


def save_state(state: dict[str, Any]) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def conversation_messages(conv: dict[str, Any]) -> list[str]:
    """Speaker-prefixed turns with the session date in front, so the extraction
    step has the same temporal anchors our own ingest gives it."""
    keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                  key=lambda s: int(s.split("_")[1]))
    out: list[str] = []
    for sk in keys:
        date = str(conv.get(f"{sk}_date_time", ""))[:10]
        for turn in conv[sk]:
            text = str(turn.get("text") or "").strip()
            if len(text) < 4:
                continue
            speaker = str(turn.get("speaker") or "")
            prefix = f"[{date}] " if date else ""
            out.append(f"{prefix}{speaker}: {text}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--only", default="", help="comma-separated sample ids to load")
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    state = load_state()
    m = Memory.from_config(CONFIG)
    print(f"Mem0 ready. already loaded: {state['done']}", flush=True)

    for sample in data:
        sid = str(sample.get("sample_id"))
        if only and sid not in only:
            continue
        if sid in state["done"]:
            print(f"[{sid}] already loaded, skipping", flush=True)
            continue
        msgs = conversation_messages(sample["conversation"])
        t0 = time.time()
        added = 0
        for i in range(0, len(msgs), args.batch):
            chunk = msgs[i : i + args.batch]
            payload = [{"role": "user", "content": c} for c in chunk]
            try:
                m.add(payload, user_id=sid, metadata={"source": "locomo", "sample": sid})
                added += len(chunk)
            except Exception as e:  # noqa: BLE001 - one bad chunk must not kill hours
                print(f"[{sid}] chunk {i} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            state["messages"] += 0  # counted below on completion
            state["add_calls"] += 1
            if (i // args.batch) % 5 == 0:
                print(f"[{sid}] {added}/{len(msgs)} msgs, {time.time() - t0:.0f}s", flush=True)
        dt = time.time() - t0
        state["done"].append(sid)
        state["messages"] += added
        state["seconds"] += round(dt, 1)
        save_state(state)
        try:
            allm = m.get_all(filters={"user_id": sid})
            n = len(allm.get("results") or [])
        except Exception:  # noqa: BLE001
            n = -1
        print(f"[{sid}] DONE: {added} messages -> {n} memories in {dt / 60:.1f} min "
              f"({dt / max(1, added):.1f}s/msg)", flush=True)

    print(json.dumps(state, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
