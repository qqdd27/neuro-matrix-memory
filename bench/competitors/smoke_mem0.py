"""Smoke-test Mem0 running entirely locally (Ollama via the OpenAI shim).

Verifies that the competitor stack actually works in this environment before
any comparison run: local LLM for extraction, local embedder, local vector
store, no API keys.

Run:  python smoke_mem0.py
"""

from __future__ import annotations

import json
import time

from mem0 import Memory

BASE = "http://127.0.0.1:11435/v1"
KEY = "local"

CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {"model": "qwen3.5:9b", "openai_base_url": BASE, "api_key": KEY},
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "bge-m3",
            "openai_base_url": BASE,
            "api_key": KEY,
            "embedding_dims": 1024,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": "D:/WEB/nm-bench/qdrant_smoke",
            # Mem0 defaults its collection to 1536 dims (OpenAI text-embedding-3);
            # our local embedder returns 1024, which otherwise fails on the first
            # add with "shapes (0,1536) and (1024,) not aligned".
            "embedding_model_dims": 1024,
        },
    },
    "history_db_path": "D:/WEB/nm-bench/mem0_smoke_history.db",
}

FACTS = [
    "Alice is a research scientist working on adoption agencies.",
    "Alice moved to Vancouver last spring for a new role.",
    "Bob fixed the car engine in the garage on Tuesday.",
    "The garage project in Berlin was cancelled in June.",
]

QUERIES = [
    "Where does Alice live now?",
    "What happened to the garage project?",
]


def main() -> int:
    t0 = time.time()
    m = Memory.from_config(CONFIG)
    print(f"Mem0 initialised in {time.time() - t0:.1f}s", flush=True)

    for i, f in enumerate(FACTS, 1):
        t = time.time()
        try:
            r = m.add(f, user_id="smoke")
            got = r.get("results") if isinstance(r, dict) else r
            n = len(got) if isinstance(got, list) else "?"
        except Exception as e:  # noqa: BLE001
            print(f"  add #{i} FAILED: {type(e).__name__}: {e}")
            continue
        print(f"  add #{i}: {time.time() - t:.1f}s -> {n} memor(y|ies) extracted", flush=True)

    total = m.get_all(filters={"user_id": "smoke"})
    items = total.get("results") if isinstance(total, dict) else total
    print(f"\nstored memories: {len(items) if items else 0}")
    for it in (items or [])[:6]:
        print("   -", str(it.get("memory"))[:100])

    for q in QUERIES:
        t = time.time()
        try:
            res = m.search(q, user_id="smoke", limit=5)
            hits = res.get("results") if isinstance(res, dict) else res
        except Exception as e:  # noqa: BLE001
            print(f"search FAILED for {q!r}: {type(e).__name__}: {e}")
            continue
        print(f"\nQ: {q}  ({time.time() - t:.2f}s, {len(hits or [])} hits)")
        for h in (hits or [])[:3]:
            score = h.get("score")
            print(f"   {score if score is None else round(float(score), 3)}  {str(h.get('memory'))[:95]}")

    try:
        with open("D:/WEB/nm-bench/shim_stats.json", "w", encoding="utf-8") as fh:
            json.dump({"queries_done": len(QUERIES)}, fh)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
