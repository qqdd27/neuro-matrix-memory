"""One metric, two systems: honest head-to-head retrieval scoring.

The fairness problem this solves: our memory stores the original dialogue turns
verbatim, so "does the gold turn's text appear in the top-k" scores us
generously, while Mem0 REWRITES turns into extracted facts ("Bob fixed the car
engine in the garage on Tuesday, 2026-09-10" for a turn that said something
else). Scoring Mem0 with literal containment would measure its paraphrasing
style, not its retrieval quality.

So both systems are scored the same way: the gold turn's content words must be
substantially present in the retrieved window.

  token-recall hit@k — fraction of the gold turn's distinct content words (>=3
  chars, stop-words removed) that appear anywhere in the concatenated top-k
  retrieved texts; >= 0.5 counts as a hit.

Literal containment is also reported, for reference only.

Run (our side):    python eval_common.py --system neuromatrix --k 8
Run (mem0 side):   python eval_common.py --system mem0 --k 8
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, "D:/WEB/neuro-matrix-memory")

from neuro_matrix.entities import STOPWORDS  # noqa: E402
from neuro_matrix.store import NeuroMatrixStore  # noqa: E402

BENCH = Path("D:/WEB/nm-bench")
DATA = Path("D:/WEB/neuro-matrix-memory/scripts/bench_data/locomo10.json")
CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")
HIT_THRESHOLD = 0.5


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS}


def token_recall(gold: str, context: str) -> float:
    g = content_tokens(gold)
    if not g:
        return 0.0
    return len(g & content_tokens(context)) / len(g)


# ---------------------------------------------------------------- our system

def neuromatrix_context_factory(sample: dict) -> Callable[[str, int], list[str]]:
    conv = sample["conversation"]
    keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                  key=lambda s: int(s.split("_")[1]))
    path = Path(tempfile.mkdtemp()) / "nm.db"
    store = NeuroMatrixStore(str(path), llm=None, llm_daily_budget=0)
    base_ts = time.time() - 400 * 86400
    for i, sk in enumerate(keys):
        date = str(conv.get(f"{sk}_date_time", ""))
        ts = base_ts + i * 86400
        m = re.search(r"(\d{1,2})[^\d]{1,3}(\w+)[^\d]{1,3}(\d{4})", date)
        if m:
            try:
                import datetime as _dt
                ts = time.mktime(_dt.datetime.strptime(
                    f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").timetuple())
            except ValueError:
                pass
        for turn in conv[sk]:
            text = str(turn.get("text") or "").strip()
            if len(text) < 4:
                continue
            store.remember(f"{turn.get('speaker', '')}: {text}", source="turn", ts=ts)

    def ctx(query: str, k: int) -> list[str]:
        return [str(h.get("text") or "") for h in store.search(query, limit=k, include_dossiers=False)]

    ctx.close = store.close  # type: ignore[attr-defined]
    return ctx


# ------------------------------------------------------------- mem0 (local)

def mem0_context_factory(sample: dict) -> Callable[[str, int], list[str]]:
    from mem0 import Memory

    cfg = {
        "llm": {"provider": "openai",
                "config": {"model": "qwen3.5:9b",
                           "openai_base_url": "http://127.0.0.1:11435/v1", "api_key": "local"}},
        "embedder": {"provider": "openai",
                     "config": {"model": "bge-m3",
                                "openai_base_url": "http://127.0.0.1:11435/v1", "api_key": "local",
                                "embedding_dims": 1024}},
        "vector_store": {"provider": "qdrant",
                         "config": {"path": str(BENCH / "qdrant_locomo"),
                                    "embedding_model_dims": 1024}},
        "history_db_path": str(BENCH / "mem0_locomo_history.db"),
    }
    m = Memory.from_config(cfg)
    sid = str(sample.get("sample_id"))

    def ctx(query: str, k: int) -> list[str]:
        try:
            r = m.search(query, filters={"user_id": sid}, limit=k)
        except Exception:  # noqa: BLE001
            return []
        hits = r.get("results") if isinstance(r, dict) else r
        return [str(h.get("memory") or "") for h in (hits or [])]

    return ctx


FACTORIES = {"neuromatrix": neuromatrix_context_factory, "mem0": mem0_context_factory}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(FACTORIES))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--samples", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    samples = data[: args.samples] if args.samples else data
    factory = FACTORIES[args.system]

    per_cat: dict[str, list[int]] = {}
    literal: dict[str, list[int]] = {}
    lat: list[float] = []
    t_start = time.time()
    for sample in samples:
        if args.system == "mem0":
            loaded = json.loads((BENCH / "ingest_state.json").read_text(encoding="utf-8")) if (BENCH / "ingest_state.json").exists() else {"done": []}
            if str(sample.get("sample_id")) not in loaded.get("done", []):
                continue
        conv = sample["conversation"]
        dmap: dict[str, str] = {}
        for sk in (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")):
            for turn in conv[sk]:
                if turn.get("dia_id"):
                    dmap[str(turn["dia_id"])] = f"{turn.get('speaker', '')}: {turn.get('text', '')}"
        ctx = factory(sample)
        for qa in sample["qa"]:
            gold_texts = [dmap[e] for e in (qa.get("evidence") or []) if e in dmap]
            if not gold_texts:
                continue
            cat = CATEGORY_NAMES.get(int(qa.get("category") or 0), "other")
            t0 = time.perf_counter()
            texts = ctx(str(qa["question"]), args.k)
            lat.append((time.perf_counter() - t0) * 1000)
            blob = "\n".join(texts)
            hit = any(token_recall(g, blob) >= HIT_THRESHOLD for g in gold_texts)
            per_cat.setdefault(cat, []).append(1 if hit else 0)
            literal.setdefault(cat, []).append(1 if any(g in blob for g in gold_texts) else 0)
        try:
            ctx.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        print(f"  [{sample.get('sample_id')}] scored", flush=True)

    flat = [v for vals in per_cat.values() for v in vals]
    overall = sum(flat) / len(flat) * 100 if flat else 0.0
    print(f"\n=== {args.system} | token-recall hit@{args.k} (threshold {HIT_THRESHOLD}) ===")
    for c, vals in sorted(per_cat.items()):
        print(f"[{c:<11}] n={len(vals):>4}  hit@{args.k} = {sum(vals):>4}/{len(vals):<4} = {sum(vals) / len(vals) * 100:.1f}%"
              f"   (literal: {sum(literal[c]) / len(literal[c]) * 100:.1f}%)")
    print(f"OVERALL: {sum(flat)}/{len(flat)} = {overall:.1f}%"
          f"   (literal overall: {sum(sum(v) for v in literal.values()) / max(1, len(flat)) * 100:.1f}%)")
    if lat:
        lat.sort()
        print(f"search latency: p50={statistics.median(lat):.1f}ms "
              f"p95={lat[int(0.95 * len(lat)) - 1]:.1f}ms")
    print(f"total wall time: {(time.time() - t_start) / 60:.1f} min")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"system": args.system, "k": args.k, "overall": overall,
             "per_cat": {c: {"n": len(v), "hit": sum(v)} for c, v in per_cat.items()}},
            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
