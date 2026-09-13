"""Answer quality head-to-head: same reader, same prompt, same questions.

Retrieval hits alone do not decide which memory is better — what matters is
whether the model can answer from what the memory handed over. This runs the
identical reader prompt used by our own harness (copied verbatim, deliberately
NOT tuned) over both systems' retrieved windows, and scores token-F1 against
LoCoMo's gold answers.

Fairness rules encoded here:
  * same reader (local qwen3.5:9b through the shim), same prompt, same k
  * same questions (same sample, same --llm-sample chunking and seed)
  * same scoring function as our own runs
  * our side reuses exactly the ingest our harness uses

Run:  python eval_f1_common.py --system mem0 --sample 60 --limit-samples 1
      python eval_f1_common.py --system neuromatrix --sample 60 --limit-samples 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, "D:/WEB/neuro-matrix-memory")
sys.path.insert(0, "D:/WEB/nm-bench")

from eval_common import (  # noqa: E402
    BENCH, DATA, CATEGORY_NAMES, mem0_context_factory, neuromatrix_context_factory,
)

READER_SYSTEM = ("Answer the question using ONLY the facts below. Extract "
                 "the SHORTEST possible answer span (2-8 words, a name, "
                 "date, or short phrase) — reuse the facts' own wording "
                 "verbatim wherever possible, never a full sentence, "
                 "never restate the question, no filler words like "
                 "'likely' or 'according to the facts'. Each fact may be "
                 "prefixed with [YYYY-MM-DD]: that is the date of the "
                 "event, so for 'when' questions answer with that date. If "
                 "the facts do not contain the answer, still give your best "
                 'single best short guess. Return JSON {"answer": "..."}.')

_NORM_RE = re.compile(r"[^0-9a-zа-яё ]+")


def _norm(text: str) -> list[str]:
    return _NORM_RE.sub(" ", str(text).lower()).split()


def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def ask(question: str, context: list[str]) -> str:
    """One reader call through the shim (no hidden reasoning phase)."""
    import urllib.request

    body = json.dumps({
        "model": "qwen3.5:9b",
        "messages": [
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": "FACTS:\n" + "\n".join(f"- {t}" for t in context[:12])
                                          + f"\n\nQUESTION: {question}"},
        ],
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11435/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"].get("content") or ""
        try:
            return str(json.loads(content).get("answer") or "")
        except json.JSONDecodeError:
            return content.strip()[:80]
    except Exception as e:  # noqa: BLE001
        print(f"    reader failed: {type(e).__name__}: {e}", flush=True)
        return ""


FACTORIES = {"neuromatrix": neuromatrix_context_factory, "mem0": mem0_context_factory}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(FACTORIES))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sample", type=int, default=60, help="questions per conversation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-samples", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))[: args.limit_samples]
    rng = random.Random(args.seed)
    factory = FACTORIES[args.system]
    per_cat: dict[str, list[float]] = {}
    t_start = time.time()

    for sample in data:
        if args.system == "mem0":
            state = BENCH / "ingest_state.json"
            done = json.loads(state.read_text(encoding="utf-8")).get("done", []) if state.exists() else []
            if str(sample.get("sample_id")) not in done:
                print(f"[{sample.get('sample_id')}] not ingested — skipping", flush=True)
                continue
        conv = sample["conversation"]
        dmap: dict[str, str] = {}
        for sk in (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")):
            for turn in conv[sk]:
                if turn.get("dia_id"):
                    dmap[str(turn["dia_id"])] = f"{turn.get('speaker', '')}: {turn.get('text', '')}"
        qa_all = [q for q in sample["qa"] if q.get("evidence")]
        qa = rng.sample(qa_all, min(args.sample, len(qa_all))) if args.sample else qa_all
        ctx = factory(sample)
        for i, item in enumerate(qa, 1):
            cat = CATEGORY_NAMES.get(int(item.get("category") or 0), "other")
            texts = ctx(str(item["question"]), args.k)
            gold = item.get("answer") or item.get("adversarial_answer") or ""
            pred = ask(str(item["question"]), texts)
            per_cat.setdefault(cat, []).append(f1(pred, gold))
            if i % 10 == 0:
                print(f"  [{sample.get('sample_id')}] {i}/{len(qa)}", flush=True)
        try:
            ctx.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    print(f"\n=== {args.system} | answer F1 (same reader, same prompt) ===")
    allv: list[float] = []
    for c, vals in sorted(per_cat.items()):
        allv.extend(vals)
        print(f"[{c:<11}] n={len(vals):>3}  avg F1 = {statistics.mean(vals) * 100:.1f}%")
    if allv:
        print(f"OVERALL QA-accuracy (F1): {statistics.mean(allv) * 100:.1f}%  (n={len(allv)})")
    print(f"wall time: {(time.time() - t_start) / 60:.1f} min")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"system": args.system, "overall_f1": statistics.mean(allv) * 100 if allv else 0,
             "per_cat": {c: statistics.mean(v) * 100 for c, v in per_cat.items()}, "n": len(allv)},
            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
