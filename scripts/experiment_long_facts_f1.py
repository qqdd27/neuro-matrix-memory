"""Does write-time splitting help once we score ANSWERS, not hits?

Earlier the same question was answered with evidence-hit@8, which rewards the
wrong thing: a larger fact "contains" the gold turn more often for free (measured
88.3% for joined blocks vs 61.3% for the original turns).  That number says
nothing about whether a model can answer from the retrieved window.

This variant keeps the write shapes and scores token-F1 with the same local
reader used everywhere else, on the same questions, so the comparison is about
usefulness:

  turns    — the original short turns (LoCoMo as published)
  blocks   — turns joined into ~1200-character facts, i.e. the shape live
             databases actually have (whole assistant answers are ~1000 chars)
  chunked  — those same blocks split into sentence-sized facts at write time

Run:  python scripts/experiment_long_facts_f1.py [--sample 60] [--k 8]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402
from scripts.experiment_long_facts import (  # noqa: E402
    blocks_from_turns, split_sentences, DATA, CATEGORY_NAMES,
)

SHIM = "http://127.0.0.1:11435/v1/chat/completions"
READER_SYSTEM = ("Answer the question using ONLY the facts below. Extract "
                 "the SHORTEST possible answer span (2-8 words, a name, "
                 "date, or short phrase) — reuse the facts' own wording "
                 "verbatim wherever possible, never a full sentence, "
                 "never restate the question, no filler words like "
                 "'likely' or 'according to the facts'. Each fact may be "
                 "prefixed with [YYYY-MM-DD]: that is the date of the "
                 "event, so for 'when' questions answer with that date. If "
                 "the facts do not contain the answer, still give your "
                 'single best short guess. Return JSON {"answer": "..."}.')


def f1(pred: str, gold: str) -> float:
    from collections import Counter
    import re

    norm = lambda s: re.sub(r"[^0-9a-zа-яё ]+", " ", str(s).lower()).split()  # noqa: E731
    p, g = norm(pred), norm(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def ask(question: str, context: list[str]) -> str:
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
    req = urllib.request.Request(SHIM, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        content = payload["choices"][0]["message"].get("content") or ""
        try:
            return str(json.loads(content).get("answer") or "")
        except json.JSONDecodeError:
            return content.strip()[:80]
    except Exception:  # noqa: BLE001
        return ""


def build(turns: list[str], mode: str, tmp: Path):
    store = NeuroMatrixStore(str(tmp / f"{mode}.db"), llm=None, llm_daily_budget=0)
    base = time.time() - 400 * 86400
    truth: dict[str, int] = {}
    if mode == "turns":
        for i, t in enumerate(turns):
            fid = store.remember(t, source="turn", ts=base + i)
            if fid:
                truth[t] = fid
        return store, truth
    blocks = blocks_from_turns(turns)
    for ids in blocks:
        blob = " ".join(turns[i] for i in ids)
        ts = base + max(ids)
        if mode == "blocks":
            fid = store.remember(blob, source="turn", ts=ts)
            if fid:
                for i in ids:
                    truth.setdefault(turns[i], fid)
        else:
            for frag in split_sentences(blob):
                fid = store.remember(frag, source="turn", ts=ts)
                if fid is None:
                    continue
                for i in ids:
                    if turns[i] in frag or frag in turns[i]:
                        truth.setdefault(turns[i], fid)
    return store, truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dialogs", type=int, default=1)
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))[: args.dialogs]
    tmp = Path("D:/WEB/nm-f1exp")
    tmp.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[float]] = {}
    for mode in ("turns", "blocks", "chunked"):
        scores: list[float] = []
        t0 = time.time()
        for sample in data:
            conv = sample["conversation"]
            keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
                          key=lambda s: int(s.split("_")[1]))
            dia: dict[str, str] = {}
            turns: list[str] = []
            for sk in keys:
                for turn in conv[sk]:
                    text = str(turn.get("text") or "").strip()
                    if len(text) < 4:
                        continue
                    line = f"{turn.get('speaker', '')}: {text}"
                    turns.append(line)
                    if turn.get("dia_id"):
                        dia[str(turn["dia_id"])] = line
            store, truth = build(turns, mode, tmp)
            qa = [q for q in sample["qa"] if q.get("evidence")][: args.sample]
            for item in qa:
                texts = [str(h.get("text") or "") for h in
                         store.search(str(item["question"]), limit=args.k, include_dossiers=False)]
                gold = item.get("answer") or item.get("adversarial_answer") or ""
                scores.append(f1(ask(str(item["question"]), texts), gold))
            store.close()
        results[mode] = scores
        print(f"  {mode}: F1 {statistics.mean(scores) * 100:.1f}% (n={len(scores)}, {time.time() - t0:.0f}s)", flush=True)

    base = statistics.mean(results["turns"]) * 100
    print()
    print(f"{'mode':<9} {'answer F1':>10} {'Δ vs turns':>11}  avg facts/context")
    for mode, sc in results.items():
        m = statistics.mean(sc) * 100
        print(f"{mode:<9} {m:>9.1f}% {m - base:>+10.1f}")
    print()
    print("Read: 'blocks' is the shape live databases have. If it is far below")
    print("'turns' and 'chunked' recovers, write-time splitting is worth shipping;")
    print("if all three are similar, the earlier 88.3% was purely a metric artefact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
