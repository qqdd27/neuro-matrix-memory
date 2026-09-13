"""Does splitting long facts at write time fix retrieval?

Motivation (measured on a live database): stored facts average 1013 characters
and reach the 2000-character cap, because whole assistant answers are stored as
single facts.  A fact then contains several unrelated topics, so BM25 — which
normalises by document length — cannot rank it well, and the graph links
everything to everything.  LoCoMo cannot see this because its facts are single
dialogue turns, so this script reproduces the real shape: it re-writes the same
conversations as long blocks, then measures whether splitting them into
sentence-sized facts restores retrieval.

Three configurations, same questions, same gold evidence:

  turns    — the original short turns (the published baseline shape)
  blocks   — turns concatenated into ~1200-character facts (the live shape)
  chunked  — the same blocks, split into sentences at write time

Run:  python scripts/experiment_long_facts.py [--sample 150] [--k 8]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "bench_data" / "locomo10.json"

CATEGORY_NAMES = {
    1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial",
}

# Split on sentence enders, keeping reasonably long fragments together.  Ru and
# En punctuation both appear in real transcripts.
_SENT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str, *, min_len: int = 25) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 <= 220:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return [s for s in out if len(s) >= min_len] or [text]


def blocks_from_turns(turns: list[str], *, target: int = 1200) -> list[list[str]]:
    """Concatenate turns into fixed-size blocks, remembering which turn ids each
    block covers so a gold turn still maps onto its block (or fragment)."""
    blocks: list[list[int]] = []
    buf: list[int] = []
    size = 0
    for i, t in enumerate(turns):
        if size + len(t) + 1 > target and buf:
            blocks.append(buf)
            buf, size = [], 0
        buf.append(i)
        size += len(t) + 1
    if buf:
        blocks.append(buf)
    return blocks


def load_questions(sample: dict, limit: int | None) -> list[dict]:
    qs = sample.get("qa") or []
    if limit:
        qs = qs[:limit]
    return qs


def evidence_turn_texts(item: dict, dia_to_text: dict[str, str]) -> set[str]:
    """Gold evidence is given as dia_ids ("D1:3"); the main harness maps them to
    the ORIGINAL TURN TEXT and checks that text resurfaced.  Matching by index is
    wrong (dia ids are per-session, storage indices are global), which silently
    produced a 4.5% baseline in an earlier version of this script."""
    ev = item.get("evidence") or []
    if isinstance(ev, str):
        ev = [ev]
    out: set[str] = set()
    for e in ev:
        txt = dia_to_text.get(str(e).strip())
        if txt:
            out.add(txt)
    return out


def build_store(turns: list[str], mode: str) -> tuple[NeuroMatrixStore, dict[str, int]]:
    """Returns (store, turn_text -> fact_id).  For block/chunked modes every turn
    text that went into a stored fact maps to that fact, so a gold turn still
    resolves."""
    path = Path(tempfile.mkdtemp()) / f"{mode}.db"
    store = NeuroMatrixStore(str(path), llm=None, llm_daily_budget=0)
    mapping: dict[str, int] = {}
    base_ts = time.time() - 400 * 86400
    if mode == "turns":
        for i, t in enumerate(turns):
            fid = store.remember(t, source="turn", ts=base_ts + i)
            if fid:
                mapping[t] = fid
        return store, mapping
    blocks = blocks_from_turns(turns)
    for turn_ids in blocks:
        blob = " ".join(turns[i] for i in turn_ids)
        ts = base_ts + max(turn_ids)
        if mode == "blocks":
            fid = store.remember(blob, source="turn", ts=ts)
            if fid:
                for i in turn_ids:
                    mapping[turns[i]] = fid
        else:  # chunked
            for frag in split_sentences(blob):
                fid = store.remember(frag, source="turn", ts=ts)
                if fid is None:
                    continue
                for i in turn_ids:
                    if turns[i] in frag or frag in turns[i]:
                        mapping.setdefault(turns[i], fid)
    return store, mapping


def store_of_texts(mapping: dict[str, int], texts: set[str]) -> set[int]:
    return {mapping[t] for t in texts if t in mapping}


def run(mode: str, sample: dict, limit: int | None, k: int, *, embedder=None) -> dict:
    conv = sample["conversation"]
    keys = sorted((x for x in conv if x.startswith("session_") and not x.endswith("_date_time")),
                  key=lambda s: int(s.split("_")[1]))
    # dia_id -> original turn text, exactly as the main harness builds it.
    dia_to_text: dict[str, str] = {}
    turns: list[str] = []
    for sk in keys:
        for turn in conv[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if not text:
                continue
            line = f"{speaker}: {text}"
            turns.append(line)
            if turn.get("dia_id"):
                dia_to_text[str(turn["dia_id"])] = line
    t0 = time.perf_counter()
    store, mapping = build_store(turns, mode)
    ingest_s = time.perf_counter() - t0
    facts = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    avg_len = store._conn.execute("SELECT AVG(length(text)) FROM facts").fetchone()[0] or 0

    per_cat: dict[str, list[int]] = {}
    lat: list[float] = []
    for item in load_questions(sample, limit):
        cat = CATEGORY_NAMES.get(int(item.get("category", 0) or 0), "other")
        want = store_of_texts(mapping, evidence_turn_texts(item, dia_to_text))
        if not want:
            continue
        t1 = time.perf_counter()
        hits = store.search(str(item.get("question", "")), limit=k, include_dossiers=False)
        lat.append((time.perf_counter() - t1) * 1000)
        got = {int(h["fact_id"]) for h in hits}
        per_cat.setdefault(cat, []).append(1 if (want & got) else 0)
    store.close()
    flat = [v for vals in per_cat.values() for v in vals]
    return {
        "mode": mode,
        "facts": facts,
        "avg_len": round(avg_len),
        "ingest_s": round(ingest_s, 1),
        "overall": (sum(flat) / len(flat) * 100) if flat else 0.0,
        "n": len(flat),
        "per_cat": {c: (sum(v) / len(v) * 100, len(v)) for c, v in sorted(per_cat.items())},
        "p50_ms": round(statistics.median(lat), 1) if lat else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150, help="questions per conversation (0 = all)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dialogs", type=int, default=3)
    args = ap.parse_args()

    if not DATA.exists():
        print(f"missing dataset: {DATA}")
        return 1
    data = json.loads(DATA.read_text(encoding="utf-8"))
    print(f"dataset: {DATA.name} | dialogs used: {args.dialogs} | k={args.k}")
    results = []
    for mode in ("turns", "blocks", "chunked"):
        agg: dict[str, list[int]] = {}
        tot = 0
        for sample in data[: args.dialogs]:
            r = run(mode, sample, args.sample or None, args.k)
            tot += r["n"]
            for c, (pct, n) in r["per_cat"].items():
                agg.setdefault(c, []).extend([1] * round(pct / 100 * n) + [0] * (n - round(pct / 100 * n)))
            if mode == "turns":
                facts, avg_len, ingest = r["facts"], r["avg_len"], r["ingest_s"]
                p50 = r["p50_ms"]
        flat = [v for vals in agg.values() for v in vals]
        overall = sum(flat) / len(flat) * 100 if flat else 0.0
        results.append((mode, overall, len(flat), agg))

    base = results[0][1]
    print()
    print(f"{'mode':<9} {'overall':>8} {'Δ vs turns':>11} {'n':>6}  per-category")
    for mode, overall, n, agg in results:
        cats = "  ".join(f"{c}={sum(v)/len(v)*100:.1f}%" for c, v in sorted(agg.items()))
        print(f"{mode:<9} {overall:>7.1f}% {overall - base:>+10.1f}  {n:>6}  {cats}")
    print()
    print("read: if 'blocks' loses a lot to 'turns' and 'chunked' recovers it, then")
    print("write-time splitting (not ranking) is the bottleneck on real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
