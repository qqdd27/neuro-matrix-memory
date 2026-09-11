"""eval_locomo.py — evaluate NeuroMatrix retrieval against LoCoMo
(snap-research/locomo, arXiv:2402.17753 "Evaluating Very Long-Term
Conversational Memory of LLM Agents"), the external, non-self-authored
benchmark this project had never been run against.

Honest scope: LoCoMo's published numbers are QA ACCURACY after an LLM reads
retrieved context and answers the question (retrieval + generation). This
script measures the retrieval half only — EVIDENCE HIT@k: for each question,
does the original conversation turn(s) the gold answer depends on
(`evidence`, e.g. "D2:3") show up among NeuroMatrix's top-k `search()` hits?
That is the ceiling on what any downstream reader (LLM or human) could
possibly answer correctly from what this memory handed back. It is NOT the
same number LoCoMo leaderboards report (no LLM reader is invoked here, by
design — zero extra dependency, zero API cost) and this script says so in
its own output rather than implying a like-for-like comparison.

Each of LoCoMo's 10 long conversations (19-32 sessions, both speakers) is
ingested into a FRESH store (speaker name prefixed onto every turn, so a
TitleCase anchor is always present) with timestamps taken from each
session's real date/time. QA pairs use LoCoMo's own category scheme
(1=single-hop, 2=temporal, 3=multi-hop, 4=open-domain, 5=adversarial).

Data license: LoCoMo (snap-research/locomo) is released under CC BY-NC 4.0
(non-commercial). It is fetched on demand into the git-ignored
scripts/bench_data/ directory the first time this script runs and is never
vendored into this (MIT-licensed) repository — only the aggregate NUMBERS
this script prints/writes are ours to keep, not the dataset's copyrighted
dialogue text.

Run:
    python scripts/eval_locomo.py
    python scripts/eval_locomo.py --k 8 --out docs/locomo-report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402

_LOCOMO_URL = ("https://raw.githubusercontent.com/snap-research/locomo/"
               "main/data/locomo10.json")


def _ensure_dataset(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching LoCoMo dataset (CC BY-NC 4.0, snap-research/locomo) -> {path} ...")
    urllib.request.urlretrieve(_LOCOMO_URL, path)

CATEGORY_NAMES = {
    1: "single-hop", 2: "temporal", 3: "multi-hop",
    4: "open-domain", 5: "adversarial",
}

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
_DT_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s*on\s*(\d{1,2})\s*([A-Za-z]+),\s*(\d{4})",
    re.IGNORECASE)


def _parse_dt(s: str, fallback: float) -> float:
    """'1:56 pm on 8 May, 2023' -> unix ts. Falls back to a monotonic stub
    (fallback + small increment) if the format ever changes upstream --
    ordering matters far more to this eval than absolute calendar accuracy."""
    m = _DT_RE.search(s or "")
    if not m:
        return fallback
    hh, mm, ap, day, mon, year = m.groups()
    hh = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    month = _MONTHS.get(mon.capitalize())
    if month is None:
        return fallback
    try:
        import datetime
        dt = datetime.datetime(int(year), month, int(day), hh, int(mm))
        return dt.timestamp()
    except ValueError:
        return fallback


def _dia_map(conv: dict) -> dict[str, str]:
    """dia_id ('D3:7') -> original turn text (speaker-prefixed, matching what
    was actually stored) for evidence lookup."""
    out: dict[str, str] = {}
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]))
    for k in session_keys:
        for turn in conv[k]:
            out[turn["dia_id"]] = f"{turn['speaker']}: {turn['text']}"
    return out


def run_sample(sample: dict, k: int) -> dict:
    conv = sample["conversation"]
    session_keys = sorted(
        (key for key in conv if key.startswith("session_") and not key.endswith("_date_time")),
        key=lambda key: int(key.split("_")[1]))
    path = os.path.join(tempfile.mkdtemp(), f"{sample['sample_id']}.db")
    store = NeuroMatrixStore(path)
    base_ts = time.time() - 400 * 86400
    for i, sk in enumerate(session_keys):
        dt_str = conv.get(f"{sk}_date_time", "")
        ts = _parse_dt(dt_str, base_ts + i * 86400)
        sid = f"{sample['sample_id']}_{sk}"
        for turn in conv[sk]:
            store.remember(f"{turn['speaker']}: {turn['text']}", source="turn",
                           session_id=sid, ts=ts, importance=1.0)

    dmap = _dia_map(conv)
    per_cat: dict[int, list[bool]] = {}
    lat = []
    for qa in sample["qa"]:
        cat = int(qa.get("category") or 0)
        ev_ids = qa.get("evidence") or []
        ev_texts = [dmap[e] for e in ev_ids if e in dmap]
        if not ev_texts:
            continue  # no gold evidence to check against (rare, skip)
        t0 = time.time()
        hits = store.search(qa["question"], limit=k)
        lat.append((time.time() - t0) * 1000)
        hit_blob = "\n".join(h.get("text", "") for h in hits)
        found = any(ev in hit_blob for ev in ev_texts)
        per_cat.setdefault(cat, []).append(found)
    store.close()
    return {"sample_id": sample["sample_id"], "per_cat": per_cat, "latency_ms": lat}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).parent / "bench_data" / "locomo10.json"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="debug: only run the first N conversations")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    _ensure_dataset(data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit_samples:
        data = data[: args.limit_samples]

    t0 = time.time()
    all_per_cat: dict[int, list[bool]] = {}
    all_lat: list[float] = []
    per_sample_summary = []
    for sample in data:
        r = run_sample(sample, args.k)
        for cat, results in r["per_cat"].items():
            all_per_cat.setdefault(cat, []).extend(results)
        all_lat.extend(r["latency_ms"])
        n = sum(len(v) for v in r["per_cat"].values())
        hits = sum(sum(v) for v in r["per_cat"].values())
        per_sample_summary.append((r["sample_id"], hits, n))
    elapsed = time.time() - t0

    total_n = sum(len(v) for v in all_per_cat.values())
    total_hit = sum(sum(v) for v in all_per_cat.values())

    print(f"=== LoCoMo evidence-hit@{args.k} (retrieval ceiling, NOT a QA-accuracy "
          f"comparison to published LoCoMo leaderboard numbers) ===\n")
    print(f"{len(data)} conversations, {total_n} scored questions "
          f"(of {sum(len(s['qa']) for s in data)} total; unscored ones had no "
          f"resolvable evidence dia_id), evaluated in {elapsed:.1f}s\n")
    lines_md = [
        "# LoCoMo evidence-hit@{} report".format(args.k), "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- conversations: {len(data)}, questions scored: {total_n}",
        f"- overall evidence-hit@{args.k}: **{total_hit}/{total_n} "
        f"({total_hit/total_n:.1%})**" if total_n else "- no questions scored",
        "", "| Category | n | hit@{} |".format(args.k), "|---|---|---|",
    ]
    for cat in sorted(all_per_cat):
        v = all_per_cat[cat]
        rate = sum(v) / len(v) if v else 0.0
        name = CATEGORY_NAMES.get(cat, str(cat))
        print(f"[{name:11s}] n={len(v):4d}  hit@{args.k} = {sum(v):4d}/{len(v):<4d} = {rate:.1%}")
        lines_md.append(f"| {cat} {name} | {len(v)} | {sum(v)}/{len(v)} ({rate:.1%}) |")
    overall_rate = (total_hit / total_n) if total_n else 0.0
    print(f"\nOVERALL evidence-hit@{args.k}: {total_hit}/{total_n} = {overall_rate:.1%}")
    if all_lat:
        sl = sorted(all_lat)
        print(f"search() latency: p50={sl[len(sl)//2]:.2f}ms  "
              f"p95={sl[int(len(sl)*0.95)]:.2f}ms  max={sl[-1]:.2f}ms")
    print("\nPer-conversation:")
    for sid, hits, n in per_sample_summary:
        print(f"  {sid:10s} {hits:3d}/{n:<3d} = {hits/n:.1%}" if n else f"  {sid} (no scored q)")

    if args.out:
        Path(args.out).write_text("\n".join(lines_md) + "\n", encoding="utf-8")
        print(f"\nreport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
