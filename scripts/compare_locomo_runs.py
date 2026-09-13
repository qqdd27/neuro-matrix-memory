"""Compare two LoCoMo run reports (e.g. with-memory vs --no-memory baseline).

Both reports are the markdown that ``eval_locomo.py --out`` writes, so this
needs no re-run and no API access — run it after the fact:

    python scripts/compare_locomo_runs.py docs/locomo-local-baseline.md docs/locomo-local-memory.md

It prints the two sections that matter side by side:
  * QA-accuracy (F1) — what a reader model actually answered, per category
  * evidence-hit@k  — the retrieval ceiling, when the run measured it
    (a --no-memory run deliberately measures none)

The delta column is the whole point: "our memory scored X" only means
something next to "the bare model scored Y" on the identical question set.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# | 1 single-hop | 281 | 85/281 (30.2%) |
_HIT_RE = re.compile(r"^\|\s*(?P<cat>[\w\s\-]+?)\s*\|\s*(?P<n>\d+)\s*\|\s*(?P<a>\d+)/(?P<b>\d+)\s*\((?P<pct>[\d.]+)%\)\s*\|")
# | 1 single-hop | 281 | avg F1 |
_F1_RE = re.compile(r"^\|\s*(?P<cat>[\w\s\-]+?)\s*\|\s*(?P<n>\d+)\s*\|\s*(?P<pct>[\d.]+)%\s*\|")
_OVERALL_HIT_RE = re.compile(r"OVERALL evidence-hit@\d+:\s*(?P<a>\d+)/(?P<b>\d+)\s*=\s*(?P<pct>[\d.]+)%")
# The console prints "OVERALL QA-accuracy (F1): 15.7%", the markdown report
# writes "**Overall F1: 15.7%**" — accept both, or the delta row silently
# reads nan.
_OVERALL_F1_RE = re.compile(r"OVERALL QA-accuracy \(F1\):\s*(?P<pct>[\d.]+)%")
_OVERALL_F1_MD_RE = re.compile(r"\*{0,2}Overall F1:\*{0,2}\s*(?P<pct>[\d.]+)%")


def parse(path: str) -> dict:
    """Pull {hits: {cat: pct}, f1: {cat: pct}, overall_hit, overall_f1, meta} from a report."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict = {"hits": {}, "f1": {}, "overall_hit": None, "overall_f1": None,
                 "k": None, "model": None, "questions": None, "file": path}
    m = re.search(r"evidence-hit@(\d+)", text)
    if m:
        out["k"] = int(m.group(1))
    m = re.search(r"questions scored:\s*(\d+)", text)
    if m:
        out["questions"] = int(m.group(1))
    m = re.search(r"model=([\w.:\-]+)", text)
    if m:
        out["model"] = m.group(1)

    # Section-scoped parsing: F1 rows and hit rows look similar, so the hit
    # table is only read before the QA-accuracy header.
    split = text.split("QA-accuracy (F1)")
    hit_part, f1_part = split[0], (split[1] if len(split) > 1 else "")
    for line in hit_part.splitlines():
        m = _HIT_RE.match(line.strip())
        if m and m.group("cat").strip().lower() not in ("category",):
            out["hits"][m.group("cat").strip()] = float(m.group("pct"))
    for line in f1_part.splitlines():
        m = _F1_RE.match(line.strip())
        if m and m.group("cat").strip().lower() not in ("category",):
            out["f1"][m.group("cat").strip()] = float(m.group("pct"))
    m = _OVERALL_HIT_RE.search(text)
    if m and int(m.group("b")):
        out["overall_hit"] = float(m.group("pct"))
    m = _OVERALL_F1_RE.search(text) or _OVERALL_F1_MD_RE.search(text)
    if m:
        out["overall_f1"] = float(m.group("pct"))
    return out


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{d:+.1f} п.п." if abs(d) >= 0.05 else "≈0"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    base_path, mem_path = argv[1], argv[2]
    base, mem = parse(base_path), parse(mem_path)

    print("=" * 74)
    print(f"  baseline (без памяти): {base_path}")
    print(f"  с памятью            : {mem_path}")
    for label, r in (("baseline", base), ("memory", mem)):
        bits = [f"k={r['k']}" if r["k"] else None,
                f"model={r['model']}" if r["model"] else None,
                f"questions={r['questions']}" if r["questions"] else None]
        print(f"  {label:<9}: " + "  ".join(b for b in bits if b))
    print("=" * 74)

    if mem["f1"] or base["f1"]:
        print("\nQA-accuracy (F1) — что модель реально ответила")
        print(f"  {'категория':<16} {'без памяти':>12} {'с памятью':>12} {'Δ':>10}")
        for cat in sorted(set(base["f1"]) | set(mem["f1"])):
            print(f"  {cat:<16} {base['f1'].get(cat, float('nan')):>11.1f}% "
                  f"{mem['f1'].get(cat, float('nan')):>11.1f}% "
                  f"{_delta(base['f1'].get(cat), mem['f1'].get(cat)):>10}")
        print(f"  {'OVERALL':<16} {base['overall_f1'] if base['overall_f1'] is not None else float('nan'):>11.1f}% "
              f"{mem['overall_f1'] if mem['overall_f1'] is not None else float('nan'):>11.1f}% "
              f"{_delta(base['overall_f1'], mem['overall_f1']):>10}")
    else:
        print("\nQA-accuracy (F1): в отчётах нет (запускали без --llm)")

    if mem["hits"]:
        print(f"\nevidence-hit@{mem['k']} — потолок поиска (в baseline не измеряется)")
        print(f"  {'категория':<16} {'hit@k':>12}")
        for cat in sorted(mem["hits"]):
            print(f"  {cat:<16} {mem['hits'][cat]:>11.1f}%")
        if mem["overall_hit"] is not None:
            print(f"  {'OVERALL':<16} {mem['overall_hit']:>11.1f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
