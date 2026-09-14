"""Collect every measured metric from the report files into one table.

Every number this project quotes lives in a report under docs/.  Re-typing them
into a summary by hand is how a stale or misattributed figure gets repeated, so
this reads the reports, extracts what each one measured, and writes
docs/METRICS.md plus a printable table.

Extraction is deliberately conservative: a metric is only reported when the
report states it plainly (labelled section, n, value) — a missing value is shown
as "—" rather than guessed.

Run: python scripts/metrics_summary.py [--out docs/METRICS.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
CATS = {"single-hop": "одиночный факт", "temporal": "время", "multi-hop": "связь фактов",
        "open-domain": "открытые", "adversarial": "ловушки"}


def _sample_size(text: str) -> str:
    m = re.search(r"## (?:QA-accuracy \(F1\)|Date accuracy[^,]*,)[^\n]*?n=(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"questions scored: (\d+)", text)
    return m.group(1) if m else "?"


def _model(text: str) -> str:
    m = re.search(r"model=([\w.:\-]+)", text)
    if m:
        return m.group(1)
    low = text.lower()
    if "1.5b" in low:
        return "qwen2.5:1.5b"
    if "9b" in low:
        return "qwen3.5:9b"
    return "—"


def _overall_f1(text: str) -> str:
    m = re.search(r"\*\*Overall F1: ([\d.]+)%\*\*", text)
    return f"{m.group(1)}%" if m else "—"


def _overall_hit(text: str, k: str = "8") -> str:
    m = re.search(rf"overall evidence-hit@{k}: \*\*\d+/\d+ \(([\d.]+)%\)\*\*", text)
    if not m:  # plain-text form, from a console log kept as a report
        m = re.search(rf"OVERALL evidence-hit@{k}: \d+/\d+ = ([\d.]+)%", text)
    return f"{m.group(1)}%" if m else "—"


def _overall_date(text: str) -> str:
    m = re.search(r"\*\*Overall date accuracy: ([\d.]+)%\*\*", text)
    if not m:
        m = re.search(r"OVERALL date accuracy: \d+/\d+ = ([\d.]+)%", text)
    return f"{m.group(1)}%" if m else "—"


def _overall_judge(text: str) -> str:
    m = re.search(r"\*\*Overall judge accuracy: ([\d.]+)%\*\*", text)
    if not m:
        m = re.search(r"OVERALL judge accuracy: ([\d.]+)%", text)
    return f"{m.group(1)}%" if m else "—"


def _per_cat_judge(text: str) -> dict[str, str]:
    return _md_body_rows(_section(text, "## Judge accuracy"))


def _section(text: str, header: str) -> str:
    i = text.find(header)
    if i < 0:
        return ""
    j = text.find("\n## ", i + 1)
    return text[i: j if j > 0 else len(text)]


def _md_body_rows(section: str) -> dict[str, str]:
    """Rows of a markdown table keyed by the category name in the second column."""
    out: dict[str, str] = {}
    for m in re.finditer(r"^\|\s*\d?\s*([a-z][a-z\-]+)\s*\|\s*\d+\s*\|\s*([^|]+?)\s*\|",
                         section, re.M):
        cat, value = m.group(1).strip(), m.group(2).strip()
        if cat in CATS:
            pct = re.search(r"\(([\d.]+)%\)", value) or re.search(r"([\d.]+)%", value)
            if pct:
                out[cat] = f"{pct.group(1)}%"
    return out


def _per_cat_f1(text: str) -> dict[str, str]:
    return _md_body_rows(_section(text, "## QA-accuracy (F1)"))


def _per_cat_date(text: str) -> dict[str, str]:
    return _md_body_rows(_section(text, "## Date accuracy"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DOCS / "METRICS.md"))
    args = ap.parse_args()

    rows = []
    for path in sorted(DOCS.glob("locomo-local-*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append({
            "file": path.name,
            "n": _sample_size(text),
            "model": _model(text),
            "hit": _overall_hit(text),
            "f1": _overall_f1(text),
            "date": _overall_date(text),
            "judge": _overall_judge(text),
            "cats": _per_cat_f1(text),
            "dcats": _per_cat_date(text),
            "jcats": _per_cat_judge(text),
        })

    lines = ["# NeuroMatrix — все измеренные метрики", "",
             "Собрано из файлов отчётов `docs/locomo-local-*.md` автоматически "
             "(`scripts/metrics_summary.py`); «—» означает, что отчёт этой метрики не содержит.",
             "",
             "## Сводка по прогонам", "",
             "| отчёт | n | модель | попадание факта@8 | ответы F1 | точность даты | судья (смысл) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['n']} | {r['model']} | {r['hit']} | {r['f1']} | "
                     f"{r['date']} | {r['judge']} |")

    lines += ["", "## Ответы F1 по категориям", "",
              "| отчёт | " + " | ".join(CATS.values()) + " |",
              "|---|" + "---|" * len(CATS)]
    for r in rows:
        if r["cats"]:
            lines.append(f"| {r['file']} | " + " | ".join(
                r["cats"].get(c, "—") for c in CATS) + " |")

    lines += ["", "## Точность даты по категориям", "",
              "| отчёт | " + " | ".join(CATS.values()) + " |",
              "|---|" + "---|" * len(CATS)]
    for r in rows:
        if r["dcats"]:
            lines.append(f"| {r['file']} | " + " | ".join(
                r["dcats"].get(c, "—") for c in CATS) + " |")

    lines += ["", "## Судья: смысл, а не написание", "",
              "| отчёт | " + " | ".join(CATS.values()) + " |",
              "|---|" + "---|" * len(CATS)]
    for r in rows:
        if r["jcats"]:
            lines.append(f"| {r['file']} | " + " | ".join(
                r["jcats"].get(c, "—") for c in CATS) + " |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'отчёт':38s} {'n':>5s} {'модель':14s} {'hit@8':>7s} {'F1':>7s} {'дата':>7s} {'судья':>7s}")
    for r in rows:
        print(f"{r['file'][:38]:38s} {r['n']:>5s} {r['model'][:14]:14s} "
              f"{r['hit']:>7s} {r['f1']:>7s} {r['date']:>7s} {r['judge']:>7s}")
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
