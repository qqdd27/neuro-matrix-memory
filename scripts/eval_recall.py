"""eval_recall.py — measure how well neuromatrix recalls its own knowledge.

Two modes (search mirrors prefetch: store.search top-k):

  --auto N          build N questions from the top dossier entities; ground
                    truth = the dossier summary itself must resurface.
  --scenarios F     manual JSON: [{"label": "...", "query": "...",
                    "must_contain": ["token", ...]}, ...]

Output: per-item PASS/FAIL with the best hit preview, overall recall@k, and
optional markdown report (--out report.md).

Run:
  python scripts/eval_recall.py --db "C:/Users/.../neuromatrix.db" --auto 10
  python scripts/eval_recall.py --db ... --scenarios scripts/scenarios.example.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402


def _auto_items(store: NeuroMatrixStore, n: int) -> list[dict]:
    conn = store._conn
    rows = conn.execute(
        "SELECT e.key, d.summary FROM dossiers d "
        "JOIN entities e ON e.id = d.entity_id "
        "ORDER BY e.hits DESC LIMIT ?", (n,)).fetchall()
    items = []
    for r in rows:
        summary = (r["summary"] or "").strip()
        if not summary:
            continue
        items.append({
            "label": f"dossier:{r['key']}",
            "query": str(r["key"]),
            "must_contain": [summary[:48]],
        })
    return items


def _load_scenarios(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def evaluate(store: NeuroMatrixStore, items: list[dict], k: int) -> list[dict]:
    results = []
    for it in items:
        query = it["query"]
        must = [str(m).lower() for m in it.get("must_contain", [])]
        try:
            hits = store.search(query, limit=k)
        except Exception as exc:  # noqa: BLE001
            hits = []
            results.append({"label": it["label"], "query": query, "pass": False,
                            "error": str(exc), "best": ""})
            continue
        texts = [h.get("text", "") for h in hits]
        blob = "\n".join(texts).lower()
        ok = all(m in blob for m in must)
        best = (texts[0][:130].replace("\n", " ") if texts else "")
        results.append({"label": it["label"], "query": query, "pass": ok,
                        "best": best, "k": len(hits)})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to neuromatrix.db")
    ap.add_argument("--auto", type=int, default=0,
                    help="build N questions from top dossier entities")
    ap.add_argument("--scenarios", default=None,
                    help="manual scenarios JSON file")
    ap.add_argument("--k", type=int, default=5, help="top-k (default 5)")
    ap.add_argument("--out", default=None, help="optional markdown report path")
    args = ap.parse_args(argv)

    store = NeuroMatrixStore(args.db)
    items: list[dict] = []
    if args.scenarios:
        items += _load_scenarios(args.scenarios)
    if args.auto:
        items += _auto_items(store, args.auto)
    if not items:
        store.close()
        ap.error("provide --auto N and/or --scenarios FILE")
    try:
        results = evaluate(store, items, args.k)
    finally:
        store.close()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    recall = passed / total if total else 0.0
    print(f"\n=== recall@{args.k}: {passed}/{total} = {recall:.0%} ===")
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[{mark}] {r['label']} :: {r['query']}")
        if not r["pass"]:
            print(f"       best: {r['best'][:120]}")
    if args.out:
        lines = [
            "# NeuroMatrix recall report",
            "",
            f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
            f"- db: `{args.db}`",
            f"- recall@{args.k}: **{passed}/{total} ({recall:.0%})**",
            "",
            "| Result | Item | Query | Best hit |",
            "|---|---|---|---|",
        ]
        for r in results:
            lines.append(f"| {'✅' if r['pass'] else '❌'} | {r['label']} | "
                         f"{r['query']} | {r['best'][:90]} |")
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nreport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
