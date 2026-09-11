"""eval_semantic_gap.py — honest measurement of the retrieval architecture's
real ceiling, not the recall it was tuned against.

NeuroMatrix retrieval is hybrid but NOT semantic: entity/alias graph +
FTS5 lexical match. It recalls extremely well whenever a query repeats an
ANCHOR (id_777, TON, Supabase, ...) — that is most real usage, and is what
scripts/eval_recall.py and bench_recall.py measure (both score 100% because
every one of their scenarios repeats an anchor token).

This script measures the one case those suites cannot see: a question that
shares NEITHER an anchor entity NOR any content word with the stored fact —
pure conceptual paraphrase. That is the honest, currently-unclosed gap of a
lexical+graph engine with no embedding layer. Expect (and report) a LOW
number here — the point is to know the real boundary, not to inflate it.

Run:  python scripts/eval_semantic_gap.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402

# Each case: a stored fact, and a query that a human would consider an
# obvious paraphrase of the SAME fact but that shares no anchor entity and
# (deliberately) no content word with it either.
CASES = [
    {
        "label": "provider switch reason (no shared anchor or word)",
        "fact": "Переключились на Supabase, потому что нужен был встроенный "
                "auth и realtime подписки.",
        "query": "Почему сменили предыдущего поставщика бэкенда?",
    },
    {
        "label": "offline capability (no shared anchor or word)",
        "fact": "id_42 хранит данные локально и продолжает работать без сети.",
        "query": "Что из наших инструментов не требует подключения к интернету?",
    },
    {
        "label": "root-cause of an outage (no shared anchor or word)",
        "fact": "Сервис TON падал из-за исчерпания лимита запросов к API.",
        "query": "Почему у нас недавно был сбой на проде?",
    },
    {
        "label": "control case: shared anchor (should recall fine)",
        "fact": "Переключились на Supabase, потому что нужен был встроенный "
                "auth и realtime подписки.",
        "query": "Что там с Supabase?",
    },
]


def main() -> int:
    path = os.path.join(tempfile.mkdtemp(), "semgap.db")
    store = NeuroMatrixStore(path)
    results = []
    for c in CASES:
        store.remember(c["fact"], source="turn:assistant")
        hits = store.search(c["query"], limit=5)
        found = any(c["fact"] in h.get("text", "") for h in hits)
        results.append({**c, "recalled": found, "top_hit": hits[0]["text"][:90] if hits else ""})
    store.close()

    print("=== semantic-gap diagnostic (NOT a pass/fail suite — a boundary measurement) ===\n")
    for r in results:
        mark = "RECALLED" if r["recalled"] else "MISSED  "
        print(f"[{mark}] {r['label']}")
        print(f"    fact : {r['fact']}")
        print(f"    query: {r['query']}")
        if not r["recalled"]:
            print(f"    top hit instead: {r['top_hit'] or '(nothing)'}")
        print()

    no_anchor_cases = [r for r in results if "control case" not in r["label"]]
    hit_rate = sum(r["recalled"] for r in no_anchor_cases) / len(no_anchor_cases)
    control = next(r for r in results if "control case" in r["label"])
    print(f"No-shared-anchor paraphrase recall: {sum(r['recalled'] for r in no_anchor_cases)}/"
          f"{len(no_anchor_cases)} = {hit_rate:.0%}  <- the real, currently unclosed ceiling")
    print(f"Control (shared anchor) recall:     {'1/1 = 100%' if control['recalled'] else '0/1 = 0%'}"
          "  <- confirms the engine is not simply broken, it is anchor-bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
