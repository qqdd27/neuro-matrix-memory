"""bench_recall.py — recall@k over realistic RU/EN scenarios.

Prints measured numbers for the README (no marketing).  --quick runs a
reduced set for CI.

Run:  python scripts/bench_recall.py [--quick]
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402

SCENARIOS = [
    # (label, [seed dialogs], query, expected_substring)
    ("cross-session id (RU)", [
        ("Криптовалюта TON — это id_777, кошелёк для тестов.", "Да, TON это id_777, тестовый кошелёк."),
        ("Что там по id_777?", "id_777 (TON) активен, выплаты раз в неделю."),
    ], "id_777", "выплаты"),
    ("decision trail", [
        ("Выбираем БД для проекта.", "Проект выбрал Supabase, нужен SQL и RLS."),
        ("Переходим на Firebase, офлайн критичен.", "Firebase из-за offline-синка, сроки горят."),
    ], "Firebase", "Firebase"),
    ("alias probe", [
        ("Кошелёк TON — это id_777.", "id_777 (TON) под тесты."),
    ], "кошелёк", "id_777"),
    ("episodic recency", [
        ("Статус токена XTK?", "XTK был делистан на бирже."),
        ("Статус токена XTK?", "XTK снова листится, объёмы растут."),
    ], "XTK", "снова"),
    ("cross-session link (EN)", [
        ("Wallet TON is id_777 for tests.", "id_777 works on TON."),
    ], "TON", "id_777"),
]


def run(quick: bool = False) -> dict:
    scenarios = SCENARIOS if not quick else SCENARIOS[:2]
    results = []
    t_start = time.time()
    for label, dialogs, query, expect in scenarios:
        path = os.path.join(tempfile.mkdtemp(), "bench.db")
        store = NeuroMatrixStore(path)
        t0 = time.time()
        for user, assistant in dialogs:
            store.add_turn(user, assistant, session_id=label)
        hits = store.search(query, limit=5)
        dt = (time.time() - t0) * 1000
        ok = any(expect.lower() in (h["text"] or "").lower() for h in hits)
        store.close()
        results.append({"scenario": label, "recall@5": ok, "latency_ms": round(dt, 1),
                        "hits": len(hits)})
    return {"scenarios": results,
            "total": len(results), "hit": sum(1 for r in results if r["recall@5"]),
            "total_ms": round((time.time() - t_start) * 1000, 1)}


if __name__ == "__main__":
    quick = "--quick" in sys.argv[1:]
    report = run(quick=quick)
    for r in report["scenarios"]:
        mark = "OK " if r["recall@5"] else "MISS"
        print(f"{mark} {r['scenario']:<24} latency {r['latency_ms']:>7} ms  hits {r['hits']}")
    print(f"\n{report['hit']}/{report['total']} scenarios recalled "
          f"in {report['total_ms']} ms total")
    sys.exit(0 if report["hit"] == report["total"] else 1)
