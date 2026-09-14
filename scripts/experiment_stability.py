"""Does memory keep what it already knew when new facts arrive?

The question the fruit-fly material actually raises for THIS project, and the only
one we never measured. Everything else here is measured obsessively (does recall
find the right fact, does an answer improve), but never: *does adding new facts
degrade the recall of facts that were already there?* A memory that quietly
rewrites its own past as it grows is a product defect no retrieval metric shows.

Method — the same questions on both sides of the growth:

1. write N "old" facts, each on its own topic, and derive one question per fact
   from that fact's OWN words (so the target is unambiguous);
2. measure recall@k and the rank of the correct fact → BEFORE;
3. append M unrelated new facts (the memory keeps growing, which is the point);
4. re-run the SAME questions → AFTER, against a store that is a superset.

Nothing is tuned and no setting is changed between the two arms; the only
difference is how much the store contains. A drop is forgetting, a stable number
is the property we want to be able to claim.

Run: python scripts/experiment_stability.py [--old 300] [--new 3000] [--k 8]
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402

TOPICS = [
    "deployment pipeline", "cache invalidation", "scraper scheduling", "database backup",
    "cloudflare caching", "websocket reconnect", "docker build cache", "memory profiler",
    "columnar storage", "rate limiting", "session tokens", "vector index",
    "time series rollup", "log rotation", "schema migration", "sharding strategy",
    "consensus quorum", "task queue", "object storage", "cold start latency",
    "connection pooling", "feature flags", "load balancer", "data retention",
    "encryption at rest", "tracing spans", "alert thresholds", "budget forecast",
    "incident review", "release checklist", "onboarding guide", "cost optimization",
]
NOUNS = [
    "harbour", "lantern", "anchor", "compass", "sextant", "atlas", "beacon", "ridge",
    "furnace", "meridian", "cavern", "orchard", "terrace", "basalt", "garnet", "pylon",
]
CITIES = [
    "Riga", "Porto", "Bergen", "Aarhus", "Ghent", "Turku", "Split", "Krakow",
    "Tallinn", "Ljubljana", "Bilbao", "Aalborg", "Bristol", "Gdansk", "Utrecht",
]


def make_old_facts(n: int, rng: random.Random) -> list[tuple[str, str]]:
    """(text, question) pairs: the question is answerable ONLY from that fact."""
    out: list[tuple[str, str]] = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        noun = NOUNS[(i * 7) % len(NOUNS)]
        city = CITIES[(i * 5) % len(CITIES)]
        text = (f"On project {noun}{i} the team rolled out {topic} in {city} "
                f"and measured latency {100 + (i % 400)} ms.")
        question = f"what did project {noun}{i} roll out in {city}"
        out.append((text, question))
    return out


def make_new_facts(n: int, rng: random.Random, hard: bool = False) -> list[str]:
    """Unrelated filler with the same shape but disjoint identifiers.

    ``hard`` makes the filler LEXICALLY NEAR-IDENTICAL to the old facts: same
    topics, same cities, only the project number differs.  That is the case worth
    measuring — with a unique rare token in every fact the answer is decided by a
    single word and any store passes, which tells us nothing about competition.
    """
    out: list[str] = []
    for i in range(n):
        topic = TOPICS[(i * 3) % len(TOPICS)]
        noun = NOUNS[(i * 11) % len(NOUNS)]
        city = CITIES[(i * 13) % len(CITIES)]
        if hard:
            # same topic AND same city as the old facts, different project only
            topic = TOPICS[i % len(TOPICS)]
            city = CITIES[(i * 5) % len(CITIES)]
            noun = NOUNS[(i * 7) % len(NOUNS)]
            out.append(f"On project {noun}{900000 + i} the team rolled out {topic} in {city} "
                       f"and measured latency {100 + (i % 400)} ms.")
            continue
        out.append(f"On project filler{noun}{i} the team discussed {topic} in {city} "
                   f"and noted throughput {500 + (i % 900)} rps.")
    return out


def evaluate(store: NeuroMatrixStore, pairs: list[tuple[str, str]], k: int) -> tuple[float, float, float]:
    """(recall@k, mean rank of the target, p50 latency ms)."""
    hits = 0
    ranks: list[int] = []
    lat: list[float] = []
    for text, question in pairs:
        target = store._conn.execute(
            "SELECT id FROM facts WHERE text = ? LIMIT 1", (text,)).fetchone()
        if target is None:
            continue
        t0 = time.perf_counter()
        got = store.search(question, limit=k)
        lat.append((time.perf_counter() - t0) * 1000)
        ids = [h["fact_id"] for h in got]
        if int(target["id"]) in ids:
            hits += 1
            ranks.append(ids.index(int(target["id"])) + 1)
        else:
            ranks.append(k + 1)
    n = max(1, len(ranks))
    return (hits / n * 100.0, statistics.mean(ranks) if ranks else float(k + 1),
            statistics.median(lat) if lat else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=int, default=300)
    ap.add_argument("--new", type=int, default=3000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hard", action="store_true",
                    help="filler shares topic AND city with the old facts, so only "
                         "the project number distinguishes them (real competition)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    old = make_old_facts(args.old, rng)
    tmp = Path(tempfile.mkdtemp(prefix="nm-stability-"))
    store = NeuroMatrixStore(str(tmp / "stability.db"))
    base = time.time() - 400 * 86400

    wrote = 0
    for i, (text, _q) in enumerate(old):
        if store.remember(text, source="turn", ts=base + i, importance=1.0):
            wrote += 1
    print(f"старых фактов записано: {wrote}/{len(old)}")

    before = evaluate(store, old, args.k)
    print(f"ДО роста:    recall@{args.k} = {before[0]:.1f}% | средний ранг = {before[1]:.2f} | p50 = {before[2]:.1f} ms")

    filler = make_new_facts(args.new, rng, hard=bool(args.hard))
    for i, text in enumerate(filler):
        store.remember(text, source="turn", ts=base + args.old + i, importance=1.0)
    total = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(f"новых фактов добавлено: {len(filler)} | всего в базе: {total}")
    store._cache.clear()

    after = evaluate(store, old, args.k)
    print(f"ПОСЛЕ роста: recall@{args.k} = {after[0]:.1f}% | средний ранг = {after[1]:.2f} | p50 = {after[2]:.1f} ms")

    print()
    d_recall = after[0] - before[0]
    d_rank = after[1] - before[1]
    print(f"Δ recall: {d_recall:+.1f} п.п. | Δ средний ранг: {d_rank:+.2f} | Δ латентность: {after[2] - before[2]:+.1f} ms")
    print()
    if d_recall >= -1.0:
        print("ВЕРДИКТ: память НЕ забывает — при росте базы в "
              f"{total / max(1, wrote):.0f}x recall старых фактов удержан.")
    else:
        print("ВЕРДИКТ: ЕСТЬ забывание — старые факты вытесняются новыми, это дефект.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
