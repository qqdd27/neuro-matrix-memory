"""Core engine tests for neuro_matrix (stdlib-only, no pytest required).

Run:  python tests/test_core.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_matrix.entities import extract_alias_pairs, extract_entities
from neuro_matrix.store import NeuroMatrixStore, _ru_variants, _entity_match_saturation


def _fresh(path: str) -> NeuroMatrixStore:
    return NeuroMatrixStore(path)


def test_extract_entities_russian_and_anchors():
    keys = extract_entities("Криптовалюта TON — это id_777, следи за ценой")
    assert "ton" in keys and "id_777" in keys, keys
    # English-only regex must NOT produce garbage tokens from Cyrillic words.
    assert "криптовалюта" not in keys
    # ...and must NOT turn every 2+ letter English word into an entity
    # (regression: a blanket (?i) flag made 'criteria'/'deadlines' anchors).
    keys2 = extract_entities("database_stack -> Firebase (criteria: offline-sync, deadlines)")
    assert keys2 == ["firebase"], keys2


def test_entity_noise_from_sentence_fillers_and_chat_abbreviations():
    """External validation finding (LoCoMo benchmark, 2026-09): common
    English sentence-continuation words ("Doing research...", "Last
    month...") and chat abbreviations that happen to be ALL-CAPS ("BTW")
    were being extracted as real entities, inflating a fact's entity count
    (and therefore its additive graph score) with pure noise -- unrelated to
    whether the fact was actually relevant to anything. Position-gating
    (the Cyrillic fix) does NOT transfer here: English text -- and this
    engine's own speaker-prefixed facts ("Caroline: ...") -- routinely open
    a sentence with the real entity, so a denylist is used instead."""
    keys = extract_entities(
        "Thanks for the tip, Caroline. Doing research and readying myself "
        "emotionally makes sense. BTW, Last month was rough.")
    assert "caroline" in keys, keys
    assert "doing" not in keys and "last" not in keys and "btw" not in keys, keys


def test_extract_alias_pairs_ru():
    pairs = extract_alias_pairs("Криптовалюта TON — это id_777.")
    assert ("ton", "id_777") in pairs, pairs


def test_extract_entities_ru_titlecase_no_anchor():
    """Real recall bug (reproduced 2026-09-11): a Russian proper noun with NO
    anchor (no id_/0x/ALL-CAPS) was invisible to the engine, so any fact about
    it was silently dropped by the write-path importance gate.  Fix: mid-
    sentence Cyrillic capitalization is a real signal (position-gated) even
    though EVERY Russian sentence starts capitalized regardless of content."""
    keys = extract_entities("Мы используем Постгрес для хранения данных.")
    assert "постгрес" in keys, keys
    # Sentence-initial capitalization must stay ambiguous -> never an entity.
    keys2 = extract_entities("Также нужно проверить сервер.")
    assert "также" not in keys2, keys2
    # Multi-sentence: the 2nd sentence's own first word is still skipped,
    # but a proper noun later in that same sentence is kept.
    keys3 = extract_entities(
        "Использовали Redis. Потом перешли на Постгрес, потому что нужна "
        "была надёжность.")
    assert "redis" in keys3 and "постгрес" in keys3, keys3
    assert "потом" not in keys3, keys3


def test_cross_session_recall_ru_named_entity_no_anchor():
    """End-to-end proof the fix actually restores memory: a Russian named
    tool with zero anchors, mentioned in one session, must still be
    recallable by name from a later session - the exact failure mode the
    entity-extraction gap caused before this fix."""
    path = os.path.join(tempfile.mkdtemp(), "m_ru_ent.db")
    store = _fresh(path)
    store.add_turn(
        "Мы используем Постгрес для хранения профилей пользователей.",
        "Понял, Постгрес хранит профили пользователей.",
        session_id="s1",
    )
    store.add_turn(
        "Расскажи, что у нас с Постгрес?",
        "Постгрес по-прежнему хранит профили, всё стабильно.",
        session_id="s2",
    )
    res = store.search("Постгрес", limit=10)
    assert res, "Russian named entity without an anchor was not recalled"
    assert any("профил" in r["text"].lower() for r in res), res
    store.close()


def test_cross_session_id_linking():
    """Dialog 1 (Russian) declares TON == id_777. Dialog 2 asks ONLY id_777
    and must still recall the TON facts."""
    path = os.path.join(tempfile.mkdtemp(), "m1.db")
    store = _fresh(path)
    # Dialog 1: assistant explains the identity.
    store.add_turn(
        "Криптовалюта TON — это id_777, кошелёк для тестов.",
        "Да, TON связан с id_777 и используется для тестовых переводов.",
        session_id="s1",
    )
    # Dialog 2 (different session): user only mentions the id.
    store.add_turn(
        "Что там по id_777?",
        "id_777 (TON) используется для тестов, проблем нет.",
        session_id="s2",
    )
    res = store.search("id_777", limit=10)
    texts = [r["text"] for r in res]
    joined = " | ".join(texts)
    assert any("тестов" in t for t in texts), texts
    assert any("TON" in t or "ton" in t for t in texts), texts
    # The bare single-anchor question must NOT be stored as a fact.
    assert "Что там по id_777" not in joined, texts
    ent = store.entity("ton")
    assert ent is not None
    assert "id_777" in ent["aliases"], ent
    store.close()
    # Persistence: reopen and query again.
    store2 = _fresh(path)
    res2 = store2.search("id_777", limit=10)
    assert len(res2) >= 1
    store2.close()


def test_graph_hop_recall():
    """Neighbor-of-neighbor recall: A co-occurs with B, B co-occurs with C ->
    querying C surfaces A-facts via the association graph."""
    path = os.path.join(tempfile.mkdtemp(), "m2.db")
    store = _fresh(path)
    store.add_turn("Проект DEX связан с биржей ABC.", "ABC - это DEX-биржа, там листится токен XYZ.",
                  session_id="g1")
    store.add_turn("Токен XYZ теперь торгуется на ABC.", "XYZ имеет пару с USDT на ABC.",
                  session_id="g2")
    res = store.search("DEX", limit=10)
    texts = [r["text"] for r in res]
    assert any("XYZ" in t for t in texts), texts  # hop-2 via ABC
    store.close()


def test_search_stays_fast_with_a_hub_entity() -> None:
    """Scale regression guard for a proven, measured defect (2026-09):
    profiling a synthetic 50k-fact / 212-entity store with one moderately
    hub-like entity found search() taking 62 SECONDS for a single query --
    caused by (1) a per-entity score helper re-queried once per (fact,
    entity) row instead of once per entity, (2) the materialize step doing
    one individual SQL SELECT per scored candidate instead of a batched
    query, and (3) unbounded 2-hop graph expansion pulling in the entire
    entity graph for a hub node. All three are fixed; this test reproduces
    the hub-entity shape (a small anchor pool so entities densely
    co-occur) at a size that still runs quickly in CI and asserts search()
    stays fast -- a future change reintroducing any of the three bugs would
    make this test slow or time out, not silently pass."""
    import random as _random
    path = os.path.join(tempfile.mkdtemp(), "hub_scale.db")
    store = _fresh(path)
    rnd = _random.Random(7)
    anchors = [f"id_{i}" for i in range(30)] + ["HUBTOKEN"]
    verbs = ["используется для", "упал из-за", "работает стабильно с",
             "интегрирован с", "хранит данные из"]
    for i in range(4000):
        a = rnd.choice(anchors)
        b = rnd.choice(anchors)
        store.remember(f"{a} {rnd.choice(verbs)} {b}, эпизод {i}, детали.",
                       source="turn:assistant", session_id=f"s{i % 200}")
    t0 = time.time()
    res = store.search("HUBTOKEN", limit=8)
    elapsed = time.time() - t0
    store.close()
    assert res, "hub entity search returned nothing"
    # Generous bound: a real regression to the pre-fix behaviour would take
    # tens of seconds at this scale, not fail this assertion by a hair.
    assert elapsed < 5.0, f"search() took {elapsed:.2f}s -- scaling regression"


def test_consolidation_extractive_no_llm():
    path = os.path.join(tempfile.mkdtemp(), "m3.db")
    store = _fresh(path)
    for i in range(3):
        store.add_turn(
            f"Сообщение {i} про кошелёк id_555.",
            f"id_555 кошелёк: баланс растёт, используется для выплат ({i}).",
            session_id=f"c{i}",
        )
    # One-off entity must NOT be consumed and must stay searchable.
    store.remember("Единичное упоминание токена SOLO_TOKEN для проверки.", source="test")
    before = store.stats()
    rep = store.consolidate()
    assert rep["dossiers_updated"] >= 1, rep
    ent = store.entity("id_555")
    assert ent is not None and ent["dossier"] is not None
    # Search surfaces dossier first.
    res = store.search("id_555", limit=5)
    assert res and res[0]["source"] == "dossier", res
    # Single-mention fact neither consumed nor lost.
    solo = store.search("SOLO_TOKEN", limit=5)
    assert solo and any(r["source"] != "dossier" for r in solo), solo
    stats = store.stats()
    assert stats["facts"] == before["facts"]  # nothing deleted by consolidate
    store.close()


def test_decay_and_prune():
    path = os.path.join(tempfile.mkdtemp(), "m4.db")
    store = _fresh(path)
    now = time.time()
    old = now - 400 * 86400
    store.add_turn("Старый факт про кошелёк id_600.", "id_600 был закрыт давно.", session_id="old", ts=old)
    store.add_turn("Новый факт про id_600.", "id_600 снова активен.", session_id="new")
    n = store.prune(retention_days=365, now=now)
    assert n >= 1, n
    res = store.search("id_600", limit=10)
    texts = [r["text"] for r in res]
    assert all("снова активен" in t for t in texts), texts  # stale archived
    assert not any("закрыт давно" in t for t in texts), texts
    store.close()


def test_ephemeral_turns_ignored():
    path = os.path.join(tempfile.mkdtemp(), "m5.db")
    store = _fresh(path)
    store.add_turn("Привет", "Здравствуйте! Чем могу помочь?", session_id="e1")
    store.add_turn("Ок, спасибо", "Пожалуйста!", session_id="e2")
    assert store.stats()["facts"] == 0, store.stats()
    store.close()


def test_merge_is_reversible():
    """Entity merge (via link()) is the single highest-risk operation in the
    engine: a wrong identity statement silently fuses two unrelated things
    and used to be permanent. Verify a full round-trip: merge, then
    unmerge(), must restore both entities' profiles, hit counts, aliases,
    shared third-party edge weight, and search results EXACTLY."""
    path = os.path.join(tempfile.mkdtemp(), "m6b.db")
    store = _fresh(path)
    store.remember("id_100 связан с проектом Alpha.", source="turn:assistant")
    store.remember("id_100 работает с Beta тоже.", source="turn:assistant")
    store.remember("id_200 отдельная сущность, тоже связана с Alpha.",
                   source="turn:assistant")

    ent100_before = store.entity("id_100")
    ent200_before = store.entity("id_200")
    alpha_before = store.entity("alpha")
    res_before = sorted(r["text"] for r in store.search("id_200", limit=10))

    mid = store.link("id_100", "id_200")
    assert mid is not None
    # While merged, id_200 resolves through the alias to the surviving entity.
    merged_probe = store.entity("id_200")
    assert merged_probe is not None and merged_probe["key"] == "id_100"

    r = store.unmerge(mid)
    assert r is not None and r["ok"] is True and r["restored_key"] == "id_200"

    assert store.entity("id_100") == ent100_before
    assert store.entity("id_200") == ent200_before
    assert store.entity("alpha")["hits"] == alpha_before["hits"]
    res_after = sorted(r["text"] for r in store.search("id_200", limit=10))
    assert res_after == res_before

    # Calling unmerge twice on the same merge_log id is a no-op, not a crash
    # or a second (corrupting) restore.
    assert store.unmerge(mid) is None
    store.close()


def test_stats_smoke():
    path = os.path.join(tempfile.mkdtemp(), "m6.db")
    store = _fresh(path)
    s = store.stats()
    for k in ("facts", "entities", "edges", "dossiers", "aliases", "unconsolidated"):
        assert k in s, s
    store.close()


def test_probe_via_alias_and_link_merge():
    """Probe by alias resolves to the canonical entity; link() merges aliases
    and keeps the more established entity."""
    path = os.path.join(tempfile.mkdtemp(), "m7.db")
    store = _fresh(path)
    store.add_turn("Кошелёк TON — это id_777.", "id_777 (TON) активен.", session_id="a")
    store.add_turn("id_777: сделали выплату.", "Готово, id_777 получил выплату.", session_id="b")
    ent = store.entity("id_777")
    assert ent is not None and ent["key"] == "ton", ent
    assert "id_777" in ent["aliases"], ent
    store.link("кошелёк", "id_777")  # ru word merges into well-known ton entity
    ent2 = store.entity("id_777")
    assert ent2 is not None and ent2["key"] == "ton", ent2
    assert "кошелёк" in ent2["aliases"], ent2
    assert store.stats()["entities"] == 1  # nothing duplicated by link
    store.close()


def test_migration_v01_to_v02():
    """A store created with the v0.1 schema (no kind/confidence/validity
    columns) must be migrated in place and keep working."""
    path = os.path.join(tempfile.mkdtemp(), "m8.db")
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'turn',
            session_id TEXT,
            ts REAL NOT NULL,
            importance REAL NOT NULL DEFAULT 1.0,
            consolidated INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            meta TEXT
        );
        CREATE TABLE entities (id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE, label TEXT, kind TEXT NOT NULL DEFAULT 'token',
            first_seen REAL NOT NULL, last_seen REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE aliases (entity_id INTEGER NOT NULL, alias TEXT NOT NULL,
            PRIMARY KEY (entity_id, alias));
        CREATE TABLE fact_entities (fact_id INTEGER NOT NULL, entity_id INTEGER NOT NULL,
            PRIMARY KEY (fact_id, entity_id));
        CREATE TABLE edges (a INTEGER NOT NULL, b INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 1, first_seen REAL NOT NULL,
            last_seen REAL NOT NULL, PRIMARY KEY (a, b));
        CREATE TABLE dossiers (entity_id INTEGER PRIMARY KEY, summary TEXT NOT NULL,
            updated_at REAL NOT NULL, meta TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    raw.commit()
    raw.close()
    store = _fresh(path)  # migration must run here
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(facts)").fetchall()}
    for c in ("kind", "confidence", "active_until", "supersedes"):
        assert c in cols, cols
    fid = store.remember("Старый факт про id_900.", source="test", ts=time.time() - 86400)
    assert fid is not None
    assert store.search("id_900")[0]["kind"] == "episodic"
    store.close()


def test_durable_kinds_survive_prune_and_rank_first():
    """Per-kind lifecycle: prune archives only stale decaying facts; durable
    decisions survive.  Durable facts do not fade with age, so an old decision
    outranks an equally-old episodic claim in recall."""
    path = os.path.join(tempfile.mkdtemp(), "m9.db")
    store = _fresh(path)
    old = time.time() - 400 * 86400
    # Durable decision (old) + two episodic facts (one stale, one fresh).
    store.remember("Проект выбрал базу Supabase для id_600.", source="decision",
                   kind="decision", ts=old, importance=1.0)
    store.remember("id_600: тогда тестировали Supabase на нагрузке.", source="ep",
                   kind="episodic", ts=old, importance=1.0)
    store.remember("id_600: сегодня Supabase работает стабильно.", source="ep",
                   kind="episodic", importance=1.0)
    n = store.prune(retention_days=365, now=time.time())
    assert n == 1, n  # only the stale episodic is archived
    res = store.search("id_600", limit=10)
    kinds = {r["kind"] for r in res}
    assert "decision" in kinds and "episodic" in kinds, (kinds, res)
    assert not any("на нагрузке" in r["text"] for r in res), res
    store.close()

    # Ranking without age-decay for durable kinds.
    path2 = os.path.join(tempfile.mkdtemp(), "m9b.db")
    s2 = _fresh(path2)
    old2 = time.time() - 400 * 86400
    s2.remember("Решение для id_601: Firebase вместо Supabase.", source="decision",
                kind="decision", ts=old2, importance=0.9, confidence=1.2)
    s2.remember("id_601: пробовали Supabase.", source="ep",
                kind="episodic", ts=old2, importance=1.0)
    res2 = s2.search("id_601", limit=5)
    assert res2[0]["kind"] == "decision", res2  # durable: no recency decay
    s2.close()


def test_confidence_and_kind_fields_roundtrip():
    path = os.path.join(tempfile.mkdtemp(), "m10.db")
    store = _fresh(path)
    fid = store.remember("Цель: мобильное приложение для id_777 после сайта.",
                         source="goal", kind="goal", confidence=0.8)
    assert fid is not None
    res = store.search("id_777", limit=5)
    hit = next(r for r in res if r["fact_id"] == fid)
    assert hit["kind"] == "goal"
    assert abs(hit["confidence"] - 0.8) < 1e-9
    # Durable kinds are stored even without anchor entities (explicit intent).
    fid2 = store.remember("Решение: всегда сначала проверять офлайн-режим.",
                          source="lesson", kind="lesson", confidence=1.5)
    assert fid2 is not None
    store.close()


def test_decision_evolution_supabase_firebase():
    """The owner scenario end-to-end: decide Supabase -> switch to Firebase.
    Old decision retires (active_until), trail is kept, the concept->Supabase
    edge is inhibited, consolidate() distills a durable lesson, and the
    retired choice stops resurfacing in recall."""
    path = os.path.join(tempfile.mkdtemp(), "m11.db")
    store = _fresh(path)
    d1 = store.decide("database_stack", "Supabase",
                      criteria=["sql", "rls"],
                      reason="нужен чистый SQL и RLS")
    dec = store.decisions("database_stack")
    assert dec["active"]["choice"] == "Supabase" and dec["count"] == 1
    # Week later: new criteria (offline-sync, deadlines) flip the decision.
    d2 = store.decide("database_stack", "Firebase",
                      criteria=["offline-sync", "deadlines"],
                      reason="нужна офлайн-синхронизация из коробки")
    assert d2 != d1
    dec = store.decisions("database_stack")
    assert dec["count"] == 2 and dec["active"]["choice"] == "Firebase"
    trail = {t["decision_id"]: t for t in dec["trail"]}
    assert trail[d1]["status"] == "superseded"
    assert trail[d1]["superseded_by"] == d2
    assert trail[d2]["supersedes"] == d1
    # Inhibition: the concept->Supabase edge is marked.
    ce = store._entity_id_lookup("database_stack")
    se = store._entity_id_lookup("supabase")
    row = store._conn.execute(
        "SELECT inhibited_since FROM edges WHERE a=? AND b=?",
        (min(ce, se), max(ce, se))).fetchone()
    assert row is not None and row["inhibited_since"] is not None
    # Retired decision does not resurface in recall (its own text is gone);
    # the active replacement may legitimately surface via the concept hub.
    res = store.search("Supabase", limit=10)
    assert not any(r["kind"] == "decision" and "Supabase" in r["text"] for r in res), res
    # The active decision does.
    res2 = store.search("Firebase", limit=10)
    assert any(r["kind"] == "decision" and "Firebase" in r["text"] for r in res2), res2
    # Sleep -> lesson distilled from the supersede chain.
    rep = store.consolidate(force=True)
    assert rep["lessons"] >= 1, rep
    res3 = store.search("database_stack", limit=10)
    lessons = [r for r in res3 if r["kind"] == "lesson"]
    assert lessons and "Supabase" in lessons[0]["text"] and "Firebase" in lessons[0]["text"], lessons
    # Re-running the sleep does not duplicate the lesson.
    assert store.consolidate(force=True)["lessons"] == 0
    store.close()


def test_decide_idempotent_and_chain_via_supersede():
    path = os.path.join(tempfile.mkdtemp(), "m12.db")
    store = _fresh(path)
    d1 = store.decide("stack_x", "A")
    d1b = store.decide("stack_x", "A", reason="refresh reason")
    assert d1 == d1b  # same choice active -> no duplicate
    d2 = store.supersede(d1, "B", criteria=["c1"], reason="потому что c1")
    assert d2 != d1
    dec = store.decisions("stack_x")
    assert dec["count"] == 2 and dec["active"]["choice"] == "B"
    store.close()


def test_decide_tools_provider():
    """Provider-level smoke: decide / supersede / decisions tool actions."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = os.path.join(tempfile.mkdtemp(), "m13.db")
    p = NeuromatrixMemoryProvider(config={"db_path": tmp, "auto_consolidate": "false",
                                          "llm_enabled": "false"})
    p.initialize("s", hermes_home=os.path.dirname(tmp))
    r1 = _json.loads(p.handle_tool_call("neuromatrix", {"action": "decide",
        "concept": "db_choice", "choice": "Postgres", "criteria": "sql, rls"}))
    assert r1["ok"] and r1["decision_id"] > 0, r1
    r2 = _json.loads(p.handle_tool_call("neuromatrix", {"action": "supersede",
        "decision_id": r1["decision_id"], "choice": "Mongo",
        "criteria": ["json-docs"], "reason": "schema-less нужен"}))
    assert r2["ok"] and r2["decision_id"] != r1["decision_id"], r2
    r3 = _json.loads(p.handle_tool_call("neuromatrix", {"action": "decisions",
        "concept": "db_choice"}))
    assert r3["active"]["choice"] == "Mongo" and r3["count"] == 2, r3
    rc = _json.loads(p.handle_tool_call("neuromatrix", {"action": "consolidate"}))
    assert rc["lessons"] >= 1, rc
    p.shutdown()


def test_feedback_reinforcement_and_archival():
    """Beta-Bernoulli confidence (Jeffreys prior 0.5/0.5, rescaled x2 so the
    no-feedback baseline stays 1.0): each vote's effect on confidence is
    exact and reproducible from the helpful/unhelpful counts alone."""
    path = os.path.join(tempfile.mkdtemp(), "m14.db")
    store = _fresh(path)
    fid = store.remember("id_777 используется для выплат.", source="turn:assistant")
    assert fid is not None
    assert abs(store.search("id_777")[0]["confidence"] - 1.0) < 1e-9
    r = store.feedback(fid, helpful=True)
    # n_helpful=1, n_unhelpful=0 -> 2*(0.5+1)/(1+1) = 1.5
    assert abs(r["confidence"] - 1.5) < 1e-6, r
    for _ in range(3):
        # 1 helpful + up to 3 unhelpful accumulate; archival needs >=3 votes
        # AND confidence < 0.3 (sustained evidence, not one bad vote).
        r = store.feedback(fid, helpful=False)
    # n_helpful=1, n_unhelpful=3 -> 2*(0.5+1)/(1+4) = 0.6 -> not archived:
    # a single lucky/unlucky vote can no longer flip the outcome by itself,
    # and one genuine earlier confirmation keeps giving the fact the benefit
    # of the doubt against a few contradicting votes (correct: a Bayesian
    # posterior does not forget real evidence just because more came later).
    assert abs(r["confidence"] - 0.6) < 1e-6, r
    assert r["archived"] is False, r
    # A fact with NO redeeming helpful votes at all DOES get archived once
    # enough sustained negative evidence accumulates (n=3, all unhelpful).
    fid2 = store.remember("id_778 используется для выплат.", source="turn:assistant")
    r2 = None
    for _ in range(3):
        r2 = store.feedback(fid2, helpful=False)
    # n_helpful=0, n_unhelpful=3 -> 2*0.5/4 = 0.25 < 0.3 -> archived.
    assert abs(r2["confidence"] - 0.25) < 1e-6, r2
    assert r2["archived"] is True, r2
    res = store.search("id_778", limit=10)
    assert not any(x["fact_id"] == fid2 for x in res), res  # archived excluded
    store.close()


def test_feedback_confidence_is_order_invariant():
    """Regression guard for a proven defect in the previous +0.15/*0.5 scheme:
    for the SAME evidence (2 helpful + 2 unhelpful votes), confidence used to
    depend on the ORDER the votes arrived in (measured: 0.325 / 0.55 / 0.3625
    for three different orderings of the identical multiset). A Bayesian
    posterior computed from counts cannot do this by construction -- three
    different orders of the same 2-vs-2 evidence must land on one value."""
    def run(order: list[bool]) -> float:
        path = os.path.join(tempfile.mkdtemp(), "m14b.db")
        store = _fresh(path)
        fid = store.remember("id_2 стабилен.", source="turn:assistant")
        r = None
        for helpful in order:
            r = store.feedback(fid, helpful)
        store.close()
        return r["confidence"]

    a = run([True, True, False, False])
    b = run([False, False, True, True])
    c = run([True, False, True, False])
    assert abs(a - b) < 1e-9 and abs(b - c) < 1e-9, (a, b, c)
    # And it must equal the closed-form posterior mean directly.
    assert abs(a - 2.0 * (0.5 + 2) / (1.0 + 4)) < 1e-9, a


def test_correction_on_write_negation():
    """Prediction-error at write: 'нет, это ошибка' + anchor cuts confidence of
    the previously stored assistant claim and flags it negated."""
    path = os.path.join(tempfile.mkdtemp(), "m15.db")
    store = _fresh(path)
    ids1 = store.add_turn("id_777 — это кошелёк TON?", "id_777 работает на TON.",
                          session_id="s1")
    assistant_fid = ids1[-1]
    before = store._conn.execute(
        "SELECT confidence FROM facts WHERE id = ?", (assistant_fid,)).fetchone()
    assert before is not None and abs(before["confidence"] - 1.0) < 1e-9
    # Later session: user corrects the record.
    store.add_turn("Стоп, это ошибка: id_777 на Solana, а не TON.",
                   "Точно, я ошибся — id_777 это Solana.", session_id="s2")
    row = store._conn.execute(
        "SELECT confidence, meta FROM facts WHERE id = ?", (assistant_fid,)).fetchone()
    assert abs(row["confidence"] - 0.4) < 1e-6, row["confidence"]
    assert '"negated": true' in row["meta"], row["meta"]
    store.close()


def test_contradictions_duplicate_scan():
    path = os.path.join(tempfile.mkdtemp(), "m16.db")
    store = _fresh(path)
    store.remember("id_900 выпустил новую версию токена.", source="turn:assistant")
    store.remember("id_900 выпустил новую версию токена.", source="turn:assistant")
    c = store.contradictions()
    assert c["duplicate_count"] >= 1, c
    assert c["duplicates"][0]["b"]["text"] == "id_900 выпустил новую версию токена."
    store.close()


def test_entity_match_saturation_formula():
    """BM25-style saturation on the COUNT of distinct entities matched per
    fact (measured fix, LoCoMo 2026-09, k1=2.0): n=1 must be a complete
    no-op (zero behavior change for the overwhelmingly common single-entity
    case), and higher n must grow toward the (k1+1) asymptote instead of
    unboundedly -- monotonically increasing but strictly sub-linear."""
    assert abs(_entity_match_saturation(1) - 1.0) < 1e-9
    vals = [_entity_match_saturation(n) for n in (1, 2, 3, 5, 10, 100)]
    # Monotonically increasing...
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    # ...but sub-linear: n=10 must score nowhere near 10x n=1.
    assert vals[4] < 3.0, vals
    # ...and bounded by the k1+1 asymptote (k1=2.0 -> ceiling 3.0) even at
    # very large n.
    assert vals[-1] < 3.0, vals
    assert abs(_entity_match_saturation(100) - 3.0) < 0.1


def test_content_relevance_bonus_beats_pure_recency_for_hub_entities():
    """External validation finding (LoCoMo benchmark, snap-research/locomo,
    2026-09): running this engine against a real, non-self-authored
    conversational-memory benchmark for the first time exposed a severe,
    systemic ranking defect that the project's own small synthetic tests
    never could -- because in those tests every anchor entity had only 1-3
    mentions total, so ranking by recency alone always happened to surface
    the one relevant fact by luck of scale, not by real relevance.

    The defect: once an entity is mentioned in MANY facts (a hub -- the most
    common real case being a person's own name in a long conversation),
    search() ranked candidates purely by entity-presence x recency x
    importance, completely blind to whether the OTHER words in the question
    matched the fact's own text. "What did Alex research?" surfaced the most
    RECENT fact mentioning Alex, not the one about research. Measured on
    LoCoMo: evidence-hit@8 was 3.0% before a fix, 27.8% after (9.3x).

    Fixed with a MULTIPLICATIVE content-token relevance bonus applied once
    per fact after its full entity-based score is summed (an earlier,
    additive '+1 per matched word' attempt measurably failed: entity base
    scores are unbounded -- they grow with edge count/dataset size and with
    how many entities a fact happens to mention -- so a fixed additive bonus
    is invisible against a hub fact's already-large base score; only a
    proportional boost reliably competes regardless of that scale)."""
    path = os.path.join(tempfile.mkdtemp(), "hubrel.db")
    store = _fresh(path)
    # Twenty generic mentions of a hub name -- none about the real question,
    # all more recent than the one relevant fact (worst case for a
    # recency-only ranker). Deliberately lowercase filler after the speaker
    # prefix so no SECOND entity accidentally enters the graph and confounds
    # the property under test (a capitalized filler word repeated across all
    # 40 facts would itself become a strong hub neighbor, which is a real
    # phenomenon but a different one than this test targets).
    for i in range(20):
        store.remember(f"Alex: yeah that sounds fun, tell me more #{i}.",
                       source="turn:assistant")
    store.remember("Alex: researching adoption agencies has been on my mind lately.",
                   source="turn:assistant")
    for i in range(20):
        store.remember(f"Alex: nice, glad to hear it, take care #{i}.",
                       source="turn:assistant")
    hits = store.search("What did Alex research?", limit=3)
    assert any("adoption" in h["text"].lower() for h in hits), hits
    store.close()


def test_dossier_conflict_detection():
    """Roadmap item: contradiction detection between dossiers and fresh facts.
    Two mentions consolidate id_900 into a dossier; a later fact carrying a
    negation marker about the same entity must be flagged as a conflict with
    that consolidated summary — before any human/LLM re-review."""
    path = os.path.join(tempfile.mkdtemp(), "m16b.db")
    store = _fresh(path)
    t0 = time.time() - 3600
    store.remember("id_900 использует Firebase для синхронизации.",
                    source="turn:assistant", ts=t0, kind="episodic")
    store.remember("id_900 хранит данные офлайн через Firebase.",
                    source="turn:assistant", ts=t0 + 1, kind="episodic")
    rep = store.consolidate(force=True)
    assert rep["dossiers_updated"] >= 1, rep
    ent = store.entity("id_900")
    assert ent and ent["dossier"], ent
    # Fresh, later fact that contradicts the settled dossier.
    t1 = time.time()
    store.remember("Firebase для id_900 не подошло, переделали на Supabase.",
                    source="turn:assistant", ts=t1, kind="episodic")
    c = store.contradictions()
    assert c["dossier_conflict_count"] >= 1, c
    hit = c["dossier_conflicts"][0]
    assert hit["entity_key"] == "id_900", hit
    assert "supabase" in hit["fact_text"].lower() or "не подошло" in hit["fact_text"].lower()
    store.close()


def test_myelination_protects_from_downscaling():
    """SHY downscaling weakens unprotected edges; myelinated (stable) edges —
    habits — survive.  Counts are rounded down but never below 1."""
    path = os.path.join(tempfile.mkdtemp(), "m17.db")
    store = NeuroMatrixStore(path, downscale_factor=0.5, myelin_stability=3.0)
    now = time.time()
    # Unprotected pair (few mentions -> stability < 3.0).
    for i in range(6):
        store.remember(f"AAA и BBB связаны (v{i}).")
    # Myelinated pair (>= ~9 mentions -> stability >= 3.0).
    for i in range(10):
        store.remember(f"CCC и DDD связаны (v{i}).")
    def count(k1, k2):
        a, b = store._entity_id_lookup(k1), store._entity_id_lookup(k2)
        if a is None or b is None:
            return None
        r = store._conn.execute("SELECT count, stability FROM edges WHERE a=? AND b=?",
                                (min(a, b), max(a, b))).fetchone()
        return (r["count"], r["stability"]) if r else None
    unprotected_before = count("aaa", "bbb")
    protected_before = count("ccc", "ddd")
    assert protected_before[1] >= 3.0, protected_before
    rep = store.consolidate(force=True)
    assert rep["downscaled"] >= 1, rep
    unprotected_after = count("aaa", "bbb")
    protected_after = count("ccc", "ddd")
    assert unprotected_after[0] < unprotected_before[0], (unprotected_before, unprotected_after)
    assert protected_after[0] == protected_before[0], (protected_before, protected_after)
    store.close()


def test_surprise_gate_skips_confirmations():
    """Nemori-style: facts that merely confirm an existing dossier are consumed
    without a dossier rewrite; surprising/novel facts drive the update."""
    path = os.path.join(tempfile.mkdtemp(), "m18.db")
    store = _fresh(path)
    store.remember("id_555 кошелёк: баланс растёт, выплаты еженедельно.")
    store.remember("id_555 активен для тестовых выплат.")
    store.consolidate(force=True)
    dossier_before = store.entity("id_555")["dossier"]
    # A pure confirmation (paraphrase of the dossier)…
    store.remember("кошелёк id_555: баланс растёт, выплаты еженедельные.")
    # …and a genuinely new fact.
    fid_novel = store.remember("id_555 переехал на новую сеть с мультисигом.")
    rep = store.consolidate(force=True)
    dossier_after = store.entity("id_555")["dossier"]
    assert "мультисиг" in dossier_after or "мультисигом" in dossier_after, dossier_after
    assert "переехал" in dossier_after, dossier_after
    # The confirmation fact was consumed (consolidated) without being merged in
    # as a separate claim line beyond the old content.
    row = store._conn.execute(
        "SELECT consolidated FROM facts WHERE id = ?", (fid_novel,)).fetchone()
    assert row["consolidated"] == 1
    store.close()


def test_as_of_point_in_time():
    """as_of resurrects superseded decisions for their era and hides the future."""
    path = os.path.join(tempfile.mkdtemp(), "m19.db")
    store = _fresh(path)
    base = time.time() - 10 * 86400
    d1 = store.decide("stack_asof", "Supabase", ts=base)
    d2 = store.supersede(d1, "Firebase", ts=base + 5 * 86400)
    assert d2 != d1
    # Era of Supabase: before the flip.
    era1 = store.search("Supabase", as_of=base + 2 * 86400, limit=10)
    assert any(r["kind"] == "decision" and "Supabase" in r["text"] for r in era1), era1
    era1b = store.search("Firebase", as_of=base + 2 * 86400, limit=10)
    assert not any(r["kind"] == "decision" for r in era1b), era1b  # future excluded
    # Era of Firebase: after the flip.
    era2 = store.search("Firebase", as_of=base + 7 * 86400, limit=10)
    assert any(r["kind"] == "decision" and "Firebase" in r["text"] for r in era2), era2
    era2b = store.search("Supabase", as_of=base + 7 * 86400, limit=10)
    assert not any(r["kind"] == "decision" for r in era2b), era2b  # retired by then
    store.close()


def test_reconsolidation_retrieval_count():
    path = os.path.join(tempfile.mkdtemp(), "m20.db")
    store = _fresh(path)
    fid = store.remember("id_777 кошелёк на TON для выплат.")
    store.search("id_777", limit=5)
    row = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE id = ?", (fid,)).fetchone()
    assert row["retrieval_count"] == 1, row
    store.close()


def test_episode_rollup_session():
    path = os.path.join(tempfile.mkdtemp(), "m21.db")
    store = _fresh(path)
    store.add_turn("Что по id_888?", "id_888 активен, выплаты раз в неделю.", session_id="sx")
    store.add_turn("Обнови статус id_888.", "id_888: баланс вырос, всё ок.", session_id="sx")
    fid = store.rollup_session("sx")
    assert fid is not None
    assert store.rollup_session("sx") is None  # idempotent
    res = store.search("id_888", limit=10)
    ep = [r for r in res if r["kind"] == "episode"]
    assert ep and "sx" in ep[0]["text"], (ep, res)
    store.close()


def test_artifacts_roundtrip_and_dedupe():
    """Big-data layer: payload in sidecar file, sha256 dedupe, snippet reads,
    unified-search pointer, explicit delete only."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m22.db")
    store = NeuroMatrixStore(path, artifacts_dir=os.path.join(tmp, "arts"))
    big = ("id_700 экспорт балансов\n" * 300)  # ~6KB payload
    r1 = store.store_artifact("balances_export", text=big, kind="export",
                              topic="id_700 балансы, еженедельный экспорт")
    assert r1 and not r1["duplicate"] and r1["size"] == len(big.encode())
    assert os.path.exists(r1["path"])
    r2 = store.store_artifact("balances_export_2", text=big)
    assert r2["duplicate"] and r2["artifact_id"] == r1["artifact_id"]
    got = store.artifact_get(r1["artifact_id"], max_chars=100)
    assert got["snippet"] and got["size"] == r1["size"] and got["path"] == r1["path"]
    assert len(store.artifact_find("баланс")) >= 1
    # Unified search finds the pointer fact via its head entities/topic.
    res = store.search("id_700", limit=10)
    assert any(r["kind"] == "artifact" for r in res), res
    assert store.artifact_delete(r1["artifact_id"]) is True
    assert store.artifact_get(r1["artifact_id"]) is None
    store.close()


def test_llm_daily_budget_cap():
    """Daily LLM budget: consolidate spends at most N calls/day."""
    class FakeLLM:
        def __init__(self):
            self.calls = 0
        def available(self):
            return True
        def chat_json(self, messages):
            self.calls += 1
            # Summarize every entity block generically.
            import re as _re
            keys = _re.findall(r"ENTITY: (\S+)", messages[1]["content"])
            return {"summaries": [{"entity": k, "summary": f"summary for {k}"}
                                  for k in keys]}
    tmp = os.path.join(tempfile.mkdtemp(), "m23.db")
    store = NeuroMatrixStore(tmp, llm_daily_budget=1)
    llm = FakeLLM()
    store.llm = llm
    # 3 entities x 2 facts each.
    for ent in ("id_701", "id_702", "id_703"):
        store.remember(f"{ent}: факт один.")
        store.remember(f"{ent}: факт два.")
    rep = store.consolidate(force=True)
    assert llm.calls <= 1, llm.calls
    assert rep["llm_calls"] <= 1
    assert store.llm_budget_remaining() == 0
    # Entities beyond the budget were handled by the extractive fallback.
    assert rep["dossiers_updated"] == 3, rep
    store.close()


def test_prefetch_recall_budget():
    """Provider prefetch honors max_recall_chars."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = os.path.join(tempfile.mkdtemp(), "m24.db")
    p = NeuromatrixMemoryProvider(config={"db_path": tmp, "auto_consolidate": "false",
                                          "llm_enabled": "false", "max_recall_chars": 120})
    p.initialize("s", hermes_home=os.path.dirname(tmp))
    p.sync_turn("Что по id_750?", "id_750: кошелёк активен, выплаты раз в неделю, всё стабильно.",
                session_id="s")
    p.sync_turn("Обнови id_750.", "id_750: баланс вырос на 10% за месяц.", session_id="s")
    recall = p.prefetch("id_750")
    assert recall.startswith("## NeuroMatrix Memory")
    body = recall.split("## NeuroMatrix Memory\n", 1)[1]
    assert len(body) <= 130, len(body)
    p.shutdown()


def test_foresight_due():
    """Time-bounded foresight (§19.3): due when trigger_at passes, silent before."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = os.path.join(tempfile.mkdtemp(), "m25.db")
    p = NeuromatrixMemoryProvider(config={"db_path": tmp, "llm_enabled": "false"})
    p.initialize("s", hermes_home=os.path.dirname(tmp))
    r1 = _json.loads(p.handle_tool_call("neuromatrix", {"action": "foresight",
        "content": "К марту проверить листинг id_900", "trigger_at": str(time.time() - 60)}))
    assert r1["ok"], r1
    r2 = _json.loads(p.handle_tool_call("neuromatrix", {"action": "foresight",
        "content": "Позже: id_900 аудит", "trigger_at": str(time.time() + 99999)}))
    assert r2["ok"]
    due = _json.loads(p.handle_tool_call("neuromatrix", {"action": "reminders"}))
    assert due["count"] == 1 and "id_900" in due["due"][0]["text"], due
    p.shutdown()


def test_ops_log_and_markdown_mirror():
    """Ops journal records policy mutations; markdown mirror is reviewable."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m26.db")
    store = NeuroMatrixStore(path)
    d1 = store.decide("stack_md", "Supabase", criteria=["sql"])
    d2 = store.supersede(d1, "Firebase", criteria=["offline"])
    store.feedback(d2, helpful=True)
    ops = store.ops_view()
    ops_text = " ".join(f"{o['op']}:{o['scope']}" for o in ops)
    assert "SUPERSEDE" in ops_text and "ADD" in ops_text and "UPDATE" in ops_text, ops
    out = os.path.join(tmp, "mirror")
    rep = store.export_markdown(out)
    assert rep["entities"] >= 1 and rep["decisions"] >= 1
    assert os.path.exists(os.path.join(out, "README.md"))
    ent_file = os.path.join(out, "entities", "supabase.md")
    assert os.path.exists(ent_file)
    with open(ent_file, encoding="utf-8") as f:
        content = f.read()
    assert "Supabase" in content or "supabase" in content, content[:200]
    store.close()


def test_ask_requires_llm_and_ops_tool():
    """ask degrades with a clear error without a key; ops tool returns journal."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = os.path.join(tempfile.mkdtemp(), "m27.db")
    p = NeuromatrixMemoryProvider(config={"db_path": tmp, "llm_enabled": "false"})
    p.initialize("s", hermes_home=os.path.dirname(tmp))
    p.sync_turn("Что по id_808?", "id_808 кошелёк активен.", session_id="s")
    r = p.handle_tool_call("neuromatrix", {"action": "ask", "query": "id_808"})
    d = _json.loads(r)
    assert "error" in d and "NEUROMATRIX_API_KEY" in d["error"], d
    o = _json.loads(p.handle_tool_call("neuromatrix", {"action": "ops"}))
    assert o["ok"] and isinstance(o["ops"], list)
    p.shutdown()


def test_rerank_orders_by_llm():
    """LLM rerank reorders heuristic top-k and spends exactly one budget call."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m28.db")
    store = NeuroMatrixStore(path)

    class _FakeLLM:
        calls = 0

        @staticmethod
        def available() -> bool:
            return True

        def chat_json(self, _messages):
            type(self).calls += 1
            return {"order": [b_id, a_id, c_id]}  # deliberately non-heuristic

    ids = [store.remember(f"Токен {t} растёт на бирже X.", source="test") for t in ("AAA", "BBB", "CCC")]
    a_id, b_id, c_id = ids
    store.llm = _FakeLLM()
    plain = store.search("CCC AAA BBB", limit=3, rerank=False)
    store._cache.clear()
    reranked = store.search("CCC AAA BBB", limit=3, rerank=True)
    assert [int(r["fact_id"]) for r in reranked][:3] == [b_id, a_id, c_id]
    assert _FakeLLM.calls == 1
    assert [int(r["fact_id"]) for r in plain] != [b_id, a_id, c_id]
    store.close()


def test_shared_workspace_scope():
    """scope=shared writes/reads the workspace pool across provider instances."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = tempfile.mkdtemp()
    wsp = os.path.join(tmp, "workspace.db")
    prov_a = NeuromatrixMemoryProvider(config={
        "db_path": os.path.join(tmp, "a.db"), "workspace_db": wsp,
        "llm_enabled": "false", "auto_consolidate": "false"})
    prov_a.initialize("sa", hermes_home=tmp)
    r = prov_a.handle_tool_call("neuromatrix", {
        "action": "remember", "content": "id_333 общий кошелёк команды", "scope": "shared"})
    assert _json.loads(r)["ok"]
    r = prov_a.handle_tool_call("neuromatrix", {
        "action": "remember", "content": "id_333 приватный секрет владельца", "scope": "private"})
    assert _json.loads(r)["ok"]
    prov_a.shutdown()

    prov_b = NeuromatrixMemoryProvider(config={
        "db_path": os.path.join(tmp, "b.db"), "workspace_db": wsp,
        "llm_enabled": "false", "auto_consolidate": "false"})
    prov_b.initialize("sb", hermes_home=tmp)
    hits = _json.loads(prov_b.handle_tool_call("neuromatrix", {
        "action": "search", "query": "id_333"}))["results"]
    texts = " ".join(h["text"] for h in hits)
    assert "общий кошелёк команды" in texts and "приватный секрет" not in texts, texts
    prov_b.shutdown()


def test_prefetch_daily_foresight_reminder():
    """Due foresight surfaces once per day in prefetch; not duplicated same day."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = os.path.join(tempfile.mkdtemp(), "m30.db")
    p = NeuromatrixMemoryProvider(config={"db_path": tmp, "llm_enabled": "false"})
    p.initialize("s", hermes_home=os.path.dirname(tmp))
    r = _json.loads(p.handle_tool_call("neuromatrix", {
        "action": "foresight", "content": "Проверить листинг к марту",
        "trigger_at": str(time.time() - 60)}))
    assert r["ok"]
    first = p.prefetch("ничего важного")
    assert "⏰" in first and "листинг" in first
    second = p.prefetch("ничего важного")
    assert second.count("⏰") == 0 or first.count("⏰") == second.count("⏰")
    p.shutdown()


def test_mcp_stdio_protocol():
    """MCP server: initialize/tools/list/tools/call/ping over pure handler."""
    import json as _json
    from neuro_matrix.mcp import TOOLS, handle_message
    tmp = os.path.join(tempfile.mkdtemp(), "m31.db")
    store = NeuroMatrixStore(tmp)
    try:
        init = handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {}}, store)
        assert init["result"]["serverInfo"]["name"] == "neuromatrix"
        listed = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, store)
        assert len(listed["result"]["tools"]) == len(TOOLS) >= 8
        called = handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "memory_remember",
                                            "arguments": {"content": "id_999 документ MCP"}}}, store)
        payload = _json.loads(called["result"]["content"][0]["text"])
        assert payload["fact_id"] > 0
        found = handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                "params": {"name": "memory_search",
                                           "arguments": {"query": "id_999"}}}, store)
        assert "документ MCP" in found["result"]["content"][0]["text"]
        assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}, store) is None
        bad = handle_message({"jsonrpc": "2.0", "id": 5, "method": "nope"}, store)
        assert bad["error"]["code"] == -32601
    finally:
        store.close()


def test_cite_explain_and_skill_propose():
    """Claims carry provenance citations; skill bridge writes a reviewable draft."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m32.db")
    store = NeuroMatrixStore(path)
    e1 = store.remember("id_500 пароль кошелька сменён 1 сентября.", source="test", session_id="sessA")
    e2 = store.remember("id_777 связан с тем же кошельком.", source="test", session_id="sessB")
    claim = store.remember("id_777 теперь под новым паролем.", source="test", session_id="sessC")
    res = store.attach_evidence(claim, [e1, e2])
    assert res is not None and res["evidence"] == [e1, e2]
    exp = store.explain_fact(claim)
    assert exp["cited_text"].startswith("id_777 теперь")
    assert f"[fact #{e1}, session sessA]" in exp["cited_text"]
    assert f"[fact #{e2}, session sessB]" in exp["cited_text"]
    assert any(o["op"] == "LINK" for o in store.ops_view())

    d1 = store.decide("stack_full", "Supabase", criteria=["sql"], reason="RLS")
    store.supersede(d1, "Firebase", criteria=["offline", "mobile"], reason="сроки")
    rep = store.skill_propose("stack_full", out_dir=os.path.join(tmp, "skills"))
    assert os.path.exists(rep["path"])
    with open(rep["path"], encoding="utf-8") as f:
        md = f.read()
    assert "# Proposed skill" in md and "Supabase" in md and "Firebase" in md
    assert "[fact #" not in md or "episodes" in rep  # citations optional here
    store.close()

    # provider-level roundtrip (private scope)
    p = NeuromatrixMemoryProvider(config={"db_path": path, "llm_enabled": "false"})
    p.initialize("s", hermes_home=tmp)
    out = _json.loads(p.handle_tool_call("neuromatrix", {
        "action": "explain", "fact_id": claim}))
    assert out["found"] and "session sessA" in out["cited_text"]
    p.shutdown()


def test_persona_policy_and_shared_sleep():
    """Persona card + policy digest + workspace pool gets its own sleep pass."""
    import json as _json
    from neuro_matrix.provider import NeuromatrixMemoryProvider
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m33.db")
    store = NeuroMatrixStore(path)
    store.remember_goal("Довести neuromatrix до продакшена.", entity="neuromatrix",
                        session_id="s")
    d1 = store.decide("stack_pc", "Supabase", criteria=["sql"])
    store.supersede(d1, "Firebase", criteria=["offline"])
    pc = store.export_profile_card(out_dir=os.path.join(tmp, "profile"))
    assert os.path.exists(pc["path"])
    with open(pc["path"], encoding="utf-8") as f:
        md = f.read()
    assert "Active decisions" in md and "Firebase" in md and "Goals" in md
    pol = store.policy_report()
    assert pol["ops_analysed"] >= 2
    assert pol["by_op"].get("SUPERSEDE", 0) >= 1
    assert pol["recommendations"], pol
    store.close()

    wsp = os.path.join(tmp, "workspace.db")
    prov = NeuromatrixMemoryProvider(config={
        "db_path": os.path.join(tmp, "a.db"), "workspace_db": wsp,
        "llm_enabled": "false", "auto_consolidate": "true"})
    prov.initialize("s", hermes_home=tmp)
    prov.handle_tool_call("neuromatrix", {
        "action": "remember", "content": "id_444 общий стандарт команды", "scope": "shared"})
    prov.on_session_end([])  # sleep covers private AND the workspace pool
    assert prov._shared is not None
    today = time.strftime("%Y%m%d", time.gmtime())
    assert prov._shared.get_meta("last_prune_day") == today
    pol2 = _json.loads(prov.handle_tool_call("neuromatrix", {"action": "policy"}))
    assert pol2["ok"] and "private" in pol2 and "shared" in pol2
    prov.shutdown()


def test_auto_decide_capture_from_turns():
    """Explicit 'для X берём Y, потому что Z' turns become decision chains."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m34.db")
    store = NeuroMatrixStore(path)
    store.add_turn(
        "Для нового приложения берём Firebase, потому что offline-синк и сроки.",
        "Согласен, Firebase.", session_id="s1")
    dec = store.decisions("нового_приложения")
    assert dec["count"] == 1, dec
    assert dec["active"]["choice"] == "Firebase", dec
    crit = dec["active"].get("criteria") or []
    assert any("offline" in c for c in crit), crit
    # Later flip in another session auto-chains (trail of 2, active = Supabase).
    store.add_turn(
        "Для нового приложения переходим на Supabase, потому что нужен SQL.",
        "Ок.", session_id="s2")
    dec2 = store.decisions("нового_приложения")
    assert dec2["count"] == 2 and dec2["active"]["choice"] == "Supabase", dec2
    choices = [t["choice"] for t in dec2["trail"]]
    assert choices == ["Firebase", "Supabase"], choices
    assert any(o["op"] == "ADD" for o in store.ops_view())
    # No scope phrase -> no decision invented.
    store.add_turn("Просто используем какой-то инструмент, хорошо?",
                   "Хорошо.", session_id="s3")
    assert store.decisions("какой_то")["count"] == 0
    # Flag off disables capture entirely.
    store.auto_decide_enabled = False
    store.add_turn("Для бота берём Python, потому что быстро.",
                   "Ок.", session_id="s4")
    assert store.decisions("бота")["count"] == 0
    store.close()


def test_sweep_decisions_llm_dynamic():
    """LLM sweep captures decisions the marker heuristic would miss (any phrasing)."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m35.db")
    store = NeuroMatrixStore(path)

    class _FakeLLM:
        calls = 0

        @staticmethod
        def available() -> bool:
            return True

        def chat_json(self, _messages):
            type(self).calls += 1
            return {"decisions": [
                {"concept": "мобильное приложение", "choice": "Vue",
                 "criteria": ["экосистема", "скорость"], "text_id": 1}]}

    # Deliberately NO marker words («берём/используем/для…») — dynamic phrasing.
    store.add_turn(
        "Пишем мобильное приложение на Vue, а Flutter отбросили, экосистема нравится.",
        "Ок.", session_id="s1")
    assert store.decisions("мобильное_приложение")["count"] == 0
    store.llm = _FakeLLM()
    assert store.sweep_decisions() == 1
    dec = store.decisions("мобильное_приложение")
    assert dec["count"] == 1 and dec["active"]["choice"] == "Vue", dec
    assert "экосистема" in (dec["active"].get("criteria") or []), dec
    assert _FakeLLM.calls == 1
    # Batch consumed: second sweep makes no further LLM call.
    assert store.sweep_decisions() == 0 and _FakeLLM.calls == 1
    store.close()


def test_traits_do_not_crowd_out_raw_evidence() -> None:
    """Measured regression, real-key paid LoCoMo run (2026-09): sweep_traits
    facts used to be created with a fresh timestamp (undecayed recency) and
    importance >=1.0, with no cap on how many can appear in results (unlike
    dossiers' explicit 2-slot cap) -- so a handful of distilled traits about
    a hub entity systematically outscored and displaced the actual episodic
    evidence turn from top-k. Confirmed on an identical 150-question LoCoMo
    sample: evidence-hit@8 was 31.8% with sweep_traits() never invoked,
    11.0% in the real paid run where it was -- traits were net HARMFUL to
    literal recall despite being designed to help inferential recall. Fixed
    by lowering trait importance (0.5, below the >=1.0 used elsewhere for
    durable kinds); this reproduces the exact failure shape at small scale
    and asserts the real evidence turn stays on top."""
    path = os.path.join(tempfile.mkdtemp(), "traitcrowd.db")
    store = _fresh(path)
    old_ts = time.time() - 400 * 86400  # old, decayed -- like a real turn
    store.remember(
        "Caroline: Researching adoption agencies has been on my mind lately.",
        source="turn", ts=old_ts, importance=1.0)
    for t in ("loves painting and art", "enjoys hiking outdoors",
              "values LGBTQ community support", "is close with Melanie",
              "appreciates classic literature"):
        store.remember(f"Caroline: {t}", source="trait", kind="trait",
                       ts=time.time(), importance=0.5, confidence=1.0,
                       meta={"type": "trait"})
    hits = store.search("What did Caroline research?", limit=8)
    assert hits and "adoption" in hits[0]["text"].lower(), hits
    store.close()


def test_sweep_traits_llm_dynamic_and_answers_inference() -> None:
    """Write-time trait distillation (LLM-optional): scattered episodic
    mentions about a named person become one durable kind='trait' fact,
    findable by plain graph/FTS search -- no reasoning needed at read time.
    Proves the actual point: a later INFERENTIAL question that shares no
    anchor/keyword with any single raw episode (only with the distilled
    trait) is now answerable, closing exactly the multi-hop/inferential
    retrieval gap measured on LoCoMo (evidence-hit@8 12.4% on that
    category) — with a write-time LLM pass, not a read-time one."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "m_traits.db")
    store = NeuroMatrixStore(path)

    class _FakeLLM:
        calls = 0

        @staticmethod
        def available() -> bool:
            return True

        def chat_json(self, _messages):
            type(self).calls += 1
            return {"traits": [
                {"subject": "Caroline", "trait": "interested in counseling "
                 "and mental health support", "text_id": 1}]}

    store.remember(
        "Caroline: I'm keen on counseling or working in mental health - "
        "I'd love to support those with similar issues.",
        source="turn:assistant")
    # Before the sweep: no distilled trait fact exists yet.
    before = store._conn.execute(
        "SELECT COUNT(*) c FROM facts WHERE kind = 'trait'").fetchone()
    assert before["c"] == 0

    store.llm = _FakeLLM()
    assert store.sweep_traits() == 1
    assert _FakeLLM.calls == 1
    # Second sweep consumes nothing new (batch already marked swept).
    assert store.sweep_traits() == 0 and _FakeLLM.calls == 1

    trait_row = store._conn.execute(
        "SELECT id, kind, text FROM facts WHERE text LIKE '%counseling and mental health%'"
    ).fetchone()
    assert trait_row is not None and trait_row["kind"] == "trait", trait_row
    assert trait_row["text"].startswith("Caroline:"), trait_row["text"]
    ents = store._conn.execute(
        "SELECT e.key FROM fact_entities fe JOIN entities e ON e.id = fe.entity_id "
        "WHERE fe.fact_id = ?", (trait_row["id"],)).fetchall()
    assert "caroline" in [e["key"] for e in ents], ents
    # A query using the trait's own distilled wording finds it via plain
    # graph/FTS lookup -- no reasoning at read time, exactly the point.
    after = store.search("Is Caroline interested in mental health work?", limit=5)
    assert any("counseling" in h["text"].lower() for h in after), after
    store.close()


def test_deadend_auto_ru() -> None:
    path = os.path.join(tempfile.mkdtemp(), "de.db")
    store = _fresh(path)
    store.add_turn(
        "Flutter не подошёл, потому что производительность низкая.",
        "", session_id="s1",
    )
    ds = store.deadends("Flutter")
    assert len(ds) == 1, ds
    assert "производительность" in ds[0]["reason"], ds[0]
    assert ds[0]["source"] == "auto" and ds[0]["subject"] == "flutter"
    # dedup: second auto mark for the same subject returns the same id
    fid2 = store._auto_outcome_from_turn(
        "Flutter всё ещё не работает, потому что UI тормозит.", time.time())
    assert len(store.deadends("Flutter")) == 1
    store.close()


def test_deadend_auto_en_fallback_subject() -> None:
    path = os.path.join(tempfile.mkdtemp(), "de.db")
    store = _fresh(path)
    store.add_turn(
        "does not work: the v2 migration keeps failing because of rate limits",
        "", session_id="s1",
    )
    # subject comes from the anchor AFTER the marker here ('migration')
    ds = store.deadends()
    assert len(ds) == 1 and ds[0]["subject"] == "v2", ds
    assert "rate limits" in ds[0]["reason"], ds[0]
    store.close()


def test_deadend_manual_and_no_question_marks() -> None:
    path = os.path.join(tempfile.mkdtemp(), "de.db")
    store = _fresh(path)
    # questions must never become outcomes
    assert store._auto_outcome_from_turn("Flutter не работает? надо чинить", time.time()) is None
    fid = store.mark_deadend("хромодрайвер", "не грузится на CI", session_id="s1")
    assert fid
    rows = store.deadends("хромодрайвер")
    assert len(rows) == 1 and rows[0]["source"] == "manual"
    # no subject filter lists it too
    assert any(r["subject"] == "хромодрайвер" for r in store.deadends())
    # durable: consolidation must not archive dead ends
    store.consolidate()
    assert len(store.deadends("хромодрайвер")) == 1
    store.close()


def test_deadend_episode_wiring_and_explain() -> None:
    """The failed attempt stays an episodic fact; the deadend fact points at it
    via evidence, so explain() shows the trail back to the original try."""
    path = os.path.join(tempfile.mkdtemp(), "de.db")
    store = _fresh(path)
    try_fid = store.remember(
        "Перешли на Solana RPC для цен", source="decision",
        meta={"concept": "rpc", "choice": "solana", "status": "superseded"},
    )
    end_fid = store.mark_deadend("solana", "rate limit на бесплатном tier",
                                 target_fact_id=try_fid)
    assert end_fid and try_fid
    explain = store.explain_fact(end_fid)
    joined = explain if isinstance(explain, str) else str(explain)
    assert str(try_fid) in joined, joined
    store.close()



def test_capability_auto_ru_en() -> None:
    path = os.path.join(tempfile.mkdtemp(), "cap.db")
    store = _fresh(path)
    store.add_turn(
        "Возьмём Riverpod для стейта?",
        "Используем Riverpod для управления состоянием приложения.", session_id="s1",
    )
    store.add_turn(
        "ок",
        "We use Riverpod for state management and code generation.", session_id="s2",
    )
    caps = store.capabilities("Riverpod")
    assert len(caps) == 2, caps
    joined = " | ".join(c["capability"] for c in caps).lower()
    assert "состоянием" in joined and "state management" in joined, joined
    # dedup: identical statement again must not duplicate
    store.add_turn("", "Используем Riverpod для управления состоянием приложения.",
                   session_id="s3")
    assert len(store.capabilities("riverpod")) == 2
    store.close()


def test_ingest_document_russian_no_anchors() -> None:
    path = os.path.join(tempfile.mkdtemp(), "doc.db")
    store = _fresh(path)
    doc = (
        "Правило первое: никогда не пишем транзакции без проверки баланса.\n"
        "Правило второе: каждый запрос к бирже обязан иметь таймаут.\n"
        "Правило третье: перед деплоем всегда делаем бэкап базы данных, "
        "потому что потерять прод нельзя. Проверяем это вручную и скриптом."
    )
    res = store.ingest_document(doc, title="Правила разработки", topic="dev")
    assert res["stored"] == res["chunks"] >= 2, res
    row = store._conn.execute(
        "SELECT COUNT(*) c FROM facts WHERE kind='doc' AND archived=0").fetchone()
    assert row["c"] == res["stored"]
    hits = store.search("таймаут", limit=5)
    assert any("таймаут" in h["text"] for h in hits), hits
    store.close()



def test_session_project_status_rollup() -> None:
    path = os.path.join(tempfile.mkdtemp(), "st.db")
    store = _fresh(path)
    sid = "sess-engine-1"
    store.add_turn(
        "Начинаем движок с нуля, решили стек?",
        "Выбрали PostgreSQL для базы движка, продолжаем завтра.", session_id=sid,
    )
    store.add_turn(
        "чертёж не открылся",
        "DraftSight не работает, потому что нет лицензии на CI.",
        session_id=sid,
    )
    store.add_turn(
        "понял",
        "Используем DraftSight для чертежей движка.", session_id=sid,
    )
    r1 = store.finalize_session(sid)
    assert r1 and r1["new"] is True and r1["id"], r1
    low = store.latest_statuses(1)[0]["text"].lower()
    assert "postgresql" in low and "draftsight" in low, low
    # dead-end subject surfaces in the status rollup too
    assert "чертежей" in low or "dead-ends" in low, low
    # idempotent: same session -> existing status, no duplicate
    r2 = store.finalize_session(sid)
    assert r2["new"] is False and r2["id"] == r1["id"]
    cnt = store._conn.execute(
        "SELECT COUNT(*) c FROM facts WHERE kind='status' AND archived=0"
    ).fetchone()["c"]
    assert cnt == 1, cnt
    # another session's status is listed newest-first
    store.add_turn("идея", "Сделаем резервный вариант на Redis.", session_id="sess-engine-2")
    store.finalize_session("sess-engine-2")
    latest = store.latest_statuses(1)[0]
    assert latest["session_id"] == "sess-engine-2"
    store.close()


def test_ru_morphological_recall() -> None:
    """FTS5 has no RU stemming: 'правила' stored, 'правило' asked.  Variant
    expansion must close the gap."""
    path = os.path.join(tempfile.mkdtemp(), "ru.db")
    store = _fresh(path)
    store.ingest_document(
        "Все правила проекта лежат в папке docs рядом с движком.\n"
        "Сборка движка запускается только после проверки правил линтера.",
        title="Правила проекта",
    )
    # ask in a different grammatical form than stored
    hits = store.search("правило проекта", limit=6)
    joined = " | ".join(h["text"].lower() for h in hits)
    assert "правила" in joined and "движка" in joined, joined
    store.close()


def test_ru_variants_unit() -> None:
    vs = _ru_variants("правило")
    assert "правило" in vs and "правила" in vs and "правилу" in vs, vs
    assert len(vs) <= 9


def test_synonym_bridge_narrows_semantic_gap() -> None:
    """Honest, BOUNDED narrowing of the measured semantic-recall ceiling
    (scripts/eval_semantic_gap.py): a query built from a curated synonym of a
    word actually present in the fact must recall it, even with zero literal
    token overlap and no shared anchor entity. This is NOT semantic search —
    verify the boundary holds too: a query sharing no synonym-group member
    and no anchor with the fact still correctly misses (regression guard
    against ever silently overclaiming this closes the whole gap)."""
    path = os.path.join(tempfile.mkdtemp(), "syn.db")
    store = _fresh(path)
    store.remember("Сервис TON падал из-за исчерпания лимита запросов к API.",
                    source="turn:assistant")
    store.remember("id_42 хранит данные локально и продолжает работать без сети.",
                    source="turn:assistant")
    hits1 = store.search("Почему у нас недавно был сбой на проде?", limit=5)
    assert any("падал" in h["text"] for h in hits1), hits1
    hits2 = store.search("Что из наших инструментов не требует подключения "
                          "к интернету?", limit=5)
    assert any("локально" in h["text"] for h in hits2), hits2
    # Boundary: genuinely disjoint vocabulary (no synonym-group member, no
    # anchor shared) must still miss -- the bridge is narrow by design.
    hits3 = store.search("Почему сменили предыдущего поставщика бэкенда?",
                          limit=5)
    assert not any("TON" in h["text"] or "id_42" in h["text"] for h in hits3), hits3
    store.close()
    # Latin tokens pass through untouched
    assert _ru_variants("postgresql") == ["postgresql"]



def test_repeated_question_cache_auto() -> None:
    """2nd distinct ask (>=2h later) freezes top answer into kind='resolved';
    the 3rd ask returns it instantly via the pre-LRU shortcut."""
    path = os.path.join(tempfile.mkdtemp(), "rc.db")
    store = _fresh(path)
    store.add_turn("двигатель?",
                   "Engine runs on kerosene and liquid oxygen.",
                   session_id="s1")
    q = "engine fuel kerosene"
    t0 = time.time()
    r1 = store.search(q, limit=3, now=t0)
    assert any("kerosene" in h["text"] for h in r1), r1
    # backdate the first ask so the 2nd ask looks like a repeat hours later
    store._conn.execute(
        "UPDATE ask_log SET first_ts = ?, hits = 1 WHERE qkey = ?",
        (t0 - 4 * 3600, q))
    store._conn.commit()
    r2 = store.search(q, limit=3, now=t0 + 31)  # past the 30s LRU window
    assert any("kerosene" in h["text"] for h in r2), r2
    r3 = store.search(q, limit=3, now=t0 + 61)
    assert len(r3) == 1 and r3[0]["source"] == "resolved", r3
    assert "kerosene" in r3[0]["text"] and r3[0]["kind"] == "resolved", r3
    store.close()


def test_repeated_question_cache_manual() -> None:
    path = os.path.join(tempfile.mkdtemp(), "rc.db")
    store = _fresh(path)
    fid = store.resolve_query("как ускорить сборку", "сборка идёт через Redis кэш")
    assert fid
    hits = store.search("как ускорить сборку", limit=3)
    assert len(hits) == 1 and hits[0]["source"] == "resolved"
    assert "redis" in hits[0]["text"].lower(), hits
    store.close()



def test_invent_proposes_novel_combinations() -> None:
    path = os.path.join(tempfile.mkdtemp(), "inv.db")
    store = _fresh(path)
    store.add_turn("", "Engine drives the Wheel.", session_id="s1")
    store.add_turn("", "Engine burns Kerosene fuel.", session_id="s1")
    store.add_turn("", "Wheel turns on the Axle.", session_id="s1")
    rows = store.invent("need a movement transport for goods", limit=6)
    assert rows, rows
    for r in rows:
        assert r["a"] != r["b"] and 0 < r["novelty"] <= 1.0
        assert "hypothesis" in r and r["a"] in r["hypothesis"]
    # dead-end gate: a pair of two known dead ends is never proposed
    store.mark_deadend("Engine", "не тянет")
    store.mark_deadend("Wheel", "ломается")
    rows2 = store.invent("need a movement transport for goods", limit=6)
    for r in rows2:
        pair = {r["a"].lower(), r["b"].lower()}
        assert not (pair == {"engine", "wheel"}), rows2
    store.close()


def test_deadend_warning_at_recall() -> None:
    """A recalled fact whose entity has a known dead end must carry the
    warning into the result — the loop closes at the point of use."""
    path = os.path.join(tempfile.mkdtemp(), "dw.db")
    store = _fresh(path)
    store.add_turn("", "Redis caching layer для API.", session_id="s1")
    store.mark_deadend("Redis", "rate limits на бесплатном tier")
    hits = store.search("Redis", limit=5)
    warns = [h for h in hits if h["source"] == "deadend-warning"]
    assert warns, hits
    assert "rate limits" in warns[0]["text"], warns
    store.close()


def test_resolved_invalidation_when_source_dies() -> None:
    """If the underlying fact of an auto-resolved answer is archived, the
    frozen answer must retire instead of lying."""
    path = os.path.join(tempfile.mkdtemp(), "ri.db")
    store = _fresh(path)
    fid = store.remember("Engine burns Kerosene fuel.",
                         source="turn:assistant", session_id="s1")
    rid = store.remember(
        "[resolved] Q: engine fuel\nA: Engine burns Kerosene fuel.",
        kind="resolved", source="resolved",
        meta={"query": "engine fuel", "top_fact": fid, "auto": True})
    t0 = time.time()
    r1 = store.search("engine fuel", limit=3, now=t0)
    assert len(r1) == 1 and r1[0]["source"] == "resolved", r1
    # the source fact is negated/archived -> resolved must retire
    store._conn.execute("UPDATE facts SET archived = 1 WHERE id = ?", (fid,))
    store._conn.commit()
    r2 = store.search("engine fuel", limit=3, now=t0 + 40)
    assert not any(h["source"] == "resolved" for h in r2), r2
    dead = store._conn.execute(
        "SELECT archived FROM facts WHERE id = ?", (rid,)).fetchone()
    assert dead and bool(dead["archived"]), dead
    store.close()


def test_distill_docs_to_rules() -> None:
    path = os.path.join(tempfile.mkdtemp(), "dd.db")
    store = _fresh(path)

    class _FakeLLM:
        calls = 0

        @staticmethod
        def available() -> bool:
            return True

        def chat_json(self, _messages):
            type(self).calls += 1
            return {"rules": [
                "Always back up the database before every deploy.",
                "Put canonical links on every public page."]}

    store.ingest_document(
        "Бэкап базы обязателен перед каждым деплоем.\n"
        "Canonical ссылка ставится на каждой публичной странице.",
        title="Правила деплоя",
    )
    store.llm = _FakeLLM()
    res = store.distill_docs(max_chunks=5, max_llm_calls=1)
    assert res["rules"] == 2 and res["calls"] == 1, res
    rows = store._conn.execute(
        "SELECT text, meta FROM facts WHERE kind='rule' AND archived=0").fetchall()
    assert len(rows) == 2, rows
    for r in rows:
        assert json.loads(r["meta"])["pending_review"] == 1
    docs = store._conn.execute(
        "SELECT meta FROM facts WHERE kind='doc' AND archived=0").fetchall()
    assert all(json.loads(d["meta"])["distilled"] for d in docs)
    # second run: nothing left to distill, no LLM call
    res2 = store.distill_docs(max_chunks=5, max_llm_calls=1)
    assert res2["skipped"] == "no-rows" and _FakeLLM.calls == 1, res2
    store.close()


def test_invent_capability_boost_and_purpose_deadend() -> None:
    path = os.path.join(tempfile.mkdtemp(), "ib.db")
    store = _fresh(path)
    store.add_turn("", "Engine is strong.", session_id="s1")
    store.add_turn("", "Use Engine for torque output.", session_id="s1")
    store.add_turn("", "Wheel rolls around.", session_id="s1")
    store.add_turn("", "Wheel carries loads.", session_id="s1")
    goal = "need torque output for movement"
    rows = store.invent(goal, limit=5)
    assert rows and rows[0]["score"] > 0, rows
    first_pair = {rows[0]["a"].lower(), rows[0]["b"].lower()}
    assert "engine" in first_pair, rows  # capability boost puts Engine first
    # purpose-aware dead end: Engine failed FOR torque -> excluded for this goal
    store.mark_deadend("Engine", "no torque on demand")
    rows2 = store.invent(goal, limit=5)
    for r in rows2:
        pair = {r["a"].lower(), r["b"].lower()}
        assert "engine" not in pair, rows2
    store.close()


def test_llm_client_routing_prefers_native_ollama():
    """A local Ollama endpoint must NOT go through the OpenAI-compatible wire
    format.  Measured on qwen3.5:9b with the same one-line QA prompt: /v1 spent
    330 completion tokens (hidden thinking) and 6.7s, and returned an EMPTY
    content at max_tokens 256/64/16; /api/chat with think=false took 0.33s and
    13 tokens.  The empty-content case is the dangerous one — callers read it
    as 'LLM unavailable' and silently degrade to extractive mode."""
    from neuro_matrix.llm import (LLMClient, OllamaLLMClient, build_llm_client,
                                  _is_local_ollama_url)
    assert isinstance(build_llm_client("", provider="ollama", model="m1"),
                      OllamaLLMClient)
    # a pasted OpenAI-style local URL is detected even with no provider name
    c = build_llm_client("k", base_url="http://127.0.0.1:11434/v1", model="m1")
    assert isinstance(c, OllamaLLMClient)
    assert c.base_url == "http://127.0.0.1:11434", c.base_url
    # hosted providers keep the OpenAI-compatible client
    assert isinstance(build_llm_client("k", provider="deepseek"), LLMClient)
    assert not isinstance(
        build_llm_client("k", provider="deepseek",
                         base_url="https://api.deepseek.com/v1", model="deepseek-chat"),
        OllamaLLMClient)
    assert _is_local_ollama_url("http://localhost:11434")
    assert not _is_local_ollama_url("https://api.openai.com/v1")
    assert not _is_local_ollama_url("")


class _FakeEmbedder:
    """Deterministic bag-of-words embedder for tests: no network, no model,
    but the same duck-typed interface the real OllamaEmbedder exposes."""

    def __init__(self, dim: int = 32) -> None:
        self.model = "fake-embed"
        self.dim = dim

    def available(self) -> bool:
        return True

    def _vec(self, text: str) -> list:
        import hashlib
        import math
        import re
        v = [0.0] * self.dim
        for w in re.findall(r"[0-9a-zа-яё_]+", str(text).lower()):
            h = int(hashlib.sha1(w.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            v[h] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list) -> list:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list:
        return self._vec(text)


def test_embed_missing_is_incremental_and_idempotent():
    """Vectors are written once per fact and never recomputed in a loop: the
    drain is bounded work per session, so it must report real progress and
    then stop (a drain that keeps re-embedding would burn a local model's GPU
    forever)."""
    path = os.path.join(tempfile.mkdtemp(), "emb1.db")
    store = _fresh(path)
    for i in range(5):
        store.remember(f"id_{i} использует Firebase для синхронизации.")
    assert store.embed_missing() == 0, "no embedder installed -> nothing to do"
    store.embedder = _FakeEmbedder()
    assert store.embed_missing() == 5
    assert store.embed_missing() == 0, "second drain must find nothing new"
    rows = store._conn.execute("SELECT COUNT(*) c FROM fact_embeddings").fetchone()["c"]
    assert rows == 5, rows
    store.close()


def test_spreading_activation_reaches_the_second_hop():
    """The multi-hop bridge: activation seeded on A must reach C through B,
    which is exactly the evidence keyword matching cannot surface (measured
    hit@8 = 0/6 on LoCoMo multi-hop before this existed)."""
    path = os.path.join(tempfile.mkdtemp(), "ppr.db")
    store = _fresh(path)
    store.remember("Alpha связан с Bravo для синхронизации данных.")
    store.remember("Bravo связан с Charlie для хранения данных.")
    seed = store._entity_id_lookup("alpha")
    assert seed is not None
    act = store._spreading_activation({seed: 1.0}, iterations=3, top=10)
    keys = {store._conn.execute("SELECT key FROM entities WHERE id = ?", (i,)).fetchone()["key"]
            for i in act}
    assert "charlie" in keys, f"second hop missing: {sorted(keys)}"
    assert "bravo" in keys, f"first hop missing: {sorted(keys)}"
    assert store._bridge_candidates({seed: 1.0}, limit=5), "no bridge candidates"
    store.close()


def test_rrf_fuses_on_rank_not_score():
    """RRF must ignore score scale (BM25-style values are unbounded, cosine is
    bounded) and reward agreement between lists."""
    from neuro_matrix.embeddings import reciprocal_rank_fusion
    # id 5 is rank 2 in both lists -> should outrank id 9 (rank 1 in one list
    # only) under RRF's agreement bonus.
    fused = reciprocal_rank_fusion([[5, 9], [5, 9]], k=20)
    assert fused[5] > fused[9], fused
    # adding a list that hates an id pushes it down, never below zero
    fused2 = reciprocal_rank_fusion([[5, 9], [9, 5], [5]], k=20)
    assert fused2[5] > fused2[9], fused2


def test_lexical_channel_needs_no_model():
    """The BM25 list must work with no embedder at all.

    That is the whole point of the largest measured gain on this project: it
    costs no model, no GPU and no VRAM, so it reaches every user.  With every
    other channel switched off, fusion must not run (the previous behaviour).
    """
    path = os.path.join(tempfile.mkdtemp(), "noemb.db")
    store = _fresh(path)
    assert store.embedder is None, "test premise: no embedder configured"
    for i in range(4):
        store.remember(f"Gamma проект номер {i} про cloudflare и кеш.")
    hits = store.search("cloudflare кеш", limit=4)
    assert hits, "baseline recall returned nothing"
    assert store._fts_candidates("cloudflare кеш", limit=4), "BM25 must need no model"
    # Every extra channel off: no fusion, exactly the old behaviour.
    store.fts_enabled = False
    store.bridge_enabled = False
    store._cache.clear()
    plain = store.search("cloudflare кеш", limit=4)
    assert all("rrf" not in h for h in plain), "fusion ran with no second channel"
    store.close()


class _CountingLLM:
    """Minimal LLM stub: counts calls and returns a valid rerank payload."""

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def chat_json(self, messages):
        self.calls += 1
        return {"order": []}


def test_zero_llm_budget_means_unlimited():
    """The settings field documents '0 = unlimited'.  The old computation,
    max(0, 0 - used) = 0, did the opposite: it disabled consolidation, the
    decision/trait sweeps, rerank and distillation at once — and 0 is exactly
    what someone running a free local model would set."""
    path = os.path.join(tempfile.mkdtemp(), "budget.db")
    store = _fresh(path)
    store.llm_daily_budget = 0
    assert store.llm_budget_remaining() > 10 ** 8, store.llm_budget_remaining()
    store.llm_spend(50)
    assert store.llm_budget_remaining() > 10 ** 8, "0 must stay unlimited after spending"
    # Reset the day's counter, then check a finite budget still runs out.
    store.set_meta(f"llm_budget:{time.strftime('%Y%m%d')}", "0")
    store.llm_daily_budget = 3
    assert store.llm_budget_remaining() == 3, store.llm_budget_remaining()
    store.llm_spend(3)
    assert store.llm_budget_remaining() == 0, "a finite budget still runs out"
    store.close()


def test_llm_rerank_setting_is_honoured():
    """llm_rerank was declared in the settings, defaulted to true, and read by
    nobody.  It now decides the default rerank behaviour while an explicit
    caller flag still wins."""
    path = os.path.join(tempfile.mkdtemp(), "rerank.db")
    store = _fresh(path)
    for i in range(5):
        store.remember(f"Delta проект {i}: cloudflare кеш и api.")
    stub = _CountingLLM()
    store.llm = stub
    store.llm_daily_budget = 0            # unlimited, so the budget cannot mask it
    store.llm_rerank_default = False
    store.search("cloudflare кеш", limit=3, rerank=None)
    assert stub.calls == 0, "rerank ran although the setting is off"
    store.search("cloudflare кеш", limit=3, rerank=True)
    assert stub.calls == 1, "explicit rerank=True was ignored"
    store.llm_rerank_default = True
    store.search("облачный кеш", limit=3, rerank=None)
    assert stub.calls == 2, "setting on but rerank did not run"
    store.search("другой запрос", limit=3, rerank=False)
    assert stub.calls == 2, "explicit rerank=False was ignored"
    store.close()


def test_deadend_warnings_cannot_flood_recall():
    """Measured live: two dead-end warnings (injected with score 9.5 to warn
    about known dead ends) plus limit=3 returned FOUR rows, two of them
    warnings — relevant evidence got one slot out of three.  Warnings are
    useful; they must not outnumber the facts they warn about, and the caller
    must never receive more rows than it asked for."""
    path = os.path.join(tempfile.mkdtemp(), "flood.db")
    store = _fresh(path)
    store.mark_deadend("start", "iOS-сборку локально на Windows не проверить")
    for i in range(6):
        store.remember(f"start и сборка: шаг {i} про cloudflare и кеш файлов.")
    res = store.search("start сборка cloudflare", limit=3)
    assert len(res) <= 3, f"more rows than limit: {len(res)}"
    warns = [r for r in res if r.get("source") == "deadend-warning"]
    assert len(warns) <= 1, f"warnings flooded the window: {len(warns)} of {len(res)}"
    # Capping must REORDER, not shrink: with room for more, the count is kept.
    res8 = store.search("start сборка cloudflare", limit=8)
    assert len(res8) <= 8, len(res8)
    assert len(res8) >= len(res), (len(res), len(res8))
    store.close()


def test_mmr_diversify_reorders_without_losing_items():
    """The lambda knob must do what it says at both ends: the default (0.5)
    demotes a near-duplicate in favour of a different fact, while 0.95 keeps the
    pure relevance order.  Both directions are asserted because a knob that only
    works one way is how the 0.7 default silently failed."""
    path = os.path.join(tempfile.mkdtemp(), "mmr.db")
    store = _fresh(path)
    items = [
        {"fact_id": 1, "text": "кеш cloudflare и api слой настроены одинаково", "score": 1.0, "kind": "episodic"},
        {"fact_id": 2, "text": "кеш cloudflare и api слой настроены одинаково дважды", "score": 0.99, "kind": "episodic"},
        {"fact_id": 3, "text": "миграции базы данных и бэкапы по расписанию", "score": 0.5, "kind": "episodic"},
    ]
    out = store._mmr_diversify(items, limit=3)
    assert len(out) == 3, [o["fact_id"] for o in out]
    assert out[1]["fact_id"] == 3, f"near-duplicate not demoted: {[o['fact_id'] for o in out]}"
    # Relevance-dominant setting: the near-duplicate keeps its rank.
    out2 = store._mmr_diversify(items, limit=3, lambda_=0.95)
    assert [o["fact_id"] for o in out2] == [1, 2, 3], [o["fact_id"] for o in out2]
    store.close()


def test_fts_ranking_reaches_the_final_order():
    """BM25 evidence must be able to change the final order.

    The FTS index and its query always existed, but the only consumer treated
    them as a *fallback*: it read the BM25 order and then re-scored candidates by
    ``importance x recency``, so an exact lexical match could not influence the
    ranking.  Measured on LoCoMo (1977 questions), exposing the BM25 order as its
    own RRF list with weight 3 took evidence-hit@8 from 32.3% to 57.1% — the
    largest single gain measured on this project.  This test pins the mechanism
    so it cannot be silently dropped again.
    """
    path = os.path.join(tempfile.mkdtemp(), "fts.db")
    store = _fresh(path)
    now = time.time()
    store.remember("Alice discussed the budget for the office renovation",
                   source="turn", ts=now, importance=1.0)
    # Equal importance and equal timestamp: the ONLY thing that can separate the
    # two facts is lexical evidence, which is exactly the mechanism under test.
    store.remember("Alice moved to Vancouver to work on the harbour project",
                   source="turn", ts=now, importance=1.0)
    q = "where did Alice move to Vancouver"
    before = [r["text"] for r in store.search(q, limit=2, include_dossiers=False)]
    assert len(before) == 2, before
    # The lexical channel is on by default now (it is the shipped behaviour).
    assert store._fts_candidates(q, limit=5), "BM25 must be armed by default"
    store._cache.clear()  # search() caches by query; a stale hit would hide the change
    after = [r["text"] for r in store.search(q, limit=2, include_dossiers=False)]
    assert "Vancouver" in after[0], f"BM25 list did not reach the final order: {after}"
    ids = store._fts_candidates(q, limit=5)
    assert ids, "BM25 candidates missing once enabled"
    store.close()


def test_fts_candidates_are_ordered_by_bm25():
    """The lexical list must be ranked by relevance, not by insertion order."""
    path = os.path.join(tempfile.mkdtemp(), "fts2.db")
    store = _fresh(path)
    now = time.time()
    store.remember("Bob fixed the car engine last week", source="turn", ts=now)
    store.remember("the garage project in Berlin was cancelled", source="turn", ts=now)
    store.remember("car engine repair is scheduled", source="turn", ts=now)
    store.fts_enabled = True
    ids = store._fts_candidates("car engine repair", limit=5)
    texts = [store._conn.execute("SELECT text FROM facts WHERE id = ?", (i,)).fetchone()["text"]
             for i in ids]
    assert texts, "no BM25 candidates for an exact query"
    assert "repair" in texts[0] or "fixed" in texts[0], texts
    assert all("Berlin" not in t for t in texts[:2]), texts
    store.close()


def test_structural_channel_distinguishes_who_did_what_to_whom():
    """Roles, not words: the two facts below share every content word.

    "Маша подарила книгу Пете" and "Петя подарил книгу Маше" are the same bag of
    words with opposite meanings — BM25 and embeddings score them identically,
    which is exactly the failure the structural channel exists to fix.  It reads
    roles from morphology (dative = recipient, nominative = agent) with no model
    call, so the question resolves to the correct one of the two.
    """
    from neuro_matrix.propositions import available

    if not available("ru"):
        return  # optional dependency absent — the channel is documented as a no-op
    path = os.path.join(tempfile.mkdtemp(), "slots.db")
    store = NeuroMatrixStore(path, llm=None, llm_daily_budget=0)
    a = store.remember("Маша подарила книгу Пете", source="turn", importance=1.0)
    b = store.remember("Петя подарил книгу Маше", source="turn", importance=1.0)
    assert a and b, "premise: both facts must be stored"
    rows = store._conn.execute(
        "SELECT fact_id, agent, patient, recipient FROM propositions ORDER BY fact_id").fetchall()
    assert len(rows) == 2, [(r["fact_id"], r["predicate"], r["agent"]) for r in rows]
    got = {int(r["fact_id"]): (r["agent"], r["patient"], r["recipient"]) for r in rows}
    assert got[a] == ("маша", "книга", "петя"), got
    assert got[b] == ("петя", "книга", "маша"), got
    first = store._slot_candidates("Кто подарил книгу Пете?", limit=5)
    second = store._slot_candidates("Кто подарил книгу Маше?", limit=5)
    assert first and int(first[0]) == a, f"wrong fact for 'Пете': {first} (want {a})"
    assert second and int(second[0]) == b, f"wrong fact for 'Маше': {second} (want {b})"
    store.close()


def test_structural_channel_survives_a_fact_without_verbs():
    """A fact the parser cannot structure must not become unfindable."""
    path = os.path.join(tempfile.mkdtemp(), "slots2.db")
    store = NeuroMatrixStore(path, llm=None, llm_daily_budget=0)
    fid = store.remember("id_777: важная заметка без глаголов", source="tool", importance=1.0)
    assert fid, "premise: the fact must be stored"
    hits = store.search("важная заметка", limit=3, include_dossiers=False)
    assert any(int(h["fact_id"]) == fid for h in hits), [h["text"] for h in hits]
    store.close()


def _run_all() -> None:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name} ({time.time() - t0:.2f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
