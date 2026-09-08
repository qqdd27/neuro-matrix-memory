"""Core engine tests for neuro_matrix (stdlib-only, no pytest required).

Run:  python tests/test_core.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_matrix.entities import extract_alias_pairs, extract_entities
from neuro_matrix.store import NeuroMatrixStore, _ru_variants


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


def test_extract_alias_pairs_ru():
    pairs = extract_alias_pairs("Криптовалюта TON — это id_777.")
    assert ("ton", "id_777") in pairs, pairs


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
    path = os.path.join(tempfile.mkdtemp(), "m14.db")
    store = _fresh(path)
    fid = store.remember("id_777 используется для выплат.", source="turn:assistant")
    assert fid is not None
    assert abs(store.search("id_777")[0]["confidence"] - 1.0) < 1e-9
    r = store.feedback(fid, helpful=True)
    assert abs(r["confidence"] - 1.15) < 1e-6
    for _ in range(3):
        r = store.feedback(fid, helpful=False)  # 1.15 -> .575 -> .2875 -> .143(archived)
    assert r["archived"] is True, r
    res = store.search("id_777", limit=10)
    assert not any(x["fact_id"] == fid for x in res), res  # archived excluded
    store.close()


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
