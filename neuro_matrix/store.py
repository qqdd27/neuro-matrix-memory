"""NeuroMatrix core store.

Local-first, zero-dependency temporal memory: SQLite + FTS5.

Model (research-informed, see README):
  facts        — episodic statements (one per stored line/turn), with source,
                 session_id, timestamp, importance, consolidated/archived flags.
  entities     — canonical entity registry (crypto ids, tokens, names).
  aliases      — ``entity_id -> alias`` (multilingual: ton <-> id_777 <-> токен).
  fact_entities— which entities a fact mentions (provenance of every edge).
  edges        — weighted association matrix between entities:
                 weight decays lazily with age (reinforcement decays slower
                 than the co-occurrence count suggests; effective strength =
                 count * exp(-age_hours / edge_half_life_hours)).
  dossiers     — consolidated long-term summaries per entity ("sleep" output).

Retrieval is hybrid: exact alias/entity expansion (graph 2-hop) + FTS5
lexical fallback, ranked by association strength * recency.  No embedding
model, no network — <1 ms typical on local SQLite.
"""

from __future__ import annotations

import functools
import difflib
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Optional

from .embeddings import cosine, pack_vector, reciprocal_rank_fusion, unpack_vector

from .entities import (
    STOPWORDS,
    extract_alias_pairs,
    extract_entities,
    strip_anchor_noise,
)

# Fact lifecycle kinds.  DECAYING kinds age (recency ranking + prune);
# durable kinds never fade — they die only via `supersedes` or explicit action.
ALL_KINDS = (
    "episodic", "episode", "decision", "goal", "constraint",
    "lesson", "correction", "foresight", "artifact", "deadend",
    "capability", "doc", "status", "resolved", "rule", "trait",
)
DECAYING_KINDS = frozenset({"episodic", "episode"})

# Negative-reinforcement markers (prediction-error / dopamine, §23): when the
# user signals a mistake, prior claims sharing the anchors get their confidence
# cut and are flagged negated — the learning-theory twin of supersede.
NEG_MARKERS = (
    "ошибк", "не подошло", "неправильно", "не так", "неверно", "не верно",
    "передел", "заменили", "заменим", "на самом деле", "вместо", "отменя",
    "wrong", "mistake", "didn't work", "doesn't work", "not right",
    "instead", "revert", "drop it",
)

# Outcome marking (dead ends, §closed-loop learning): phrases that flag an
# attempt/approach as failed.  Detection is conservative: subject = nearest
# strong anchor BEFORE the marker; reason = the clause after it.
OUTCOME_MARKERS = (
    "не работает", "не заработал", "не сработал", "не сработало", "не вышло",
    "не получилось", "не взлетело", "не подошло", "не подошёл", "не подошла",
    "не подошел", "не подходит", "сломал",
    "сломался", "упал", "ошибк", "глючит", "не поддерживает", "не хватает",
    "недостаточно", "dead end", "не справил", "doesn't work", "didn't work",
    "not working", "not feasible", "failed", "broken", "broke", "error",
    "doesn't fit", "didn't fit", "can't do", "cannot do", "does not work",
)
OUTCOME_REASON_RE = re.compile(
    r"(?:потому\s+что|так\s+как|из-за|из\s+за|причина|поскольку|"
    r"ошибк|проблема|because|due\s+to|reason|error|the\s+problem)"
)

# Capability markers (§capabilities): '<Provider> <используется для/подходит
# для/...> <что даёт или решает>'.  Provider = strong anchor just before the
# marker; capability = the phrase after it (first sentence/clause).
CAPABILITY_MARKERS = (
    "используется для", "используем ", "применяем ", "применяется для",
    "нужен для", "нужна для", "нужно для", "подходит для", "подходит под",
    "для ", "юзаем ", "используется в", "is used for", "we use ", "used for",
    "helps with", "solves ", "good for", "perfect for", "for ",
)

# RU morphological recall: FTS5 (unicode61) has no Russian stemming, so
# 'правила' stored vs 'правило' asked never match.  Expand RU query tokens to
# surface variants (strip known ending -> re-apply common endings) and OR them
# inside the MATCH expression.  Deterministic, no dictionary, no LLM.
_RU_SUFFIXES = (
    "ами", "ями", "ыми", "ого", "его", "ому", "ему", "ую", "юю",
    "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ей", "ый", "ий",
    "ах", "ях", "ам", "ям", "ов", "ев", "ом", "ем", "ою", "ею",
    "а", "я", "у", "ю", "е", "ы", "и", "ь", "о",
)
_RU_ALTS = ("а", "ы", "е", "у", "ой", "ом", "ам", "ах", "ов", "ами",
            "ями", "ая", "ые", "ое", "ого", "ую", "ем", "ей")


# Curated synonym bridge (RU/EN, ops/tech vocabulary): a small, deliberately
# narrow set of groups where the words are near-synonyms in this domain,
# used ONLY as an extra OR-branch in the FTS lexical fallback — never as an
# entity/alias merge. This is an honest, bounded narrowing of the measured
# semantic-recall gap (see scripts/eval_semantic_gap.py), not a fix for it:
# it recovers a paraphrase that swaps one of these specific words for another
# in the same group, but does nothing for two sentences sharing no group
# member at all (that needs an embedding model, which conflicts with this
# project's zero-runtime-dependency, fully local design — see README/roadmap
# rather than silently pretending an n-gram trick closes that gap).
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"сбой", "авария", "инцидент", "падение", "упал", "упала",
               "упало", "упали", "падал", "падала", "падало", "падали",
               "рухнул", "рухнула", "рухнули", "легло", "легли",
               "накрылось", "накрылись", "отвалилось", "отвалились",
               "outage", "incident", "crash", "crashed", "down", "failure"}),
    frozenset({"провайдер", "поставщик", "вендор", "provider", "vendor",
               "supplier"}),
    frozenset({"интернет", "сеть", "сети", "онлайн", "internet", "network",
               "online"}),
    frozenset({"баг", "ошибка", "дефект", "проблема", "bug", "defect",
               "issue", "glitch"}),
    frozenset({"офлайн", "локально", "автономно", "offline", "locally"}),
    frozenset({"переделали", "перешли", "сменили", "switched", "migrated",
               "moved"}),
)
_SYNONYM_INDEX: dict[str, frozenset[str]] = {
    w: g for g in _SYNONYM_GROUPS for w in g
}


def _synonym_variants(tok: str) -> list[str]:
    group = _SYNONYM_INDEX.get(tok.lower())
    return sorted(group - {tok.lower()}) if group else []


def _ru_variants(tok: str) -> list[str]:
    out = [tok]
    for suf in _RU_SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            base = tok[:-len(suf)]
            if base not in out and len(base) >= 3:
                out.append(base)
            for alt in _RU_ALTS:
                cand = base + alt
                if cand not in out and len(cand) >= 3:
                    out.append(cand)
                if len(out) >= 9:
                    break
            break
    return out[:9]


# Repeated-question cache: the 2nd distinct ask (>=2h after the first) of the
# same normalized query freezes the best answer into a durable kind='resolved'
# fact; later identical asks return it instantly (no graph walk, no FTS).
RESOLVED_TTL = 90 * 86400.0


def _qnorm(query: str) -> str:
    return re.sub(r"[^0-9a-zа-яё_]+", " ", query.lower()).strip()[:160]

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'episodic',
    source TEXT NOT NULL DEFAULT 'turn',
    session_id TEXT,
    ts REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 1.0,
    confidence REAL NOT NULL DEFAULT 1.0,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    active_until REAL,
    supersedes INTEGER REFERENCES facts(id),
    consolidated INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_ts ON facts(ts);
CREATE INDEX IF NOT EXISTS idx_facts_consolidated ON facts(consolidated);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text, content='facts', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF text ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    label TEXT,
    kind TEXT NOT NULL DEFAULT 'token',
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_entities_key ON entities(key);

CREATE TABLE IF NOT EXISTS aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE IF NOT EXISTS edges (
    a INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    b INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    inhibited_since REAL,
    stability REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS dossiers (
    entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    updated_at REAL NOT NULL,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'document',
    topic TEXT,
    sha256 TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    path TEXT NOT NULL,
    head TEXT,
    fact_id INTEGER REFERENCES facts(id) ON DELETE SET NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256);
CREATE INDEX IF NOT EXISTS idx_artifacts_fact ON artifacts(fact_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_name ON artifacts(name);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS ops_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    op TEXT NOT NULL,
    scope TEXT,
    fact_id INTEGER,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_ops_ts ON ops_log(ts);

CREATE TABLE IF NOT EXISTS ask_log (
    qkey TEXT PRIMARY KEY,
    first_ts REAL NOT NULL,
    last_ts REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS merge_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    keep_id INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_merge_log_keep ON merge_log(keep_id);

CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    INTEGER PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fact_emb_model ON fact_embeddings(model);
"""


# BM25-style term-frequency saturation (§entity-match saturation, measured
# fix, LoCoMo benchmark 2026-09), applied to the COUNT of distinct query
# entities matched in one fact instead of word frequency: entity-path
# scoring used to SUM one additive term per matched entity, so a fact that
# happened to co-mention several query-adjacent entities (some incidental,
# not actually relevant) could outscore a precisely-matched single-entity
# fact by sheer count, independent of relevance. sat(n) grows toward an
# asymptote of (k1+1) instead of unboundedly: sat(1)==1.0 (single-entity
# facts score EXACTLY as before — zero behavior change for the common
# case), sat(5)~=1.92 for k1=2.0 (a 5-entity fact's score is its average
# per-entity weight times ~1.92, not a naive 5x sum).
#
# k1=2.0 — the honest story: chosen via a small sweep against LoCoMo
# (0.8/1.0/1.5/2.0/2.5/4.0 -> overall evidence-hit@8 31.7/31.5/32.1/32.3/
# 30.9/31.2%), not derived from first principles, so this IS tuned on one
# benchmark. What makes that defensible rather than a magic number picked
# to game one score: (1) 2.0 sits at the upper end of BM25's own
# conventional k1 range in the IR literature (1.2-2.0), a value with
# reasons independent of this project; (2) every candidate was checked
# against the zero-cost evidence-hit@8 metric only — no paid QA-accuracy
# (LLM reader) calls were spent choosing it; (3) the effect is a genuine
# trade, reported honestly, not a pure win: overall evidence-hit@8 went
# 27.8% -> 32.3% (+4.5pts / +16% relative), but temporal (42.8%->36.2%) and
# multi-hop (12.4%->7.9%) REGRESSED — questions that need several entities
# genuinely combined lose some signal that unsaturated summation gave them,
# traded for single/dual-entity precision no longer drowned out by
# incidental multi-entity noise. A second, external benchmark (LongMemEval)
# would be the honest next check that this generalizes past LoCoMo's
# specific question shapes rather than being overfit to them.
_ENTITY_SATURATION_K1 = 2.0


def _entity_match_saturation(n_matched: int, k1: float = _ENTITY_SATURATION_K1) -> float:
    return (n_matched * (k1 + 1.0)) / (n_matched + k1)


def logger_debug_rerank(exc: Exception) -> None:  # pragma: no cover — best-effort noise
    try:
        import logging
        logging.getLogger("neuro_matrix").debug("rerank skipped: %s", exc)
    except Exception:
        pass


class NeuroMatrixStore:
    def __init__(
        self,
        path: str,
        *,
        edge_half_life_hours: float = 24.0 * 365,   # associations age slowly
        recency_half_life_hours: float = 48.0,      # ranking: recent > old
        llm: Optional[Any] = None,
        min_facts_per_dossier: int = 2,
        downscale_factor: float = 0.95,             # SHY sleep downscaling rate
        myelin_stability: float = 3.0,              # edges >= this are protected
        artifacts_dir: Optional[str] = None,        # sidecar payload files
        artifact_max_bytes: int = 100 * 1024 * 1024,
        llm_daily_budget: int = 20,                 # shared consolidation budget
        auto_decide: bool = True,                   # capture explicit decisions
    ) -> None:
        self.path = path
        self.edge_half_life_hours = edge_half_life_hours
        self.recency_half_life_hours = recency_half_life_hours
        self.min_facts_per_dossier = min_facts_per_dossier
        self.llm = llm  # optional LLMClient-like object (chat_json(messages))
        self.downscale_factor = downscale_factor
        self.myelin_stability = myelin_stability
        self.artifacts_dir = artifacts_dir or os.path.join(
            os.path.dirname(os.path.abspath(path)), "artifacts")
        self.artifact_max_bytes = artifact_max_bytes
        self.llm_daily_budget = llm_daily_budget
        self.auto_decide_enabled = auto_decide
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")
        except sqlite3.Error:
            pass
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._migrate()  # idempotent ALTERs for stores created before v0.2
        self._lock = threading.RLock()  # guards every public op (bg sync threads)
        # In-process LRU for repeated queries within/across turns.
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._cache_ttl = 30.0
        # Optional dense/hybrid retrieval.  `embedder` stays None unless a
        # caller installs one (the provider does, when embeddings are enabled
        # and a local model is available): with no embedder every new code
        # path below is skipped, so behaviour is exactly the previous one.
        self.embedder: Optional[Any] = None
        self.hybrid_enabled = True
        # RRF rank constant: 20 rather than the TREC default 60 — published
        # ablations put the optimum at 10-20 for corpora of this size
        # (hundreds to a few thousand facts), where rank differences mean more.
        self.hybrid_rrf_k = 20
        self.hybrid_max_candidates = 400
        self._vec_cache: dict[int, list[float]] = {}

    # ------------------------------------------------------------------ write

    def _migrate(self) -> None:
        """Idempotent schema migration (v0.1 -> v0.2): adds kind / validity /
        supersedes / confidence columns to stores created before v0.2."""
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(facts)").fetchall()}
        if "kind" not in cols:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN kind TEXT NOT NULL DEFAULT 'episodic'")
        if "active_until" not in cols:
            self._conn.execute("ALTER TABLE facts ADD COLUMN active_until REAL")
        if "supersedes" not in cols:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN supersedes INTEGER REFERENCES facts(id)")
        if "confidence" not in cols:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0")
        if "retrieval_count" not in cols:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN retrieval_count INTEGER NOT NULL DEFAULT 0")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(active_until)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_supersedes ON facts(supersedes)")
        ecols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(edges)").fetchall()}
        if "inhibited_since" not in ecols:
            self._conn.execute("ALTER TABLE edges ADD COLUMN inhibited_since REAL")
        if "stability" not in ecols:
            self._conn.execute(
                "ALTER TABLE edges ADD COLUMN stability REAL NOT NULL DEFAULT 0")
        self._conn.commit()

    def remember(
        self,
        text: str,
        *,
        source: str = "turn",
        session_id: str = "",
        ts: Optional[float] = None,
        importance: float = 1.0,
        kind: str = "episodic",
        confidence: float = 1.0,
        active_until: Optional[float] = None,
        supersedes: Optional[int] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Store one statement.  Extracts entities, links them (weighted
        co-occurrence) and processes explicit alias statements.  ``kind``
        selects the lifecycle: DECAYING kinds age, durable kinds persist."""
        kind = kind if kind in ALL_KINDS else "episodic"
        text = strip_anchor_noise(text)
        if len(text) < 3:
            return None
        now = ts if ts is not None else time.time()
        entities = self._resolve_entities(text, now)
        # Importance filter (write-path lever): a statement without any
        # durable anchor is treated as ephemeral conversation, not memory.
        # Durable kinds are explicit user/agent intent — stored regardless.
        if not entities and kind in DECAYING_KINDS:
            return None
        cur = self._conn.execute(
            "INSERT INTO facts (text, kind, source, session_id, ts, importance, "
            "confidence, active_until, supersedes, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (text, kind, source, session_id, now, importance, confidence,
             active_until, supersedes,
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        fact_id = int(cur.lastrowid)
        for ent_id in entities:
            self._conn.execute(
                "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                (fact_id, ent_id),
            )
            self._conn.execute(
                "UPDATE entities SET last_seen=?, hits=hits+1 WHERE id=?",
                (now, ent_id),
            )
        for i, a_id in enumerate(entities):
            for b_id in entities[i + 1:]:
                self._touch_edge(a_id, b_id, now)
        self._conn.commit()
        self._cache.clear()
        return fact_id

    def add_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        ts: Optional[float] = None,
    ) -> list[int]:
        """Persist a completed (user, assistant) turn — maps to MemoryProvider
        ``sync_turn``.  User side gets importance 1.0, assistant side 0.6
        (assistant restatements are weaker evidence)."""
        ids: list[int] = []
        now_ = ts if ts is not None else time.time()
        if user_content and not _is_trivial(user_content):
            self._detect_and_apply_correction(user_content, now_)
        # Importance gate on the user side: bare single-anchor questions
        # ("what about id_777?") are ephemeral — the assistant's answer below
        # carries the durable statement.  Store the user turn only when it is
        # substantive (>=2 anchors, an alias statement, or a long message) and
        # is not system/background boilerplate.
        if user_content and not _is_trivial(user_content) and not _is_notification(user_content):
            user_keys = extract_entities(user_content)
            user_worth = (
                len(user_keys) >= 2
                or bool(extract_alias_pairs(user_content))
                or len(user_content) > 120
            )
            if user_worth:
                fid = self.remember(
                    user_content, source="turn:user", session_id=session_id, ts=ts,
                    importance=1.0, meta={"role": "user"},
                )
                if fid:
                    ids.append(fid)
        if assistant_content and not _is_trivial(assistant_content):
            fid = self.remember(
                assistant_content, source="turn:assistant", session_id=session_id,
                ts=ts, importance=0.6, meta={"role": "assistant"},
            )
            if fid:
                ids.append(fid)
        if getattr(self, "auto_decide_enabled", True) and (user_content or assistant_content):
            try:
                self._auto_decide_from_turn(
                    user_content or "", assistant_content or "", session_id, now_)
            except Exception:
                # Auto-extraction must never break the write path.
                import logging
                logging.getLogger("neuro_matrix").debug(
                    "auto_decide skipped", exc_info=True)
        if getattr(self, "outcome_enabled", True):
            for txt in (user_content or "", assistant_content or ""):
                if txt and not _is_notification(txt) and not _is_trivial(txt):
                    try:
                        self._auto_outcome_from_turn(txt, now_, session_id)
                    except Exception:
                        import logging
                        logging.getLogger("neuro_matrix").debug(
                            "auto_outcome skipped", exc_info=True)
        if getattr(self, "capability_enabled", True):
            for txt in (assistant_content, user_content or ""):
                if txt and not _is_notification(txt) and not _is_trivial(txt):
                    try:
                        self._extract_capabilities(txt, now_, session_id)
                    except Exception:
                        pass
        return ids

    # ------------------------------------------------- auto decision capture

    _DECIDE_SCOPE_RE = re.compile(
        r"(?:для|по|в|на|за|for|with|in)\s+([^.,;:!?\n]{2,70}?)\s*"
        r"(?:мы\s+|we\s+)?(?:бер[её]м|используем|переход[ия]м\s+на|перешли\s+на|"
        r"выбрали|остановились\s+на|ставим|делаем\s+на|применяем|выбираю|"
        r"выбрал|go\s+with|we\s+go\s+with|we\s+use|we\s+chose|we\s+picked|"
        r"switching\s+to|moving\s+to|use|pick|chose|adopt)",
        re.IGNORECASE,
    )
    _DECIDE_REASON_RE = re.compile(
        r"(?:потому\s+что|так\s+как|из-за|из\s+за|причина|причины|поскольку|"
        r"для\s+того\s+чтобы|чтобы|because|due\s+to|reason|as\s+we\s+need)",
        re.IGNORECASE,
    )
    _DECIDE_SKIP = {"the", "this", "that", "these", "for", "with", "our",
                    "your", "new", "one", "not", "them", "then", "use"}

    def _auto_decide_from_turn(self, user_text: str, assistant_text: str,
                               session_id: str, now: float) -> None:
        """Heuristic capture of explicit user decisions, e.g.
        "для нового приложения берём Firebase, потому что offline и сроки".

        Conservative on purpose: fires only when a scope phrase precedes a
        decision verb AND the chosen thing is a strong anchor (TitleCase /
        ALL-CAPS / id_ / 0x).  Nothing invented — no scope, no decision.
        Reasons after "потому что / because / ..." become criteria, which is
        exactly what later lessons and supersede chains need."""
        def _find(text: str) -> None:
            m = self._DECIDE_SCOPE_RE.search(text)
            if not m:
                return
            scope = m.group(1)
            # The scope match consumed the verb; everything after it is the
            # choice clause.
            tail = text[m.end():]
            tail = tail.split("\n", 1)[0]
            tail = re.split(r"[.!?;]", tail, maxsplit=1)[0][:120]
            concept = self._concept_key(scope)
            if not concept:
                return
            rm = self._DECIDE_REASON_RE.search(tail)
            if rm:
                clause = tail[rm.end():]
                reasons = [c.strip().rstrip(".,").lower()[:64]
                           for c in re.split(r"[,;]", clause)][:5]
                reasons = [c for c in reasons if c and not _is_trivial(c)
                           and len(c) >= 3]
                pre = tail[:rm.start()]
            else:
                pre = tail
            # First strong anchor AFTER the verb, by position (TitleCase /
            # ALL-CAPS / id_ / 0x).  Position beats anchor-class: in
            # 'переходим на Supabase, потому что нужен SQL' the choice is
            # Supabase even though SQL is also an anchor.
            display = ""
            for mm in re.finditer(
                    r"(?i)\b(?:id[-_]?[a-z0-9]{1,20}|0x[0-9a-f]{2,})\b|"
                    r"[A-Za-zА-Яа-яЁё]{2,}", pre):
                tok = mm.group(0)
                tl = tok.lower()
                if tl in self._DECIDE_SKIP:
                    continue
                if ((tok[0].isupper() and any(ch.islower() for ch in tok)
                     and len(tok) >= 2)
                        or (tok.isupper() and len(tok) >= 2)
                        or re.fullmatch(r"(?i)(?:id[-_]?\w+|0x[0-9a-f]+)", tok)):
                    display = tok
                    break
            if not display:
                return
            try:
                self.decide(concept, display, criteria=reasons, reason="",
                            session_id=session_id)
            except Exception:
                pass

        _find(user_text)

    def sweep_decisions(self, *, batch: int = 8, max_items: int = 6) -> int:
        """Dynamic (LLM) decision capture: read recent user turns and let the
        model extract explicit decisions as structured JSON, then apply them
        through ``decide()``.  Phrasing-independent — complements the cheap
        marker heuristic, not bound to any fixed words.  Spends 1 LLM call."""
        llm = self.llm
        if llm is None or not getattr(llm, "available", lambda: False)() \
                or self.llm_budget_remaining() <= 0:
            return 0
        rows = self._conn.execute(
            "SELECT id, text, meta FROM facts WHERE source = 'turn:user' "
            "AND kind = 'episodic' AND archived = 0 AND consolidated = 0 "
            "AND (meta IS NULL OR meta NOT LIKE '%\"swept\"%') "
            "ORDER BY ts DESC LIMIT ?", (batch,)).fetchall()
        if not rows:
            return 0
        texts = "\n".join(f"[{r['id']}] {r['text'][:700]}" for r in rows)
        content = llm.chat_json([
            {"role": "system",
             "content": (
                 "You read user turns from an agent memory. Extract ONLY "
                 "explicit, unambiguous decisions where the user chose something "
                 "for a scope (e.g. 'для нового приложения берём Firebase, "
                 "потому что offline', 'we are building the app on Vue'). "
                 'Return JSON {"decisions": [{"concept": "...", "choice": "...", '
                 '"criteria": ["..."], "text_id": <id>}]} with at most 6 items. '
                 "Skip questions, opinions and vague talk — do not invent.")},
            {"role": "user", "content": texts}])
        self.llm_spend(1)
        applied = 0
        if isinstance(content, dict):
            for d in (content.get("decisions") or [])[:max_items]:
                if not isinstance(d, dict):
                    continue
                concept = str(d.get("concept") or "").strip()
                choice = str(d.get("choice") or "").strip()
                if len(concept) < 3 or len(choice) < 2:
                    continue
                crit = [str(c).strip().lower()[:64]
                        for c in (d.get("criteria") or []) if c][:6]
                try:
                    self.decide(concept[:80], choice[:80], criteria=crit,
                                reason="", session_id="")
                    applied += 1
                except Exception:
                    continue
        # Mark the whole batch swept (do not re-ask the same turns).
        for r in rows:
            meta = json.loads(r["meta"] or "{}") or {}
            meta["swept"] = 1
            self._conn.execute(
                "UPDATE facts SET meta = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), r["id"]))
        self._conn.commit()
        self._cache.clear()
        return applied

    def sweep_traits(self, *, batch: int = 12, max_items: int = 8) -> int:
        """Write-time trait/interest/preference distillation (LLM-optional,
        budget-gated — same shape as ``sweep_decisions``, applied to a
        different question).

        Biological grounding: Complementary Learning Systems theory
        (McClelland, McNaughton & O'Reilly 1995; schema-dependent
        consolidation, Tse et al. 2007) — scattered hippocampal episodic
        traces ("went camping", "saw a meteor shower", "roasted
        marshmallows") are consolidated during sleep into one general
        neocortical semantic schema ("loves the outdoors"). This is a
        DIFFERENT, complementary mechanism from the SHY/synaptic-homeostasis
        downscaling this engine already implements elsewhere (``_downscale_
        edges``): that one weakens noise, this one abstracts signal.
        Mathematically: each episodic mention is one noisy observation of an
        unobserved latent trait; the distilled trait is a posterior estimate
        from several observations — the same Bayesian lineage as the
        Beta-Bernoulli ``feedback()`` confidence model, just with a
        text-valued instead of scalar latent, which needs an LLM rather than
        a closed-form update.

        Why this exists: single-shot lexical/graph retrieval structurally
        cannot answer an INFERENTIAL question ("what field would X pursue?"
        from scattered interest mentions) at read time — there is no
        entity/keyword bridge between "counseling" mentions and "education
        fields" to traverse. Doing the inference ONCE, at write/consolidation
        time, and storing its result as an ordinary durable fact means the
        read path stays pure graph/FTS lookup (fast, cheap, local, no
        per-query LLM call) — this is the "prospective indexing" pattern
        (write-time LLM enrichment, read-time pure retrieval) rather than a
        read-time reasoning loop. Roadmapped in docs/FRONTIER-RESEARCH.md
        §7 as `kind=profile` / PersonaMem-v2-style distillation; implemented
        here as `kind='trait'`.
        """
        llm = self.llm
        if llm is None or not getattr(llm, "available", lambda: False)() \
                or self.llm_budget_remaining() <= 0:
            return 0
        rows = self._conn.execute(
            "SELECT id, text, meta FROM facts WHERE kind = 'episodic' "
            "AND archived = 0 AND consolidated = 0 "
            "AND (meta IS NULL OR meta NOT LIKE '%\"trait_swept\"%') "
            "ORDER BY ts DESC LIMIT ?", (batch,)).fetchall()
        if not rows:
            return 0
        texts = "\n".join(f"[{r['id']}] {r['text'][:700]}" for r in rows)
        content = llm.chat_json([
            {"role": "system",
             "content": (
                 "You read conversation turns from an agent's memory. Extract "
                 "durable statements about a NAMED person's interests, traits, "
                 "values, or preferences that could help answer a LATER "
                 "inferential question about them (e.g. 'likes hiking and the "
                 "outdoors' from a camping story; 'interested in counseling "
                 "and mental health' from career talk). "
                 'Return JSON {"traits": [{"subject": "Name", "trait": "...", '
                 '"text_id": <id>}]} with at most 8 items, subject = the '
                 "person's name as it appears in the text. Skip vague small "
                 "talk — do not invent traits the text does not support.")},
            {"role": "user", "content": texts}])
        self.llm_spend(1)
        applied = 0
        if isinstance(content, dict):
            now = time.time()
            for t in (content.get("traits") or [])[:max_items]:
                if not isinstance(t, dict):
                    continue
                subject = str(t.get("subject") or "").strip()
                trait = str(t.get("trait") or "").strip()
                if len(subject) < 2 or len(trait) < 3:
                    continue
                text = f"{subject}: {trait}"[:300]
                # importance=0.5 (NOT the >=1.0 used elsewhere for durable
                # kinds): measured regression, real-key paid LoCoMo run,
                # 2026-09 -- trait facts are created with ts=now and kind
                # 'trait' is not in DECAYING_KINDS, so they get a full,
                # undecayed recency multiplier while the ORIGINAL episodic
                # evidence they were distilled from (real timestamps,
                # already old) is heavily decayed. Combined with an
                # importance >=1.0 and NO cap on how many trait facts can
                # appear in results (dossiers get an explicit 2-slot cap for
                # exactly this reason; traits had none), a handful of traits
                # about a hub entity systematically outscored and displaced
                # the real evidence turn from top-k. Confirmed: evidence-hit
                # @8 on an identical 150-question LoCoMo sample was 31.8%
                # with sweep_traits() never invoked, 11.0% in the real paid
                # run where it was -- traits were net HARMFUL to literal
                # recall despite being designed to help inferential recall.
                # A low importance keeps a trait findable when it is the
                # ONLY relevant signal (nothing else competes) while no
                # longer letting it systematically outrank real evidence.
                fid = self.remember(
                    text, source="trait", kind="trait", ts=now,
                    importance=0.5, confidence=1.0,
                    meta={"type": "trait", "subject": subject.lower()})
                if fid is not None:
                    self._attach_entity(fid, subject, now)
                    applied += 1
        # Mark the whole batch swept regardless of outcome (do not re-ask
        # the same turns forever — mirrors sweep_decisions).
        for r in rows:
            meta = json.loads(r["meta"] or "{}") or {}
            meta["trait_swept"] = 1
            self._conn.execute(
                "UPDATE facts SET meta = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), r["id"]))
        self._conn.commit()
        self._cache.clear()
        return applied

    @staticmethod
    def _concept_key(scope_text: str) -> str:
        """'для нового приложения мы' -> 'нового_приложения' (stable key)."""
        tokens = [t for t in re.findall(r"[a-zа-яё0-9_]+", scope_text.lower())
                  if t not in {"мы", "будем", "хотим", "нашего", "наш", "этого",
                               "своего", "свой", "проекта", "the", "our", "for",
                               "with", "a", "an", "to", "of"}]
        key = "_".join(tokens[:4])[:48]
        return key

    def link(self, alias_a: str, alias_b: str, *,
             label: Optional[str] = None) -> Optional[int]:
        """Explicit cross-session identity statement: alias_a <-> alias_b refer
        to the same canonical entity (user tool / alias extraction).  The more
        established entity survives the merge (higher hit count wins).
        Returns the ``merge_log`` id (pass to ``unmerge()`` to undo this exact
        merge, e.g. if the identity statement turns out to be wrong)."""
        now = time.time()
        a = self._entity_for_key(_norm(alias_a), now)
        b = self._entity_for_key(_norm(alias_b), now)
        if a is None or b is None or a == b:
            return None
        ra = self._conn.execute(
            "SELECT hits, last_seen FROM entities WHERE id = ?", (a,)).fetchone()
        rb = self._conn.execute(
            "SELECT hits, last_seen FROM entities WHERE id = ?", (b,)).fetchone()
        if rb is not None and (ra is None or (rb["hits"], rb["last_seen"]) > (ra["hits"], ra["last_seen"])):
            a, b = b, a
        mid = self._merge_entities(a, b, now)
        self._conn.commit()
        return mid

    # ------------------------------------------------------------- retrieval

    # ------------------------------------------------- dense / hybrid retrieval
    def embed_missing(self, *, batch: int = 48, limit: int = 400) -> int:
        """Compute and store vectors for facts that do not have one yet.

        Deliberately NOT called from ``remember``: one HTTP round-trip per
        stored fact would make writing as slow as the embedding server, and a
        memory that stalls while being written to is worse than one that
        indexes a little later.  The provider drains this in batches after a
        session; the benchmark drains it once after ingest.  A search never
        pays for the whole corpus.

        Returns the number of vectors added (0 when embeddings are off).
        """
        emb = self.embedder
        if emb is None or not getattr(emb, "available", lambda: False)():
            return 0
        model = str(getattr(emb, "model", "") or "unknown")
        rows = self._conn.execute(
            "SELECT f.id, f.text FROM facts f "
            "LEFT JOIN fact_embeddings e ON e.fact_id = f.id AND e.model = ? "
            "WHERE e.fact_id IS NULL AND f.archived = 0 AND f.text IS NOT NULL "
            "ORDER BY f.ts DESC LIMIT ?", (model, int(limit))).fetchall()
        added = 0
        for i in range(0, len(rows), max(1, int(batch))):
            chunk = rows[i:i + int(batch)]
            vecs = emb.embed([str(r["text"]) for r in chunk])
            if not vecs or len(vecs) != len(chunk):
                break
            for row, vec in zip(chunk, vecs):
                if not vec:
                    continue
                self._conn.execute(
                    "INSERT OR REPLACE INTO fact_embeddings "
                    "(fact_id, model, dim, vec, created_at) VALUES (?, ?, ?, ?, ?)",
                    (int(row["id"]), model, len(vec), pack_vector(vec), time.time()))
                self._vec_cache[int(row["id"])] = vec
                added += 1
            self._conn.commit()
        return added

    def _load_vectors(self, ids: list[int], model: str) -> dict[int, list[float]]:
        """Decode stored vectors, cache-first.  The in-process cache matters:
        without it every search would re-decode a few thousand float32 rows in
        pure Python, which is the one place this design could get slow."""
        out: dict[int, list[float]] = {}
        need: list[int] = []
        for i in ids:
            cached = self._vec_cache.get(i)
            if cached is not None:
                out[i] = cached
            else:
                need.append(i)
        for i in range(0, len(need), 400):
            chunk = need[i:i + 400]
            ph = ",".join("?" * len(chunk))
            for r in self._conn.execute(
                    f"SELECT fact_id, vec FROM fact_embeddings "
                    f"WHERE model = ? AND fact_id IN ({ph})",
                    [model, *chunk]).fetchall():
                vec = unpack_vector(r["vec"])
                fid = int(r["fact_id"])
                self._vec_cache[fid] = vec
                out[fid] = vec
        return out

    def _materialize_hits(self, ids: list[int], *, dense_only: bool = False) -> list[dict[str, Any]]:
        """Fetch result rows for fact ids found by the dense retriever but
        absent from the heuristic candidate window."""
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, text, kind, source, session_id, ts, importance, "
            f"confidence, retrieval_count, active_until, supersedes, meta "
            f"FROM facts WHERE id IN ({ph})", ids).fetchall()
        matched: dict[int, list[str]] = {}
        for mr in self._conn.execute(
                f"SELECT fe.fact_id, e.key FROM fact_entities fe "
                f"JOIN entities e ON e.id = fe.entity_id "
                f"WHERE fe.fact_id IN ({ph})", ids).fetchall():
            matched.setdefault(int(mr["fact_id"]), []).append(mr["key"])
        out = []
        for row in rows:
            fid = int(row["id"])
            out.append({
                "fact_id": fid,
                "text": row["text"],
                "kind": row["kind"] or "episodic",
                "source": row["source"],
                "session_id": row["session_id"] or "",
                "ts": row["ts"],
                "importance": row["importance"],
                "confidence": row["confidence"] or 1.0,
                "retrieval_count": row["retrieval_count"] or 0,
                "active_until": row["active_until"],
                "supersedes": row["supersedes"],
                # Dense-only hits have no heuristic score; 0.0 keeps them from
                # out-ranking scored evidence, and a positive min_recall_score
                # gate will (correctly) drop them.
                "score": 0.0,
                "dense_only": True,
                "via": matched.get(fid, [])[:6],
            })
        return out

    def _spreading_activation(self, seeds: dict[int, float], *, iterations: int = 3,
                              damping: float = 0.5, top: int = 12) -> dict[int, float]:
        """Personalized PageRank over the association graph (HippoRAG-style).

        Why: the association graph is the one asset this memory has that a
        vector store does not, and multi-hop questions are exactly where
        single-shot retrieval fails (measured here: hit@8 = 0/6 on LoCoMo's
        multi-hop category, while denser categories scored ~33-41%).  The
        published result for this family of methods is a mean +7 F1 over a
        strong dense retriever on associative benchmarks, Recall@5 MuSiQue
        69.7 -> 74.7 (HippoRAG 2, arXiv 2502.14802) — and unlike HippoRAG we do
        not have to build the graph, it is already here (thousands of weighted
        edges per store).

        Power iteration on the edge list, damping 0.5: a query entity's
        activation flows to its neighbours, then to theirs, which is what turns
        "who did X work with" into a fact that never mentions X.  Stability
        (myelination) scales edge conductance, so reinforced paths carry
        further than one-off co-occurrences.
        """
        if not seeds:
            return {}
        total = sum(float(v) for v in seeds.values()) or 1.0
        p = {int(k): float(v) / total for k, v in seeds.items()}
        act = dict(p)
        for _ in range(max(1, int(iterations))):
            nodes = [n for n, w in act.items() if w > 1e-6]
            if not nodes:
                break
            ph = ",".join("?" * len(nodes))
            rows = self._conn.execute(
                f"SELECT a, b, count, stability FROM edges "
                f"WHERE (a IN ({ph}) OR b IN ({ph})) AND inhibited_since IS NULL "
                f"ORDER BY count DESC LIMIT 4000",
                [*nodes, *nodes]).fetchall()
            if not rows:
                break
            edges: list[tuple[int, int, float]] = []
            deg: dict[int, float] = {}
            for r in rows:
                a, b = int(r["a"]), int(r["b"])
                w = float(r["count"] or 1.0) * (1.0 + float(r["stability"] or 0.0))
                edges.append((a, b, w))
                deg[a] = deg.get(a, 0.0) + w
                deg[b] = deg.get(b, 0.0) + w
            new: dict[int, float] = {}
            for a, b, w in edges:
                for src, dst in ((a, b), (b, a)):
                    pa = act.get(src, 0.0)
                    if pa <= 0.0:
                        continue
                    da = deg.get(src) or 1.0
                    new[dst] = new.get(dst, 0.0) + damping * pa * (w / da)
            for n, w in p.items():
                new[n] = new.get(n, 0.0) + (1.0 - damping) * w
            act = new
        # Seeds themselves are already covered by the entity path — the value
        # here is what activation reached *through* them.
        rest = {n: w for n, w in act.items() if n not in p}
        return dict(sorted(rest.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top))])

    def _bridge_candidates(self, seeds: dict[int, float], *, limit: int = 20) -> list[int]:
        """Facts reachable only through the graph — the 'bridge' evidence a
        multi-hop question needs and keyword matching cannot produce."""
        active = self._spreading_activation(seeds, top=max(8, int(limit) // 2))
        if not active:
            return []
        ids = list(active)
        ph = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT fe.entity_id, fe.fact_id FROM fact_entities fe "
            f"JOIN facts f ON f.id = fe.fact_id "
            f"WHERE fe.entity_id IN ({ph}) AND f.archived = 0 "
            f"AND f.active_until IS NULL ORDER BY f.ts DESC LIMIT ?",
            [*ids, int(limit) * 4]).fetchall()
        best: dict[int, float] = {}
        for r in rows:
            fid = int(r["fact_id"])
            w = float(active.get(int(r["entity_id"]), 0.0))
            if w > best.get(fid, 0.0):
                best[fid] = w
        return [fid for fid, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:int(limit)]

    def _apply_fusion(self, query: str, results: list[dict[str, Any]], limit: int,
                      q_entities: Optional[dict[int, float]] = None) -> list[dict[str, Any]]:
        """Fuse up to three ranked lists — heuristic, dense (embeddings) and
        bridge (graph activation) — with Reciprocal Rank Fusion.

        RRF decides ORDER only: the ``score`` field is left untouched so a
        caller's min-score gate keeps meaning what it meant before (an RRF
        value is ~1/20, an entirely different scale).  Any list that cannot be
        produced (no embedder, no graph) is simply absent; with fewer than two
        lists the original order is returned unchanged.
        """
        if len(results) < 2:
            return results
        lists: list[list[int]] = []
        heuristic_ids = [int(r["fact_id"]) for r in results if r.get("fact_id") is not None]
        if heuristic_ids:
            lists.append(heuristic_ids)

        emb = self.embedder
        if emb is not None and getattr(emb, "available", lambda: False)():
            try:
                qvec = emb.embed_one(query)
                if qvec:
                    model = str(getattr(emb, "model", "") or "unknown")
                    all_ids = [int(r["fact_id"]) for r in self._conn.execute(
                        "SELECT fact_id FROM fact_embeddings WHERE model = ?",
                        (model,)).fetchall()]
                    vecs = self._load_vectors(all_ids, model)
                    scored = [(fid, cosine(qvec, vecs[fid])) for fid in all_ids if fid in vecs]
                    scored.sort(key=lambda kv: kv[1], reverse=True)
                    # Floor measured on bge-m3, not guessed: a genuinely related
                    # fact sits around cosine 0.75 while an unrelated one lands
                    # near 0.35, so a 0.15 floor would have let the entire corpus
                    # into the fusion.  0.45 keeps the tail out while still
                    # admitting paraphrase matches.
                    dense = [fid for fid, sim in scored[: max(limit * 6, 60)] if sim > 0.45]
                    if dense:
                        lists.append(dense)
            except Exception:  # noqa: BLE001 - dense must never break recall
                pass

        if q_entities:
            try:
                bridge = self._bridge_candidates(q_entities, limit=max(limit * 3, 24))
                if bridge:
                    lists.append(bridge)
            except Exception:  # noqa: BLE001 - graph must never break recall
                pass

        if len(lists) < 2:
            return results
        fused = reciprocal_rank_fusion(lists, k=int(self.hybrid_rrf_k))
        by_id = {int(r["fact_id"]): r for r in results if r.get("fact_id") is not None}
        missing = [i for i in fused if i not in by_id]
        if missing:
            for r in self._materialize_hits(missing[: max(limit * 2, 20)]):
                by_id[int(r["fact_id"])] = r
        order = sorted(by_id, key=lambda i: fused.get(i, 0.0), reverse=True)
        out = []
        for fid in order:
            item = by_id[fid]
            item["rrf"] = round(float(fused.get(fid, 0.0)), 6)
            out.append(item)
        return out

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        include_dossiers: bool = True,
        session_id: str = "",
        now: Optional[float] = None,
        as_of: Optional[float] = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """Hybrid associative recall.

        1. Expand query entities over aliases + weighted 2-hop graph.
        2. Score facts mentioning any expanded entity (strength * recency).
        3. FTS5 lexical fallback for words that are not entities.
        4. Prepend matching consolidated dossiers (highest-priority logic).

        ``as_of`` (point-in-time, §7): returns only facts known and valid at
        that moment — superseded decisions resurrect for their era, future
        facts are excluded.  Recency ranking always uses real ``now``.
        Retrieved facts are rehearsed (§22.1): retrieval_count +1 and the
        fact's association edges gain a little stability.
        """
        now = now if now is not None else time.time()
        act_t = as_of if as_of is not None else now
        cache_key = f"{query}|{limit}|{include_dossiers}|{as_of!r}|rerank={rerank}"
        # Repeated-question cache: an identical normalized ask that was
        # resolved before returns the frozen answer instantly (checked BEFORE
        # the 30s LRU so even a same-minute repeat gets the cached answer).
        if as_of is None and not rerank:
            try:
                _qn = _qnorm(query)
                if _qn:
                    _res = self._conn.execute(
                        "SELECT id, text, ts, meta FROM facts WHERE kind='resolved' "
                        "AND archived=0 AND json_extract(meta,'$.query')=? "
                        "COLLATE NOCASE AND ts > ? ORDER BY ts DESC LIMIT 1",
                        (_qn, now - RESOLVED_TTL)).fetchone()
                    if _res:
                        # Invalidation: if the underlying top fact died
                        # (archived / negated / expired), the frozen answer is
                        # a lie — retire the resolved row and fall through to a
                        # fresh search instead of serving a stale answer.
                        _rmeta = json.loads(_res["meta"] or "{}") or {}
                        _top = _rmeta.get("top_fact")
                        _stale = False
                        if _top:
                            _tf = self._conn.execute(
                                "SELECT archived, active_until, meta FROM facts "
                                "WHERE id = ?", (int(_top),)).fetchone()
                            _stale = _tf is None or bool(_tf["archived"])
                            if not _stale and _tf["active_until"] is not None \
                                    and float(_tf["active_until"]) <= now:
                                _stale = True
                            if not _stale:
                                _tm = json.loads(_tf["meta"] or "{}") or {}
                                if _tm.get("negated"):
                                    _stale = True
                        if _stale:
                            try:
                                self._conn.execute(
                                    "UPDATE facts SET archived = 1 WHERE id = ?",
                                    (int(_res["id"]),))
                            except sqlite3.Error:
                                pass
                        else:
                            try:
                                self._rehearse(int(_res["id"]), now)
                            except Exception:
                                pass
                            return [{
                                "fact_id": int(_res["id"]),
                                "text": str(_res["text"]),
                                "kind": "resolved",
                                "source": "resolved",
                                "session_id": "",
                                "ts": float(_res["ts"]),
                                "importance": 1.0,
                                "confidence": 1.0,
                                "retrieval_count": 0,
                                "active_until": None,
                                "supersedes": None,
                                "score": 9.0,
                                "via": [],
                            }]
            except sqlite3.Error:
                pass
        hit = self._cache.get(cache_key)
        if hit and now - hit[0] < self._cache_ttl:
            return hit[1]

        q_entities = self._resolve_query_entities(query, now)
        if as_of is not None:
            # Point-in-time reads must be precise: no graph-hop expansion (the
            # concept hub would drag in decisions from other eras) and no
            # current dossiers (they reflect today, not then).
            q_entities = {}
            for key in extract_entities(query):
                eid = self._entity_id_lookup(key)
                if eid is not None:
                    q_entities[eid] = max(q_entities.get(eid, 0.0), 2.0)
        results: list[dict[str, Any]] = []
        seen_facts: set[int] = set()
        scored: dict[int, float] = {}
        # Computed once, used both to boost entity-path candidates below and
        # as the FTS lexical fallback query further down.
        tokens = [
            t for t in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{3,}", query.lower())
            if t not in STOPWORDS
        ]

        if q_entities:
            eids = list(q_entities)
            scored, seen_facts = self._score_facts_by_entities(
                eids, now, as_of=as_of, content_tokens=tokens)

        # Lexical fallback via FTS5 for remaining query tokens.
        if tokens and len(scored) < limit * 4:
            _match_parts: list[str] = []
            for t in tokens[:6]:
                if re.search(r"[а-яё]", t) and len(t) >= 4:
                    _vs = list(_ru_variants(t))
                else:
                    _vs = [t]
                # Curated synonym bridge (bounded, see _SYNONYM_GROUPS): OR in
                # the token's synonym-group siblings alongside its own
                # morphological variants, same weight tier as the rest of
                # this fallback -- a narrow, honest widening, not a semantic
                # search. Checked against every morphological variant (not
                # just the raw token) since the group holds base/dictionary
                # forms ("интернет") while the query carries an inflected one
                # ("интернету") that only the stripped-suffix variant matches.
                for _v in list(_vs):
                    for _sv in _synonym_variants(_v):
                        if _sv not in _vs:
                            _vs.append(_sv)
                if len(_vs) == 1:
                    _match_parts.append(f'"{_vs[0]}"')
                else:
                    _match_parts.append(
                        "( " + " OR ".join(f'"{v}"' for v in _vs) + " )")
            match_q = " OR ".join(_match_parts)
            ts_cond = " AND f.ts <= ?" if as_of is not None else ""
            try:
                params: list[Any] = [match_q, act_t]
                if as_of is not None:
                    params.append(as_of)
                rows = self._conn.execute(
                    "SELECT f.id, f.text, f.source, f.ts, f.importance, "
                    "f.kind, f.confidence, "
                    "bm25(facts_fts) AS rank "
                    "FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                    "WHERE facts_fts MATCH ? AND f.archived = 0 "
                    "AND (f.active_until IS NULL OR f.active_until > ?) "
                    + ts_cond +
                    " ORDER BY rank LIMIT 40",
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for r in rows:
                fid = int(r["id"])
                if fid in scored:
                    continue
                kind = r["kind"] or "episodic"
                rec = 1.0 if kind not in DECAYING_KINDS else math.exp(
                    -max(0.0, (now - r["ts"]) / 3600.0) / self.recency_half_life_hours)
                scored[fid] = (0.35 * float(r["importance"])
                               * float(r["confidence"] or 1.0) * (0.5 + 0.5 * rec))

        # Materialize + sort. `scored` can hold as many entries as facts
        # mentioning ANY resolved query entity — for a hub entity (a project
        # name that recurs across a large fraction of all facts) that can be
        # tens of thousands of rows even though only `limit` are ever
        # returned. Sort by score FIRST and materialize only a bounded
        # candidate window with two batched queries, instead of one
        # individual SELECT per candidate (previously O(all candidates) DB
        # round-trips regardless of `limit` — measured at 50k facts: this
        # step alone issued 100k+ separate queries; batching + capping brings
        # the whole search() call from ~2s to sub-50ms on the same data).
        _cap = max(limit * 8, 60)
        _top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:_cap]
        if _top:
            _ids = [fid for fid, _ in _top]
            _score_map = dict(_top)
            _ph = ",".join("?" * len(_ids))
            _frows = self._conn.execute(
                f"SELECT id, text, kind, source, session_id, ts, importance, "
                f"confidence, retrieval_count, active_until, supersedes, meta "
                f"FROM facts WHERE id IN ({_ph})", _ids).fetchall()
            _matched_map: dict[int, list[str]] = {}
            for _mr in self._conn.execute(
                    f"SELECT fe.fact_id, e.key FROM fact_entities fe "
                    f"JOIN entities e ON e.id = fe.entity_id "
                    f"WHERE fe.fact_id IN ({_ph})", _ids).fetchall():
                _matched_map.setdefault(int(_mr["fact_id"]), []).append(_mr["key"])
            for row in _frows:
                fid = int(row["id"])
                results.append({
                    "fact_id": fid,
                    "text": row["text"],
                    "kind": row["kind"] or "episodic",
                    "source": row["source"],
                    "session_id": row["session_id"] or "",
                    "ts": row["ts"],
                    "importance": row["importance"],
                    "confidence": row["confidence"] or 1.0,
                    "retrieval_count": row["retrieval_count"] or 0,
                    "active_until": row["active_until"],
                    "supersedes": row["supersedes"],
                    "score": round(float(_score_map[fid]), 4),
                    "via": _matched_map.get(fid, [])[:6],
                })
        results.sort(key=lambda x: x["score"], reverse=True)

        # Hybrid retrieval (optional, v0.9): fuse the heuristic ranking with
        # (a) dense similarity via local embeddings and (b) graph activation
        # bridges for multi-hop questions.  Placed before the LLM rerank so a
        # rerank sees the fused candidate order, and before dossier injection
        # so dossiers still win their 2 slots.  Each source is independent and
        # optional: with none of them available this is a no-op.
        if self.hybrid_enabled and as_of is None:
            try:
                results = self._apply_fusion(query, results, limit, q_entities)
            except Exception:  # noqa: BLE001 - fusion must never break recall
                pass

        # LLM rerank (§10, optional): heuristic top-k -> LLM order.  Only when a
        # client is configured, budget remains and the caller asked for it.
        if rerank and as_of is None and results and self.llm is not None \
                and getattr(self.llm, "available", lambda: False)() \
                and self.llm_budget_remaining() > 0:
            try:
                cand = results[: max(limit * 3, 8)]
                lines = "\n".join(
                    f"[{h['fact_id']}] {h['kind']}: {h['text'][:300]}" for h in cand)
                content = self.llm.chat_json([
                    {"role": "system",
                     "content": (
                         "Rank the memory items by relevance to the question. "
                         'Return JSON {"order": [<ids from most to least relevant>]}.')},
                    {"role": "user", "content": f"QUESTION: {query}\n\nITEMS:\n{lines}"}])
                self.llm_spend(1)
                if isinstance(content, dict) and isinstance(content.get("order"), list):
                    wanted = [int(i) for i in content["order"] if isinstance(i, (int, float))]
                    by_id = {int(h["fact_id"]): h for h in cand}
                    ordered = [by_id[i] for i in wanted if i in by_id]
                    ordered += [h for h in cand if int(h["fact_id"]) not in by_id]
                    results = ordered + [h for h in results if int(h["fact_id"]) not in by_id]
            except Exception as e:  # noqa: BLE001  — rerank must never break recall
                logger_debug_rerank(e)

        # Dossiers first — consolidated knowledge outranks raw episodes, but at
        # most 2 slots: a flat 5.0 score across many dossiers used to flood the
        # top and bury far more relevant raw facts (seen live: 60+ rows returned
        # for limit=3).
        if include_dossiers and as_of is None and not q_entities:
            # Lowercase queries ('cloudflare') are not anchors, so the entity
            # path never fires and the entity's dossier stays hidden under
            # weak FTS rows.  Match query tokens to entity keys/aliases
            # case-insensitively as a dossier-only fallback; when no exact
            # key exists, try typo-tolerant close matches ('cloudflre' ->
            # cloudflare) — retrieval suggestion only, never a persisted merge.
            _toks = re.findall(r"[a-z0-9_]{2,}", query.lower())[:6]
            if _toks:
                _ph = ",".join("?" * len(_toks))
                _ids = [r["id"] for r in self._conn.execute(
                    f"SELECT id FROM entities WHERE lower(key) IN ({_ph}) "
                    f"UNION SELECT e.id FROM aliases a JOIN entities e "
                    f"ON e.id = a.entity_id WHERE lower(a.alias) IN ({_ph})",
                    _toks + _toks).fetchall()]
                if not _ids and max(len(t) for t in _toks) >= 5:
                    _cand = [r["key"] for r in self._conn.execute(
                        "SELECT key FROM entities WHERE hits > 0 AND length(key) BETWEEN ? AND ?",
                        (min(len(t) for t in _toks) - 2, max(len(t) for t in _toks) + 2)
                    ).fetchall()]
                    for _t in _toks:
                        _near = difflib.get_close_matches(_t, _cand, n=1, cutoff=0.8)
                        if _near:
                            _row = self._conn.execute(
                                "SELECT id FROM entities WHERE key = ?", (_near[0],)).fetchone()
                            if _row and _row["id"] not in _ids:
                                _ids.append(_row["id"])
                if _ids:
                    q_entities = {i: 1.0 for i in _ids}
        if include_dossiers and q_entities and as_of is None:
            dossiers = self._dossiers_for_entities(list(q_entities))
            if dossiers:
                # Keep only dossiers that plausibly answer THIS query (their
                # own key/alias is a query token, the summary shares one, or
                # the entity itself was matched — incl. fuzzy typo matches) —
                # otherwise the whole graph's dossiers flood every recall.
                qtokens = re.findall(r"[a-zа-яё0-9_]{2,}", query.lower())
                qkeys = {_norm(t) for t in qtokens}
                _qrows = self._conn.execute(
                    f"SELECT key FROM entities WHERE id IN "
                    f"({','.join('?' * len(q_entities))})",
                    list(q_entities)).fetchall() if q_entities else []
                _allowed_keys = {_norm(r["key"]) for r in _qrows}
                dossiers = [
                    d for d in dossiers
                    if (d.get("via") and _norm(str(d["via"][0])) in qkeys)
                    or (d.get("via") and _norm(str(d["via"][0])) in _allowed_keys)
                    or any(w in str(d.get("text", "")).lower() for w in qtokens)
                ]
            used = dossiers[:2]
            out = used + results[: max(0, limit - len(used))]
        else:
            out = results[:limit]
        out = out[:limit]

        # De-duplicate identical texts in one recall (same notice or turn often
        # arrives twice across sessions) — first occurrence keeps its rank.
        _seen: set[str] = set()
        _deduped: list[dict[str, Any]] = []
        for _item in out:
            _txt = str(_item.get("text", ""))
            if _txt in _seen:
                continue
            _seen.add(_txt)
            _deduped.append(_item)
        out = _deduped

        # Dead-end warnings at the point of need: when a recalled fact belongs
        # to an entity with a known dead end, say so right here — 'we already
        # tried this path, here is why it failed' — instead of letting the
        # caller repeat it.
        if as_of is None:
            try:
                _keyset: list[str] = []
                for _it in out:
                    for _v in (_it.get("via") or []):
                        _ks = str(_v).lower()
                        if _ks and _ks not in _keyset:
                            _keyset.append(_ks)
                if _keyset:
                    _de_rows = self._conn.execute(
                        "SELECT json_extract(meta,'$.subject') s, "
                        "json_extract(meta,'$.reason') r, ts FROM facts "
                        "WHERE kind='deadend' AND archived=0 "
                        "AND json_extract(meta,'$.subject') IS NOT NULL "
                        "ORDER BY ts DESC LIMIT 12").fetchall()
                    _warns: list[dict[str, Any]] = []
                    for _dr in _de_rows:
                        if len(_warns) >= 2:
                            break
                        if str(_dr["s"]).lower() in _keyset:
                            _warns.append({
                                "fact_id": None,
                                "text": (f"⚠ Known dead end: {_dr['s']} — "
                                         f"{str(_dr['r'] or '')[:120]}"),
                                "kind": "deadend",
                                "source": "deadend-warning",
                                "session_id": "",
                                "ts": float(_dr["ts"]),
                                "importance": 1.0,
                                "confidence": 1.0,
                                "retrieval_count": 0,
                                "active_until": None,
                                "supersedes": None,
                                "score": 9.5,
                                "via": [],
                            })
                    if _warns:
                        out = _warns + out
            except sqlite3.Error:
                pass

        # Reconsolidation on retrieval: rehearse returned facts (skip dossier
        # stubs and historical point-in-time reads — the past is not rehearsed).
        if as_of is None:
            try:
                for item in out:
                    if item.get("fact_id") and item.get("source") != "dossier":
                        self._rehearse(int(item["fact_id"]), now)
            except sqlite3.Error:
                pass

        self._cache[cache_key] = (now, out)
        # Repeated-ask tracking: 2nd distinct ask (>=2 h later) freezes the
        # top answer into a durable 'resolved' fact for instant future reuse.
        if as_of is None:
            try:
                self._note_ask(query, out, now)
            except Exception:
                pass
        return out

    def entity(self, key: str) -> Optional[dict[str, Any]]:
        """Entity profile by canonical key OR any alias."""
        key = _norm(key)
        row = self._conn.execute(
            "SELECT id, key, label, kind, hits, first_seen, last_seen "
            "FROM entities WHERE key = ?", (key,),
        ).fetchone()
        if row is None:
            row = self._conn.execute(
                "SELECT e.id, e.key, e.label, e.kind, e.hits, e.first_seen, e.last_seen "
                "FROM aliases al JOIN entities e ON e.id = al.entity_id "
                "WHERE al.alias = ?", (key,),
            ).fetchone()
        if row is None:
            return None
        aliases = [
            r["alias"]
            for r in self._conn.execute(
                "SELECT alias FROM aliases WHERE entity_id = ?", (row["id"],)
            ).fetchall()
        ]
        d = self._conn.execute(
            "SELECT summary, updated_at FROM dossiers WHERE entity_id = ?",
            (row["id"],),
        ).fetchone()
        return {
            "key": row["key"],
            "label": row["label"] or row["key"],
            "kind": row["kind"],
            "hits": row["hits"],
            "aliases": aliases,
            "dossier": d["summary"] if d else None,
            "dossier_updated_at": d["updated_at"] if d else None,
        }

    def stats(self) -> dict[str, Any]:
        def one(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])

        return {
            "facts": one("SELECT COUNT(*) FROM facts"),
            "archived": one("SELECT COUNT(*) FROM facts WHERE archived=1"),
            "unconsolidated": one(
                "SELECT COUNT(*) FROM facts WHERE consolidated=0 AND archived=0"),
            "entities": one("SELECT COUNT(*) FROM entities"),
            "aliases": one("SELECT COUNT(*) FROM aliases"),
            "edges": one("SELECT COUNT(*) FROM edges"),
            "dossiers": one("SELECT COUNT(*) FROM dossiers"),
            "path": self.path,
        }

    # ----------------------------------------------- decisions & intent ledger

    def _ensure_entity_key(self, key: str, label: Optional[str] = None,
                           now: Optional[float] = None) -> int:
        """Create/fetch an entity by explicit key (concepts like
        'database_stack' are not anchor-extractable — intent API guarantees
        the entity regardless of heuristics)."""
        now = now if now is not None else time.time()
        key = _norm(key)
        if not key:
            raise ValueError("empty entity key")
        row = self._conn.execute(
            "SELECT id FROM entities WHERE key = ?", (key,)).fetchone()
        if row:
            return int(row["id"])
        al = self._conn.execute(
            "SELECT entity_id FROM aliases WHERE alias = ?", (key,)).fetchone()
        if al:
            return int(al["entity_id"])
        cur = self._conn.execute(
            "INSERT INTO entities (key, label, kind, first_seen, last_seen, hits) "
            "VALUES (?, ?, 'concept', ?, ?, 0)",
            (key, label or key, now, now))
        self._conn.commit()
        return int(cur.lastrowid)

    def _active_decision_for(self, concept_eid: int) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT f.id, f.meta FROM facts f "
            "JOIN fact_entities fe ON fe.fact_id = f.id AND fe.entity_id = ? "
            "WHERE f.kind = 'decision' AND f.archived = 0 "
            "AND f.active_until IS NULL ORDER BY f.ts DESC, f.id DESC LIMIT 1",
            (concept_eid,)).fetchone()
        return dict(row) if row else None

    def decide(
        self,
        concept: str,
        choice: str,
        *,
        criteria: Optional[list[str]] = None,
        scope_tags: Optional[list[str]] = None,
        reason: str = "",
        session_id: str = "",
        ts: Optional[float] = None,
        supersedes: Optional[int] = None,
        auto_supersede: bool = True,
        meta: Optional[dict[str, Any]] = None,
    ) -> int:
        """Record an active decision for a concept.  Choosing a different
        option while one is active retires the old one (supersede chain) and
        inhibits the old association — unless ``auto_supersede=False``."""
        now = ts if ts is not None else time.time()
        criteria = [str(c) for c in (criteria or [])]
        scope = [str(s) for s in (scope_tags or [])]
        dmeta = {
            "concept": concept, "concept_n": _norm(concept),
            "choice": choice, "choice_n": _norm(choice),
            "criteria": criteria, "scope": scope, "reason": reason or "",
        }
        if meta:
            dmeta.update(meta)
        ce = self._ensure_entity_key(concept, label=concept, now=now)
        active = self._active_decision_for(ce) if supersedes is None else None
        if active is not None:
            am = json.loads(active["meta"] or "{}") or {}
            if am.get("choice_n") == _norm(choice):
                # Same choice already active — refresh reason only (idempotent).
                if reason:
                    am["reason"] = reason
                    self._conn.execute(
                        "UPDATE facts SET meta = ? WHERE id = ?",
                        (json.dumps(am, ensure_ascii=False), active["id"]))
                    self._conn.commit()
                return int(active["id"])
            if auto_supersede:
                return self.supersede(
                    int(active["id"]), choice, criteria=criteria,
                    scope_tags=scope, reason=reason or "changed by decide",
                    session_id=session_id, ts=ts)
        text = f"{concept} \u2192 {choice}"
        if criteria:
            text += f" (criteria: {', '.join(criteria)})"
        fid = self.remember(
            text, source="decision", kind="decision", session_id=session_id,
            ts=ts, importance=1.2, confidence=1.0,
            active_until=None, supersedes=supersedes, meta=dmeta)
        if fid is None:
            raise RuntimeError("decision insert failed")
        self._link_concept_choice(fid, ce, choice, now, delta=2.0)
        self._oplog("ADD", f"decision:{_norm(concept)}", fid,
                    detail=f"{choice} (criteria: {', '.join(criteria)})", now=now)
        self._cache.clear()
        return fid

    def supersede(
        self,
        decision_id: int,
        choice: str,
        *,
        criteria: Optional[list[str]] = None,
        scope_tags: Optional[list[str]] = None,
        reason: str = "",
        session_id: str = "",
        ts: Optional[float] = None,
    ) -> int:
        """Retire an active decision and open a new one; inhibit the edge
        concept->old choice so the rejected option stops resurfacing."""
        now = ts if ts is not None else time.time()
        old = self._conn.execute(
            "SELECT id, text, meta FROM facts WHERE id = ? AND kind = 'decision' "
            "AND archived = 0", (decision_id,)).fetchone()
        if old is None:
            raise ValueError(f"decision fact {decision_id} not found")
        om = json.loads(old["meta"] or "{}") or {}
        concept = om.get("concept") or om.get("concept_n")
        if not concept:
            raise ValueError("legacy decision without concept meta")
        self._conn.execute(
            "UPDATE facts SET active_until = ? WHERE id = ?", (now, decision_id))
        old_choice = om.get("choice")
        ce = self._ensure_entity_key(concept, label=concept, now=now)
        if old_choice:
            oc = self._entity_id_lookup(_norm(old_choice))
            if oc is not None:
                lo, hi = (ce, oc) if ce < oc else (oc, ce)
                self._conn.execute(
                    "UPDATE edges SET inhibited_since = ? WHERE a = ? AND b = ?",
                    (now, lo, hi))
        self._conn.commit()
        self._oplog("SUPERSEDE", f"decision:{_norm(concept)}", decision_id,
                    detail=f"{om.get('choice', '')} -> {choice}", now=now)
        self._cache.clear()
        return self.decide(
            concept, choice, criteria=criteria, scope_tags=scope_tags,
            reason=reason or om.get("reason", ""), session_id=session_id,
            ts=ts, supersedes=decision_id, auto_supersede=False)

    def _link_concept_choice(self, fact_id: int, concept_eid: int,
                             choice: str, now: float, delta: float) -> None:
        """Attach concept + choice entities to a decision fact, reinforce the
        concept->choice edge and clear a previous inhibition (re-pick)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, concept_eid))
        try:
            choice_eid = self._ensure_entity_key(choice, label=choice, now=now)
        except ValueError:
            choice_eid = None
        if choice_eid is not None and choice_eid != concept_eid:
            self._conn.execute(
                "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                (fact_id, choice_eid))
            lo, hi = (concept_eid, choice_eid) if concept_eid < choice_eid else (choice_eid, concept_eid)
            self._conn.execute(
                "INSERT INTO edges (a, b, count, first_seen, last_seen, inhibited_since) "
                "VALUES (?, ?, ?, ?, ?, NULL) ON CONFLICT(a, b) DO UPDATE SET "
                "count = edges.count + ?, last_seen = ?, inhibited_since = NULL",
                (lo, hi, delta, now, now, delta, now))
        self._conn.execute(
            "UPDATE entities SET last_seen = ?, hits = hits + 1 WHERE id = ?",
            (now, concept_eid))
        self._conn.commit()

    def attach_evidence(self, fact_id: int, evidence_ids: list[int],
                        now: Optional[float] = None) -> Optional[dict[str, Any]]:
        """EMem-style provenance (§19.1): mark source facts as the evidence of a
        synthesized claim.  The claim can then be quoted with ``[fact #N, session]``
        markers instead of being trusted on faith."""
        row = self._conn.execute(
            "SELECT meta, text FROM facts WHERE id = ? AND archived = 0",
            (fact_id,)).fetchone()
        if row is None:
            return None
        now = now if now is not None else time.time()
        keep: list[int] = []
        ev_ids = [int(i) for i in evidence_ids if i and i != fact_id]
        if not ev_ids:
            return None
        meta = json.loads(row["meta"] or "{}") or {}
        prev = [int(i) for i in (meta.get("evidence") or [])]
        for i in prev + ev_ids:
            if i not in keep:
                keep.append(i)
        meta["evidence"] = keep[-32:]
        self._conn.execute(
            "UPDATE facts SET meta = ? WHERE id = ?",
            (json.dumps(meta, ensure_ascii=False), fact_id))
        self._oplog("LINK", f"evidence:{fact_id}", fact_id,
                    detail=",".join(str(i) for i in keep[-32:]), now=now)
        self._conn.commit()
        self._cache.clear()
        return {"fact_id": fact_id, "evidence": keep[-32:], "text": row["text"]}

    def explain_fact(self, fact_id: int) -> Optional[dict[str, Any]]:
        """Render a claim with its evidence trail and citation markers."""
        row = self._conn.execute(
            "SELECT text, kind, session_id, ts, meta FROM facts WHERE id = ?",
            (fact_id,)).fetchone()
        if row is None:
            return None
        meta = json.loads(row["meta"] or "{}") or {}
        ev = [int(i) for i in (meta.get("evidence") or [])]
        evidence: list[dict[str, Any]] = []
        if ev:
            q = ",".join("?" * len(ev))
            for r in self._conn.execute(
                    f"SELECT id, text, session_id, ts FROM facts WHERE id IN ({q})",
                    ev).fetchall():
                evidence.append({"fact_id": int(r["id"]), "text": r["text"][:500],
                                 "session_id": r["session_id"] or "", "ts": r["ts"]})
        evidence.sort(key=lambda x: x["ts"])
        markers = []
        for e in evidence:
            sid = e["session_id"]
            markers.append(f"[fact #{e['fact_id']}, session {sid or '?'}]")
        cited = row["text"] + (" " + " ".join(markers) if markers else "")
        return {"fact_id": fact_id, "text": row["text"], "kind": row["kind"],
                "session_id": row["session_id"] or "", "cited_text": cited,
                "evidence": evidence}

    def export_profile_card(self, out_dir: Optional[str] = None) -> dict[str, Any]:
        """PersonaMem-style reviewable profile card: who this memory says the
        user/agent is — top entities, active decisions, goals, constraints,
        lessons.  Deterministic; the human audits the mirror."""
        out_dir = out_dir or os.path.join(
            os.path.dirname(os.path.abspath(self.path)), "profile")
        os.makedirs(out_dir, exist_ok=True)
        st = self.stats()
        lines = ["# NeuroMatrix persona card (auto-generated, reviewable)",
                 "> Derived from stored memory — a mirror for humans to audit.",
                 "> Regenerate with `export_profile_card()` / CLI `profile-card`.",
                 "",
                 f"- facts: {st['facts']} | entities: {st['entities']} "
                 f"({st['aliases']} aliases) | dossiers: {st['dossiers']}",
                 ""]
        ents = self._conn.execute(
            "SELECT e.key, e.label, e.kind, e.hits FROM entities e "
            "ORDER BY e.hits DESC LIMIT 15").fetchall()
        if ents:
            lines += ["## Top entities", ""]
            for e in ents:
                lines.append(f"- {e['label'] or e['key']} (`{e['key']}`, "
                             f"kind={e['kind']}, {e['hits']} mentions)")
            lines.append("")
        decs = self._conn.execute(
            "SELECT meta, ts FROM facts WHERE kind = 'decision' AND archived = 0 "
            "AND active_until IS NULL ORDER BY ts DESC LIMIT 20").fetchall()
        if decs:
            lines += ["## Active decisions", ""]
            for d in decs:
                m = json.loads(d["meta"] or "{}") or {}
                lines.append(f"- **{m.get('choice')}** ({m.get('concept')}) — "
                             f"{', '.join(m.get('criteria') or []) or 'no criteria'}")
            lines.append("")
        goals = self._conn.execute(
            "SELECT text FROM facts WHERE kind = 'goal' AND archived = 0 "
            "ORDER BY ts DESC LIMIT 10").fetchall()
        if goals:
            lines += ["## Goals", ""] + [f"- {r['text']}" for r in goals] + [""]
        cons = self._conn.execute(
            "SELECT text FROM facts WHERE kind = 'constraint' AND archived = 0 "
            "ORDER BY ts DESC LIMIT 10").fetchall()
        if cons:
            lines += ["## Constraints", ""] + [f"- {r['text']}" for r in cons] + [""]
        less = self._conn.execute(
            "SELECT text FROM facts WHERE kind = 'lesson' AND archived = 0 "
            "ORDER BY ts DESC LIMIT 10").fetchall()
        if less:
            lines += ["## Lessons", ""] + [f"- {r['text']}" for r in less]
        path = os.path.join(out_dir, "persona.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return {"path": path, "entities": len(ents), "decisions": len(decs),
                "goals": len(goals), "constraints": len(cons), "lessons": len(less)}

    def policy_report(self, limit: int = 2000) -> dict[str, Any]:
        """Distill the memory-ops journal into a learned-policy summary
        (§19.4, RL-ready).  Not a neural policy — an honest, deterministic
        digest of what the memory actually does and which interventions help."""
        rows = self._conn.execute(
            "SELECT op, scope, detail, ts, fact_id FROM ops_log "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        by_op: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        flips: dict[str, int] = {}
        feedback = {"helpful": 0, "unhelpful": 0}
        links = 0
        for r in rows:
            by_op[r["op"]] = by_op.get(r["op"], 0) + 1
            s = (r["scope"] or "").split(":")[0]
            by_scope[s] = by_scope.get(s, 0) + 1
            if r["op"] == "SUPERSEDE":
                flips[s] = flips.get(s, 0) + 1
            if r["op"] == "LINK":
                links += 1
            if r["op"] == "UPDATE" and (r["detail"] or "").startswith("helpful="):
                k = "helpful" if "helpful=True" in r["detail"] else "unhelpful"
                feedback[k] += 1
        scope = by_scope or {}
        recommendations: list[str] = []
        for sname, n in sorted(flips.items(), key=lambda x: -x[1]):
            if n >= 2:
                recommendations.append(
                    f"concept area '{sname}': {n} decision flips — distil a "
                    "lesson / re-check criteria before the next decide")
        total_fb = feedback["helpful"] + feedback["unhelpful"]
        if total_fb >= 3 and feedback["unhelpful"] / total_fb > 0.6:
            recommendations.append(
                "feedback skews unhelpful: raise min_recall_score or tighten "
                "the anchor gate (reducing noisy recall)")
        if not recommendations:
            recommendations.append(
                "default-NOOP is the learned baseline: no policy change yet "
                "(collect more ops/feedback)")
        return {"ops_analysed": len(rows), "by_op": by_op, "by_scope": scope,
                "decision_flips": flips, "feedback": feedback,
                "evidence_links": links, "recommendations": recommendations}

    def skill_propose(self, concept: str,
                      out_dir: Optional[str] = None) -> dict[str, Any]:
        """Memp/MemTool procedure bridge (§19.5): distill a decision concept
        (trail + lessons + recent episodes) into a reviewable skill draft on
        disk.  Human reviews it before it ever becomes a real skill."""
        dec = self.decisions(concept)
        out_dir = out_dir or os.path.join(
            os.path.dirname(os.path.abspath(self.path)), "skills")
        os.makedirs(out_dir, exist_ok=True)
        trail = dec.get("trail") or []
        trail_ids = {int(t["decision_id"]) for t in trail}
        lessons: list[str] = []
        rows = self._conn.execute(
            "SELECT id, text, meta, ts FROM facts WHERE kind = 'lesson' "
            "AND archived = 0 ORDER BY id DESC LIMIT 300").fetchall()
        for r in rows:
            m = json.loads(r["meta"] or "{}") or {}
            old, new = m.get("old_decision"), m.get("new_decision")
            same_concept = (m.get("concept") and _norm(m["concept"]) == _norm(concept))
            if (old in trail_ids and new in trail_ids) or same_concept:
                lessons.append(r["text"])
        lessons = list(dict.fromkeys(reversed(lessons)))
        episodes: list[dict[str, Any]] = []
        ce = self._entity_id_lookup(_norm(concept))
        if ce is not None:
            for r in self._conn.execute(
                    "SELECT f.id, f.text, f.session_id, f.ts FROM facts f "
                    "JOIN fact_entities fe ON fe.fact_id = f.id AND fe.entity_id = ? "
                    "WHERE f.kind IN ('episodic','episode') AND f.archived = 0 "
                    "ORDER BY f.ts DESC LIMIT 6", (ce,)).fetchall():
                episodes.append({"fact_id": int(r["id"]), "text": r["text"][:300],
                                 "session_id": r["session_id"] or ""})
        episodes.reverse()
        active = dec.get("active")
        safe = re.sub(r"[^\w\-]+", "_", _norm(concept))[:80] or "concept"
        lines = [f"# Proposed skill: {concept}",
                 f"> Reviewable draft — promote to a real skill only after human review.",
                 "", "## Trigger",
                 f"Use when the topic **{concept}** comes up and a decision or procedure is needed.",
                 "", "## Context (why we chose what)",
                 "| # | Choice | Status | Criteria | Reason |"]
        for i, t in enumerate(trail, 1):
            lines.append(f"| {i} | {t['choice']} | {t['status']} | "
                         f"{', '.join(t['criteria'] or []) or '-'} | "
                         f"{t.get('reason') or '-'} |")
        if active:
            lines.append("")
            lines.append(f"**Active:** {active['choice']} "
                         f"(decision #{active['decision_id']})")
        lines.append("")
        lines.append("## Proposed procedure")
        if lessons:
            lines += [f"- {l}" for l in lessons]
        else:
            lines.append("- (no consolidated lessons yet — propose again after a "
                         "decision change or corrections)")
        if episodes:
            lines.append("")
            lines.append("## Recent episodes backing this")
            for e in episodes:
                lines.append(f"- {e['text']} — [fact #{e['fact_id']}, "
                             f"session {e['session_id'] or '?'}]")
        path = os.path.join(out_dir, f"{safe}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return {"path": path, "concept": concept, "lessons": len(lessons),
                "episodes": len(episodes), "active": (active or {}).get("choice")}
    def decisions(self, concept: str) -> dict[str, Any]:
        """Current active decision + full evolution trail for a concept."""
        cn = _norm(concept)
        ce = self._entity_id_lookup(cn)
        if ce is None:
            return {"concept": concept, "active": None, "trail": [], "count": 0}
        rows = self._conn.execute(
            "SELECT f.id, f.text, f.meta, f.ts, f.active_until, f.supersedes "
            "FROM facts f JOIN fact_entities fe ON fe.fact_id = f.id "
            "WHERE fe.entity_id = ? AND f.kind = 'decision' AND f.archived = 0 "
            "ORDER BY f.ts ASC, f.id ASC", (ce,)).fetchall()
        trail = []
        for r in rows:
            m = json.loads(r["meta"] or "{}") or {}
            trail.append({
                "decision_id": int(r["id"]),
                "choice": m.get("choice") or r["text"],
                "criteria": m.get("criteria") or [],
                "scope": m.get("scope") or [],
                "reason": m.get("reason") or "",
                "ts": r["ts"],
                "status": "active" if r["active_until"] is None else "superseded",
                "superseded_at": r["active_until"],
                "supersedes": r["supersedes"],
            })
        by_id = {t["decision_id"]: t for t in trail}
        for t in trail:
            if t["supersedes"] in by_id:
                by_id[t["supersedes"]]["superseded_by"] = t["decision_id"]
        active = next((t for t in reversed(trail) if t["status"] == "active"), None)
        return {"concept": concept, "active": active, "trail": trail,
                "count": len(trail)}

    def plan_foresight(self, text: str, trigger_at: float, *,
                       entity: Optional[str] = None, session_id: str = "",
                       meta: Optional[dict[str, Any]] = None) -> Optional[int]:
        """Time-bounded future signal (§19.3 / EverMemOS): surfaces when due or
        whenever the topic reappears.  Durable; dies only via explicit ops."""
        now = time.time()
        fid = self.remember(
            text, source="foresight", kind="foresight", session_id=session_id,
            ts=now, importance=1.0,
            meta={**(meta or {}), "type": "foresight", "trigger_at": trigger_at})
        if fid is not None and entity:
            self._attach_entity(fid, entity, now)
        return fid

    def foresights_due(self, now: Optional[float] = None,
                       limit: int = 10) -> list[dict[str, Any]]:
        """Foresight signals whose trigger time has arrived."""
        now = now if now is not None else time.time()
        out: list[dict[str, Any]] = []
        rows = self._conn.execute(
            "SELECT id, text, meta, ts FROM facts WHERE kind = 'foresight' "
            "AND archived = 0 AND active_until IS NULL "
            "ORDER BY ts ASC LIMIT 100").fetchall()
        for r in rows:
            m = json.loads(r["meta"] or "{}") or {}
            try:
                trig = float(m.get("trigger_at") or 0.0)
            except (TypeError, ValueError):
                continue
            if trig <= now:
                out.append({"foresight_id": int(r["id"]), "text": r["text"],
                            "trigger_at": trig, "created_at": r["ts"]})
                if len(out) >= limit:
                    break
        return out

    def remember_goal(self, text: str, *, entity: Optional[str] = None,
                      session_id: str = "", meta: Optional[dict[str, Any]] = None,
                      ts: Optional[float] = None) -> Optional[int]:
        """Durable user goal (intent ledger). Optional explicit entity anchor."""
        fid = self.remember(text, source="goal", kind="goal",
                            session_id=session_id, ts=ts, importance=1.1,
                            meta={**(meta or {}), "type": "goal"})
        if fid is not None and entity:
            self._attach_entity(fid, entity, ts if ts is not None else time.time())
        return fid

    def remember_constraint(self, text: str, *, entity: Optional[str] = None,
                            session_id: str = "", meta: Optional[dict[str, Any]] = None,
                            ts: Optional[float] = None) -> Optional[int]:
        """Durable user constraint (intent ledger)."""
        fid = self.remember(text, source="constraint", kind="constraint",
                            session_id=session_id, ts=ts, importance=1.1,
                            meta={**(meta or {}), "type": "constraint"})
        if fid is not None and entity:
            self._attach_entity(fid, entity, ts if ts is not None else time.time())
        return fid

    def _attach_entity(self, fact_id: int, entity: str, now: float) -> None:
        try:
            eid = self._ensure_entity_key(entity, label=entity, now=now)
        except ValueError:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            (fact_id, eid))
        self._conn.execute(
            "UPDATE entities SET last_seen = ?, hits = hits + 1 WHERE id = ?",
            (now, eid))
        self._conn.commit()
        self._cache.clear()

    def feedback(self, fact_id: int, helpful: bool) -> Optional[dict[str, Any]]:
        """Reinforcement lever (prediction-error reward), Beta-Bernoulli
        calibrated: confidence is the posterior mean of "this fact is
        helpful" under a Jeffreys prior Beta(0.5, 0.5), rescaled so "no
        feedback yet" == the pre-existing neutral baseline (1.0) and full,
        one-sided agreement asymptotically approaches 2.0 / 0.0 as evidence
        accumulates:

            confidence = 2 * (0.5 + helpful_n) / (1 + helpful_n + unhelpful_n)

        This replaces a fixed +0.15 / *0.5 step per vote, which was provably
        unsound: for the SAME total evidence (e.g. 2 helpful + 2 unhelpful),
        the old formula gave a different confidence depending purely on the
        ORDER the votes arrived in (0.325 vs 0.55 vs 0.3625, measured) —
        Bayesian posterior updates from counts are order-invariant by
        construction, so this class of bug cannot recur. A single vote also
        no longer moves confidence by an identical amount regardless of how
        much prior evidence exists (old: vote #1 and vote #20 both +0.15);
        each new vote now has a diminishing effect as the evidence pool
        grows, which is the statistically correct behaviour (more history ->
        more inertia). Archival requires BOTH low posterior confidence and
        enough accumulated votes (>=3) — a single unlucky vote can no longer
        archive a fact outright."""
        row = self._conn.execute(
            "SELECT confidence, meta FROM facts WHERE id = ? AND archived = 0",
            (fact_id,)).fetchone()
        if row is None:
            return None
        meta = json.loads(row["meta"] or "{}") or {}
        fb = meta.setdefault("feedback", [])
        fb.append({"helpful": bool(helpful), "ts": time.time()})
        helpful_n = sum(1 for e in fb if e.get("helpful"))
        unhelpful_n = len(fb) - helpful_n
        conf = 2.0 * (0.5 + helpful_n) / (1.0 + helpful_n + unhelpful_n)
        archived = int(conf < 0.3 and len(fb) >= 3)
        self._conn.execute(
            "UPDATE facts SET confidence = ?, archived = ?, meta = ? WHERE id = ?",
            (conf, archived, json.dumps(meta, ensure_ascii=False), fact_id))
        self._oplog("UPDATE", f"feedback:{fact_id}", fact_id,
                    detail=f"helpful={bool(helpful)} -> confidence {round(conf, 3)} "
                           f"(n_helpful={helpful_n}, n_unhelpful={unhelpful_n})")
        self._conn.commit()
        self._cache.clear()
        return {"fact_id": fact_id, "confidence": round(conf, 4),
                "archived": bool(archived), "helpful_count": helpful_n,
                "unhelpful_count": unhelpful_n}

    # ------------------------------------------------ capabilities + doc ingest

    def _extract_capabilities(self, text: str, now: float,
                              session_id: str = "") -> Optional[int]:
        """Deterministic capability capture: '<Provider> используется для
        <что даёт>' -> durable kind='capability' fact tying the provider
        entity to what it is good for.  'Используем Riverpod для управления
        состоянием' -> provider=riverpod, capability=управления состоянием."""
        if not text or len(text) < 10 or "?" in text or "؟" in text:
            return None
        tl = text.lower()
        best = None
        for m in CAPABILITY_MARKERS:
            i = tl.find(m)
            while i != -1:
                if i >= 5:
                    before = text[max(0, i - 140):i]
                    keys = extract_entities(before)
                    if keys:
                        for k in reversed(keys):
                            pos = before.lower().rfind(k.lower())
                            if pos != -1 and (len(before) - pos - len(k)) <= 48:
                                best = (k, i, len(m))
                                break
                    if best:
                        break
                i = tl.find(m, i + 1)
            if best:
                break
        if not best:
            return None
        provider, idx, mlen = best
        after = text[idx + mlen: idx + mlen + 200]
        cap = after.split("?")[0].split(".")[0].split("!")[0].split(";")[0]
        cap = cap.strip(" ,;:—–-")
        if len(cap) < 4:
            return None
        cap = cap[:160]
        for _stop in ("продолжаем", "дальше ", "потом ", "кроме того",
                      "также ", "ещё ", "и завтра"):
            _j = cap.lower().find(_stop)
            if _j > 0:
                cap = cap[:_j].strip(" ,;:—–-")
                break
        if len(cap) < 4:
            return None
        with self._lock:
            dup = self._conn.execute(
                "SELECT id FROM facts WHERE kind='capability' AND archived=0 "
                "AND json_extract(meta,'$.provider')=? COLLATE NOCASE "
                "AND json_extract(meta,'$.capability')=? COLLATE NOCASE LIMIT 1",
                (provider, cap)).fetchone()
            if dup:
                return int(dup["id"])
            fid = self.remember(
                f"[capability] {provider}: {cap}", kind="capability",
                source="capability", session_id=session_id, ts=now,
                importance=1.0,
                meta={"provider": provider, "capability": cap, "source": "auto"},
            )
            return fid

    def capabilities(self, provider: Optional[str] = None,
                     limit: int = 25) -> list[dict]:
        """What do we know <provider> is good for (durable capability facts)."""
        with self._lock:
            q = ("SELECT id, text, ts, meta FROM facts "
                 "WHERE kind='capability' AND archived=0")
            params: list = []
            if provider:
                q += " AND json_extract(meta,'$.provider')=? COLLATE NOCASE"
                params.append(provider)
            q += " ORDER BY ts DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(q, params).fetchall()
            out = []
            for r in rows:
                m = json.loads(r["meta"] or "{}") or {}
                out.append({
                    "id": int(r["id"]),
                    "provider": m.get("provider", ""),
                    "capability": m.get("capability", ""),
                    "source": m.get("source", "auto"),
                    "ts": float(r["ts"]),
                })
            return out

    def ingest_document(self, text: str, *, title: str = "", topic: str = "",
                        source: str = "ingest", session_id: str = "") -> dict:
        """Bulk knowledge preload: split a document into sentence-aligned
        chunks and store each as a DURABLE kind='doc' fact.  No anchors
        required — pure-Russian/technical docs are stored regardless (unlike
        decaying episodic turns).  Use for reference material the AI should
        always be able to recall ('как строим приложение', SEO rulebook…)."""
        chunks = self._chunk_text(text)
        stored = 0
        for i, ch in enumerate(chunks):
            fid = self.remember(
                ch, kind="doc", source=source, session_id=session_id,
                ts=None, importance=0.9,
                meta={"title": title[:120], "topic": topic[:80],
                      "chunk": i, "total": len(chunks)},
            )
            if fid:
                stored += 1
        return {"stored": stored, "chunks": len(chunks), "title": title}

    @staticmethod
    def _chunk_text(text: str, max_len: int = 900) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        chunks: list[str] = []
        for para in paras:
            if len(para) <= max_len:
                chunks.append(para)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", para)
            cur = ""
            for s in sentences:
                if cur and len(cur) + len(s) + 1 > max_len:
                    chunks.append(cur)
                    cur = ""
                cur = f"{cur} {s}".strip() if cur else s
            if cur:
                chunks.append(cur)
        return chunks or ([text] if text.strip() else [])

    # ------------------------------------------------ outcome marking (dead ends)

    def _auto_outcome_from_turn(self, user_text: str, now: float,
                                session_id: str = "") -> Optional[int]:
        """Closed-loop learning, write path: 'Flutter не подошёл, потому что
        производительность низкая.' -> a durable deadend fact about Flutter
        with the reason.  Conservative: subject = nearest strong anchor before
        the failure marker (fallback: first anchor after it), reason = the
        clause after the marker.  Questions are never outcomes."""
        if not user_text or "?" in user_text or len(user_text) < 10:
            return None
        tl = user_text.lower()
        marker, idx = None, -1
        for m in OUTCOME_MARKERS:
            i = tl.find(m)
            if i != -1 and (idx == -1 or i < idx):
                marker, idx = m, i
        if marker is None:
            return None
        before = user_text[max(0, idx - 260):idx]
        keys = extract_entities(before)
        subject = keys[-1] if keys else None
        if subject is None:
            after_head = user_text[idx + len(marker): idx + len(marker) + 120]
            keys = extract_entities(after_head)
            if keys:
                # reason-ish words ('потому', 'слишком') are not subjects
                subject = next((k for k in keys
                                if k.lower() not in self._DECIDE_SKIP), None)
            if subject is None:
                # lowercase tech words ('v2 migration') are not anchors, but
                # right after a failure marker they usually ARE the subject
                _junk = self._DECIDE_SKIP | {"of", "the", "on", "in", "at",
                                             "to", "from", "and", "for", "with",
                                             "by", "rate", "limits", "error"}
                for _t in re.findall(r"[a-zA-Z0-9_]{2,}", after_head):
                    if _t.lower() not in _junk:
                        subject = _t
                        break
        if not subject or len(subject) < 2 or subject.lower() in self._DECIDE_SKIP:
            return None
        after = user_text[idx + len(marker): idx + len(marker) + 260]
        seg = after
        rm = OUTCOME_REASON_RE.search(after)
        if rm:
            seg = after[rm.end():]
        seg = seg.split("?")[0].split(".")[0].split("!")[0]
        seg = seg.strip(" ,;:-—").replace("и теперь", "").replace("и надо", "")
        seg = seg.strip(" ,;:-—")
        if (len(seg) < 3 or seg.lower().startswith(
                ("мы ", "я ", "надо ", "нужно ", "давай", "перейд", "осталось", "попробу"))):
            seg = ""
        reason = seg[:160] if seg else marker
        try:
            return self.mark_deadend(subject, reason, source="auto", ts=now,
                                     session_id=session_id)
        except Exception:
            return None

    def mark_deadend(self, subject: str, reason: str, *, source: str = "manual",
                     ts: Optional[float] = None,
                     target_fact_id: Optional[int] = None,
                     session_id: str = "") -> Optional[int]:
        """Record a failed attempt as a durable fact: '[dead-end] <subject>:
        <reason>'.  Kind 'deadend' never decays — the whole point is that the
        system keeps knowing what NOT to re-try.  A same-subject deadend within
        14 days is deduped (returns the existing id)."""
        with self._lock:
            now_ = ts if ts is not None else time.time()
            dup = self._conn.execute(
                "SELECT id FROM facts WHERE kind='deadend' AND archived=0 "
                "AND json_extract(meta,'$.subject')=? COLLATE NOCASE "
                "AND ts > ? LIMIT 1",
                (subject, now_ - 14 * 86400)).fetchone()
            if dup:
                return int(dup["id"])
            text = f"[dead-end] {subject}: {reason}".strip()
            fid = self.remember(
                text, kind="deadend", source="deadend", session_id=session_id, ts=now_,
                importance=1.0,
                meta={"subject": subject, "reason": reason,
                      "outcome": "failed", "source": source},
            )
            if not fid:
                return None
            if target_fact_id:
                try:
                    self.attach_evidence(fid, [target_fact_id])
                except Exception:
                    pass
            self._conn.execute(
                "INSERT INTO ops_log (ts, op, scope, fact_id, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (now_, "MARK", "deadend", fid, f"{subject} :: {reason[:120]}"))
            return fid

    def deadends(self, subject: Optional[str] = None, limit: int = 20) -> list[dict]:
        """Active dead ends — optionally filtered by subject entity (NOCASE).
        Ordered newest-first; only non-archived rows (durable kind)."""
        with self._lock:
            q = ("SELECT id, text, ts, meta FROM facts "
                 "WHERE kind='deadend' AND archived=0")
            params: list = []
            if subject:
                q += " AND json_extract(meta,'$.subject')=? COLLATE NOCASE"
                params.append(subject)
            q += " ORDER BY ts DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(q, params).fetchall()
            out = []
            for r in rows:
                m = json.loads(r["meta"] or "{}") or {}
                out.append({
                    "id": int(r["id"]),
                    "subject": m.get("subject", ""),
                    "reason": m.get("reason", ""),
                    "source": m.get("source", "manual"),
                    "ts": float(r["ts"]),
                    "text": str(r["text"])[:200],
                })
            return out

    # ------------------------------------------------ session project status

    def finalize_session(self, session_id: str) -> Optional[dict]:
        """Session project-status rollup: at session end, distil what happened
        into ONE durable kind='status' fact — decisions taken, dead ends hit,
        capabilities discovered, entities touched.  The next session's first
        prefetch surfaces it, so work on a long-running project (an engine,
        an SEO push…) continues where it stopped instead of re-deriving state.
        Deterministic, no LLM; idempotent per session."""
        if not session_id:
            return None
        with self._lock:
            ex = self._conn.execute(
                "SELECT id FROM facts WHERE kind='status' AND session_id=? "
                "AND archived=0 LIMIT 1", (session_id,)).fetchone()
            if ex:
                return {"id": int(ex["id"]), "session_id": session_id, "new": False}
            n = self._conn.execute(
                "SELECT COUNT(*) c FROM facts WHERE session_id=? AND archived=0",
                (session_id,)).fetchone()["c"]
            if not n:
                return None
            end_ts = self._conn.execute(
                "SELECT MAX(ts) m FROM facts WHERE session_id=?", (session_id,)
            ).fetchone()["m"] or time.time()
            dec = [str(r["text"])[:120] for r in self._conn.execute(
                "SELECT text FROM facts WHERE session_id=? AND kind='decision' "
                "AND archived=0 ORDER BY ts DESC LIMIT 6", (session_id,)).fetchall()]
            de = [str(r["s"]) for r in self._conn.execute(
                "SELECT DISTINCT json_extract(meta,'$.subject') s FROM facts "
                "WHERE session_id=? AND kind='deadend' AND archived=0 "
                "AND json_extract(meta,'$.subject') IS NOT NULL LIMIT 6",
                (session_id,)).fetchall()]
            cp = [str(r["p"]) for r in self._conn.execute(
                "SELECT DISTINCT json_extract(meta,'$.provider') p FROM facts "
                "WHERE session_id=? AND kind='capability' AND archived=0 "
                "AND json_extract(meta,'$.provider') IS NOT NULL LIMIT 6",
                (session_id,)).fetchall()]
            ents = [str(r["key"]) for r in self._conn.execute(
                "SELECT e.key, COUNT(*) c FROM fact_entities fe "
                "JOIN entities e ON e.id = fe.entity_id "
                "JOIN facts f ON f.id = fe.fact_id "
                "WHERE f.session_id=? AND f.archived=0 "
                "GROUP BY e.key ORDER BY c DESC, e.last_seen DESC LIMIT 8",
                (session_id,)).fetchall()]
            day = time.strftime("%Y-%m-%d", time.localtime(end_ts))
            text = (
                f"[session-status] {day} sess:{session_id[:12]}: "
                f"decisions: {('; '.join(dec)) or '—'} | "
                f"dead-ends: {(', '.join(de)) or '—'} | "
                f"capabilities: {(', '.join(cp)) or '—'} | "
                f"touched: {(', '.join(ents)) or '—'}"
            )[:1700]
            fid = self.remember(
                text, kind="status", source="session-status",
                session_id=session_id, ts=end_ts, importance=1.0,
                meta={"session_id": session_id, "decisions": dec,
                      "deadends": de, "capabilities": cp,
                      "entities": ents, "facts": int(n)},
            )
            if not fid:
                return None
            return {"id": fid, "session_id": session_id, "new": True}

    def latest_statuses(self, limit: int = 3) -> list[dict]:
        """Newest session-status rollups (for project continuity at start of
        a new session)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, session_id, ts FROM facts "
                "WHERE kind='status' AND archived=0 "
                "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            return [{
                "id": int(r["id"]),
                "session_id": str(r["session_id"] or ""),
                "text": str(r["text"])[:500],
                "ts": float(r["ts"]),
            } for r in rows]

    # ------------------------------------------------------------ invention

    def invent(self, goal: str, limit: int = 8, top_n: int = 60) -> list[dict[str, Any]]:
        """'Fantasy' on the graph: given a problem, propose NOVEL combinations
        of known components (wheel x engine = car).  Novelty = 1/(1+pair
        co-occurrence count); relevance = shared graph neighbours with the
        goal's entities.  Deterministic, no LLM, nothing is stored — these are
        hypotheses for a human/agent gate, never facts."""
        with self._lock:
            gids: set[int] = set()
            for k in extract_entities(goal):
                row = self._conn.execute(
                    "SELECT id FROM entities WHERE key = ? COLLATE NOCASE",
                    (k,)).fetchone()
                if row:
                    gids.add(int(row["id"]))
            rows = self._conn.execute(
                "SELECT id, key FROM entities WHERE hits >= 2 "
                "ORDER BY hits DESC, last_seen DESC LIMIT ?", (top_n,)).fetchall()
            cands = [(int(r["id"]), str(r["key"])) for r in rows]
            de_rows = self._conn.execute(
                "SELECT json_extract(meta,'$.subject') s, "
                "json_extract(meta,'$.reason') r, ts FROM facts "
                "WHERE kind='deadend' AND archived=0 "
                "AND json_extract(meta,'$.subject') IS NOT NULL "
                "ORDER BY ts DESC LIMIT 60").fetchall()
            de: set[str] = set()
            de_reason: dict[str, str] = {}
            for r_ in de_rows:
                _s = str(r_["s"]).lower()
                de.add(_s)
                if _s not in de_reason:
                    de_reason[_s] = str(r_["r"] or "")
            # Purpose-aware dead-end exclusion: an entity whose KNOWN failure
            # reason matches the goal's words is excluded FOR THIS GOAL (it
            # stays combinable for other goals — a dead end is goal-specific:
            # 'лошадь' for speed died, not for farm work).
            gtok = {t for t in re.findall(r"[a-zа-яё0-9_]{4,}", goal.lower())}
            purpose_exclude = {
                s for s, reason in de_reason.items()
                if (gtok and s in gtok)
                or (reason and any(t in reason.lower() for t in gtok))
            }

            def _neighbors(eid: int) -> dict[int, int]:
                rs = self._conn.execute(
                    "SELECT CASE WHEN a = ? THEN b ELSE a END nid, count "
                    "FROM edges WHERE a = ? OR b = ?", (eid, eid, eid)).fetchall()
                return {int(r["nid"]): int(r["count"]) for r in rs}

            neigh = {cid: _neighbors(cid) for cid, _ in cands}
            rels = {cid: len(set(neigh[cid]) & gids) + (1 if cid in gids else 0)
                    for cid in neigh}
            # Capability boost: a component documented as 'used FOR <goal-ish
            # phrase>' gets a relevance bump (typed-affordance seed).
            cap_rows = self._conn.execute(
                "SELECT json_extract(meta,'$.provider') p, "
                "json_extract(meta,'$.capability') c FROM facts "
                "WHERE kind='capability' AND archived=0 "
                "AND json_extract(meta,'$.provider') IS NOT NULL "
                "LIMIT 400").fetchall()
            cap_boost: dict[str, int] = {}
            for r_ in cap_rows:
                _p = str(r_["p"]).lower()
                _c = str(r_["c"] or "").lower()
                if gtok and _c and any(t in _c for t in gtok):
                    cap_boost[_p] = cap_boost.get(_p, 0) + 2
            for _cid, _key in cands:
                rels[_cid] = rels[_cid] + cap_boost.get(_key.lower(), 0)
            ordered = [
                c for c in sorted(cands, key=lambda c: rels[c[0]], reverse=True)
                if c[1].lower() not in purpose_exclude
            ][:26]

            out: list[dict[str, Any]] = []
            seen: set[tuple[int, int]] = set()
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    a_id, a_key = ordered[i]
                    b_id, b_key = ordered[j]
                    if (a_id, b_id) in seen:
                        continue
                    seen.add((a_id, b_id))
                    cnt = neigh[a_id].get(b_id, 0) + neigh[b_id].get(a_id, 0)
                    novelty = round(1.0 / (1.0 + cnt), 3)
                    rel = rels[a_id] + rels[b_id]
                    score = round(rel * 1.0 + novelty * 2.0, 3)
                    out.append({
                        "a": a_key, "b": b_key, "a_id": a_id, "b_id": b_id,
                        "novelty": novelty, "edge_count": cnt, "relevance": rel,
                        "score": score,
                        "a_deadend": a_key.lower() in de,
                        "b_deadend": b_key.lower() in de,
                    })
            out.sort(key=lambda x: x["score"], reverse=True)
            res: list[dict[str, Any]] = []
            for o in out:
                # already-tried territory: both components are known dead ends
                if o["a_deadend"] and o["b_deadend"]:
                    continue
                o["hypothesis"] = (
                    f"Combine '{o['a']}' x '{o['b']}' to solve: {goal[:90]}")
                res.append(o)
                if len(res) >= limit:
                    break
            return res

    # ------------------------------------------------------------ distiller

    def distill_docs(self, *, max_chunks: int = 4, max_llm_calls: int = 1) -> dict:
        """Doc -> rules distiller: convert ingested doc chunks (kind='doc')
        into crisp durable rules (kind='rule', pending_review=1) with one LLM
        call per up-to-4-chunk batch.  Budget-gated and optional — without an
        LLM or budget it degrades to a no-op, never blocks the write path.
        Distilled chunks are flagged so they are not distilled twice."""
        llm = self.llm
        if llm is None or not getattr(llm, "available", lambda: False)() \
                or self.llm_budget_remaining() <= 0:
            return {"rules": 0, "skipped": "no-llm-or-budget"}
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, meta FROM facts WHERE kind='doc' AND archived=0 "
                "AND (meta IS NULL OR meta NOT LIKE '%\"distilled\":true%') "
                "ORDER BY ts DESC LIMIT ?", (max_chunks,)).fetchall()
            if not rows:
                return {"rules": 0, "skipped": "no-rows"}
            rules_total = 0
            made = 0
            for start in range(0, len(rows), 4):
                if self.llm_budget_remaining() <= 0 or made >= max_llm_calls:
                    break
                batch = rows[start:start + 4]
                items = "\n".join(
                    f"[{r['id']}] {str(r['text'])[:900]}" for r in batch)
                resp = None
                try:
                    resp = llm.chat_json([
                        {"role": "system",
                         "content": (
                             "You are a memory distiller for an agent. Turn "
                             "reference/document chunks into crisp, durable, "
                             "self-contained RULES the agent must follow in "
                             "this project (one sentence each, imperative or "
                             "factual). Keep the author's meaning exactly — do "
                             "not add advice that is not in the text. "
                             'Return JSON {"rules": [string, ...]}, at most 6.')},
                        {"role": "user",
                         "content": f"DOCS:\n{items}\n\nReturn the rules."}])
                    self.llm_spend(1)
                    made += 1
                except Exception:
                    resp = None
                rules: list[str] = []
                if isinstance(resp, dict):
                    rules = [str(x).strip() for x in resp.get("rules", [])
                             if isinstance(x, str) and len(str(x).strip()) >= 10]
                for rid_row in batch:
                    try:
                        rm = json.loads(rid_row["meta"] or "{}") or {}
                        rm["distilled"] = True
                        self._conn.execute(
                            "UPDATE facts SET meta = ? WHERE id = ?",
                            (json.dumps(rm, ensure_ascii=False,
                                        separators=(",", ":")),
                             int(rid_row["id"])))
                    except (sqlite3.Error, ValueError):
                        pass
                for rule in rules[:6]:
                    rule = rule[:500]
                    try:
                        self.remember(
                            rule, kind="rule", source="distill", importance=1.0,
                            meta={"from_doc": int(batch[0]["id"]),
                                  "pending_review": 1},
                        )
                        rules_total += 1
                    except sqlite3.Error:
                        pass
            return {"rules": rules_total, "chunks": len(rows), "calls": made}

    # ------------------------------------------------ repeated-question cache

    def _note_ask(self, query: str, out: list[dict[str, Any]], now: float) -> None:
        """Track distinct asks per normalized query; the 2nd one (>=2 h after
        the first) freezes the top result into a durable 'resolved' fact."""
        qn = _qnorm(query)
        if len(qn) < 4 or not out:
            return
        row = self._conn.execute(
            "SELECT first_ts, hits FROM ask_log WHERE qkey = ?", (qn,)).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO ask_log (qkey, first_ts, last_ts, hits) "
                "VALUES (?, ?, ?, 1)", (qn, now, now))
            return
        hits = int(row["hits"]) + 1
        self._conn.execute(
            "UPDATE ask_log SET hits = ?, last_ts = ? WHERE qkey = ?",
            (hits, now, qn))
        if hits == 2 and now - float(row["first_ts"]) >= 7200.0:
            _exists = self._conn.execute(
                "SELECT id FROM facts WHERE kind='resolved' AND archived=0 "
                "AND json_extract(meta,'$.query')=? COLLATE NOCASE LIMIT 1",
                (qn,)).fetchone()
            if _exists:
                return
            top = next((h for h in out
                        if h.get("source") != "dossier" and h.get("text")), None)
            if not top or float(top.get("score", 0.0)) < 0.15:
                return
            answer = str(top["text"])[:400].replace("\n", " ")
            text = f"[resolved] Q: {qn[:100]}\nA: {answer}"
            try:
                self.remember(
                    text, kind="resolved", source="resolved", ts=now,
                    importance=1.0,
                    meta={"query": qn, "answer": answer, "auto": True,
                          "top_fact": top.get("fact_id"), "hits": 2},
                )
            except sqlite3.Error:
                pass

    def resolve_query(self, query: str, answer: str,
                      ts: Optional[float] = None) -> Optional[int]:
        """Manually freeze a question->answer pair as a durable resolved fact
        (the agent/user says 'запомни ответ на это')."""
        qn = _qnorm(query)
        if len(qn) < 3 or not answer:
            return None
        now_ = ts if ts is not None else time.time()
        answer = str(answer)[:400].replace("\n", " ")
        text = f"[resolved] Q: {qn[:100]}\nA: {answer}"
        fid = self.remember(
            text, kind="resolved", source="resolved", ts=now_, importance=1.0,
            meta={"query": qn, "answer": answer, "auto": False},
        )
        if fid:
            self._conn.execute(
                "INSERT INTO ops_log (ts, op, scope, fact_id, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (now_, "ADD", "resolved", fid, f"{qn[:80]} :: {answer[:80]}"))
        return fid

    def _detect_and_apply_correction(self, text: str, now: float) -> int:
        """Write-path correction (prediction error): negative markers + shared
        anchors -> cut confidence of the newest matching assistant/durable
        claims and flag them negated (deterministic, no LLM)."""
        tl = text.lower()
        if not any(m in tl for m in NEG_MARKERS):
            return 0
        keys = extract_entities(text)
        if not keys:
            return 0
        ph = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT DISTINCT f.id, f.confidence, f.meta FROM facts f "
            f"JOIN fact_entities fe ON fe.fact_id = f.id "
            f"JOIN entities e ON e.id = fe.entity_id "
            f"WHERE e.key IN ({ph}) AND f.archived = 0 "
            f"AND f.active_until IS NULL "
            f"AND f.kind IN ('episodic','decision','goal','constraint') "
            f"AND f.source IN ('turn:assistant','decision','goal','constraint') "
            f"AND (f.meta IS NULL OR f.meta NOT LIKE '%\"negated\": true%') "
            f"ORDER BY f.ts DESC LIMIT 3",
            keys).fetchall()
        applied = 0
        for r in rows:
            conf = max(0.1, float(r["confidence"] or 1.0) * 0.4)
            m = json.loads(r["meta"] or "{}") or {}
            m["negated"] = True
            m["negated_at"] = now
            m["negated_by"] = strip_anchor_noise(text)[:200]
            self._conn.execute(
                "UPDATE facts SET confidence = ?, meta = ? WHERE id = ?",
                (conf, json.dumps(m, ensure_ascii=False), r["id"]))
            applied += 1
        if applied:
            self._conn.commit()
            self._cache.clear()
        return applied

    def contradictions(self) -> dict[str, Any]:
        """Contradiction scan: near-duplicate episodic claims per entity
        (potential conflicting statements) + entities with several active
        durable intent facts + fresh facts that conflict with an entity's
        already-consolidated dossier. Deterministic review list (LLM
        resolution later)."""
        rows = self._conn.execute(
            "SELECT fe.entity_id, f.id, f.text, f.kind FROM fact_entities fe "
            "JOIN facts f ON f.id = fe.fact_id "
            "WHERE f.archived = 0 AND f.active_until IS NULL "
            "AND f.kind IN ('episodic','decision','goal','constraint') "
            "ORDER BY f.ts DESC LIMIT 1500").fetchall()
        by_ent: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            by_ent.setdefault(int(r["entity_id"]), []).append(
                {"id": int(r["id"]), "text": r["text"], "kind": r["kind"]})
        duplicates: list[dict[str, Any]] = []
        conflict_ents: list[dict[str, Any]] = []
        for eid, facts in by_ent.items():
            durables = [f for f in facts if f["kind"] != "episodic"]
            if len(durables) >= 3:
                conflict_ents.append({"entity_id": eid,
                                      "facts": durables[:6]})
            seen: list[dict[str, Any]] = []
            for f in facts:
                for g in seen:
                    if self._jaccard(f["text"], g["text"]) >= 0.62:
                        duplicates.append({"entity_id": eid,
                                           "a": g, "b": f})
                        break
                else:
                    seen.append(f)
                if len(duplicates) >= 20:
                    break
            if len(duplicates) >= 20:
                break
        return {"duplicates": duplicates[:20], "conflict_entities": conflict_ents[:10],
                "duplicate_count": len(duplicates),
                "dossier_conflicts": (dc := self._dossier_conflicts())[:10],
                "dossier_conflict_count": len(dc)}

    def _dossier_conflicts(self) -> list[dict[str, Any]]:
        """Conflict-monitoring pass (prediction-error / ACC signal, mirrors the
        write-path ``_detect_and_apply_correction`` but against consolidated
        long-term memory instead of the last few turns): for every entity that
        already has a dossier, find facts about that SAME entity, recorded
        AFTER the dossier was last consolidated, that carry a negation /
        reversal marker (NEG_MARKERS) — i.e. the entity's settled long-term
        summary and its most recent mention plausibly disagree.  Deterministic,
        no LLM; a human (or a future LLM pass) resolves the flagged pairs by
        re-consolidating."""
        out: list[dict[str, Any]] = []
        dossiers = self._conn.execute(
            "SELECT d.entity_id, d.summary, d.updated_at, e.key FROM dossiers d "
            "JOIN entities e ON e.id = d.entity_id").fetchall()
        for d in dossiers:
            # ``>=`` not ``>``: consolidate() stamps ``updated_at`` from the same
            # wall clock, and coarse platform tick resolution (Windows: ~1 ms)
            # can give a fact written right after a consolidation the *same*
            # timestamp — a strict comparison silently dropped it, so conflicts
            # were detected only ~half the time.  ``consolidated = 0`` keeps
            # facts already folded into the dossier out of the list.
            rows = self._conn.execute(
                "SELECT f.id, f.text, f.ts FROM fact_entities fe "
                "JOIN facts f ON f.id = fe.fact_id "
                "WHERE fe.entity_id = ? AND f.archived = 0 AND f.ts >= ? "
                "AND f.consolidated = 0 "
                "AND f.kind IN ('episodic','decision','goal','constraint','capability') "
                "AND (f.meta IS NULL OR f.meta NOT LIKE '%\"negated\": true%') "
                "ORDER BY f.ts DESC LIMIT 20",
                (int(d["entity_id"]), float(d["updated_at"]))).fetchall()
            for r in rows:
                tl = str(r["text"]).lower()
                if any(m in tl for m in NEG_MARKERS):
                    out.append({
                        "entity_id": int(d["entity_id"]), "entity_key": d["key"],
                        "dossier_summary": str(d["summary"])[:300],
                        "dossier_updated_at": d["updated_at"],
                        "fact_id": int(r["id"]), "fact_text": r["text"][:300],
                        "fact_ts": r["ts"],
                    })
                    if len(out) >= 40:
                        return out
        return out

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"[0-9A-Za-zА-Яа-яЁё]{2,}", a.lower()))
        tb = set(re.findall(r"[0-9A-Za-zА-Яа-яЁё]{2,}", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _downscale_edges(self, now: float) -> int:
        """SHY sleep downscaling (§22.3): proportionally weaken all edges whose
        stability is below the myelin threshold (habits survive, noise fades).
        A factor of 1.0 disables the pass; stability never decays itself."""
        f = float(getattr(self, "downscale_factor", 0.95))
        if f <= 0.0 or f >= 1.0:
            return 0
        cur = self._conn.execute(
            "UPDATE edges SET count = MAX(1, CAST(ROUND(count * ?) AS INTEGER)) "
            "WHERE stability < ?",
            (f, float(getattr(self, "myelin_stability", 3.0))))
        self._conn.commit()
        return int(cur.rowcount or 0)

    def rollup_session(self, session_id: str, summary: Optional[str] = None,
                       ts: Optional[float] = None) -> Optional[int]:
        """Episode rollup (§8): one kind='episode' fact per session carrying its
        topics (top entities by mention count).  Idempotent per session."""
        if not session_id:
            return None
        now = ts if ts is not None else time.time()
        dup = self._conn.execute(
            "SELECT id FROM facts WHERE kind = 'episode' AND meta LIKE ?",
            (f'%"session": "{session_id}"%',)).fetchone()
        if dup:
            return None
        facts = self._conn.execute(
            "SELECT id, text FROM facts WHERE session_id = ? AND archived = 0 "
            "AND kind IN ('episodic','episode')", (session_id,)).fetchall()
        topics_rows = self._conn.execute(
            "SELECT e.key FROM fact_entities fe "
            "JOIN facts f ON f.id = fe.fact_id "
            "JOIN entities e ON e.id = fe.entity_id "
            "WHERE f.session_id = ? AND f.archived = 0 "
            "GROUP BY e.id ORDER BY COUNT(*) DESC LIMIT 6",
            (session_id,)).fetchall()
        if not facts and not topics_rows:
            return None
        topics = [r["key"] for r in topics_rows]
        parts = [f"Session {session_id}: {len(facts)} stored claims."]
        if topics:
            parts.append("Topics: " + ", ".join(topics))
        if summary:
            parts.append(summary[:300])
        text = " | ".join(parts)[:600]
        fid = self.remember(text, source="episode", kind="episode",
                            session_id=session_id, ts=now, importance=0.8,
                            meta={"type": "episode", "session": session_id,
                                  "fact_count": len(facts), "topics": topics})
        if fid is None:
            return None
        for key in topics:
            eid = self._entity_id_lookup(key)
            if eid is not None:
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) "
                    "VALUES (?, ?)", (fid, eid))
        self._conn.execute("UPDATE facts SET consolidated = 1 WHERE id = ?", (fid,))
        self._conn.commit()
        self._cache.clear()
        return fid

    # -------------------------------------------------------------- artifacts

    def store_artifact(
        self,
        name: str,
        *,
        text: Optional[str] = None,
        file_path: Optional[str] = None,
        kind: str = "document",
        topic: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
        session_id: str = "",
        ts: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Big-data layer (§11): payloads live in sidecar files keyed by sha256;
        only metadata + a head snippet (first ~500 chars) enter the DB/index.
        Content-dedupes by sha256.  Never auto-deleted."""
        if (text is None) == (file_path is None):
            raise ValueError("provide exactly one of text= or file_path=")
        if file_path is not None:
            if os.path.getsize(file_path) > self.artifact_max_bytes:
                raise ValueError("artifact exceeds artifact_max_bytes")
            with open(file_path, "rb") as f:
                data = f.read()
        else:
            data = text.encode("utf-8")
        if not data:
            raise ValueError("empty artifact")
        digest = hashlib.sha256(data).hexdigest()
        now = ts if ts is not None else time.time()
        dup = self._conn.execute(
            "SELECT id, fact_id FROM artifacts WHERE sha256 = ?", (digest,)).fetchone()
        if dup is not None:
            return {"artifact_id": int(dup["id"]), "duplicate": True,
                    "sha256": digest, "size": len(data)}
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, digest)
        with open(path, "wb") as f:
            f.write(data)
        head = data[:500].decode("utf-8", errors="replace")
        topic = (topic or f"{name}: {head[:120]}")[:300]
        # Unified-search pointer fact (durable kind, anchor-extractable head).
        fact_text = f"[artifact:{name}] {topic}"
        if head:
            fact_text += " | " + head[:180]
        fid = self.remember(fact_text, source="artifact", kind="artifact",
                            session_id=session_id, ts=now, importance=0.5,
                            meta={"type": "artifact", "name": name, **({} if not meta else meta)})
        cur = self._conn.execute(
            "INSERT INTO artifacts (name, kind, topic, sha256, size, path, head, fact_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, kind, topic, digest, len(data), path, head, fid, now))
        self._conn.commit()
        self._cache.clear()
        return {"artifact_id": int(cur.lastrowid), "duplicate": False,
                "sha256": digest, "size": len(data), "path": path, "fact_id": fid}

    def artifact_get(self, artifact_id: Optional[int] = None, *,
                     name: Optional[str] = None, sha256: Optional[str] = None,
                     max_chars: int = 4000) -> Optional[dict[str, Any]]:
        row = None
        if artifact_id is not None:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        elif sha256:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE sha256 = ?", (sha256,)).fetchone()
        elif name:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE name = ? ORDER BY id DESC LIMIT 1",
                (name,)).fetchone()
        if row is None:
            return None
        head = (row["head"] or "")[:max_chars]
        return {"artifact_id": int(row["id"]), "name": row["name"],
                "kind": row["kind"], "topic": row["topic"],
                "sha256": row["sha256"], "size": row["size"],
                "path": row["path"], "fact_id": row["fact_id"],
                "snippet": head, "truncated": len(head or "") >= max_chars}

    def artifact_find(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        like = f"%{query}%"
        rows = self._conn.execute(
            "SELECT id, name, kind, topic, sha256, size, created_at FROM artifacts "
            "WHERE name LIKE ? OR topic LIKE ? ORDER BY id DESC LIMIT ?",
            (like, like, limit)).fetchall()
        return [dict(r) for r in rows]

    def artifact_delete(self, artifact_id: int) -> bool:
        """Explicit removal only — artifacts are never auto-deleted."""
        row = self._conn.execute(
            "SELECT path, fact_id FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            return False
        self._conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        if row["fact_id"]:
            self._conn.execute(
                "UPDATE facts SET archived = 1 WHERE id = ?", (row["fact_id"],))
        self._conn.commit()
        try:
            if row["path"] and os.path.exists(row["path"]):
                os.remove(row["path"])
        except OSError:
            pass
        self._cache.clear()
        return True

    # -------------------------------------------------------- LLM daily budget

    def llm_budget_remaining(self) -> int:
        day = time.strftime("%Y%m%d")
        used = int(self.get_meta(f"llm_budget:{day}", "0") or 0)
        return max(0, int(getattr(self, "llm_daily_budget", 20)) - used)

    def llm_spend(self, n: int = 1) -> int:
        day = time.strftime("%Y%m%d")
        used = int(self.get_meta(f"llm_budget:{day}", "0") or 0) + n
        self.set_meta(f"llm_budget:{day}", str(used))
        return used

    def _oplog(self, op: str, scope: str, fact_id: Optional[int] = None,
               detail: Optional[str] = None, now: Optional[float] = None) -> None:
        """Memory-ops journal (§19.4 / RL-ready interface): every policy-level
        mutation (ADD/UPDATE/DELETE/SUPERSEDE...) is recorded — the exact
        action space a learned memory policy later consumes."""
        self._conn.execute(
            "INSERT INTO ops_log (ts, op, scope, fact_id, detail) VALUES (?, ?, ?, ?, ?)",
            (now if now is not None else time.time(), op, scope, fact_id,
             (detail or "")[:500]))

    def ops_view(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, ts, op, scope, fact_id, detail FROM ops_log "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def export_markdown(self, directory: str) -> dict[str, int]:
        """Reviewable-memory mirror (§7/§19.5): human-readable markdown tree of
        entities (dossiers, aliases), decision trails and lessons.  Regenerate
        anytime; users may audit/edit the source of truth in the DB via tools."""
        import re as _re
        os.makedirs(os.path.join(directory, "entities"), exist_ok=True)
        os.makedirs(os.path.join(directory, "decisions"), exist_ok=True)
        idx: list[str] = ["# NeuroMatrix Memory — human-readable mirror\n",
                          "Regenerate with `neuromatrix export-markdown` / `store.export_markdown()`.\n"]
        n_entities = n_decisions = n_lessons = 0

        def _safe(name: str) -> str:
            return _re.sub(r"[^\w\-]+", "_", (name or "unnamed"))[:80] or "unnamed"

        lessons = self._conn.execute(
            "SELECT text, ts FROM facts WHERE kind = 'lesson' AND archived = 0 "
            "ORDER BY ts DESC LIMIT 50").fetchall()
        n_lessons = len(lessons)

        ents = self._conn.execute(
            "SELECT e.id, e.key, e.label, e.kind, e.hits FROM entities e "
            "ORDER BY e.hits DESC LIMIT 500").fetchall()
        for ent in ents:
            eid = int(ent["id"])
            aliases = [r["alias"] for r in self._conn.execute(
                "SELECT alias FROM aliases WHERE entity_id = ?", (eid,)).fetchall()]
            d = self._conn.execute(
                "SELECT summary, updated_at FROM dossiers WHERE entity_id = ?",
                (eid,)).fetchone()
            decs = self._conn.execute(
                "SELECT f.meta, f.ts, f.active_until FROM facts f "
                "JOIN fact_entities fe ON fe.fact_id = f.id AND fe.entity_id = ? "
                "WHERE f.kind = 'decision' AND f.archived = 0 ORDER BY f.ts",
                (eid,)).fetchall()
            lines = [f"# Entity: {ent['label'] or ent['key']}",
                     f"- key: `{ent['key']}` | kind: {ent['kind']} | mentions: {ent['hits']}"]
            if aliases:
                lines.append("- aliases: " + ", ".join(f"`{a}`" for a in aliases))
            if d:
                lines += ["", "## Dossier", d["summary"]]
            if decs:
                lines += ["", "## Decisions"]
                for dc in decs:
                    m = json.loads(dc["meta"] or "{}") or {}
                    status = "active" if dc["active_until"] is None else "superseded"
                    lines.append(f"- **{m.get('choice')}** ({status}) — " +
                                 ", ".join(m.get("criteria") or []))
            path = os.path.join(directory, "entities", _safe(ent["key"]) + ".md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            idx.append(f"- [{ent['label'] or ent['key']}](entities/{_safe(ent['key'])}.md)")
            n_entities += 1

        concepts = self._conn.execute(
            "SELECT DISTINCT meta FROM facts WHERE kind = 'decision' AND archived = 0 "
            "ORDER BY ts DESC LIMIT 200").fetchall()
        seen_c: set[str] = set()
        for row in concepts:
            m = json.loads(row["meta"] or "{}") or {}
            concept = m.get("concept")
            if not concept or concept in seen_c:
                continue
            seen_c.add(concept)
            dec = self.decisions(concept)
            lines = [f"# Decisions: {concept}", ""]
            for t in dec["trail"]:
                lines.append(f"- **{t['choice']}** [{t['status']}] — " +
                             ", ".join(t["criteria"] or []) +
                             (f" — {t['reason']}" if t.get("reason") else ""))
            path = os.path.join(directory, "decisions", _safe(concept) + ".md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            n_decisions += 1

        with open(os.path.join(directory, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(idx) + "\n")
        return {"entities": n_entities, "decisions": n_decisions,
                "lessons": n_lessons, "directory": directory}

    def _decision_lessons(self, now: float) -> int:
        """Deterministic / LLM distillation of supersede chains into durable
        lessons (kind='lesson'), once per chain.  This is the 'round horizon'
        step: switching decisions becomes an explicit experience, not a loss."""
        chains = self._conn.execute(
            "SELECT d1.id AS old_id, d1.meta AS old_meta, d1.text AS old_text, "
            "d2.id AS new_id, d2.meta AS new_meta "
            "FROM facts d1 JOIN facts d2 ON d2.supersedes = d1.id "
            "WHERE d1.kind = 'decision' AND d2.kind = 'decision' "
            "AND d1.archived = 0 AND d2.archived = 0").fetchall()
        made = 0
        todo = []
        for ch in chains:
            dup = self._conn.execute(
                "SELECT COUNT(*) c FROM facts WHERE kind = 'lesson' AND meta LIKE ?",
                (f'%"old_decision": {ch["old_id"]}%',)).fetchone()
            if dup and dup["c"] > 0:
                continue
            om = json.loads(ch["old_meta"] or "{}") or {}
            nm = json.loads(ch["new_meta"] or "{}") or {}
            todo.append({
                "old_id": int(ch["old_id"]), "new_id": int(ch["new_id"]),
                "concept": nm.get("concept") or om.get("concept") or "",
                "old_choice": om.get("choice") or "",
                "new_choice": nm.get("choice") or "",
                "old_criteria": om.get("criteria") or [],
                "new_criteria": nm.get("criteria") or [],
            })
        if not todo:
            return 0
        # LLM-enhanced summaries (batched), fallback to structured text.
        summaries: dict[int, str] = {}
        llm_ok = bool(self.llm is not None and getattr(self.llm, "available", lambda: False)())
        if llm_ok:
            for start in range(0, len(todo), 4):
                chunk = todo[start:start + 4]
                blocks = "\n".join(
                    f"ID {c['old_id']}: concept={c['concept']} "
                    f"{c['old_choice']}(criteria {c['old_criteria']}) -> "
                    f"{c['new_choice']}(criteria {c['new_criteria']})"
                    for c in chunk)
                content = self.llm.chat_json([{
                    "role": "system",
                    "content": (
                        "You distill agent decision changes into one-line durable "
                        "lessons (plain text, no markdown). For each ID return "
                        '{"lessons":[{"id":<old_id>,"summary":"..."}]}'),
                }, {"role": "user", "content": blocks}])
                if isinstance(content, dict) and isinstance(content.get("lessons"), list):
                    for item in content["lessons"]:
                        try:
                            summaries[int(item["id"])] = str(item["summary"])
                        except (KeyError, TypeError, ValueError):
                            pass
        for c in todo:
            summary = summaries.get(c["old_id"]) or (
                f"[lesson] {c['concept']}: {c['old_choice']} \u2192 {c['new_choice']} "
                f"(criteria {c['old_criteria']} \u2192 {c['new_criteria']})")
            text = summary[:400]
            fid = self.remember(
                text, source="lesson", kind="lesson", ts=now,
                importance=1.3, confidence=1.0,
                meta={"type": "decision_lesson", "old_decision": c["old_id"],
                      "new_decision": c["new_id"]})
            if fid is not None:
                ce = self._entity_id_lookup(_norm(c["concept"]))
                if ce is not None:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) "
                        "VALUES (?, ?)", (fid, ce))
                made += 1
        self._conn.commit()
        if made:
            self._cache.clear()
        return made

    # -------------------------------------------------------- consolidation

    def consolidate(
        self,
        *,
        max_llm_calls: int = 4,
        force: bool = False,
    ) -> dict[str, Any]:
        """'Sleep' cycle: group fresh (unconsolidated) facts per entity and
        merge them into long-term dossiers.

        LLM path (when a client is configured): batch up to 4 entities per
        call, JSON in/out, merges old dossier + fresh facts into one paragraph.
        Extractive path (no LLM): keeps the top-3 most important/recent facts
        verbatim.  Single-mention entities are left alone (never silently
        dropped — the raw fact stays searchable until prune()).
        """
        now = time.time()
        report: dict[str, Any] = {"buckets": 0, "dossiers_updated": 0,
                                  "llm_calls": 0, "facts_consumed": 0,
                                  "lessons": 0, "downscaled": 0}
        # Decision-evolution distillation runs every sleep, independent of the
        # episodic bucket load (chains are rare and cheap to scan).
        report["lessons"] = self._decision_lessons(now)
        # SHY synaptic downscaling: global proportional decay of unprotected
        # edges (myelinated/stability >= threshold edges are exempt).
        report["downscaled"] = self._downscale_edges(now)
        if not force:
            cur = self._conn.execute(
                "SELECT COUNT(*) c FROM facts WHERE consolidated=0 AND archived=0 "
                "AND kind IN ('episodic','episode')")
            if cur.fetchone()["c"] < self.min_facts_per_dossier:
                return report

        buckets: dict[int, list[int]] = {}
        for row in self._conn.execute(
            "SELECT DISTINCT fe.entity_id, fe.fact_id FROM fact_entities fe "
            "JOIN facts f ON f.id = fe.fact_id "
            "WHERE f.consolidated = 0 AND f.archived = 0 "
            "AND f.kind IN ('episodic','episode') "
            "ORDER BY f.ts"
        ).fetchall():
            buckets.setdefault(int(row["entity_id"]), []).append(int(row["fact_id"]))
        report["buckets"] = len(buckets)
        if not buckets:
            return report

        # Entities worth summarizing: >= min fresh facts each.
        target = [
            (eid, fids) for eid, fids in buckets.items()
            if len(fids) >= self.min_facts_per_dossier
        ]
        llm_available = bool(self.llm is not None and getattr(self.llm, "available", lambda: False)())
        calls = 0
        if llm_available:
            max_llm_calls = min(max_llm_calls, self.llm_budget_remaining())
        consumed: set[int] = set()
        def _fetch(fids: list[int]) -> list[dict[str, Any]]:
            ph = ",".join("?" * len(fids))
            return [dict(r) for r in self._conn.execute(
                f"SELECT id, text, source, ts, importance FROM facts "
                f"WHERE id IN ({ph}) ORDER BY importance DESC, ts DESC",
                fids).fetchall()]

        def _old_dossier(eid: int) -> str:
            r = self._conn.execute(
                "SELECT summary FROM dossiers WHERE entity_id = ?", (eid,)).fetchone()
            return r["summary"] if r else ""

        def _upsert(eid: int, summary: str) -> None:
            self._conn.execute(
                "INSERT INTO dossiers (entity_id, summary, updated_at, meta) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(entity_id) DO UPDATE SET "
                "summary = excluded.summary, updated_at = excluded.updated_at",
                (eid, summary, now, json.dumps({"mode": "llm" if llm_available else "extractive"})),
            )

        # --- surprise-gated selection (Nemori §22.4) -----------------------
        # Consolidate FIRST what surprised the system (no dossier yet, or a
        # claim that extends/contradicts it).  Facts that merely confirm an
        # existing dossier are consumed silently — no dossier rewrite, no
        # wasted LLM tokens on the obvious.
        summary_of: dict[int, str] = {}
        for r in self._conn.execute(
            "SELECT entity_id, summary FROM dossiers").fetchall():
            summary_of[int(r["entity_id"])] = r["summary"]

        def _split(fids: list[int], summary: str) -> tuple[list[dict[str, Any]], list[int]]:
            surpr: list[dict[str, Any]] = []
            conf: list[int] = []
            for fr in _fetch(fids):
                ov = self._jaccard(fr["text"], summary) if summary else 0.0
                if summary and ov > 0.65:
                    conf.append(int(fr["id"]))
                else:
                    surpr.append(fr)
            return surpr, conf

        def _consume(fids_to_mark: list[int]) -> None:
            for f in fids_to_mark:
                if f not in consumed:
                    consumed.add(f)
                    self._conn.execute(
                        "UPDATE facts SET consolidated=1 WHERE id=?", (f,))

        def _order_key(item: tuple[int, list[int]]) -> float:
            eid, fids = item
            s = summary_of.get(eid, "")
            if not s:
                return 0.0  # no dossier yet -> maximum surprise first
            return min(self._jaccard(f["text"], s) for f in _fetch(fids))

        target.sort(key=_order_key)

        # LLM path: batch 4 entities per call over SURPRISING facts only.
        if llm_available and target:
            llm_batch: list[tuple[int, list[dict[str, Any]]]] = []
            for eid, fids in target:
                surpr, confirms = _split(fids, summary_of.get(eid, ""))
                _consume(confirms)
                if surpr:
                    llm_batch.append((eid, surpr))
            for start in range(0, len(llm_batch), 4):
                if calls >= max_llm_calls:
                    break
                chunk = llm_batch[start:start + 4]
                prompt_blocks = []
                for eid, facts_rows in chunk:
                    key = self._conn.execute(
                        "SELECT key FROM entities WHERE id = ?", (eid,)).fetchone()["key"]
                    facts = "\n".join(
                        f"- {f['text'][:280]}" for f in facts_rows[:8])
                    old = _old_dossier(eid)
                    prompt_blocks.append(
                        f"ENTITY: {key}\nOLD DOSSIER: {old or '(none)'}\n"
                        f"FRESH FACTS:\n{facts}")
                content = self.llm.chat_json([{
                    "role": "system",
                    "content": (
                        "You are the memory-consolidation module of an agent. "
                        "For each ENTITY block, merge the old dossier with the "
                        "fresh facts into ONE concise factual paragraph (plain "
                        "text, no markdown) that keeps stable identities, ids "
                        "and relations. If a fresh fact contradicts the old "
                        "dossier, the fresh fact wins. Return JSON: "
                        '{"summaries": [{"entity": "<key>", "summary": "..."}]}'),
                }, {"role": "user", "content": "\n\n".join(prompt_blocks)}])
                calls += 1
                self.llm_spend(1)
                if isinstance(content, dict) and isinstance(content.get("summaries"), list):
                    by_key = {s.get("entity", ""): s.get("summary", "") for s in content["summaries"]}
                    for eid, facts_rows in chunk:
                        key = self._conn.execute(
                            "SELECT key FROM entities WHERE id = ?", (eid,)).fetchone()["key"]
                        summary = by_key.get(key)
                        if summary:
                            _upsert(eid, strip_anchor_noise(summary)[:2000])
                            report["dossiers_updated"] += 1
                            _consume([int(f["id"]) for f in facts_rows])

        # Extractive fallback (no LLM / LLM refused / budget exhausted).
        for eid, fids in target:
            if all(f in consumed for f in fids):
                continue
            surpr, confirms = _split(fids, summary_of.get(eid, ""))
            _consume(confirms)
            if not surpr:
                continue
            keep = surpr[:3]
            summary = " | ".join(f["text"][:200] for f in keep if f["text"])
            if summary:
                _upsert(eid, summary[:2000])
                report["dossiers_updated"] += 1
                _consume([int(f["id"]) for f in surpr])

        self._conn.commit()
        report["llm_calls"] = calls
        report["facts_consumed"] = len(consumed)
        self._cache.clear()
        return report

    def prune(self, retention_days: float = 365.0, *, now: Optional[float] = None) -> int:
        """Eviction lever: archive facts older than ``retention_days`` unless
        they are the most recent fact of an entity (keep one anchor per
        entity so graphs never go empty)."""
        now = now if now is not None else time.time()
        cutoff = now - retention_days * 86400.0
        # Only decaying kinds are auto-archived; durable facts (decisions,
        # goals, lessons...) die exclusively via supersedes/explicit removal.
        rows = self._conn.execute(
            "SELECT id FROM facts WHERE ts < ? AND archived = 0 "
            "AND kind IN ('episodic','episode')", (cutoff,)).fetchall()
        keep: set[int] = set()
        for r in self._conn.execute(
            "SELECT fe.entity_id, MAX(f.ts) mts FROM fact_entities fe "
            "JOIN facts f ON f.id = fe.fact_id GROUP BY fe.entity_id"
        ).fetchall():
            latest = self._conn.execute(
                "SELECT id FROM facts WHERE ts = ? AND archived = 0 "
                "ORDER BY id DESC LIMIT 1", (r["mts"],)).fetchone()
            if latest:
                keep.add(int(latest["id"]))
        n = 0
        for r in rows:
            if int(r["id"]) in keep:
                continue
            self._conn.execute(
                "UPDATE facts SET archived=1 WHERE id = ?", (int(r["id"]),))
            n += 1
        self._conn.commit()
        if n:
            self._cache.clear()
        return n

    def reset(self) -> None:
        self._conn.execute("DELETE FROM facts")
        self._conn.execute("DELETE FROM entities")
        self._conn.execute("DELETE FROM aliases")
        self._conn.execute("DELETE FROM fact_entities")
        self._conn.execute("DELETE FROM edges")
        self._conn.execute("DELETE FROM dossiers")
        self._conn.execute("DELETE FROM meta")
        self._conn.execute("DELETE FROM ops_log")
        self._conn.commit()
        self._cache.clear()

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))
        self._conn.commit()

    def backup_to(self, dest_path: str) -> None:
        """Online SQLite backup (safe under WAL) — full snapshot incl. schema."""
        dest = sqlite3.connect(dest_path)
        try:
            self._conn.backup(dest)
        finally:
            dest.close()

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------- internal

    def _resolve_entities(self, text: str, now: float) -> list[int]:
        """Canonicalize extracted keys into entity ids; merge alias pairs."""
        keys = extract_entities(text)
        ids: list[int] = []
        for key in keys:
            eid = self._entity_for_key(key, now)
            if eid is not None:
                ids.append(eid)
        for a, b in extract_alias_pairs(text):
            ea = self._entity_for_key(a, now)
            eb = self._entity_for_key(b, now)
            if ea is not None and eb is not None and ea != eb:
                self._merge_entities(ea, eb, now)
        # Re-resolve every original key AFTER merges (a dropped entity id must
        # never leak into fact_entities / edges).
        resolved: list[int] = []
        for key in keys:
            eid = self._entity_id_lookup(key)
            if eid is not None and eid not in resolved:
                resolved.append(eid)
        return resolved

    def _entity_for_key(self, key: str, now: float) -> Optional[int]:
        row = self._conn.execute(
            "SELECT id FROM entities WHERE key = ?", (key,)).fetchone()
        if row:
            return int(row["id"])
        alias_row = self._conn.execute(
            "SELECT entity_id FROM aliases WHERE alias = ?", (key,)).fetchone()
        if alias_row:
            return int(alias_row["entity_id"])
        cur = self._conn.execute(
            "INSERT INTO entities (key, label, kind, first_seen, last_seen, hits) "
            "VALUES (?, ?, 'token', ?, ?, 0)",
            (key, key, now, now))
        return int(cur.lastrowid)

    def _merge_entities(self, keep_id: int, drop_id: int, now: float) -> Optional[int]:
        """Fold entity ``drop_id`` into ``keep_id``: aliases move over, fact
        links and edges are re-pointed, stronger of the two remains the key.

        Entity merges are the single highest-risk, hardest-to-notice mistake
        this engine can make (a wrong alias match silently fuses two
        unrelated identities and every downstream recall inherits the error).
        Unlike every other mutation here, a merge used to be irreversible.
        Before touching any row, a full snapshot of everything about to be
        deleted or overwritten is written to ``merge_log`` (as JSON, not more
        schema); ``unmerge()`` replays it byte-for-byte. Returns the
        merge_log id so a caller can undo this exact merge later."""
        if keep_id == drop_id:
            return None
        drop = self._conn.execute(
            "SELECT key, label, kind, hits, first_seen, last_seen "
            "FROM entities WHERE id=?", (drop_id,)).fetchone()
        keep_before = self._conn.execute(
            "SELECT label, first_seen, last_seen, hits FROM entities WHERE id=?",
            (keep_id,)).fetchone()
        if drop is None or keep_before is None:
            return None
        drop_aliases = [r["alias"] for r in self._conn.execute(
            "SELECT alias FROM aliases WHERE entity_id = ?", (drop_id,)).fetchall()]
        alias_candidates = list(dict.fromkeys(drop_aliases + [drop["key"]]))
        alias_already_on_keep = {
            a: bool(self._conn.execute(
                "SELECT 1 FROM aliases WHERE entity_id = ? AND alias = ?",
                (keep_id, a)).fetchone())
            for a in alias_candidates
        }
        moved_fact_ids = [int(r["fact_id"]) for r in self._conn.execute(
            "SELECT fact_id FROM fact_entities WHERE entity_id = ?",
            (drop_id,)).fetchall()]
        fact_already_on_keep = [fid for fid in moved_fact_ids if self._conn.execute(
            "SELECT 1 FROM fact_entities WHERE fact_id = ? AND entity_id = ?",
            (fid, keep_id)).fetchone()]
        drop_edges = [dict(r) for r in self._conn.execute(
            "SELECT a, b, count, first_seen, last_seen, inhibited_since, stability "
            "FROM edges WHERE a = ? OR b = ?", (drop_id, drop_id)).fetchall()]
        keep_oth_before: dict[int, Optional[dict[str, Any]]] = {}
        for r in drop_edges:
            oth = r["a"] if r["b"] == drop_id else r["b"]
            if oth == keep_id:
                continue
            lo, hi = (keep_id, oth) if keep_id < oth else (oth, keep_id)
            row = self._conn.execute(
                "SELECT count, first_seen, last_seen, inhibited_since, stability "
                "FROM edges WHERE a = ? AND b = ?", (lo, hi)).fetchone()
            keep_oth_before[oth] = dict(row) if row else None

        snapshot = {
            "drop": dict(drop), "keep_before": dict(keep_before),
            "drop_aliases": drop_aliases,
            "alias_already_on_keep": alias_already_on_keep,
            "moved_fact_ids": moved_fact_ids,
            "fact_already_on_keep": fact_already_on_keep,
            "drop_edges": drop_edges,
            "keep_oth_before": {str(k): v for k, v in keep_oth_before.items()},
        }
        mid = int(self._conn.execute(
            "INSERT INTO merge_log (ts, keep_id, snapshot) VALUES (?, ?, ?)",
            (now, keep_id, json.dumps(snapshot, ensure_ascii=False))).lastrowid)

        for a in alias_candidates:
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?, ?)",
                (keep_id, a))
        # Re-point fact links (dedupe).
        for fid in moved_fact_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                (fid, keep_id))
        # Fold edge counts where both endpoints had edges to the dropped node.
        for r in drop_edges:
            oth = r["a"] if r["b"] == drop_id else r["b"]
            if oth == keep_id:
                continue
            self._conn.execute(
                "INSERT INTO edges (a, b, count, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(a, b) DO UPDATE SET "
                "count = edges.count + excluded.count, "
                "last_seen = MAX(edges.last_seen, excluded.last_seen)",
                (min(keep_id, oth), max(keep_id, oth),
                 r["count"], min(r["first_seen"], now), max(r["last_seen"], now)))
        self._conn.execute("DELETE FROM aliases WHERE entity_id = ?", (drop_id,))
        self._conn.execute("DELETE FROM fact_entities WHERE entity_id = ?", (drop_id,))
        self._conn.execute("DELETE FROM edges WHERE a = ? OR b = ?", (drop_id, drop_id))
        self._conn.execute("DELETE FROM entities WHERE id = ?", (drop_id,))
        self._conn.execute(
            "UPDATE entities SET label = COALESCE(NULLIF(label, key), ?), "
            "hits = hits + ?, first_seen = MIN(first_seen, ?), last_seen = ? "
            "WHERE id = ?",
            (drop["label"] or drop["key"], drop["hits"], drop["first_seen"], now, keep_id))
        return mid

    def unmerge(self, merge_id: int) -> Optional[dict[str, Any]]:
        """Undo one entity merge exactly, from its ``merge_log`` snapshot: the
        dropped entity is recreated (a NEW id — SQLite ids are never reused,
        but key/label/aliases/fact links/edges are restored verbatim) and
        ``keep_id`` reverts to its pre-merge row. Safe to call once per merge;
        a second call on the same ``merge_id`` is a no-op (``undone`` flag).
        Best-effort only against further writes to the SAME edges/links made
        strictly between the merge and this call (a real risk only if other
        merges touched the same pair in between) — the common "I linked the
        wrong two things, undo it" case is fully exact."""
        row = self._conn.execute(
            "SELECT keep_id, snapshot, undone FROM merge_log WHERE id = ?",
            (merge_id,)).fetchone()
        if row is None or int(row["undone"]):
            return None
        keep_id = int(row["keep_id"])
        snap = json.loads(row["snapshot"])
        drop = snap["drop"]
        existing = self._conn.execute(
            "SELECT id FROM entities WHERE key = ?", (drop["key"],)).fetchone()
        if existing is not None:
            # The dropped key was reclaimed by something else since the merge
            # (e.g. a brand-new unrelated entity with the same name) — refuse
            # rather than silently colliding two identities a second time.
            return {"ok": False, "reason": "drop_key_in_use", "key": drop["key"]}
        new_drop_id = int(self._conn.execute(
            "INSERT INTO entities (key, label, kind, first_seen, last_seen, hits) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (drop["key"], drop["label"], drop["kind"], drop["first_seen"],
             drop["last_seen"], drop["hits"])).lastrowid)
        # NOTE: drop["key"] must be removed from keep's aliases (the merge put
        # it there) but never re-added to new_drop_id's own aliases — an
        # entity is never its own alias, its key already IS drop["key"].
        alias_candidates = list(dict.fromkeys(snap["drop_aliases"] + [drop["key"]]))
        for a in alias_candidates:
            if not snap["alias_already_on_keep"].get(a):
                self._conn.execute(
                    "DELETE FROM aliases WHERE entity_id = ? AND alias = ?",
                    (keep_id, a))
            if a == drop["key"]:
                continue
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?, ?)",
                (new_drop_id, a))
        already_on_keep = set(snap["fact_already_on_keep"])
        for fid in snap["moved_fact_ids"]:
            self._conn.execute(
                "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                (fid, new_drop_id))
            if fid not in already_on_keep:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ? AND entity_id = ?",
                    (fid, keep_id))
        # Restore edges. keep_oth_before's keys enumerate every third-party
        # entity ("oth") the dropped entity had a folded edge with; the
        # direct keep<->drop edge (if any) is handled separately below since
        # it was never folded (it was simply deleted by the merge).
        for oth_str, before in snap["keep_oth_before"].items():
            oth = int(oth_str)
            lo, hi = (keep_id, oth) if keep_id < oth else (oth, keep_id)
            if before is None:
                self._conn.execute(
                    "DELETE FROM edges WHERE a = ? AND b = ?", (lo, hi))
            else:
                self._conn.execute(
                    "UPDATE edges SET count = ?, first_seen = ?, last_seen = ?, "
                    "inhibited_since = ?, stability = ? WHERE a = ? AND b = ?",
                    (before["count"], before["first_seen"], before["last_seen"],
                     before["inhibited_since"], before["stability"], lo, hi))
            # Recreate old_drop_id <-> oth with its original snapshot values.
            orig = next((e for e in snap["drop_edges"]
                        if (e["a"] == oth or e["b"] == oth) and oth != keep_id), None)
            if orig is not None:
                lo2, hi2 = (new_drop_id, oth) if new_drop_id < oth else (oth, new_drop_id)
                self._conn.execute(
                    "INSERT OR REPLACE INTO edges "
                    "(a, b, count, first_seen, last_seen, inhibited_since, stability) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (lo2, hi2, orig["count"], orig["first_seen"], orig["last_seen"],
                     orig["inhibited_since"], orig["stability"]))
        # The direct keep<->drop edge (if any) was skipped from folding above
        # and simply deleted by the merge — recreate it verbatim.
        direct = next((e for e in snap["drop_edges"]
                       if e["a"] == keep_id or e["b"] == keep_id), None)
        if direct is not None:
            lo3, hi3 = (new_drop_id, keep_id) if new_drop_id < keep_id else (keep_id, new_drop_id)
            self._conn.execute(
                "INSERT OR REPLACE INTO edges "
                "(a, b, count, first_seen, last_seen, inhibited_since, stability) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (lo3, hi3, direct["count"], direct["first_seen"], direct["last_seen"],
                 direct["inhibited_since"], direct["stability"]))
        kb = snap["keep_before"]
        self._conn.execute(
            "UPDATE entities SET label = ?, first_seen = ?, last_seen = ?, "
            "hits = ? WHERE id = ?",
            (kb["label"], kb["first_seen"], kb["last_seen"], kb["hits"], keep_id))
        self._conn.execute(
            "UPDATE merge_log SET undone = 1 WHERE id = ?", (merge_id,))
        self._conn.commit()
        self._cache.clear()
        return {"ok": True, "restored_key": drop["key"], "new_entity_id": new_drop_id}

    def _touch_edge(self, a: int, b: int, now: float, delta: float = 1.0) -> None:
        """Hebbian reinforcement + myelination (stability grows with every
        co-occurrence; stable edges are exempt from sleep downscaling)."""
        lo, hi = (a, b) if a < b else (b, a)
        self._conn.execute(
            "INSERT INTO edges (a, b, count, first_seen, last_seen, "
            "inhibited_since, stability) "
            "VALUES (?, ?, ?, ?, ?, NULL, 0.5) ON CONFLICT(a, b) DO UPDATE SET "
            "count = edges.count + ?, last_seen = ?, "
            "stability = MIN(5.0, edges.stability + 0.3)",
            (lo, hi, delta, now, now, delta, now))

    def _rehearse(self, fact_id: int, now: float) -> None:
        """Reconsolidation on retrieval (§22.1): recalling a fact increments
        its retrieval counter and slightly myelinates its association edges."""
        self._conn.execute(
            "UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE id = ?",
            (fact_id,))
        rows = self._conn.execute(
            "SELECT entity_id FROM fact_entities WHERE fact_id = ?",
            (fact_id,)).fetchall()
        eids = [int(r["entity_id"]) for r in rows]
        for i, a in enumerate(eids):
            for b in eids[i + 1:]:
                lo, hi = (a, b) if a < b else (b, a)
                self._conn.execute(
                    "UPDATE edges SET stability = MIN(5.0, stability + 0.1), "
                    "last_seen = ? WHERE a = ? AND b = ?", (now, lo, hi))
        self._conn.commit()

    def _edge_strength(self, a: int, b: int, now: float) -> float:
        row = self._conn.execute(
            "SELECT count, last_seen, inhibited_since FROM edges WHERE a = ? AND b = ?",
            (min(a, b), max(a, b))).fetchone()
        if row is None:
            return 0.0
        hours = max(0.0, (now - row["last_seen"]) / 3600.0)
        decay = math.exp(-hours / self.edge_half_life_hours)
        w = float(row["count"]) * decay
        if row["inhibited_since"] is not None:
            w *= 0.1
        return w

    # Graph-hop breadth cap (measured fix): a hub entity (a project name that
    # recurs across a large fraction of all facts) can have hundreds+ of
    # neighbors. Expanding ALL of them, then expanding hop-2 from ALL of
    # those, was an unbounded breadth-first walk that turned "recall facts
    # near this entity" into "touch nearly the whole entity graph" -- on a
    # synthetic 50k-fact / 212-entity store with one moderately hub-like
    # entity, this alone contributed to a 62s single search() call (profiled;
    # see _score_facts_by_entities). Real usage is typically sparse (most
    # entities have a handful of neighbors, well under this cap, so nothing
    # changes for them) -- the cap only bites for genuine hubs, which is
    # exactly where unbounded expansion was pathological.
    _HOP_BREADTH_CAP = 40

    def _resolve_query_entities(self, query: str, now: float) -> dict[int, float]:
        """Query entity ids with graph-expanded scores (2-hop, decaying)."""
        keys = extract_entities(query)
        if not keys:
            return {}
        ids: dict[int, float] = {}
        direct: set[int] = set()
        cap = self._HOP_BREADTH_CAP
        for key in keys:
            eid = self._entity_id_lookup(key)
            if eid is None:
                continue
            ids[eid] = ids.get(eid, 0.0) + 2.0
            direct.add(eid)
            # Hop 1: alias expansion (same canonical id is automatic) +
            # strongest neighbors.  Inhibited edges (superseded choices) are
            # heavily downweighted — the rejected path stops resurfacing.
            # Only the top-`cap` neighbors by decayed strength survive: a
            # sparse entity is unaffected, a hub entity is bounded.
            neighbors: list[tuple[int, float]] = []
            for r in self._conn.execute(
                "SELECT a, b, count, last_seen, inhibited_since FROM edges "
                "WHERE a = ? OR b = ?", (eid, eid)).fetchall():
                oth = r["a"] if r["b"] == eid else r["b"]
                hours = max(0.0, (now - r["last_seen"]) / 3600.0)
                strength = float(r["count"]) * math.exp(
                    -hours / self.edge_half_life_hours)
                if r["inhibited_since"] is not None:
                    strength *= 0.1
                neighbors.append((oth, strength))
            neighbors.sort(key=lambda x: x[1], reverse=True)
            for oth, strength in neighbors[:cap]:
                ids[oth] = max(ids.get(oth, 0.0), strength)
        # Hop 2 (cheap, capped): expand only from the strongest hop-1
        # entities (bounded above), not every entity gathered so far -- a
        # hub's already-capped hop-1 set must not re-explode into hundreds of
        # hop-2 queries.
        hop1_ranked = sorted(
            ((eid, sc) for eid, sc in ids.items() if eid not in direct),
            key=lambda kv: kv[1], reverse=True)[:cap]
        hop2: list[tuple[int, float]] = []
        for eid, _ in hop1_ranked:
            for r in self._conn.execute(
                "SELECT a, b, count, last_seen, inhibited_since FROM edges "
                "WHERE (a = ? OR b = ?) AND count >= 2", (eid, eid)).fetchall():
                oth = r["a"] if r["b"] == eid else r["b"]
                if oth in ids:
                    continue
                hours = max(0.0, (now - r["last_seen"]) / 3600.0)
                w = float(r["count"]) * 0.3 * math.exp(
                    -hours / self.edge_half_life_hours)
                if r["inhibited_since"] is not None:
                    w *= 0.1
                hop2.append((oth, w))
        for eid, score in hop2:
            ids[eid] = max(ids.get(eid, 0.0), score)
        # Final global cap: per-source fan-out limits above still let a
        # densely connected neighborhood re-saturate the overall set (many
        # capped sources, each contributing a different slice, can union back
        # up close to the whole graph). Directly-matched query entities are
        # always kept; graph-discovered ones are truncated to the strongest
        # 2*cap regardless of how many distinct hop-1 sources contributed
        # them -- this is what actually bounds _score_facts_by_entities'
        # downstream join to a fixed-size entity set.
        if len(ids) > cap * 2:
            kept = {eid: sc for eid, sc in ids.items() if eid in direct}
            others = sorted(
                ((eid, sc) for eid, sc in ids.items() if eid not in direct),
                key=lambda kv: kv[1], reverse=True)[: max(0, cap * 2 - len(kept))]
            kept.update(dict(others))
            ids = kept
        return ids

    def _entity_id_lookup(self, key: str) -> Optional[int]:
        row = self._conn.execute(
            "SELECT id FROM entities WHERE key = ?", (key,)).fetchone()
        if row:
            return int(row["id"])
        row = self._conn.execute(
            "SELECT entity_id FROM aliases WHERE alias = ?", (key,)).fetchone()
        return int(row["entity_id"]) if row else None

    def _score_facts_by_entities(
        self, eids: list[int], now: float, *, as_of: Optional[float] = None,
        content_tokens: Optional[list[str]] = None,
    ) -> tuple[dict[int, float], set[int]]:
        scored: dict[int, float] = {}
        seen: set[int] = set()
        placeholders = ",".join("?" * len(eids))
        params: list[Any] = list(eids)
        params.append(as_of if as_of is not None else now)
        ts_cond = " AND f.ts <= ?" if as_of is not None else ""
        if as_of is not None:
            params.append(as_of)
        rows = self._conn.execute(
            f"SELECT fe.fact_id, fe.entity_id, f.ts, f.importance, f.kind, "
            f"f.confidence, f.text "
            f"FROM fact_entities fe JOIN facts f ON f.id = fe.fact_id "
            f"WHERE fe.entity_id IN ({placeholders}) AND f.archived = 0 "
            f"AND (f.active_until IS NULL OR f.active_until > ?) "
            f"{ts_cond}",
            params).fetchall()
        # _entity_query_score(eid) depends ONLY on eid, not on the fact row --
        # but eids is small (the query's resolved entities, typically single
        # digits even after 2-hop expansion) while `rows` is one row per
        # (fact, entity) pair, which scales with how many facts mention that
        # entity. Recomputing it per row instead of per eid was a measured,
        # severe scaling bug: on a synthetic 50k-fact store with a moderately
        # hub-like entity (a project name mentioned across ~1/4 of all
        # facts), one search() call issued 99,773 redundant score queries
        # (200k+ total SQL round-trips) and took 62 SECONDS. Caching per eid
        # bounds those calls to at most len(eids) — the same search() dropped
        # to sub-100ms after this fix (verified, not assumed).
        _score_cache: dict[int, float] = {}
        # Content-relevance bonus (measured fix, external LoCoMo benchmark,
        # 2026-09): entity-path candidates used to rank PURELY by entity
        # presence x recency x importance, never checking whether the fact's
        # own text relates to anything else in the question. For a "hub"
        # entity -- the most common real case being a person's name mentioned
        # in most turns of a long conversation -- this buried the one fact
        # that actually answers "what did X research?" under dozens of
        # unrelated "Wow, X!" turns that merely name-drop X more recently.
        # Measured: on LoCoMo (snap-research/locomo), evidence-hit@8 was 3.0%
        # before this bonus existed. A small, cheap per-matched-token bonus
        # (substring check against the already-fetched fact text, no extra
        # query) lets real lexical relevance compete with recency instead of
        # being structurally invisible to entity-path scoring.
        _tokens = [t for t in (content_tokens or []) if len(t) >= 3][:6]
        _fact_text: dict[int, str] = {}
        # Per-fact weights collected first, THEN combined with a saturating
        # aggregate below -- NOT a running linear sum. A fact that happens to
        # co-mention several query-adjacent entities (some of them incidental
        # noise) used to accumulate one additive term per entity, so a
        # 5-entity fact could outscore a precisely-matched 1-entity fact by
        # sheer count, independent of relevance (measured, LoCoMo 2026-09:
        # the gold-evidence fact for "What career path has Caroline decided
        # to pursue?" ranked #271 of 419 candidates, well below several
        # facts that merely co-mentioned more entities).
        _fact_weights: dict[int, list[float]] = {}
        for r in rows:
            fid = int(r["fact_id"])
            eid = int(r["entity_id"])
            seen.add(fid)
            kind = r["kind"] or "episodic"
            if kind in DECAYING_KINDS:
                hours = max(0.0, (now - r["ts"]) / 3600.0)
                rec = math.exp(-hours / self.recency_half_life_hours)
            else:
                rec = 1.0
            if eid not in _score_cache:
                _score_cache[eid] = float(self._entity_query_score(eid, now))
            ent_score = _score_cache[eid]
            w = (ent_score * float(r["importance"])
                * float(r["confidence"] or 1.0) * (0.5 + 0.5 * rec))
            _fact_weights.setdefault(fid, []).append(w)
            if _tokens and fid not in _fact_text:
                _fact_text[fid] = (r["text"] or "").lower()
        # Entity-match saturation (see _entity_match_saturation docstring
        # above for the full, honest story on this fix and its k1 constant).
        for fid, weights in _fact_weights.items():
            scored[fid] = (sum(weights) / len(weights)) * _entity_match_saturation(len(weights))
        if _tokens:
            # Applied AFTER the full per-entity sum, as a MULTIPLIER rather
            # than a fixed additive amount: entity base scores are unbounded
            # (they scale with edge count / dataset size via
            # _entity_query_score, and with how many query entities a fact
            # happens to mention), so a fixed "+1 per matched word" is
            # meaningless noise for a hub fact scoring 50+ and everything for
            # one scoring 2 -- a fixed bonus can never reliably compete at an
            # unknown, unbounded scale. A proportional boost does, regardless
            # of how large the base score already is.
            for fid, txt_l in _fact_text.items():
                matched = sum(1 for t in _tokens if t in txt_l)
                if matched:
                    scored[fid] = scored.get(fid, 0.0) * (1.0 + 0.6 * matched)
        return scored, seen

    def _entity_query_score(self, eid: int, now: float) -> float:
        """Score contribution of an entity found inside the result fact.
        Entities directly named in the query weigh 2.0; graph-discovered
        neighbors use their (decayed) edge strength."""
        # Cheap approximation: aggregate strongest edge into any other entity.
        row = self._conn.execute(
            "SELECT MAX(count) c FROM edges WHERE a = ? OR b = ?", (eid, eid)).fetchone()
        if row is None or row["c"] is None:
            return 2.0
        return max(2.0, float(row["c"]) * 0.25)

    def _matched_entity_keys(self, fact_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT e.key FROM fact_entities fe JOIN entities e ON e.id = fe.entity_id "
            "WHERE fe.fact_id = ?", (fact_id,)).fetchall()
        return [r["key"] for r in rows][:6]

    def _dossiers_for_entities(self, eids: list[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for eid in eids:
            row = self._conn.execute(
                "SELECT d.summary, d.updated_at, e.key, e.label FROM dossiers d "
                "JOIN entities e ON e.id = d.entity_id WHERE d.entity_id = ?",
                (eid,)).fetchone()
            if row:
                out.append({
                    "text": f"[dossier:{row['key']}] {row['summary']}",
                    "source": "dossier",
                    "session_id": "",
                    "ts": row["updated_at"],
                    "importance": 3.0,
                    "score": 5.0,
                    "via": [row["key"]],
                })
        return out


def _norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", (s or "").strip().lower())


def _is_trivial(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or t.startswith("/"):
        return True
    return t in {
        "yes", "no", "ok", "okay", "sure", "thanks", "thank you", "y", "n",
        "hi", "hey", "hello", "continue", "go ahead", "got it", "done",
        "спасибо", "ок", "окей", "да", "нет", "привет", "понятно", "ага",
    }


_NOTIFICATION_MARKERS = (
    "[important:",
    "[important]",
    "background process proc_",
    "background task proc_",
    "команда завершена",
    "процесс завершён",
)


def _is_notification(text: str) -> bool:
    """System/background boilerplate (process-done notices surfaced as user
    turns) is not user intent — keep the assistant's answer, drop the notice."""
    head = (text or "").strip()[:300].lower()
    return any(marker in head for marker in _NOTIFICATION_MARKERS)


def _thread_safe(names: list[str]) -> None:
    """Serialize every public DB op on the shared connection (sync_turn and
    prefetch run on different threads inside Hermes).  RLock is reentrant, so
    wrapped public methods may freely call other wrapped methods."""
    def _deco(fn):
        @functools.wraps(fn)
        def _wrapper(self, *args, **kwargs):
            with self._lock:
                return fn(self, *args, **kwargs)
        return _wrapper
    for _n in names:
        setattr(NeuroMatrixStore, _n, _deco(getattr(NeuroMatrixStore, _n)))


_thread_safe([
    "remember", "add_turn", "link", "search", "entity", "stats",
    "consolidate", "prune", "reset", "get_meta", "set_meta", "close",
    "decide", "supersede", "decisions", "remember_goal",
    "remember_constraint", "feedback", "contradictions", "rollup_session",
    "store_artifact", "artifact_get", "artifact_find", "artifact_delete",
    "llm_budget_remaining", "llm_spend", "plan_foresight", "foresights_due",
    "export_markdown", "ops_view", "attach_evidence", "explain_fact",
    "skill_propose", "export_profile_card", "policy_report", "sweep_decisions",
])

__all__ = ["NeuroMatrixStore", "extract_entities", "extract_alias_pairs"]
