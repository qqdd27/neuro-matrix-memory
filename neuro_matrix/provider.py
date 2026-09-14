"""NeuroMatrix — local-first temporal knowledge-graph memory provider for Hermes Agent.

Implements ``agent.memory_provider.MemoryProvider`` and registers itself as the
``neuromatrix`` provider (``memory.provider: neuromatrix``).

Research-informed design (2025-2026 state of the art, see README):
  * temporal fact graph (Zep/Graphiti lineage) with provenance & recency,
  * entity aliases / cross-session identity (Mem0-style write-path linking),
  * weighted association edges with exponential decay (SCM / Hindsight levers),
  * offline consolidation "sleep" at session end (Anthropic Dreaming lineage)
    that merges fresh facts into per-entity dossiers — via an OPTIONAL
    OpenAI-compatible LLM (DeepSeek etc.), or pure extractive when no key is
    configured (zero cost, fully private, works offline).
  * hybrid retrieval: alias + graph expansion over FTS5 lexical recall.

Storage is SQLite + FTS5 under ``$HERMES_HOME/neuromatrix.db`` — nothing leaves
the machine unless an LLM key for consolidation is explicitly provided.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # running inside Hermes Agent
    from agent.memory_provider import MemoryProvider, RecallStatus
    from tools.registry import tool_error
    from utils import is_truthy_value

    _IN_HERMES = True
except Exception:  # pragma: no cover - standalone dev / unit tests w/o hermes tree
    _IN_HERMES = False

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        provider_label: str
        count: int
        glyph: str = "\U0001f9e0"

    class MemoryProvider:  # type: ignore[no-redef]
        """Minimal stand-in so the provider imports & can be smoke-tested
        outside a Hermes checkout (all hooks no-op by default)."""

        @property
        def name(self) -> str:
            return "neuromatrix"

        def is_available(self) -> bool:
            return True

        def initialize(self, session_id: str, **kwargs) -> None:  # noqa: ARG002
            pass

        def get_config_schema(self) -> list:  # noqa: D401
            return []

        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:  # noqa: ARG002
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        def prefetch(self, query: str, *, session_id: str = "") -> str:  # noqa: ARG002
            return ""

        def recall_status(self) -> Optional[RecallStatus]:
            return None

        def system_prompt_block(self) -> str:
            return ""

        def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:  # noqa: ARG002
            pass

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None:  # noqa: ARG002
            pass

        def on_session_switch(self, new_session_id: str, **kwargs) -> None:  # noqa: ARG002
            pass

        def on_delegation(self, task: str, result: str, **kwargs) -> None:  # noqa: ARG002
            pass

        def backup_paths(self) -> list:
            return []

        def unavailable_reason(self) -> str:
            return ""

        def on_session_end(self, messages: List[Dict[str, Any]]) -> None:  # noqa: ARG002
            pass

        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:  # noqa: ARG002
            return ""

        def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:  # noqa: ARG002
            pass

        def shutdown(self) -> None:
            pass

    def tool_error(msg: str) -> str:  # noqa: ARG001
        return json.dumps({"error": msg})

    def is_truthy_value(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        return str(v or "").strip().lower() in {"1", "true", "yes", "on", "y"}


from .config_schema import LEGACY_CONFIG_SCHEMA as CONFIG_SCHEMA  # noqa: E402
from .llm import (  # noqa: E402
    DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient, build_llm_client,
    resolve_hermes_llm_config,
)
from .store import NeuroMatrixStore  # noqa: E402

NAME = "neuromatrix"
DISPLAY = "NeuroMatrix Memory"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

NM_TOOL_SCHEMA = {
    "name": "neuromatrix",
    "description": (
        "Cross-session associative memory (entity graph + dossiers). Use it when the answer "
        "may depend on past conversations: stable ids (id_777, 0x...), tokens, projects, user "
        "preferences, decisions, or any entity mentioned before. "
        "\n\nACTIONS:\n"
        "• search — recall facts + entity dossiers relevant to a topic/id (hybrid: alias + graph + text).\n"
        "• remember — store one durable fact verbatim (statement with entities).\n"
        "• probe — full profile of one entity: aliases, relation strength, consolidated dossier.\n"
        "• link — declare that two ids/names refer to the SAME entity (merge + strong edge).\n"
        "• consolidate — run the memory 'sleep' cycle now (merge fresh facts into dossiers).\n"
        "• stats — memory size / entity counts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "remember", "probe", "link", "decide",
                         "supersede", "decisions", "remember_goal",
                         "remember_constraint", "feedback", "contradict",
                         "artifact_store", "artifact_get", "artifact_find",
                         "artifact_delete", "foresight", "reminders", "ask",
                         "cite", "explain", "skill",
                         "persona", "policy",
                         "ops", "budget",
                         "consolidate", "stats",
                         "deadend", "deadends",
                         "capabilities", "ingest",
                         "statuses", "resolve",
                         "invent", "distill"],
            },
            "query": {"type": "string", "description": "Search query (action=search)."},
            "content": {"type": "string", "description": "Fact statement (action=remember)."},
            "entity": {"type": "string", "description": "Entity key/alias (action=probe)."},
            "entity_a": {"type": "string", "description": "First identity (action=link)."},
            "entity_b": {"type": "string", "description": "Second identity (action=link)."},
            "concept": {"type": "string", "description": "Decision concept/scope, e.g. 'database_stack' (actions decide/decisions)."},
            "choice": {"type": "string", "description": "Chosen option, e.g. 'Firebase' (actions decide/supersede)."},
            "decision_id": {"type": "integer", "description": "Decision fact id to retire (action=supersede)."},
            "criteria": {"type": "array", "items": {"type": "string"},
                         "description": "Structured decision criteria tags (decide/supersede)."},
            "scope_tags": {"type": "array", "items": {"type": "string"},
                           "description": "Optional context tags for scope-matching lessons."},
            "reason": {"type": "string", "description": "Why this choice (decide/supersede)."},
            "verdict": {"type": "string", "enum": ["helpful", "unhelpful"],
                        "description": "Reinforcement verdict (action=feedback)."},
            "fact_id": {"type": "integer", "description": "Target fact (action=feedback)."},
            "name": {"type": "string", "description": "Artifact name (artifact_store/artifact_get)."},
            "topic": {"type": "string", "description": "One-line artifact description."},
            "kind": {"type": "string", "description": "Artifact kind (document/log/export/...)."},
            "text": {"type": "string", "description": "Inline payload (artifact_store; XOR file_path)."},
            "file_path": {"type": "string", "description": "Payload file (artifact_store; XOR text)."},
            "artifact_id": {"type": "integer", "description": "Artifact id (artifact_get/artifact_delete)."},
            "trigger_at": {"type": "string", "description": "ISO time or epoch seconds when a foresight becomes due."},
            "max_chars": {"type": "integer", "description": "Snippet limit (artifact_get, default 4000)."},
            "limit": {"type": "integer", "description": "Max results (default 6)."},
            "scope": {"type": "string", "enum": ["private", "shared"],
                      "description": "Storage scope (default 'private'). 'shared' writes/reads the "
                                     "workspace pool (workspace_db) shared across profiles/agents."},
            "rerank": {"type": "boolean",
                       "description": "LLM rerank of heuristic results (action=search; spends 1 LLM call)."},
            "evidence_ids": {"type": "array", "items": {"type": "integer"},
                             "description": "Source fact ids backing a claim (action=cite)."},
            "out_dir": {"type": "string",
                        "description": "Output directory for the skill draft (action=skill)."},
        },
        "required": ["action"],
    },
}


def _load_plugin_config() -> dict:
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        return cfg_get(load_config_readonly(), "plugins", NAME, default={}) or {}
    except Exception:
        return {}


class NeuromatrixMemoryProvider(MemoryProvider):
    """Temporal knowledge-graph memory provider (local-first)."""

    pre_compress_checkpoint_api_version = 1  # best-effort evidence archive

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_plugin_config()
        self._store: Optional[NeuroMatrixStore] = None
        self._shared: Optional[NeuroMatrixStore] = None
        self._session_id = ""
        self._recall: Optional[RecallStatus] = None
        self._active = False

    # ------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return NAME

    def _start_structure_backfill(self) -> None:
        """Give pre-existing facts a structural form, off the critical path.

        Facts written before the structural channel existed have no propositions,
        so slot search cannot see them — measured on a real profile: 65 facts,
        0 propositions, every slot query returning nothing.  Bounded batches in a
        daemon thread keep provider start-up fast and never block the first turn;
        repeating is safe because only facts without propositions are touched.
        """
        try:
            import threading

            def _run() -> None:
                try:
                    n = self._store.reindex_propositions(limit=5000)
                    if n:
                        logger.info("neuromatrix: structured %d pre-existing facts", n)
                except Exception as e:  # noqa: BLE001 - never break the agent
                    logger.debug("neuromatrix structure backfill failed: %s", e)
                try:
                    # Same reasoning for the lemma index: a channel that only
                    # applies to facts written after an update is invisible on
                    # the data a user already has.
                    m = self._store.rebuild_lemmas(limit=5000)
                    if m:
                        logger.info("neuromatrix: lemmatised %d pre-existing facts", m)
                except Exception as e:  # noqa: BLE001 - never break the agent
                    logger.debug("neuromatrix lemma backfill failed: %s", e)
                try:
                    # Retroactive cleanup: dead ends invented by the old auto-detector
                    # out of the assistant's own prose.  They arrive with the highest
                    # recall score, so leaving them costs real context on every turn.
                    g = self._store.archive_garbage_deadends()
                    if g:
                        logger.info("neuromatrix: archived %d junk dead ends", g)
                except Exception as e:  # noqa: BLE001 - never break the agent
                    logger.debug("neuromatrix dead-end cleanup failed: %s", e)
                try:
                    # Same retroactive cleanup for capabilities captured out of the
                    # assistant's prose.  They sit in the cacheable prefix, so a
                    # fragment there is repeated on every turn until removed.
                    c = self._store.archive_garbage_capabilities()
                    if c:
                        logger.info("neuromatrix: archived %d junk capabilities", c)
                except Exception as e:  # noqa: BLE001 - never break the agent
                    logger.debug("neuromatrix capability cleanup failed: %s", e)

            threading.Thread(target=_run, name="neuromatrix-structure-backfill",
                             daemon=True).start()
        except Exception as e:  # noqa: BLE001
            logger.debug("neuromatrix: could not start structure backfill: %s", e)

    def is_available(self) -> bool:
        return True  # SQLite always available

    # --------------------------------------------------------------- config

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return CONFIG_SCHEMA

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret values to config.yaml under plugins.neuromatrix."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            from hermes_cli.config import read_user_config_raw
            existing = read_user_config_raw(config_path)
            existing.setdefault("plugins", {})[NAME] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.debug("neuromatrix save_config failed: %s", e)

    # ------------------------------------------------------------ lifecycle

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = str(kwargs.get("hermes_home") or os.environ.get("HERMES_HOME", ""))
        db_path = str(self._config.get("db_path") or "$HERMES_HOME/neuromatrix.db")
        db_path = db_path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        if not db_path or db_path.startswith("$"):
            db_path = os.path.join(hermes_home or ".", "neuromatrix.db")
        try:
            self._store = NeuroMatrixStore(db_path)
        except Exception as e:  # provider must degrade, never crash the agent
            logger.error("neuromatrix: failed to open %s: %s", db_path, e)
            self._store = None
            return
        if self._store is not None:
            self._store.llm_daily_budget = max(0, int(self._config.get("llm_daily_budget", 20) or 20))
            self._store.auto_decide_enabled = is_truthy_value(
                self._config.get("auto_decide", "true"))
            self._store.outcome_enabled = is_truthy_value(
                self._config.get("auto_decide", "true"))
            self._store.capability_enabled = is_truthy_value(
                self._config.get("auto_decide", "true"))
            self._start_structure_backfill()
            adir = str(self._config.get("artifacts_dir") or "").replace(
                "$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
            if adir:
                self._store.artifacts_dir = adir
        self._max_recall_chars = int(self._config.get("max_recall_chars", 1500) or 1500)
        self._min_recall_score = float(self._config.get("min_recall_score", 0.0) or 0.0)

        llm_enabled = is_truthy_value(self._config.get("llm_enabled", "true"))
        api_key = os.environ.get("NEUROMATRIX_API_KEY", "")
        client = None
        hers: dict = {}
        if llm_enabled:
            if not api_key:
                # No dedicated key: reuse the LLM provider Hermes already uses.
                hers = resolve_hermes_llm_config() or {}
                if hers:
                    client = build_llm_client(
                        hers["api_key"], provider=hers.get("provider", ""),
                        base_url=hers["base_url"], model=hers["model"],
                    )
                    logger.info("neuromatrix LLM: reusing Hermes provider %s (%s)",
                                hers.get("provider") or "?", hers["model"])
            if api_key:
                # Dedicated key: empty base/model/provider fields follow
                # Hermes' provider (a provider mismatch here would silently
                # send an Anthropic key to the OpenAI-compatible wire format
                # and vice versa — both just 401/404 and degrade to
                # extractive mode with no visible error, so this must route
                # through the same provider-aware factory as above).
                hers = hers or resolve_hermes_llm_config() or {}
                client = build_llm_client(
                    api_key,
                    provider=str(self._config.get("llm_provider") or hers.get("provider") or ""),
                    base_url=str(self._config.get("llm_base_url") or hers.get("base_url") or ""),
                    model=str(self._config.get("llm_model") or hers.get("model") or ""),
                )
            if client is not None and self._store is not None:
                self._store.llm = client
        # Optional dense retrieval: a local embedding model makes paraphrase-
        # reachable facts findable and is fused into recall by rank (RRF), so
        # it can only re-order candidates, never lose them.  Off unless asked
        # for; on any failure the embedder stayes None and recall is unchanged.
        embedder = None
        if is_truthy_value(self._config.get("embeddings_enabled")):
            try:
                from .embeddings import DEFAULT_BASE_URL as _EMB_URL, OllamaEmbedder
                base = str(self._config.get("embeddings_base_url") or "").strip()
                if not base:
                    base = str(self._config.get("llm_base_url") or "").strip()
                if ":11434" not in base:
                    # A hosted LLM endpoint (DeepSeek etc.) serves no embeddings;
                    # fall back to the local Ollama the embeddings feature needs.
                    base = _EMB_URL
                emb = OllamaEmbedder(
                    model=str(self._config.get("embeddings_model") or ""), base_url=base)
                if emb.available():
                    embedder = emb
                    if self._store is not None:
                        self._store.embedder = emb
                    logger.info("neuromatrix embeddings: %s via %s", emb.model, emb.base_url)
                else:
                    logger.info("neuromatrix embeddings requested but no model "
                                "available at %s (pull one, e.g. bge-m3)", base)
            except Exception as e:  # embeddings must never break the provider
                logger.debug("neuromatrix embeddings init failed: %s", e)
        ws = str(self._config.get("workspace_db") or "").replace(
            "$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        if ws and os.path.abspath(ws) != os.path.abspath(db_path):
            try:
                self._shared = NeuroMatrixStore(ws)
                self._shared.llm_daily_budget = self._store.llm_daily_budget
                if client is not None:
                    # Reuse the same client object (LLMClient or
                    # AnthropicLLMClient, whichever build_llm_client picked
                    # above) -- both are effectively stateless (budget
                    # tracking lives on the store, not the client), so there
                    # is no reason to reconstruct one and every reason not
                    # to: a hardcoded LLMClient(...) here would silently
                    # rebuild an Anthropic-configured client with the wrong
                    # wire format.
                    self._shared.llm = client
                if embedder is not None:
                    self._shared.embedder = embedder
                logger.info("neuromatrix shared workspace ready: %s", ws)
            except Exception as e:  # shared pool must never break the provider
                logger.error("neuromatrix workspace %s failed: %s", ws, e)
                self._shared = None
        self._retention_days = float(self._config.get("retention_days", 365) or 365)
        self._auto_consolidate = is_truthy_value(self._config.get("auto_consolidate", "true"))
        # The llm_rerank setting finally does something: it decides whether
        # search() may spend an LLM call reordering candidates when the caller
        # did not ask either way.  A local model makes this cheap; a hosted one
        # makes it seconds per recall, which is why it is a setting and not a
        # hardcoded default.
        self._store.llm_rerank_default = is_truthy_value(
            self._config.get("llm_rerank", "true")) and bool(self._store.llm)
        if self._shared is not None:
            self._shared.llm_rerank_default = self._store.llm_rerank_default
        ctx = kwargs.get("agent_context")
        self._active = ctx in (None, "", "primary", "flush")
        logger.info("neuromatrix ready: %s (active=%s, llm=%s)",
                    db_path, self._active, bool(self._store.llm))

    def system_prompt_block(self) -> str:
        if self._store is None:
            return ""
        try:
            s = self._store.stats()
        except Exception:
            return ""
        if s["facts"] == 0:
            body = ("Active (empty). Store durable facts the user would expect you to remember "
                    "via neuromatrix(action='remember').\n")
        else:
            body = (f"Active. {s['facts']} facts, {s['entities']} entities "
                    f"({s['aliases']} aliases), {s['dossiers']} dossiers. "
                    f"Recall is injected automatically each turn.\n")
        body += ("Cross-session associative recall: neuromatrix(action='search', query=...) — "
                 "stable ids (id_777, 0x...), tokens, projects, preferences, decisions; "
                 "'probe' for an entity dossier; 'link' to merge two names for one entity.")
        return f"# {DISPLAY}\n{body}"

    # ------------------------------------------------------------ recall path

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Background recall for the upcoming turn; MUST be fast (store caches).
        Respects the recall budget: min-score gate + max_recall_chars cap.
        Surfaces due foresights once per day (v0.3 cron-free bridge)."""
        self._recall = None
        if self._store is None:
            return ""
        parts: list[str] = []
        try:
            today = time.strftime("%Y%m%d", time.gmtime())
            if self._store.get_meta("nm:remind_day") != today:
                self._store.set_meta("nm:remind_day", today)
                for d in self._store.foresights_due(limit=3):
                    parts.append(f"- ⏰ {d['text'][:180]}")
            # Daily 'sleep' bridge: sessions that end without a clean shutdown
            # would leave unconsolidated facts behind forever — catch up once a
            # day on the first prefetch (budget-gated, 2 LLM calls max).
            if self._auto_consolidate and self._store.get_meta("nm:cons_day") != today:
                self._store.set_meta("nm:cons_day", today)
                uncons = self._store._conn.execute(
                    "SELECT COUNT(*) c FROM facts WHERE archived = 0 AND consolidated = 0"
                ).fetchone()["c"] if self._store is not None else 0
                if uncons >= 20 and self._store.llm_budget_remaining() > 0:
                    rep = self._store.consolidate(max_llm_calls=2)
                    if rep.get("dossiers_updated") or rep.get("lessons"):
                        logger.info("neuromatrix daily catch-up consolidate: %s", rep)
                    self._store.sweep_decisions()
        except Exception as e:
            logger.debug("neuromatrix reminders failed: %s", e)
        if query:
            results: list[dict[str, Any]] = []
            try:
                results = self._store.search(query, limit=6)
            except Exception as e:
                logger.debug("neuromatrix prefetch failed: %s", e)
            gate = getattr(self, "_min_recall_score", 0.0)
            if gate:
                results = [r for r in results if float(r.get("score", 0.0)) >= gate]
            budget = int(getattr(self, "_max_recall_chars", 1500))
            used = 0
            lines: list[str] = []
            rule_lines: list[str] = []
            for r in results:
                # Attach the EVENT date (when it happened), not just the record
                # date, so the agent sees time the way the memory knows it.
                try:
                    ev = self._store.event_time(r["fact_id"]) if self._store else None
                except Exception:  # noqa: BLE001
                    ev = None
                if ev:
                    r["event_ts"], r["event_granularity"] = ev[0], ev[1]
                line = self._format_hit(r)
                if lines and used + len(line) > budget:
                    break
                # A standing rule and an observation are not the same kind of claim:
                # one must be obeyed, the other weighed.  Kept in separate lists so
                # the rules are stated as rules instead of dissolving into the list.
                if str(r.get("kind") or "") in ("rule", "constraint", "policy"):
                    rule_lines.append(line)
                else:
                    lines.append(line)
                used += len(line) + 1
            # Chronological trace: ONE added line stating the retrieved dated events in
            # order, with the gap between consecutive ones.  Measured on 607 questions
            # with a local 1.5B reader: temporal F1 3.9% -> 24.4% (x6.3) and overall F1
            # 31.1% -> 35.0%, while evidence-hit@8 was unchanged (61.0% -> 61.1%) — the
            # retrieved window is untouched.  This is GRAVITY's finding applied: the
            # reader fails not from missing evidence but from the relations BETWEEN
            # fragments being implicit (their oracle run: all gold evidence present,
            # accuracy still 80.9%).  Every attempt to express that relation by
            # REORDERING the window measured worse (chronological context: F1 30.0%);
            # adding it as a line is what worked.  Two dated facts are the minimum,
            # because a single event has no relation to state.
            try:
                dated = [(r.get("event_ts"), str(r.get("text", "")))
                         for r in results if r.get("event_ts")]
                if len(dated) >= 2:
                    from .temporal import timeline_trace

                    trace = timeline_trace(dated)
                    if trace:
                        lines.insert(0, f"- [timeline] {trace}")
                        used += len(trace)
            except Exception as e:  # noqa: BLE001 - never break recall
                logger.debug("neuromatrix timeline failed: %s", e)
            if rule_lines:
                parts.append("Standing rules (must be followed):")
                parts.extend(rule_lines)
            parts.extend(lines)
            if lines:
                self._recall = RecallStatus(provider_label="NeuroMatrix", count=len(lines))
        # Project continuity: the NEW session's first turn should see where the
        # previous session stopped (finalize_session rollup at session end).
        if session_id and self._store is not None:
            try:
                st = self._store.latest_statuses(limit=1)
                if st and str(st[0].get("session_id") or "") != session_id:
                    parts.append(f"- 🧭 {st[0]['text'][:420]}")
            except Exception as e:
                logger.debug("neuromatrix status prefetch failed: %s", e)
        if not parts:
            return ""
        return f"## {DISPLAY}\n" + "\n".join(parts)

    def recall_status(self) -> Optional[RecallStatus]:
        return self._recall

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Store-level LRU (30 s) already makes the next prefetch() fast even
        # when the identical query repeats; nothing else to pre-warm.
        return None

    # ------------------------------------------------------------- write path

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:
        """Persist a completed (user, assistant) turn — episodic + graph write."""
        if self._store is None or not self._active:
            return
        session_id = kwargs.get("session_id") or self._session_id
        try:
            self._store.add_turn(user_content, assistant_content, session_id=session_id)
        except Exception as e:
            logger.debug("neuromatrix sync_turn failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror curated built-in memory writes into the graph (high weight)."""
        if self._store is None or action != "add" or not content:
            return
        try:
            self._store.remember(
                content,
                source=f"mirror:{target}",
                session_id=metadata.get("session_id", "") if metadata else "",
                importance=1.2,
                meta={"origin": "builtin_memory"},
            )
        except Exception as e:
            logger.debug("neuromatrix mirror failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Best-effort evidence archive before the transcript is lossily
        rewritten: store user/assistant prose rows that carry anchors."""
        if self._store is None:
            return ""
        archived = 0
        try:
            for msg in (messages or [])[-60:]:
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content")
                if not isinstance(content, str) or len(content) < 10:
                    continue
                if self._store.remember(
                    content[:600], source="precompress",
                    session_id=self._session_id, importance=0.7,
                ) is not None:
                    archived += 1
        except Exception as e:
            logger.debug("neuromatrix on_pre_compress failed: %s", e)
        # Crash-safe status rollup: sessions that never fire on_session_end
        # still get a project-status fact (idempotent per session).
        if self._store is not None and self._session_id:
            try:
                self._store.finalize_session(self._session_id)
            except Exception as e:
                logger.debug("neuromatrix finalize (pre-compress) failed: %s", e)
        return f"neuro-matrix checkpoint: archived {archived} pre-compress messages" if archived else ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """'Sleep' phase: consolidate fresh facts into dossiers + daily prune."""
        if self._store is None:
            return
        try:
            if self._active and self._session_id:
                try:
                    self._store.rollup_session(self._session_id)
                except Exception as e:
                    logger.debug("neuromatrix rollup failed: %s", e)
                try:
                    self._store.finalize_session(self._session_id)
                except Exception as e:
                    logger.debug("neuromatrix finalize failed: %s", e)
            if self._auto_consolidate:
                rep = self._store.consolidate(max_llm_calls=2)
                if rep["dossiers_updated"] or rep["lessons"]:
                    logger.info("neuromatrix consolidate: %s", rep)
            # Dynamic decision sweep — once per UTC day, phrasing-independent.
            if self._active and getattr(self._store, "auto_decide_enabled", True):
                sday = time.strftime("%Y%m%d", time.gmtime())
                if self._store.get_meta("nm:sweep_day") != sday:
                    self._store.set_meta("nm:sweep_day", sday)
                    swept = self._store.sweep_decisions()
                    if swept:
                        logger.info("neuromatrix sweep: captured %d decisions", swept)
            # Trait/interest distillation (write-time, LLM-optional) — once
            # per UTC day, mirrors the decision sweep. Turns scattered
            # episodic mentions about a named person into a durable trait
            # fact, so a later inferential question is a plain lookup
            # instead of needing live reasoning.
            if self._active:
                tday = time.strftime("%Y%m%d", time.gmtime())
                if self._store.get_meta("nm:trait_sweep_day") != tday:
                    self._store.set_meta("nm:trait_sweep_day", tday)
                    try:
                        traits = self._store.sweep_traits()
                        if traits:
                            logger.info("neuromatrix trait sweep: captured %d traits", traits)
                    except Exception as e:
                        logger.debug("neuromatrix trait sweep failed: %s", e)
            # Doc -> rules distiller: once per UTC day, budget-gated.
            if self._auto_consolidate:
                dday = time.strftime("%Y%m%d", time.gmtime())
                if self._store.get_meta("nm:distill_day") != dday:
                    self._store.set_meta("nm:distill_day", dday)
                    try:
                        rep = self._store.distill_docs(max_chunks=3, max_llm_calls=1)
                        if rep.get("rules"):
                            logger.info("neuromatrix distill: %s", rep)
                    except Exception as e:
                        logger.debug("neuromatrix distill failed: %s", e)
            # Dense index drain (bounded): index a slice of the unindexed facts
            # so hybrid recall has vectors to work with.  A slice, not a full
            # pass — the queue persists between sessions, and a session end
            # must not block for minutes on a local embedding model.
            if self._store.embedder is not None:
                try:
                    added = self._store.embed_missing(limit=256)
                    if added:
                        logger.debug("neuromatrix embeddings: indexed %d facts", added)
                except Exception as e:
                    logger.debug("neuromatrix embedding drain failed: %s", e)
            today = time.strftime("%Y%m%d")
            if self._store.get_meta("last_prune_day") != today:
                n = self._store.prune(retention_days=self._retention_days)
                self._store.set_meta("last_prune_day", today)
                if n:
                    logger.info("neuromatrix prune: archived %d old facts", n)
            if self._shared is not None:
                # The workspace pool gets its own 'sleep' pass — shared episodic
                # noise decays and consolidates too, without touching private data.
                if self._auto_consolidate:
                    srep = self._shared.consolidate(max_llm_calls=1)
                    if srep.get("dossiers_updated") or srep.get("lessons"):
                        logger.info("neuromatrix workspace consolidate: %s", srep)
                if self._shared.get_meta("last_prune_day") != today:
                    sn = self._shared.prune(retention_days=self._retention_days)
                    self._shared.set_meta("last_prune_day", today)
                    if sn:
                        logger.info("neuromatrix workspace prune: %d old facts", sn)
        except Exception as e:
            logger.debug("neuromatrix on_session_end failed: %s", e)

    # ---------------------------------------------------------------- tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [NM_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != NM_TOOL_SCHEMA["name"]:
            return tool_error(f"Unknown tool: {tool_name}")
        action = str(args.get("action", ""))
        handler = _HANDLERS.get(action)
        if handler is None:
            return tool_error(f"Unknown neuromatrix action: {action}")
        try:
            return handler(self, args)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:  # noqa: BLE001
            return tool_error(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ formatting

    @staticmethod
    def _clip(text: str, limit: int = 300) -> str:
        """Cut text at a boundary the reader can still parse.

        The old form was ``text[:300]``, which slices mid-word and mid-table-cell —
        that is where the fragments the user sees ("кап — **: 32") come from: a cut
        through the middle of a markdown row reads as knowledge even though it is
        rubble.  This prefers a sentence end, then a clause boundary, then a word
        boundary, and marks the omission so a truncated fact never looks complete.
        """
        s = str(text or "")
        s = " ".join(s.split())
        # Markdown is not prose.  A table row carried into memory reads as rubble
        # ("| гипотеза | результат | |---|---|"), and it is what makes fragments look
        # like knowledge once they are cut.  Collapse the furniture, keep the words.
        if any(ch in s for ch in ("|", "**", "```")):
            s = re.sub(r"\|+", " ", s)
            s = s.replace("**", "").replace("```", " ")
            s = re.sub(r"^[#>\-\s]+", "", s)
            s = " ".join(s.split())
        if len(s) <= limit:
            return s
        window = s[:limit]
        for sep in (". ", "! ", "? ", "; ", ", ", " — ", " - "):
            cut = window.rfind(sep)
            if cut >= limit * 0.55:
                return window[: cut + 1].rstrip() + " …"
        cut = window.rfind(" ")
        if cut <= 0:
            return window + "…"
        return window[:cut].rstrip() + " …"

    @staticmethod
    def _format_hit(r: Dict[str, Any]) -> str:
        via = f" (via {', '.join(r['via'][:3])})" if r.get("via") else ""
        # What KIND of thing this is travels with the line: a standing rule the agent
        # must obey and an observation it may weigh are not the same sort of claim, and
        # a flat list hides that difference.  Dossiers already read as summaries.
        kind = str(r.get("kind") or "episodic")
        tag = ""
        if kind in ("rule", "constraint", "policy"):
            tag = "[rule] "
        elif kind in ("deadend",):
            tag = "[dead end] "
        elif kind in ("capability",):
            tag = "[capability] "
        elif kind in ("trait",):
            tag = "[trait] "
        if r["source"] == "dossier":
            return f"- {r['text']}"
        # Event date when the fact names one (with the weekday, a calendar lookup
        # the model gets wrong), falling back to the record date otherwise.
        ev = r.get("event_ts")
        if ev:
            try:
                d = time.localtime(float(ev))
                gran = str(r.get("event_granularity") or "day")
                when = time.strftime("%Y-%m" if gran == "month" else
                                     ("%Y" if gran == "year" else "%Y-%m-%d"), d)
                if gran == "day":
                    when = f"{when} {time.strftime('%a', d)}"
                human = ""
                try:
                    from .temporal import format_human

                    human = format_human(float(ev), gran)
                except Exception:  # noqa: BLE001
                    human = ""
                if human:
                    when = f"{when} · {human}"
                return f"- {tag}{NeuromatrixMemoryProvider._clip(r['text'], 300)}{via} [event {when}]"
            except (TypeError, ValueError, OSError):
                pass
        when = time.strftime("%Y-%m-%d", time.localtime(r["ts"])) if r.get("ts") else ""
        return f"- {tag}{NeuromatrixMemoryProvider._clip(r['text'], 300)}{via} [{when}]"

    # ------------------------------------------------------------ shutdown

    def shutdown(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("neuromatrix shutdown close() failed: %s", e)
        if self._shared is not None:
            try:
                self._shared.close()
            except Exception as e:
                logger.debug("neuromatrix shutdown shared close() failed: %s", e)
        self._store = None
        self._shared = None
        self._recall = None


def _tool_search(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    if not prov._store:
        return tool_error("provider not initialized")
    query = a.get("query")
    if not query:
        return tool_error("search requires 'query'")
    rerank = is_truthy_value(a.get("rerank"))
    res = _search_merged(prov, query, int(a.get("limit", 6)), rerank=rerank)
    out = [{"text": r["text"], "source": r["source"], "score": r["score"],
            "via": r.get("via", []),
            **({"workspace": "shared"} if r.get("workspace") else {})} for r in res]
    return json.dumps({"ok": True, "count": len(out), "results": out}, ensure_ascii=False)


def _tool_remember(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("shared workspace not configured (workspace_db); scope=shared unavailable")
    fid = store.remember(a["content"], source="tool", session_id=prov._session_id)
    return json.dumps({"ok": True, "fact_id": fid})


def _tool_probe(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    if not prov._store:
        return tool_error("provider not initialized")
    ent = _probe_merged(prov, a["entity"])
    if ent is None:
        return json.dumps({"ok": True, "found": False, "entity": a["entity"]})
    return json.dumps({"ok": True, "found": True, **ent}, ensure_ascii=False)


def _tool_link(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    if not prov._store:
        return tool_error("provider not initialized")
    prov._store.link(a["entity_a"], a["entity_b"])
    return json.dumps({"ok": True, "linked": [a["entity_a"], a["entity_b"]]})


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.replace(";", ",").split(",") if s.strip()]
    if isinstance(v, list):
        return [str(s) for s in v]
    return [str(v)]


def _ws(prov: NeuromatrixMemoryProvider, a: dict) -> Optional[NeuroMatrixStore]:
    """Route an op to the private or shared store by scope."""
    scope = str(a.get("scope") or "private")
    if scope == "shared":
        return getattr(prov, "_shared", None)
    return prov._store


def _search_merged(prov: NeuromatrixMemoryProvider, query: str, limit: int,
                   rerank: bool = False) -> list[dict[str, Any]]:
    """Private pool first, then shared pool (workspace); score-merged."""
    out = []
    if prov._store:
        out = prov._store.search(query, limit=limit if not prov._shared else limit * 2,
                                 rerank=rerank)
    if prov._shared:
        seen = {r["text"] for r in out}
        for h in prov._shared.search(query, limit=limit):
            if h["text"] not in seen:
                hh = dict(h)
                hh["workspace"] = "shared"
                out.append(hh)
        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[:limit]


def _probe_merged(prov: NeuromatrixMemoryProvider, key: str) -> Optional[dict[str, Any]]:
    if prov._store:
        ent = prov._store.entity(key)
        if ent is not None:
            return ent
    if prov._shared:
        ent2 = prov._shared.entity(key)
        if ent2 is not None:
            ent2 = dict(ent2)
            ent2["workspace"] = "shared"
            return ent2
    return None


def _tool_decide(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("concept") or not a.get("choice"):
        return tool_error("decide requires 'concept' and 'choice'")
    fid = store.decide(
        a["concept"], a["choice"],
        criteria=_as_list(a.get("criteria")),
        scope_tags=_as_list(a.get("scope_tags")),
        reason=a.get("reason", ""),
        session_id=prov._session_id,
    )
    return json.dumps({"ok": True, "decision_id": fid, "concept": a["concept"],
                       "choice": a["choice"]})


def _tool_supersede(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("decision_id") or not a.get("choice"):
        return tool_error("supersede requires 'decision_id' and 'choice'")
    fid = store.supersede(
        int(a["decision_id"]), a["choice"],
        criteria=_as_list(a.get("criteria")),
        scope_tags=_as_list(a.get("scope_tags")),
        reason=a.get("reason", ""),
        session_id=prov._session_id,
    )
    return json.dumps({"ok": True, "decision_id": fid, "superseded": int(a["decision_id"]),
                       "choice": a["choice"]})


def _tool_decisions(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("concept"):
        return tool_error("decisions requires 'concept'")
    return json.dumps({"ok": True, **store.decisions(a["concept"])},
                      ensure_ascii=False)


def _tool_remember_goal(prov: NeuromatrixMemoryProvider, a: dict, kind: str) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("content"):
        return tool_error(f"{kind} requires 'content'")
    if kind == "goal":
        fid = store.remember_goal(a["content"], entity=a.get("entity"),
                                  session_id=prov._session_id)
    else:
        fid = store.remember_constraint(a["content"], entity=a.get("entity"),
                                        session_id=prov._session_id)
    return json.dumps({"ok": True, "fact_id": fid, "kind": kind})


def _tool_feedback(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    if not prov._store:
        return tool_error("provider not initialized")
    if not a.get("fact_id") or a.get("verdict") not in ("helpful", "unhelpful"):
        return tool_error("feedback requires 'fact_id' and verdict=helpful|unhelpful")
    res = prov._store.feedback(int(a["fact_id"]), helpful=a["verdict"] == "helpful")
    if res is None:
        return tool_error(f"fact {a['fact_id']} not found or archived")
    return json.dumps({"ok": True, **res})


def _tool_contradict(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    if not prov._store:
        return tool_error("provider not initialized")
    return json.dumps({"ok": True, **prov._store.contradictions()},
                      ensure_ascii=False)


def _tool_artifact_store(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("name"):
        return tool_error("artifact_store requires 'name'")
    has_text = bool(a.get("text"))
    has_file = bool(a.get("file_path"))
    if has_text == has_file:
        return tool_error("artifact_store requires exactly one of 'text'|'file_path'")
    res = store.store_artifact(
        a["name"], text=a.get("text") if has_text else None,
        file_path=a.get("file_path") if has_file else None,
        kind=a.get("kind") or "document", topic=a.get("topic"),
        session_id=prov._session_id)
    if res is None:
        return tool_error("artifact_store failed")
    return json.dumps({"ok": True, **res}, ensure_ascii=False)


def _tool_artifact_get(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    res = store.artifact_get(
        int(a["artifact_id"]) if a.get("artifact_id") is not None else None,
        name=a.get("name"), max_chars=int(a.get("max_chars", 4000)))
    if res is None:
        return json.dumps({"ok": True, "found": False})
    return json.dumps({"ok": True, "found": True, **res}, ensure_ascii=False)


def _tool_artifact_find(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    if not prov._store:
        return tool_error("provider not initialized")
    if not a.get("query"):
        return tool_error("artifact_find requires 'query'")
    out = list(prov._store.artifact_find(a["query"]))
    if prov._shared:
        out += list(prov._shared.artifact_find(a["query"]))
    return json.dumps({"ok": True, "results": out}, ensure_ascii=False)


def _tool_artifact_delete(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("artifact_id"):
        return tool_error("artifact_delete requires 'artifact_id'")
    ok = store.artifact_delete(int(a["artifact_id"]))
    return json.dumps({"ok": ok, "deleted": bool(ok)})


def _tool_deadend(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Record a failed attempt as a durable dead-end fact (closed-loop
    learning): the system keeps knowing what NOT to re-try, with the reason."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    subject = (a.get("entity") or a.get("query") or "").strip()
    reason = (a.get("reason") or a.get("content") or "").strip()
    if not subject:
        return tool_error("deadend requires 'entity' (the failed subject)")
    if not reason:
        return tool_error("deadend requires 'reason' (why it failed)")
    fid = store.mark_deadend(subject, reason, source="tool")
    return json.dumps({"ok": bool(fid), "deadend_id": fid, "subject": subject},
                      ensure_ascii=False)


def _tool_deadends(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Active dead ends, newest first; filter by subject entity (NOCASE)."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    rows = store.deadends((a.get("entity") or "").strip() or None,
                          limit=int(a.get("limit") or 20))
    return json.dumps({"ok": True, "count": len(rows), "deadends": rows},
                      ensure_ascii=False)



def _tool_attempt(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Record ONE try at ONE goal and what came of it — including failures.

    Knowledge answers "what is true"; this answers the question an agent asks while
    working: "have I tried this, and what happened".  A failure recorded here is why
    the same dead end does not get walked twice.
    """
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    goal = (a.get("goal") or "").strip()
    approach = (a.get("approach") or "").strip()
    outcome = (a.get("outcome") or "").strip().lower()
    if not goal or not approach:
        return tool_error("attempt requires 'goal' and 'approach'")
    if outcome not in ("ok", "partial", "fail"):
        return tool_error("'outcome' must be one of: ok, partial, fail")
    fid = store.record_attempt(
        goal, approach, outcome,
        reason=(a.get("reason") or "").strip(),
        evidence=(a.get("evidence") or "").strip(),
        session_id=(a.get("session_id") or "").strip())
    return json.dumps({"ok": bool(fid), "attempt_id": fid, "goal": goal,
                       "outcome": outcome}, ensure_ascii=False)


def _tool_attempts(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """What has already been tried for a goal, with the outcomes.  Newest first."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    rows = store.attempts((a.get("goal") or "").strip(),
                          limit=int(a.get("limit") or 20))
    tried = [{"goal": r.get("g"), "approach": r.get("a"), "outcome": r.get("o"),
              "reason": r.get("r"), "evidence": r.get("e"), "ts": r.get("ts")}
             for r in rows]
    failed = [r for r in tried if r.get("outcome") == "fail"]
    worked = [r for r in tried if r.get("outcome") == "ok"]
    return json.dumps({"ok": True, "count": len(tried), "attempts": tried,
                       "summary": {"failed": len(failed), "worked": len(worked)}},
                      ensure_ascii=False)



def _tool_self(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """What this instance is: its own budget, size, speed and measured weak spots.

    Not background context — called when the agent has a reason to reason about its
    own limits (context is filling up, a date must be computed, a hard multi-hop
    question is coming).
    """
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    model = store.self_model()
    if a.get("remember"):
        fid = store.remember_self(str(a["remember"]))
        model["remembered"] = fid
    return json.dumps({"ok": True, **model}, ensure_ascii=False)


def _tool_capabilities(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """What do we know <entity> is good for (durable capability facts,
    captured automatically from 'X используется для Y' statements)."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    rows = store.capabilities((a.get("entity") or "").strip() or None,
                              limit=int(a.get("limit") or 25))
    return json.dumps({"ok": True, "count": len(rows), "capabilities": rows},
                      ensure_ascii=False)


def _tool_ingest(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Bulk knowledge preload: store a document (content) as durable doc
    facts — chunked, no anchors required, always recallable afterwards."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    content = (a.get("content") or "").strip()
    if not content:
        return tool_error("ingest requires 'content' (the document text)")
    res = store.ingest_document(
        content, title=(a.get("topic") or a.get("name") or "ingested doc"),
        topic=(a.get("entity") or ""),
    )
    return json.dumps({"ok": res["stored"] > 0, **res}, ensure_ascii=False)


def _tool_statuses(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Newest session project-status rollups ('где мы остановились')."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    rows = store.latest_statuses(limit=int(a.get("limit") or 3))
    return json.dumps({"ok": True, "count": len(rows), "statuses": rows},
                      ensure_ascii=False)


def _tool_resolve(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Manually freeze a question->answer pair: the same ask then returns this
    answer instantly (no graph walk / FTS / LLM)."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    query = (a.get("query") or "").strip()
    answer = (a.get("content") or a.get("answer") or "").strip()
    if not query:
        return tool_error("resolve requires 'query' (the question)")
    if not answer:
        return tool_error("resolve requires 'content' (the answer)")
    fid = store.resolve_query(query, answer)
    return json.dumps({"ok": bool(fid), "resolved_id": fid, "query": query},
                      ensure_ascii=False)


def _tool_invent(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """'Fantasy' tool: propose novel combinations of known components for a
    goal/problem (wheel x engine -> car).  Deterministic novelty by graph
    edge strength; dead-end pairs excluded.  Hypotheses only — nothing stored
    until a human/agent approves (then decide/feedback)."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    goal = (a.get("query") or a.get("content") or "").strip()
    if not goal:
        return tool_error("invent requires 'query' (the goal/problem)")
    rows = store.invent(goal, limit=int(a.get("limit") or 8))
    return json.dumps({"ok": bool(rows), "count": len(rows), "ideas": rows},
                      ensure_ascii=False)


def _tool_distill(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Doc -> rules distiller: convert ingested doc chunks into crisp durable
    rules (pending review).  Budget-gated; no-op without an LLM/budget."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    res = store.distill_docs(max_chunks=int(a.get("limit") or 6),
                             max_llm_calls=2)
    return json.dumps({"ok": bool(res.get("rules")), **res}, ensure_ascii=False)


def _tool_explain(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Render a claim with evidence citations; falls back to the shared pool."""
    if not prov._store:
        return tool_error("provider not initialized")
    fid = int(a.get("fact_id") or 0)
    if not fid:
        return tool_error("explain requires 'fact_id'")
    store = prov._store
    res = store.explain_fact(fid)
    if res is None and prov._shared:
        res = prov._shared.explain_fact(fid)
    if res is None:
        return json.dumps({"ok": True, "found": False, "fact_id": fid})
    return json.dumps({"ok": True, "found": True, **res}, ensure_ascii=False)


def _tool_cite(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    fid = int(a.get("fact_id") or 0)
    ev = [int(i) for i in (a.get("evidence_ids") or []) if isinstance(i, (int, float))]
    if not fid or not ev:
        return tool_error("cite requires 'fact_id' and non-empty 'evidence_ids'")
    res = store.attach_evidence(fid, ev)
    if res is None:
        return tool_error(f"cite failed: fact {fid} missing/archived or empty evidence")
    return json.dumps({"ok": True, **res}, ensure_ascii=False)


def _tool_skill(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    concept = a.get("concept")
    if not concept:
        return tool_error("skill requires 'concept'")
    res = store.skill_propose(concept, out_dir=a.get("out_dir"))
    return json.dumps({"ok": True, **res}, ensure_ascii=False)


def _tool_persona(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """Generate the reviewable persona/profile card (PersonaMem mirror)."""
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    res = store.export_profile_card(out_dir=a.get("out_dir"))
    return json.dumps({"ok": True, **res}, ensure_ascii=False)


def _tool_policy(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    """Learned-policy digest from the memory-ops journal (RL-ready data)."""
    if not prov._store:
        return tool_error("provider not initialized")
    out: dict[str, Any] = {"private": prov._store.policy_report()}
    if prov._shared:
        out["shared"] = prov._shared.policy_report()
    return json.dumps({"ok": True, **out}, ensure_ascii=False)


def _tool_budget(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    if not prov._store:
        return tool_error("provider not initialized")
    return json.dumps({"ok": True, "llm_calls_remaining": prov._store.llm_budget_remaining()})


def _as_ts(v: Any) -> float:
    import datetime as _dt
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v or "").strip()
    try:
        return float(s)  # epoch seconds as string
    except ValueError:
        pass
    s = s.replace("Z", "+00:00")
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except ValueError as e:
        raise ValueError(f"cannot parse trigger_at: {v!r}") from e


def _tool_foresight(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    store = _ws(prov, a)
    if store is None:
        return tool_error("provider not initialized or shared workspace not configured")
    if not a.get("content") or not a.get("trigger_at"):
        return tool_error("foresight requires 'content' and 'trigger_at'")
    try:
        trig = _as_ts(a["trigger_at"])
    except ValueError as e:
        return tool_error(str(e))
    fid = store.plan_foresight(a["content"], trig, entity=a.get("entity"),
                               session_id=prov._session_id)
    return json.dumps({"ok": True, "foresight_id": fid, "trigger_at": trig})


def _tool_reminders(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    if not prov._store:
        return tool_error("provider not initialized")
    due = prov._store.foresights_due()
    if prov._shared:
        due += prov._shared.foresights_due()
    due = sorted(due, key=lambda d: d.get("trigger_at", 0.0))
    return json.dumps({"ok": True, "due": due, "count": len(due)}, ensure_ascii=False)


def _tool_ask(prov: NeuromatrixMemoryProvider, a: dict) -> str:
    """MemR3-style grounded answer: synthesize from retrieved memory, cite the
    facts, and state the evidence gaps explicitly (LLM required)."""
    store = prov._store
    if not store:
        return tool_error("provider not initialized")
    query = a.get("query")
    if not query:
        return tool_error("ask requires 'query'")
    llm = getattr(store, "llm", None)
    if llm is None or not getattr(llm, "available", lambda: False)():
        return tool_error("ask requires an LLM key (NEUROMATRIX_API_KEY) — use 'search' instead")
    try:
        hits = _search_merged(prov, query, 8)
    except Exception as e:  # noqa: BLE001
        return tool_error(f"search failed: {e}")
    if not hits:
        return json.dumps({"ok": True, "answer": "No relevant memory found.",
                           "cited": [], "unknowns": []})
    lines = [f"[{h['kind']} #{h['fact_id']}] {h['text'][:400]}" for h in hits]
    content = llm.chat_json([
        {"role": "system",
         "content": (
             "You answer from the agent's memory only. Ground the answer in the "
             "cited items, be concise (2-4 sentences), never invent facts. "
             'Return JSON: {"answer": "...", "cited": [<ids>], '
             '"unknowns": ["what is missing to answer fully"]}')},
        {"role": "user", "content": f"QUESTION: {query}\n\nMEMORY ITEMS:\n" + "\n".join(lines)}])
    if isinstance(content, dict):
        cited = [int(i) for i in (content.get("cited") or []) if isinstance(i, (int, float))]
        return json.dumps({"ok": True, "answer": str(content.get("answer", "")),
                           "cited": cited,
                           "unknowns": [str(u) for u in (content.get("unknowns") or [])]},
                          ensure_ascii=False)
    # Non-JSON fallback: transparent recall, no invented synthesis.
    return json.dumps({"ok": True, "answer": "Recalled (no synthesis):\n" + "\n".join(lines),
                       "cited": [int(h["fact_id"]) for h in hits if h.get("fact_id")],
                       "unknowns": []}, ensure_ascii=False)


def _tool_ops(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    if not prov._store:
        return tool_error("provider not initialized")
    return json.dumps({"ok": True, "ops": prov._store.ops_view(limit=50)},
                      ensure_ascii=False)


def _tool_consolidate(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    if not prov._store:
        return tool_error("provider not initialized")
    rep = prov._store.consolidate(max_llm_calls=2)
    return json.dumps({"ok": True, **rep})


def _tool_stats(prov: NeuromatrixMemoryProvider, a: dict) -> str:  # noqa: ARG001
    if not prov._store:
        return tool_error("provider not initialized")
    return json.dumps({"ok": True, **prov._store.stats()})


_HANDLERS = {
    "search": _tool_search,
    "remember": _tool_remember,
    "probe": _tool_probe,
    "link": _tool_link,
    "decide": _tool_decide,
    "supersede": _tool_supersede,
    "decisions": _tool_decisions,
    "remember_goal": lambda p, a: _tool_remember_goal(p, a, "goal"),
    "remember_constraint": lambda p, a: _tool_remember_goal(p, a, "constraint"),
    "feedback": _tool_feedback,
    "contradict": _tool_contradict,
    "artifact_store": _tool_artifact_store,
    "artifact_get": _tool_artifact_get,
    "artifact_find": _tool_artifact_find,
    "artifact_delete": _tool_artifact_delete,
    "foresight": _tool_foresight,
    "reminders": _tool_reminders,
    "ask": _tool_ask,
    "cite": _tool_cite,
    "explain": _tool_explain,
    "skill": _tool_skill,
    "persona": _tool_persona,
    "policy": _tool_policy,
    "ops": _tool_ops,
    "budget": _tool_budget,
    "consolidate": _tool_consolidate,
    "stats": _tool_stats,
    "deadend": _tool_deadend,
    "attempt": _tool_attempt,
    "self": _tool_self,
    "attempts": _tool_attempts,
    "deadends": _tool_deadends,
    "capabilities": _tool_capabilities,
    "ingest": _tool_ingest,
    "statuses": _tool_statuses,
    "resolve": _tool_resolve,
    "invent": _tool_invent,
    "distill": _tool_distill,
}


def register(ctx) -> None:
    """Register the neuromatrix memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(NeuromatrixMemoryProvider(config=_load_plugin_config()))
