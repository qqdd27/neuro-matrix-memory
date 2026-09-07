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


from .config_schema import CONFIG_SCHEMA  # noqa: E402
from .llm import LLMClient  # noqa: E402
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
                         "ops", "budget",
                         "consolidate", "stats"],
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
            adir = str(self._config.get("artifacts_dir") or "").replace(
                "$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
            if adir:
                self._store.artifacts_dir = adir
        self._max_recall_chars = int(self._config.get("max_recall_chars", 1500) or 1500)
        self._min_recall_score = float(self._config.get("min_recall_score", 0.0) or 0.0)

        llm_enabled = is_truthy_value(self._config.get("llm_enabled", "true"))
        api_key = os.environ.get("NEUROMATRIX_API_KEY", "")
        if llm_enabled and api_key:
            client = LLMClient(
                api_key=api_key,
                base_url=str(self._config.get("llm_base_url") or DEFAULT_BASE_URL),
                model=str(self._config.get("llm_model") or DEFAULT_MODEL),
            )
            self._store.llm = client
        ws = str(self._config.get("workspace_db") or "").replace(
            "$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        if ws and os.path.abspath(ws) != os.path.abspath(db_path):
            try:
                self._shared = NeuroMatrixStore(ws)
                self._shared.llm_daily_budget = self._store.llm_daily_budget
                if llm_enabled and api_key:
                    self._shared.llm = LLMClient(
                        api_key=api_key,
                        base_url=str(self._config.get("llm_base_url") or DEFAULT_BASE_URL),
                        model=str(self._config.get("llm_model") or DEFAULT_MODEL),
                    )
                logger.info("neuromatrix shared workspace ready: %s", ws)
            except Exception as e:  # shared pool must never break the provider
                logger.error("neuromatrix workspace %s failed: %s", ws, e)
                self._shared = None
        self._retention_days = float(self._config.get("retention_days", 365) or 365)
        self._auto_consolidate = is_truthy_value(self._config.get("auto_consolidate", "true"))
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
            for r in results:
                line = self._format_hit(r)
                if lines and used + len(line) > budget:
                    break
                lines.append(line)
                used += len(line) + 1
            parts.extend(lines)
            if lines:
                self._recall = RecallStatus(provider_label="NeuroMatrix", count=len(lines))
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
            if self._auto_consolidate:
                rep = self._store.consolidate(max_llm_calls=2)
                if rep["dossiers_updated"] or rep["lessons"]:
                    logger.info("neuromatrix consolidate: %s", rep)
            today = time.strftime("%Y%m%d")
            if self._store.get_meta("last_prune_day") != today:
                n = self._store.prune(retention_days=self._retention_days)
                self._store.set_meta("last_prune_day", today)
                if n:
                    logger.info("neuromatrix prune: archived %d old facts", n)
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
    def _format_hit(r: Dict[str, Any]) -> str:
        via = f" (via {', '.join(r['via'][:3])})" if r.get("via") else ""
        if r["source"] == "dossier":
            return f"- {r['text']}"
        when = time.strftime("%Y-%m-%d", time.localtime(r["ts"])) if r.get("ts") else ""
        return f"- {r['text'][:300]}{via} [{when}]"

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
    "ops": _tool_ops,
    "budget": _tool_budget,
    "consolidate": _tool_consolidate,
    "stats": _tool_stats,
}


def register(ctx) -> None:
    """Register the neuromatrix memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(NeuromatrixMemoryProvider(config=_load_plugin_config()))
