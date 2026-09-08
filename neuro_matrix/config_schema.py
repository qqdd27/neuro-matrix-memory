"""Dashboard config schemas for the neuromatrix memory provider.

Two contracts live here (both read from disk by name — never imported as a
package inside Hermes, so this file must stay dependency-free):

1. ``CONFIG_SCHEMA`` — the *declared* schema contract for memory-provider
   plugins (``plugins/memory/config_schema.py``): a ``ProviderConfigSchema``
   dataclass instance rendered by the generic settings panel
   (``GET /api/memory/providers/{name}/config?surface=declared``).

2. ``LEGACY_CONFIG_SCHEMA`` — the legacy list-of-dicts used by
   ``NeuroMatrixMemoryProvider.get_config_schema()`` (setup CLI + older
   consumers).

Outside Hermes the pure-data module is unavailable, so the same dataclasses
are re-declared locally (they only carry data).
"""

from __future__ import annotations

try:  # inside Hermes: the authoritative pure-data module
    from plugins.memory.config_schema import (  # type: ignore
        KIND_BOOL,
        KIND_NUMBER,
        KIND_SECRET,
        KIND_SELECT,
        KIND_TEXT,
        ProviderConfigSchema,
        ProviderField,
        ProviderFieldOption,
    )
    _DECLARED_AVAILABLE = True
except Exception:  # standalone / tests / pip install
    _DECLARED_AVAILABLE = False

    class ProviderFieldOption:  # noqa: D101
        def __init__(self, value: str, label: str, description: str = ""):
            self.value = value
            self.label = label
            self.description = description

    class ProviderField:  # noqa: D101
        def __init__(self, key: str, label: str, kind: str = "text",
                     default: str = "", description: str = "",
                     placeholder: str = "", options=(), env_key=None,
                     aliases=(), env_fallbacks=(), inline: bool = False,
                     group: str = "", info: str = "", scope: str = "host"):
            self.key = key
            self.label = label
            self.kind = kind
            self.default = default
            self.description = description
            self.placeholder = placeholder
            self.options = tuple(options)
            self.env_key = env_key
            self.aliases = tuple(aliases)
            self.env_fallbacks = tuple(env_fallbacks)
            self.inline = inline
            self.group = group
            self.info = info
            self.scope = scope

    class ProviderConfigSchema:  # noqa: D101
        def __init__(self, name: str, label: str, storage: str = "flat_json",
                     docs_url: str = "", fields=()):
            self.name = name
            self.label = label
            self.storage = storage
            self.docs_url = docs_url
            self.fields = tuple(fields)

    KIND_TEXT = "text"
    KIND_SELECT = "select"
    KIND_SECRET = "secret"
    KIND_BOOL = "bool"
    KIND_NUMBER = "number"


def _opt(value: str) -> ProviderFieldOption:
    return ProviderFieldOption(value=value, label=value)


def _tf(choices: tuple[str, ...]) -> tuple[ProviderFieldOption, ...]:
    return tuple(_opt(c) for c in choices)


TRUE_FALSE = ("true", "false")

# ── Declared schema (generic Hermes settings panel) ──────────────────────────
CONFIG_SCHEMA = ProviderConfigSchema(
    name="neuromatrix",
    label="Neuromatrix",
    storage="flat_json",
    fields=(
        ProviderField("db_path", "Database path", KIND_TEXT,
                      default="$HERMES_HOME/neuromatrix.db",
                      description="SQLite database path (expandable $HERMES_HOME)",
                      inline=True),
        ProviderField("auto_consolidate", "Auto consolidate", KIND_SELECT,
                      default="true", options=_tf(TRUE_FALSE),
                      description="Run the memory 'sleep' consolidation at session end",
                      inline=True),
        ProviderField("retention_days", "Retention (days)", KIND_NUMBER,
                      default="365",
                      description="Archive episodic facts older than N days (eviction lever)"),
        ProviderField("llm_enabled", "LLM consolidation", KIND_SELECT,
                      default="true", options=_tf(TRUE_FALSE),
                      description="Allow optional LLM consolidation (requires API key below)",
                      inline=True),
        ProviderField("llm_base_url", "LLM base URL", KIND_TEXT,
                      default="",
                      description="Only used with a dedicated API key. "
                                  "Leave EMPTY to follow Hermes' active provider URL"),
        ProviderField("llm_model", "LLM model", KIND_TEXT,
                      default="",
                      description="Only used with a dedicated API key. "
                                  "Leave EMPTY to follow Hermes' active model"),
        ProviderField("llm_api_key", "LLM API key", KIND_SECRET,
                      env_key="NEUROMATRIX_API_KEY",
                      description="Leave EMPTY to auto-reuse Hermes' active LLM provider "
                                  "(e.g. DeepSeek); set only to force a dedicated key",
                      inline=True),
        ProviderField("llm_daily_budget", "Daily LLM budget", KIND_NUMBER,
                      default="20",
                      description="Max LLM consolidation calls per day (0 = unlimited)"),
        ProviderField("max_recall_chars", "Recall cap (chars)", KIND_NUMBER,
                      default="1500",
                      description="Token economy: cap on auto-injected recall per turn"),
        ProviderField("min_recall_score", "Min recall score", KIND_NUMBER,
                      default="0",
                      description="Score gate for auto-injected recall (0 = disabled)"),
        ProviderField("artifacts_dir", "Artifacts dir", KIND_TEXT,
                      default="$HERMES_HOME/artifacts",
                      description="Sidecar payload files for the artifact layer"),
        ProviderField("workspace_db", "Workspace pool path", KIND_TEXT,
                      default="",
                      description="Shared workspace pool (SQLite) across profiles — "
                                  "scope=shared reads/writes it"),
        ProviderField("llm_rerank", "LLM rerank", KIND_SELECT,
                      default="true", options=_tf(TRUE_FALSE),
                      description="Enable optional LLM rerank of search results "
                                  "(action=search rerank=true)"),
        ProviderField("auto_decide", "Auto-capture decisions", KIND_SELECT,
                      default="true", options=_tf(TRUE_FALSE),
                      description="Record explicit choices from turns ('для X берём Y, "
                                  "потому что Z') as decision facts with criteria",
                      inline=True),
    ),
)

# ── Legacy schema (provider.get_config_schema(), setup CLI) ──────────────────
LEGACY_CONFIG_SCHEMA = [
    {
        "key": "db_path",
        "description": "SQLite database path (expandable $HERMES_HOME)",
        "default": "$HERMES_HOME/neuromatrix.db",
        "required": False,
    },
    {
        "key": "auto_consolidate",
        "description": "Run the memory 'sleep' consolidation at session end",
        "default": "true",
        "choices": ["true", "false"],
    },
    {
        "key": "retention_days",
        "description": "Archive episodic facts older than N days (eviction lever)",
        "default": "365",
        "type": "integer",
    },
    {
        "key": "llm_enabled",
        "description": "Allow optional LLM consolidation (requires API key below)",
        "default": "true",
        "choices": ["true", "false"],
    },
    {
        "key": "llm_base_url",
        "description": "Only used with a dedicated API key. "
                       "Leave EMPTY to follow Hermes' active provider URL",
        "default": "",
    },
    {
        "key": "llm_model",
        "description": "Only used with a dedicated API key. "
                       "Leave EMPTY to follow Hermes' active model",
        "default": "",
    },
    {
        "key": "llm_api_key",
        "description": "Leave EMPTY to auto-reuse Hermes' active LLM provider (e.g. DeepSeek); "
                       "set only to force a dedicated key (stored in .env)",
        "secret": True,
        "env_var": "NEUROMATRIX_API_KEY",
    },
    {
        "key": "llm_daily_budget",
        "description": "Max LLM consolidation calls per day (0 = unlimited)",
        "default": "20",
        "type": "integer",
    },
    {
        "key": "max_recall_chars",
        "description": "Token economy: cap on auto-injected recall per turn",
        "default": "1500",
        "type": "integer",
    },
    {
        "key": "min_recall_score",
        "description": "Score gate for auto-injected recall (0 = disabled)",
        "default": "0",
        "type": "number",
    },
    {
        "key": "artifacts_dir",
        "description": "Sidecar payload files for the artifact layer",
        "default": "$HERMES_HOME/artifacts",
    },
    {
        "key": "workspace_db",
        "description": "Shared workspace pool (SQLite) across profiles/agents — "
                      "scope=shared reads/writes it",
        "default": "",
    },
    {
        "key": "llm_rerank",
        "description": "Enable optional LLM rerank of search results "
                       "(action=search rerank=true)",
        "default": "true",
        "choices": ["true", "false"],
    },
    {
        "key": "auto_decide",
        "description": "Record explicit choices from turns ('для X берём Y, потому что Z') "
                       "as decision facts with criteria",
        "default": "true",
        "choices": ["true", "false"],
    },
]
