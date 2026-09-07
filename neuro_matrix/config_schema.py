"""Dashboard config panel schema for the neuromatrix memory provider.

Read from disk next to the package ``__init__.py``; keep in sync with
``NeuromatrixMemoryProvider.get_config_schema()``.
"""

CONFIG_SCHEMA = [
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
        "description": "OpenAI-compatible base URL for consolidation",
        "default": "https://api.deepseek.com/v1",
    },
    {
        "key": "llm_model",
        "description": "Model used for consolidation",
        "default": "deepseek-chat",
    },
    {
        "key": "llm_api_key",
        "description": "API key for consolidation (secret; stored in .env)",
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
        "description": "Shared workspace pool (SQLite) across profiles/agents — scope=shared reads/writes it",
        "default": "",
    },
    {
        "key": "llm_rerank",
        "description": "Enable optional LLM rerank of search results (action=search rerank=true)",
        "default": "true",
        "choices": ["true", "false"],
    },
]
