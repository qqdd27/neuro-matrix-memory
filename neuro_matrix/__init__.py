"""NeuroMatrix Memory — local-first temporal knowledge-graph memory for Hermes Agent.

Package layout:
  store.py     — zero-dependency SQLite+FTS5 engine (facts, entities, aliases,
                 weighted decaying edges, dossiers, consolidation "sleep").
  entities.py  — multilingual (en+ru) anchor extraction & alias detection.
  llm.py       — optional OpenAI-compatible client (consolidation only).
  provider.py  — Hermes ``MemoryProvider`` plugin (``memory.provider: neuromatrix``).
"""

__version__ = "0.5.2"

from .store import NeuroMatrixStore

try:  # Hermes runtime: expose the provider for discovery/instantiation
    from .provider import NAME, NeuromatrixMemoryProvider  # noqa: F401
except Exception:  # standalone / tests: the provider layer stays optional
    NAME = "neuromatrix"
    NeuromatrixMemoryProvider = None  # type: ignore[assignment]


def register(ctx) -> None:
    """Entry point used by directory installs
    (``$HERMES_HOME/plugins/neuromatrix/``) and by the pip entry point
    ``hermes_agent.memory_providers``."""
    from .provider import register as _register
    _register(ctx)


__all__ = ["NeuroMatrixStore", "register", "__version__"]
