# NeuroMatrix Memory — Hermes Agent provider

**Local-first temporal knowledge-graph memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**
Cross-session entity IDs, alias merging (EN + RU), weighted decaying associations, and
sleep-style consolidation into entity dossiers. SQLite + FTS5, zero runtime dependencies,
nothing leaves your machine unless you opt into LLM consolidation.

> **TL;DR** — "Cryptocurrency TON — это id_777" (dialog 1) → "What about id_777?" (dialog 5)
> instantly recalls the TON facts *and* the consolidated dossier — no embeddings, no cloud, ~1 ms.

## Why this design (research-informed)

Classic vector RAG is linear; human memory is associative and temporal. This provider
implements the 2025–2026 agent-memory stack, not a naive co-occurrence graph:

| Idea | Source | Implementation here |
|---|---|---|
| Temporal knowledge graph, provenance, validity | [Zep / Graphiti (arXiv 2501.13956)](https://arxiv.org/abs/2501.13956) | every fact stores source + session + timestamp; stale facts are archived, never silently overwritten |
| KG + associative retrieval beats pure vectors | [HippoRAG 2 (ICML 2025)](https://arxiv.org/abs/2502.14802) | hybrid recall: alias + weighted 2-hop graph expansion over FTS5 lexical fallback |
| LLM decides at write time what to keep | [Mem0 (ECAI 2025)](https://arxiv.org/abs/2504.19413) | importance gate on write: ephemeral turns (single-anchor questions, greetings) never become facts |
| Consolidation levers: importance → merge → decay → eviction | [Hindsight (arXiv 2512.12818)](https://arxiv.org/abs/2512.12818) | importance at write, alias/entity merge, exponential decay on edges, retention-based prune |
| Offline "sleep" consolidation between sessions | [Anthropic Dreaming (2026)](https://kenhuangus.substack.com/p/why-ai-agents-are-starting-to-dream), SCM (arXiv 2604.20943) | `on_session_end` merges fresh facts into per-entity dossiers — LLM (batched, optional) or extractive |
| Cross-session identity = hardest open problem | [State of Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026) | alias registry + explicit `link` tool (`id_777` ↔ `TON` ↔ `токен`) |

## Install

### Option A — directory (no packaging)

Hermes discovers user memory providers in `$HERMES_HOME/plugins/<name>/`; the
provider package + its `plugin.yaml` / `config_schema.py` go there as-is:

```bash
mkdir -p "$HERMES_HOME/plugins/neuromatrix"
cp -r neuro_matrix/* "$HERMES_HOME/plugins/neuromatrix/"   # __init__.py, store.py, ..., plugin.yaml, config_schema.py
hermes memory setup        # pick "neuromatrix"  (or set memory.provider: neuromatrix in config.yaml)
hermes memory status
```

### Option B — pip package (recommended for publishing)

```bash
pip install neuro-matrix-memory
hermes memory setup
```

The package publishes the `hermes_agent.memory_providers` entry point
(`neuro_matrix.provider:register`), so nothing is copied; `config_schema.py`
sits next to the package `__init__.py` for the dashboard.

## Configuration (`hermes memory setup` → `plugins.neuromatrix` in config.yaml)

| Field | Default | Meaning |
|---|---|---|
| `db_path` | `$HERMES_HOME/neuromatrix.db` | SQLite store (WAL, FTS5) |
| `auto_consolidate` | `true` | run the "sleep" cycle at session end |
| `retention_days` | `365` | archive episodic facts older than this (keeps 1 latest per entity) |
| `llm_enabled` | `true` | allow LLM consolidation |
| `llm_base_url` / `llm_model` | DeepSeek `https://api.deepseek.com/v1` / `deepseek-chat` | any OpenAI-compatible endpoint |
| `llm_api_key` | — | **secret** → stored in `.env` as `NEUROMATRIX_API_KEY` |

Without an API key the provider runs **extractive mode**: consolidation keeps the top
facts verbatim. Fully private, works offline, zero cost.

## Tools the agent gains

One tool, `neuromatrix`, with actions:
`search` (associative recall incl. dossiers, optional `rerank=true` LLM pass) ·
`remember` (episodic fact) · `probe` (entity dossier + aliases) · `link` ·
`decide` / `supersede` / `decisions` (decision evolution with trail & lessons) ·
`remember_goal` / `remember_constraint` (intent ledger) ·
`feedback` (reinforcement) · `contradict` (conflict scan) ·
`artifact_store` / `artifact_get` / `artifact_find` / `artifact_delete`
(big-data layer) · `foresight` / `reminders` (time-bounded signals) ·
`ask` (grounded synthesis) · `ops` (memory-ops journal) ·
`budget` (LLM quota) · `consolidate` (sleep cycle) · `stats`.
Writes accept `scope: private|shared` — `shared` targets the workspace pool
(`workspace_db` config) so several profiles/agents exchange durable context;
reads (`search`, `probe`, `reminders`, `artifact_find`) merge both pools.

Recalled context is injected automatically between turns (`prefetch`, bounded
by `max_recall_chars`); built-in memory writes are mirrored into the graph
(`on_memory_write`); before context compression the transcript's
anchor-bearing rows are archived (`on_pre_compress`); every session ends with
an episode rollup + the consolidation "sleep".

## CLI

```bash
hermes neuromatrix status          # stats + LLM budget
hermes neuromatrix search "id_777" # recall from the terminal
hermes neuromatrix consolidate     # run the sleep cycle
hermes neuromatrix remind          # due foresights
hermes neuromatrix ingest          # cold start from $HERMES_HOME/state.db
hermes neuromatrix export out.db   # full snapshot (online SQLite backup)
hermes neuromatrix import out.db   # restore
```

Standalone: `python -m neuro_matrix.cli <command> [--db path]`.

## CLI

```bash
python -m neuro_matrix.cli status --db ~/.hermes/neuromatrix.db
python -m neuro_matrix.cli search "id_777" --db ...
python -m neuro_matrix.cli remind-cron   # one line per due foresight (exit 0) — cron-friendly
python -m neuro_matrix.cli consolidate --db ...
python -m neuro_matrix.cli ingest [state.db]   # cold start from Hermes session db
python -m neuro_matrix.cli export backup.db && python -m neuro_matrix.cli import backup.db
```

## MCP surface

Same engine behind a standard MCP stdio server (JSON-RPC over stdio, zero
dependencies — works with any MCP client):

```bash
python -m neuro_matrix.mcp --db ~/.hermes/neuromatrix.db
```

Tools: `memory_search`, `memory_remember`, `memory_probe`,
`memory_decide`, `memory_decisions`, `memory_foresight`,
`memory_reminders`, `memory_stats`. Register in Hermes via the MCP
registry pointing at `python -m neuro_matrix.mcp`.

## Development

```bash
python tests/test_core.py          # engine: 27 scenarios (RU/EN, decisions,
                                   #   neuro-levers, as_of, artifacts, budgets)
python tests/test_contract.py      # MemoryManager lifecycle + concurrency
python scripts/bench_recall.py     # 5/5 recall@5, ~2-4 ms per query
pip install -e .                   # dev install of the provider package
```

## Storage model

- `facts` — episodes with source/session/ts/importance/consolidated/archived.
- `entities` + `aliases` — canonical registry; alias merge folds two entities into one
  (moves links & edges, keeps the stronger key).
- `fact_entities` — provenance of every mention.
- `edges` — weighted associations `(a,b,count,first_seen,last_seen)`; effective strength
  = `count · exp(-age_hours / half_life)` (lazy decay, no background job needed).
- `dossiers` — consolidated long-term summaries per entity (search priority #1).

## Roadmap

- [x] Core engine (graph + FTS5 + decay + prune + extractive consolidation)
- [x] Hermes `MemoryProvider` plugin (prefetch / sync_turn / session-end sleep / tools)
- [x] Optional batched LLM consolidation (OpenAI-compatible)
- [ ] `hermes neuromatrix` CLI (`status`, `search`, `consolidate`)
- [ ] Contradiction detection between dossiers and fresh facts
- [ ] Optional reranking when a local embedding model is available
- [ ] Hermes Skills auto-registration on activation

## License

MIT
