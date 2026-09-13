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
| Failure-driven learning — don't repeat dead ends | MemoryAgentBench (2025: forgetting is the weakest agent competency) | outcome markers on the write path: “Flutter не подошёл, потому что …” → durable `kind='deadend'` fact with the reason + evidence trail to the original attempt (auto + manual `deadend` tool) |
| Knowledge preload + capability tags | — | `ingest_document()` chunks any reference doc (RU/technical, no anchors needed) into durable `kind='doc'` facts; “X используется для Y” turns into `kind='capability'` facts tying the entity to what it is good for — the raw material for `invent` |
| Session project-status + resolved Q&A + `invent` | — | session end → durable `kind='status'` rollup surfaced on the next session's first turn; the 2nd distinct ask of a question freezes the best answer (`kind='resolved'`, instant replay, auto-retires when its source fact dies); `invent(goal)` proposes novel component pairs by graph-edge novelty — capability phrases boost relevance, purpose-matched dead ends exclude the component for that goal only; `distill_docs()` turns ingested docs into durable rules (`kind='rule'`, pending review, budget-gated, once/day) |

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

### Option C — a local model via Ollama (no cloud, no key, nothing leaves the machine)

The LLM half of this provider (consolidation, trait distillation, rerank,
artifact extraction) runs on a local model. Point it at Ollama and set the
provider — no API key is needed, and the key-less path is explicitly allowed
so a local setup is not mistaken for "no LLM configured":

```bash
ollama pull qwen3.5:9b        # or any model you already pulled
hermes memory setup           # plugins.neuromatrix → llm_provider = ollama
```

Routing is automatic: `llm_provider = ollama` **or** any base URL containing
`:11434` selects Ollama's native `/api/chat` instead of the
OpenAI-compatible `/v1`, and the model name is resolved from `/api/tags`
when left empty. That is not a style preference — it is a correctness fix,
measured on qwen3.5:9b with one short QA prompt (identical answer both ways):

| endpoint | latency | completion tokens | content |
|---|---|---|---|
| `/v1` (OpenAI-compatible), `max_tokens=800` | 6.7 s | 330 | ok |
| `/v1`, `max_tokens=256` / `64` / `16` | 4.8 / 1.3 / 0.4 s | 256 / 64 / 16 | **empty** |
| `/api/chat` with `think: false` | **0.33 s** | **13** | ok |

Reasoning-capable models (qwen3.5:*, deepseek-r1, …) spend hundreds of
hidden "thinking" tokens per call on the `/v1` path and return an **empty
`content`** once the token budget runs out — which this provider reads as
"LLM unavailable" and silently degrades to extractive mode. Local inference is
free, so the daily LLM budget is not meaningful there; note that
`llm_daily_budget = 0` currently means *the opposite* of "unlimited" (it
zeroes the remaining budget, disabling every LLM step) — use a large number
until this is fixed.

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
`ask` (grounded synthesis) · `cite` / `explain` (claim provenance with
`[fact #N, session]` citations) · `skill` (procedure → reviewable skill draft) ·
`persona` (reviewable profile card) · `policy` (learned-policy digest from the
memory-ops journal) ·
`ops` (memory-ops journal) ·
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

## Test it yourself (10 minutes, no Hermes needed)

```bash
git clone https://github.com/qqdd27/neuro-matrix-memory && cd neuro-matrix-memory
python tests/test_core.py       # 35 engine scenarios — expect "35/35 passed"
python tests/test_contract.py   # provider lifecycle + concurrency — "2/2 passed"
python scripts/bench_recall.py  # recall@5 on RU/EN scenarios — expect 5/5, ~2-4 ms

# live smoke on a scratch DB:
python -m neuro_matrix.cli status --db /tmp/nm_demo.db
python -c "
from neuro_matrix import NeuroMatrixStore
s = NeuroMatrixStore('/tmp/nm_demo.db')
s.add_turn('Кошелёк id_777 привязан к TON.', 'Понял, TON = id_777.', session_id='demo')
print([h['text'] for h in s.search('id_777')])
s.close()
"
```

Then install as a Hermes provider (Option A in “Install”), restart Hermes, run
`hermes memory setup` and pick **neuromatrix**. LLM consolidation/rerank works
out of the box: the provider **reuses Hermes' active model** (reads the model
block of config.yaml + its `*_API_KEY` env). Set `NEUROMATRIX_API_KEY` only to
force a dedicated provider — without any key everything runs in extractive mode.

## Evaluate it (baseline measured)

```bash
python scripts/eval_recall.py --db "path/to/neuromatrix.db" \
    --scenarios scripts/scenarios.example.json      # semantic questions
python scripts/eval_recall.py --db "path/to/neuromatrix.db" \
    --auto 10 --out recall.md                        # dossier self-test
```

Baseline on the live profile db (2026-09-08): **recall@5 = 100% / 100%**
(see `docs/recall-report-2026-09-08.md`). Product thesis: `docs/VISION.md`.

**Honest boundary, measured, not assumed:** that 100% holds whenever a query
repeats an anchor (an id, a name, a token) — which is most real usage, and is
exactly what both suites above test. `python scripts/eval_semantic_gap.py`
measures the one case they cannot see: a pure conceptual paraphrase that
shares **neither** an anchor entity **nor** a content word with the stored
fact. A curated, bounded RU/EN synonym bridge (ops/tech vocabulary: сбой ~
авария ~ падал, интернет ~ сеть ~ офлайн, ...) narrowed this from **0/3 to
2/3** — a query built from a synonym of a word actually present in the fact
now recalls it (`test_synonym_bridge_narrows_semantic_gap`). The 3rd case
("Почему сменили предыдущего поставщика бэкенда?" for a fact that never
mentions "поставщик"/"провайдер"/"вендор" at all, only "Supabase") still
correctly misses — genuinely disjoint vocabulary needs an embedding model to
close, which conflicts with this project's zero-runtime-dependency, fully
local design. Every recall claim in this document should be read with that
scope attached: anchor- or synonym-reachable, not fully semantic.

**External, non-self-authored benchmark (2026-09):** every number above is
measured on scenarios this project wrote and tuned its own code against —
useful, but not an honest outside check. `python scripts/eval_locomo.py`
runs against [LoCoMo](https://github.com/snap-research/locomo) (arXiv
2402.17753), a public long-term conversational-memory benchmark this engine
had never seen: 10 real multi-session dialogues (19-32 sessions each),
~2000 QA pairs with gold evidence turns. It measures **evidence-hit@8** —
does the original turn the answer depends on show up in top-8 `search()`
results — the retrieval ceiling, not LoCoMo's published QA-accuracy number
(which also requires an LLM reader; not run here by design, zero extra
dependency/cost). First run: **3.0%**. It exposed a real, systemic defect no
internal test ever could: once an entity is mentioned across many facts (a
hub — a person's own name in a long conversation, the single most common
real case), `search()` ranked purely by entity-presence × recency ×
importance, blind to whether the rest of the question's words matched the
fact's own text — so "what did X research?" surfaced X's most *recent*
mention, not the one about research. Fixed with a multiplicative
content-token relevance bonus (an additive one measurably failed — entity
base scores are unbounded, so a fixed bonus is invisible against a hub's
already-large score) plus two entity-noise fixes (English sentence-filler
words and chat abbreviations like "BTW" were inflating fact scores as fake
entities). Result: **27.8%** (×9.3), still far from solved — multi-hop
questions (12.4%) remain the hardest, an honest, expected limit for
single-shot retrieval without multi-hop reasoning. Full breakdown in
`docs/locomo-report.md`. The dataset (CC BY-NC 4.0) is fetched on demand,
never vendored into this MIT-licensed repo.

**Local-model run — does the memory help the model at all? (2026-09-13):**
the numbers above measure *our retrieval*. The question a user actually asks
is how much the memory adds to the answers of a small **local** model, which
is the configuration this provider is meant for. Same 154 sampled questions,
same reader (qwen3.5:9b on an RTX 5060 Laptop via Ollama), two runs — once
with retrieved facts, once with the reader answering alone (`--no-memory`):

| Category | without memory | with memory | Δ |
|---|---|---|---|
| 1 single-hop | 4.0% | 11.0% | +7.0 pp |
| 2 temporal | 5.2% | 4.7% | −0.5 pp |
| 3 multi-hop | 17.9% | 16.7% | −1.2 pp |
| 4 open-domain | 9.3% | 20.2% | +10.9 pp |
| 5 adversarial | 8.0% | 18.7% | +10.7 pp |
| **Overall F1** | **7.9%** | **15.7%** | **+7.8 pp (×1.99)** |

Retrieval in that run: **evidence-hit@8 = 31.8%**, reproducing the earlier
full-population 32.3% on a different sample. Two findings worth more than the
headline: (1) `temporal` has the *best* retrieval (hit@8 41.4%) and the
*worst* answer score (4.7%) — the evidence is handed to the model and the
reader still cannot extract the date, so that gap is a reader/prompt problem,
not a memory problem; (2) `multi-hop` retrieved **0 of 6** gold turns — the
one category an associative graph is supposed to win, and it currently
contributes nothing. Reproduce the comparison with:

```bash
python scripts/compare_locomo_runs.py docs/locomo-local-baseline.md docs/locomo-local-memory.md
```

**Entity-match saturation (2026-09, zero LLM cost):** the additive fix above
still let a fact that happened to co-mention several query-adjacent
entities (some incidental) outscore a precisely-matched single-entity fact
by sheer count. Fixed with BM25-style term-frequency saturation applied to
the COUNT of matched entities per fact instead of word frequency (single-
entity facts score exactly as before; a 5-entity fact's contribution is
capped toward an asymptote, not a linear 5x). Result: evidence-hit@8
**27.8% → 32.3%** (+16% relative) — a genuine trade, reported honestly, not
a pure win: temporal (42.8%→36.2%) and multi-hop (12.4%→7.9%) regressed,
because saturation also dampens the cases where several co-mentioned
entities were genuinely, not incidentally, relevant. The k1=2.0 constant was
chosen by a small sweep against LoCoMo's own zero-cost evidence-hit@8 metric
(no paid QA-accuracy calls spent on it) — it also happens to sit at the
upper end of BM25's conventional k1 range (1.2–2.0) in the IR literature,
which is the honest mitigation against reading this as a number picked to
game one benchmark. A second, external benchmark (LongMemEval) would be the
next check that this generalizes past LoCoMo's specific question shapes.

**First real QA-accuracy run (2026-09, limited/paid, `--llm`):** 150 sampled
questions, Claude Haiku as the reader, scored with token-F1 — the number
actually comparable to published leaderboard entries. Result: **9.4% F1**,
evidence-hit@8 on that same sample **11.0%** — but reproducing the identical
150-question sample for free with `sweep_traits()` never invoked scored
**31.8%**, matching the full-population rate. The gap was a real, measured
regression the paid run itself surfaced: trait facts (write-time distillation,
v0.7.4) were created with an undecayed timestamp and importance ≥1.0, no cap
on how many can appear in results (unlike dossiers' explicit 2-slot cap) —
so a handful of distilled traits about a hub entity systematically
outscored and displaced the actual evidence turn from top-k. Fixed by
lowering trait importance to 0.5. Net honest lesson: the one paid run that
actually mattered was not the QA-accuracy number itself but the side effect
it exposed in a feature that had only ever been checked with a mocked LLM —
confirming, again, that a mechanism "working" in a unit test and a
mechanism helping in practice are different claims.

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
- [x] `hermes neuromatrix` CLI (`status`, `search`, `consolidate`)
- [x] Contradiction detection between dossiers and fresh facts (`contradict`
      action → `dossier_conflicts`: any post-consolidation fact carrying a
      negation/reversal marker about an entity that already has a settled
      dossier is flagged for review before it silently drifts)
- [ ] Optional reranking when a local embedding model is available
- [ ] Hermes Skills auto-registration on activation

## License

MIT
