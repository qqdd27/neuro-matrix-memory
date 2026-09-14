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

**Re-measured after reviving the lexical channel (v0.10.0), same methodology,
same sample size, same reader:**

| Category | without memory | with memory (v0.8.0) | with memory (v0.10.0) |
|---|---|---|---|
| 1 single-hop | 4.0% | 11.0% | **19.7%** |
| 2 temporal | 5.2% | 4.7% | **7.4%** |
| 3 multi-hop | 17.9% | 16.7% | 16.7% (n=6, noise) |
| 4 open-domain | 9.3% | 20.2% | **35.3%** |
| 5 adversarial | 8.0% | 18.7% | **23.4%** |
| **Overall F1** | **7.9%** | 15.7% | **24.6%** |

So memory now multiplies this local model's answer accuracy by **×3.1** (was
×1.99), and the retrieval ceiling moved 32.3% → 53.2% on the same sample.
`temporal` is still the outlier: retrieval is fine, the reader cannot extract
dates — a prompt/reader problem that no retrieval change addresses.

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

## The lexical channel was dead (2026-09) — the largest single gain so far

The FTS5 index (`facts_fts`) and its `bm25(facts_fts)` query had existed for a
long time, so it was reasonable to assume lexical search was "done". It was not:
the only consumer used BM25 as a *fallback* — it read the BM25-ordered candidate
list and then re-scored every candidate by `importance × recency`, sorting by
that. **The text match itself never reached the ranking.** A fact containing
every query term could lose its slot to an important, recent, lexically
unrelated fact.

Fix: expose the BM25 order as its own ranked list inside RRF (`_fts_candidates`,
weight `fts_weight=4.0`), plus prefix terms so FTS5's exact-token matching stops
missing morphology (`research` → `researching`; the same gap affects inflected
Russian).

Measured on the full LoCoMo population (1977 questions, k=8, zero LLM cost):

| configuration | overall | single-hop | temporal | multi-hop | open-domain | adversarial |
|---|---|---|---|---|---|---|
| before (BM25 unused) | 32.3% | 30.2% | 36.2% | 7.9% | 34.8% | 30.7% |
| BM25 as its own list | **57.8%** | 44.5% | 64.1% | 28.1% | 61.7% | 60.3% |

**+25.5 pp overall (×1.79)** — and it needs no model, no GPU and no VRAM, so it
benefits every user equally. Cost at p50 on 5000 facts: 4.4 ms → 6.5 ms.

Two things were learned the hard way and are pinned by tests:

1. **The plateau is wide** (weights 3–10 all land at 57.1–58.0%), so this is a
   flat optimum, not a value tuned to one benchmark.
2. **Pure RRF ordering has a product cost the benchmark cannot see.** Taking the
   fused rank as the only ordering let a lexically similar throwaway turn
   outrank a `decision` fact and pushed a raw fact below `trait` rows. LoCoMo
   cannot detect this (all of its facts share one kind); the product tests
   caught it. Fixed by blending the fused rank with the evidence score
   (`fusion_blend=0.9`, where all 76 tests pass and overall stays at 57.8%,
   against 58.8% for the pure-RRF maximum).

Honest negative result from the same session: the graph-activation bridge
(co-occurrence, then PMI-weighted, then sparsified, at seven weights) never beat
leaving it off — 32.3% → 28.3% at best, 28.6% with PMI. It *does* improve the one
category it was built for (multi-hop 7.9% → 11.2% alone), so it stays implemented
and measurable, but off by default: it belongs behind a query-type router, not in
the default path.

## Meaning as structure: predicate + roles (2026-09)

The lexical fix above took recall to 57.8%, but it cannot see one thing at all:
**direction of action**.  "Маша подарила книгу Пете" and "Петя подарил книгу
Маше" contain identical content words and opposite meanings — BM25, embeddings
and the graph all score them the same.

`neuro_matrix/propositions.py` adds a structural channel that reads roles from
morphology, with **no model call and no training**:

| case | role | example |
|---|---|---|
| именительный / subject-first | agent | **Маша** подарила |
| винительный | patient | подарила **книгу** |
| дательный | recipient | подарила **Пете** |
| творительный | instrument | открыл **ключом** |
| предложный | location | был в **Берлине** |

English uses word order plus prepositions (`to` → recipient, `with` →
instrument, `in/at` → location).  Two further details were necessary and are
measured:

- **First person must be resolved to the speaker**: dialogue turns say
  "Caroline: I gave a speech", while questions ask "When did **Caroline** give a
  speech?".  Without that mapping the agent slot never matches and the channel
  stays silent on exactly the data it was built for (1529 propositions indexed,
  0-2 candidates per question).
- **Slot voting beats strict matching**: demanding every role the question names
  returned zero candidates on 20 of 20 temporal questions, because question roles
  and fact roles rarely align exactly.  Counting agreeing roles keeps the signal.

Measured on the full population (1977 questions):

| | overall | single-hop | temporal | multi-hop | open-domain | adversarial |
|---|---|---|---|---|---|---|
| lexical only | 57.8% | 44.5% | 64.1% | 28.1% | 61.7% | 60.3% |
| **+ structure** | **58.9%** | **46.3%** | **66.9%** | 29.2% | 62.5% | 60.3% |

+1.1 pp overall, concentrated exactly where roles matter.  It is optional in the
same sense as the embedder: without `pymorphy3`/`nltk` installed the channel is
an empty list and behaviour is byte-identical to before (pinned by a test).

**Honest negative: the role bridge does not help either (2026-09).** The natural
next step after the structural channel was to link facts through a *shared
participant* — `подарить(agent=маша, patient=книга)` is connected to any other
fact naming «книга» — instead of through word co-occurrence. It is implemented
(`_role_bridge_candidates`, two hops, pronouns excluded, pinned by two tests) and
it does not pay off:

| configuration | overall | multi-hop |
|---|---|---|
| structure only (shipped) | **58.9%** | 29.2% |
| + role bridge, weight 0.5 | 57.9% | 28.1% |
| + role bridge, weight 1.0 | 58.0% | 28.1% |
| + role bridge, weight 2.0 | 57.8% | 28.1% |

Consistent with the co-occurrence and PMI bridges before it: **any extra channel
that adds a broad candidate list loses to leaving it out**, because the slate is
finite and the precise lexical/role hits are exactly what gets displaced. Three
different graph formulations, one verdict — the default path stays lexical +
structural, and `role_bridge_enabled` remains off with the code kept for anyone
who wants to retest it.

**Re-measured after the structural channel (v0.11.0), same sample and reader:**

| Category | without memory | v0.8.0 | v0.10.0 (lexical) | v0.11.0 (+structure) |
|---|---|---|---|---|
| 1 single-hop | 4.0% | 11.0% | 19.7% | **21.0%** |
| 2 temporal | 5.2% | 4.7% | 7.4% | 7.4% |
| 3 multi-hop | 17.9% | 16.7% | 16.7% (n=6) | 16.7% (n=6) |
| 4 open-domain | 9.3% | 20.2% | 35.3% | **38.1%** |
| 5 adversarial | 8.0% | 18.7% | 23.4% | **32.7%** |
| **Overall F1** | **7.9%** | 15.7% | 24.6% | **28.2%** |

Memory now multiplies this local model's answer accuracy by **×3.6** (was ×1.99
before the lexical channel and ×3.1 before structure).  Note that structure
helped ANSWERS more than its +1.1 pp on retrieval suggested: answers also came
from a better-ordered window, not only from a longer list of candidates.

**Write-time splitting: measured and NOT shipped (2026-09).** The real database
stores whole assistant answers (~1000 characters), which makes morphology produce
junk roles, so splitting looked like the obvious prerequisite. Scored by answer F1
with the same reader, on one dialog, 60 questions:

| write shape | answer F1 | vs short turns |
|---|---|---|
| turns (as published) | 11.9% | — |
| **blocks (joined, live shape)** | **16.3%** | **+4.5 pp** |
| chunked (split at write time) | 15.3% | +3.4 pp |

Joining turns into big facts HELPS the reader: a longer fact carries the detail
around the answer, and splitting removes it. So the hypothesis that long facts
break retrieval was wrong, and the earlier evidence-hit@8 figure (88.3% for
blocks) was not purely an artefact either — it pointed the same way the answer
metric does. Nothing was shipped: no splitting, no chunking.

Note the honest caveat this leaves: the structural channel therefore helps
SHORT facts (where morphology is accurate) while live facts are long, and the
role data on long facts is noise. The channels that pay off on live data remain
the lexical one and the structural one's ordering effect.

**Question-type routing: measured and NOT enabled (2026-09).** Answers have
recognisable surfaces — "when" is answered by a date, "who" by a name — while all
types are retrieved by the same bag of words, so `question_types.py` classifies
the question by regex (English + Russian, no model) and promotes same-shape facts
within the window. Scored by answer F1 (the retrieval metric cannot see a
reorder at all):

| | shipped (structure) | + type routing (weight 2) |
|---|---|---|
| **overall F1** | **28.2%** | 25.1% |
| temporal | 7.4% | **8.5%** |
| single-hop | 21.0% | 20.7% |
| open-domain | 38.1% | 33.8% |
| adversarial | 32.7% | 26.6% |

It helps exactly where it was designed to help — questions whose answer is a
date — and loses overall, because promoting "the same SHAPE as the question"
displaces facts that are simply more relevant. Shape is not relevance. Off by
default (`type_boost_enabled`), kept measurable, like the bridges.

**Edge ordering ("lost in the middle"): measured and NOT enabled (2026-09).**
Placing the runner-up fact LAST (readers attend to the edges of a long context
more than to its middle) — again scored by answer F1, since the retrieval metric
cannot see a reorder:

| | shipped (structure) | + edge ordering |
|---|---|---|
| **overall F1** | **28.2%** | 22.2% |
| temporal | 7.4% | **11.8%** |
| single-hop | 21.0% | **23.3%** |
| multi-hop | 16.7% (n=6) | 11.2% |
| open-domain | 38.1% | 29.3% |
| adversarial | 32.7% | 19.2% |

Two of five categories improve markedly and the total collapses, mostly through
adversarial questions (ones whose answer is NOT in the window): reordering makes
the window look more authoritative and the model stops saying "no information".
The honest summary of the whole ordering line of work: **nothing that reorders
the window has beaten leaving it alone**, measured four ways.

## Meaning as morphology: one lemma per concept

The second language law worth applying — after roles from case — is the one that
makes a dictionary possible at all: an inflected word is a LEMMA plus features
(tense, case, number), so "бежал / бежит / бежать" and "research / researching /
researched" denote one concept. A lexical index keyed on SURFACE forms treats them
as unrelated strings, so a question phrased in one form cannot reach a fact phrased
in another.

**The important part is HOW it is applied.** Two placements were measured, and the
one that raises retrieval the most is the one that damages answers:

| | retrieval (1977 q) | answers (154 q) |
|---|---|---|
| surface index only (before) | 57.8% | 28.2% |
| lemma index REPLACING the surface one | **62.8%** | 25.3% |
| lemma index BESIDE the surface one (weight 2) | 61.2% | **28.9%** |

Replacing the surface index wins on retrieval and loses on answers: normalisation
also pulls unrelated words together ("university"/"universal"), so the window fills
with topically similar rather than exact facts and exact wording loses its weight.
As a second channel — `_lemma_candidates` beside `_fts_candidates` — the same
mechanism keeps exact matches strong and only adds:

| category | before | beside |
|---|---|---|
| single-hop | 21.0% | 27.3% |
| temporal | 7.4% | 8.5% |
| open-domain | 38.1% | 39.9% |
| adversarial | 32.7% | 29.4% |

Weight sweep (flat plateau, so the default is not a knife edge): 1 → 60.2%,
2 → 61.2%, 3 → 61.4%. Cost: pymorphy (ru) + WordNet (en), both optional, memoised
at ~0.01–0.05 ms/word, identity when absent.

English uses true WordNet LEMMATISATION, not a stemmer: stemming collides
"universal"/"university" into one form, inflating matches with noise. The pick is
fixed and tie-broken by position, never by set-iteration order — index and query
must normalise IDENTICALLY, and a non-deterministic choice would index the same
text two different ways between runs.

Why this is safe where four graph experiments were not: normalisation **narrows**.
It introduces no new candidates and cannot displace a correct hit — an existing
fact simply becomes reachable from more of its own forms. (The placement lesson
above shows even a narrowing channel can hurt when it REPLACES rather than ADDS.)

Three defects surfaced while building it, each caught by a test rather than by
reasoning:

- **Russian names were dropped by the write filter.** `extract_entities` gated
  Cyrillic proper nouns on position ("sentence-initial capitalisation is
  ambiguous"), so "Каролина изучала модели памяти" produced ZERO entities and was
  discarded entirely as ephemeral — while Latin names had already had that gate
  removed. Fixed by accepting a morphologically CONFIRMED name (Name/Surn/Patr in
  ANY parse, since pymorphy tags "каролина" as a Geox first and a Name later) in
  any position, keeping the gate for unverified words so "Новый факт…" and
  "Привет…" stay discarded.
- **A query with no resolvable entity returned nothing at all.** Fusion bailed
  out whenever the heuristic list had fewer than two entries, so the text match
  was never consulted: asking for "кеш" with no named entity produced an empty
  result even when facts contained the word. Now an empty heuristic window lets
  the lexical channel answer alone; a single hit keeps the old path.
- **Channels bypassed the lifecycle filter.** Every channel returns bare ids and
  the materialisation step re-fetched them without checking `archived` or
  `active_until`, so a superseded decision could resurface through text search.

## Understanding time: what the number was actually measuring

The temporal category looked like this project's worst result (5.6% for the
1.5B reader). A diagnosis of what the reader actually answers showed the number
was largely an artefact of FORM:

| question | gold answer | model answered | token-F1 |
|---|---|---|---|
| When did Evan lose his job? | `end of October 2023` | `2023-10-10` | 0.00 |
| When is the family reunion? | `Summer 2024` | `2024-07-01` | 0.00 |
| When did Calvin first visit Tokyo? | `between 26 March and 20 April 2023` | `2023-04-20` | 0.00 |

Every answer names the right calendar point — October 2023, summer 2024, April
2023 — and scores zero, because token-F1 compares SPELLING. The same fact written
by a machine and by a person shares no tokens.

Two changes, both measured on the same 607 questions with `qwen2.5:1.5b`:

| | record date (A) | event date, ISO (B) | event date + human form |
|---|---|---|---|
| temporal F1 | 3.8% | 2.8% | **5.5%** |
| overall F1 | 30.4% | 32.3% | 32.1% |

1. The memory now hands over the fact's EVENT date (fact_time, resolved from the
   text, anchored to the fact's own record time) with the weekday, in BOTH the
   machine form and the way a person writes it: `[event 2023-10-10 Tue ·
   October 10, 2023]`. Nothing about retrieval or ranking changed — the reader
   simply no longer has to translate, and translating was measured to lose.
2. A second metric is reported NEXT TO F1, never instead of it: **date accuracy**,
   which asks whether the answer names the same calendar point. On the same run:

| metric | value |
|---|---|
| temporal token-F1 | 5.5% |
| **temporal date accuracy** | **63.4%** (52/82) |

So this memory names the correct date in ~2 of 3 temporal questions, while token
F1 reported 5.5%. Both numbers are printed on every run, so the gap can never be
hidden by quoting only the flattering one — and the published 42-46% temporal
figures from other systems are F1, not date accuracy, so they are NOT comparable
to 63.4%.

## Anaphora: the mechanism works, the gain does not (2026-09)

A fact stored as "She opened a studio there" — or, far more commonly, "Caroline: I
just joined a group" — is unreachable by the name a question uses.  Pronouns are now
resolved, in order of certainty: **first person goes to the speaker named by the fact
itself** ("I" in Caroline's turn *is* Caroline; no inference, no dictionary), and
gendered pronouns go to a named participant of the same session whose gender is
known.  The resolution is appended to the indexed text only — the stored text is
never rewritten — so a wrong guess adds one searchable word and can neither hide a
fact nor steal a slot.

Two findings from measuring it:

* the widest anaphora is the first person, not "she"/"he" — most turns are about
  oneself, so the earlier gendered-only design was resolving the rare case;
* **learned genders must be bounded by the sentence.**  The first learner chose the
  nearest capitalised word before any pronoun and produced `sweden=f`, `seeing=f`
  and `caroline=m`.  It now requires the name and the pronoun in the same sentence
  within four words, and no longer guesses at all for "they"/"we"/"it", where the
  first candidate is a coin flip.

Measured on 607 questions it changes nothing that outgrows noise — F1 32.1% → 32.4%
(+0.3, noise ±2 at this sample), evidence-hit@8 62.9% → 62.4%.  **OFF by default**,
kept behind `anaphora_enabled` / `--no-anaphora`, numbers in
`docs/locomo-local-v019-anaphora.md`.  The reason is structural: LoCoMo questions
quote words from the answer turn, so the fact is already reachable lexically, and a
second route to something already found can displace a better hit elsewhere in a
fixed window.  The case it should help — question names a person, fact uses a
pronoun, nothing else overlaps — is absent from the benchmark and is the common case
in real use.

## Reader models: which number is comparable to which

Two readers appear in this project's measurements, and **they are not
interchangeable** — every recorded figure is pinned to the model that produced it:

- **`qwen3.5:9b`** (Ollama, RTX 5060 Laptop) — the historical reference. All
  figures up to and including v0.12.0 were produced with it. Kept for continuity;
  do not mix these with newer ones.
- **`qwen2.5:1.5b`** (~1 GB, Ollama) — the current measuring reader from
  2026-09-14 on, used for a fast loop (a full 600-question run in minutes instead
  of tens of minutes). Its baseline on the same 203-question sample was overall
  F1 28.2%, and the shift was explained rather than assumed: retrieval
  (evidence-hit@8) is IDENTICAL between the two readers (61.1%), so the memory
  hands both the same evidence quality — the difference is reader behaviour, with
  the larger model refusing to answer on 27% of adversarial questions against an
  explicit "always guess" instruction while the smaller one never refuses.

Consequence for reading any number here: a 9B figure and a 1.5B figure may be
compared in DIRECTION, never in absolute value. When a claim matters, both arms of
an A/B are run on the SAME reader, in the same session, on the SAME questions.

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
