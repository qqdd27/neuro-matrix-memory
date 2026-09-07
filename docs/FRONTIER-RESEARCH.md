# NeuroMatrix — Frontier Research & "Next Level" Plan

Survey of the 2025–2026 memory frontier, run 2026-09-07, to take the plugin to
a fundamentally different level. Every proposal is mapped to a concrete,
local-first, LLM-optional implementation — no RL training, no mandatory
embeddings.

---

## 0. Competitive landscape (must-know)

| System | What it is | Notes for us |
|---|---|---|
| **MemOS (MemTensor)** | "Memory OS" for LLM agents; **official local Hermes plugin since 2026-05** (L1 traces → L2 policies → L3 world models → crystallized Skills; FTS5+vector; multi-agent; claims 35% token savings) | Direct competitor in the Hermes plugin slot. Differentiators we own: **bi-temporal decisions & lessons**, **truth-through-time**, deterministic zero-dep core, offline extractive mode. |
| **Mem0 / Hindsight / Honcho / Holographic / Supermemory** | Existing Hermes providers | We are not a cloud vendor; we are the "local temporal-KG + reflection" niche. |
| **Zep Graphiti, HippoRAG 2, EverMemOS, EMem, MemR3** | Research systems (below) | Source of our next mechanisms. |

Positioning sentence: *"MemOS stores what happened and how to do things;
NeuroMatrix stores what is true, what we decided, why we changed our minds,
and what we still don't know — with citations back to the source turn."*

---

## 1. Event-centric atomic memory (EMem, arXiv 2511.17208)

**Finding:** a *simple* baseline — conversation decomposed into **enriched
elementary discourse units (EDUs)**: short self-contained statements with
normalized entities and **source-turn attribution**, organized as a
heterogeneous graph — matches or beats complex pipelines on LoCoMo and
LongMemEval_S **while using much shorter QA contexts**. Non-compressive: keep
detail, make it reachable.

**Why it matters for us:** our `facts` currently store whole messages (coarse
granularity → weak associative precision). The frontier unit is the **atomic
claim with provenance**.

**Adopt (v0.3):**
- sentence-level split at write: one `fact` = one claim (deterministic
  splitting first; LLM refinement later);
- every fact already carries `source`/`session_id` — expose as **citation**
  `[fact #id, session s2]` in every retrieval hit;
- retrieval returns *bundled evidence* (the claims + their source turns), not
  lone fragments → answers can be grounded and verified.

## 2. Agentic retrieval + evidence-gap tracking (MemR3, arXiv 2512.20237)

**Finding:** standard retrieve-then-answer is passive and one-shot; systems
fail because *"they don't know what they're missing"*. MemR3 = closed-loop
control: a **router** chooses retrieve / reflect / answer, and a **global
evidence-gap tracker** makes the answering process transparent.

**Adopt (v0.3, LLM-optional, budgeted):**
- `ask` becomes iterative: retrieve → cheap LLM assesses "can I answer? which
  criteria/claims have no evidence?" → one targeted re-retrieval (max 2
  rounds) → answer **explicitly lists gaps** ("offline requirement: no
  recorded criterion").
- without LLM: gap list is structural — criteria present in past decisions
  but absent in current context (already planned in v0.2 §9; this makes it
  *closed-loop*).

## 3. Foresight signals (EverMemOS, arXiv 2601.02163)

**Finding:** EverMemOS (SOTA on LoCoMo/LongMemEval) stores, alongside episodic
traces and atomic facts, **time-bounded Foresight signals** — statements about
the future that later become relevant ("user plans X by date Y"; "watch for
Z after W"). This is the user's original "thinking ahead" wish, done as data
instead of 100 templates.

**Adopt (v0.3/v0.4):**
- `kind=foresight` fact with `trigger_at` + window; surfaced when the topic
  reappears *or* when now ≥ trigger;
- optional bridge to Hermes **cron** (`hermes neuromatrix remind` → scheduled
  job wakes the agent with due foresights — "Dreaming/reminding" as a job).

## 4. Procedural memory & outcome-tagged task episodes (Memp 2508.06433, MemTool 2507.21428, mem-agent)

**Finding:** the third memory axis (procedural) is where agents actually get
better: remembering *how a task type was solved*, with outcomes, and
crystallizing it into reusable skills — tiered evolution (MemOS L1 traces →
L2 policies → L3 skills; Voyager/Reflexion lineage).

**Adopt (v0.4):**
- `kind=procedure`: distilled at session end when a task completed
  `{task_type (verb+entities), steps summary, outcome: success|failed|partial,
  tools_used}` (LLM) or deterministic template (no key);
- on a new task with matching pattern → procedure surfaced; repeated success →
  **skills suggestion returned to the host** (Hermes skills are the natural
  crystallization target; provider already supports `register_skills`);
- our `on_delegation` hook (v0.2 §15) feeds subagent outcomes in.

## 5. Multi-profile shared memory with scopes + audit (Collaborative Memory, arXiv 2505.18279)

**Finding:** multi-agent/multi-user memory = two tiers (**private** +
**shared**) with asymmetric permissions, immutable **provenance** on every
fragment, and full auditability. Hermes has profiles/subagents/bots — memory
is per-profile today; sharing requires an explicit scope layer.

**Adopt (v0.4):**
- `scope` (private|shared) + `owner` columns; optional shared pool via config
  `workspace_id` (Honcho-style);
- every write already records provenance → **audit view** of memory ops;
- read policy: shared fragments visible across profiles in the workspace;
  private never leaves the profile.

## 6. Learned memory *policy* interface — RL-ready but no RL today (Memory-R1 2508.19828, Agentic Memory 2601.01885)

**Finding:** the frontier is **learned** management — RL (PPO/GRPO) agents
outputting memory operations (ADD/UPDATE/DELETE/NOOP) and pre-selecting
evidence (Memory-R1: beats baselines with only 152 training pairs; Agentic
Memory: unified STM/LTM ops via GRPO). Training is out of scope for a local
plugin — **but the interface is not**.

**Adopt (v0.3):**
- formalize a **memory-ops layer** in code: every mutating action is an
  operation log entry `{op: ADD|UPDATE|DELETE|NOOP|SUPERSEDE, fact_id, ts,
  reason}` — the exact action space RL policies consume;
- default policy = current heuristics + optional cheap-LLM policy at decision
  points (Mem0-style ADD/UPDATE/DELETE from the conversation), `NOOP` default
  → cost control;
- future-proof: a trained policy can later slot in behind the same ops log.

## 7. User-profile distillation & reviewable memory (PersonaMem-v2, mem-agent files layout, "memory must be reviewable")

**Finding:** implicit personas distilled from scattered preferences
(PersonaMem-v2 / EverMemOS profile study) + memory exposed as **human-editable
files** (mem-agent: `memory/user.md`, `memory/entities/<name>.md`) +
governance principle: *consolidation is not neutral — output must be
reviewable* (Ken Huang on Anthropic Dreaming).

**Adopt (v0.3/v0.4):**
- `kind=profile` living "user card": preferences/goals/decisions distilled by
  consolidation (LLM) or extractive;
- **markdown mirror**: `neuromatrix export --markdown` / on-consolidate
  regeneration of a readable `memory/` folder (entities, dossiers, decisions)
  the user can audit and edit → edits import back (op log UPDATE);
- fits Hermes dashboard memory panel surface.

## 8. One engine, two surfaces: Hermes provider + MCP (strategic)

**Finding:** *"MCP is emerging as the standard interface for memory sharing
between agents — dedicated memory MCP servers make persistent memory available
to any MCP-compatible agent regardless of framework."*

**Adopt (v0.4+, strategic):** expose the same `store.py` core behind a small
MCP memory server (`memory_search`/`memory_add`/`memory_decide`/...) so any
MCP client (other agents, the owner's other products) reuses one knowledge
graph. The provider and the MCP server share the DB — no second source of
truth.

---

## Deliberately out of scope

- RL training of memory policies (Memory-R1/Agentic Memory) — we build the
  ops interface, not the training run;
- mandatory vector/embedding pipeline — hybrid FTS5+graph stays primary;
  optional vector rerank only;
- cloning MemOS's L1/L2/L3 machinery — we differentiate on truth/time/lessons.

## Suggested roadmap deltas

| Release | Additions beyond v0.2 spec |
|---|---|
| v0.3 | §1 atomic claims + citations; §2 agentic `ask` w/ gap tracker; §6 memory-ops log; §3 foresight kind |
| v0.4 | §4 procedures→skills; §5 shared scopes + audit; §7 profile card + markdown mirror |
| v0.4+ | §8 MCP memory surface; §3 cron reminders |
