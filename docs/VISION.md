# NeuroMatrix — product vision

> Anchored by the owner (2026-09-08). This is the "why" — the spec
> (`docs/V0.2-SPEC.md`) and frontier research (`docs/FRONTIER-RESEARCH.md`)
> carry the "what/how".

## Thesis

**A memory that does not store — it understands.**

Most agent memory is a warehouse: facts in, facts out, no structure. NeuroMatrix
is a **temporal knowledge graph with judgment**: it links entities into an
association web (паутина), records decisions *and the reasons they changed*,
weakens what failed, consolidates what repeated — so with every passing day the
agent answers better, because it *knows what we are doing and why*, with
citations back to the original dialogues.

Positioning vs. built-in Hermes memory and vs. MemOS-class plugins:

> Not a list of durable strings (MEMORY.md), not a vector store, not a trace →
> policy ladder — but a system that remembers **what we decided, why we changed
> our minds, and what already failed** — with provenance
> (`[fact #N, session …]`).

## Pillars

1. **Associative web** — cross-session entity linking, aliases (`id_777 ↔ TON`),
   weighted decaying edges, inhibition of superseded paths.
2. **Truth & time** — decisions evolve (decide → supersede → lesson with the
   criterion shift), `as_of` point-in-time views, per-kind lifecycle
   (episodic decays; durable dies only via supersede/removal).
3. **Learning from outcomes** — corrections damp the wrong path, feedback
   reinforces, dead-ends become structural (roadmap), the nightly "sleep"
   (consolidation, SHY, surprise gate) re-organises instead of merely storing.
4. **Economy** — recall budgets, score gates, artifacts-on-disk, daily LLM
   budget: memory must not eat the context window.
5. **Reviewable** — markdown mirrors, ops journal, persona card: the human can
   always audit what the machine believes and why.

## North-star metric

**Recall@k on the owner's own sessions**: of N real questions about past work
("what did we decide about GSC?", "why did we switch to Postgres?"), how many
does the memory answer from its top-k without the agent re-searching? Baseline
is measured by `scripts/eval_recall.py`; every feature ships against this
number, not against vibes.

## Roadmap (next, in order)

1. Eval harness + first measured number on live data.   *(this release)*
2. Outcome tagging — attempts get structural dead-end marks (failed + reason).
3. `breakthrough` tool — at a dead end, return (a) why past attempts failed,
   (b) analogies from unrelated graph clusters, (c) hypotheses that flip the
   underlying assumption.
4. Typed edges between decisions (AFFECTS / REQUIRES / BLOCKS) — the causal web.

Beyond: skills crystallisation (Memp), shared scopes at scale, learned memory
policy over the ops journal (RL-ready data already recorded).
