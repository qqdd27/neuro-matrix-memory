# Temporal direction: what the field does, what we tried, what actually worked

The temporal category is this project's weakest (F1 5.5% against 63.4% date
accuracy).  Rather than keep guessing, this session went to the published work, took
the parts that fit our constraints, implemented them, and measured each one.

## What the field does (three sources that matter)

**Mem0 — "Temporal Reasoning" (May 2026).**  Their numbers: +4.1 pts overall on
LoCoMo, +6.7 on temporal questions, +9.1 at top_50.  The mechanism is not a reorder:

* every memory gets a **time signature** at write time (when it happened, whether it
  is ongoing or completed, precision, and a memory *type*: event, state, plan,
  relationship, preference, absence);
* queries get a **temporal intent** classified at read time with no LLM call;
* the date signal is **additive** — their words: *"Temporal scoring is additive: it
  nudges ranking toward the right dated instance; semantic relevance always
  dominates"*;
* and it needs **more candidates in the pool**, because a date only discriminates
  among near-identical alternatives;
* plus a `state_key` so an evolving fact ("lives in Austin" → moved) closes the old
  instance's window instead of deleting it.

**Zep / Graphiti (arXiv 2501.13956).**  Bi-temporal: every fact carries when it was
true in the world *and* when it was ingested; contradictions invalidate rather than
delete.  We already store both timestamps (`fact_time.event_ts` and `facts.ts`).

**GRAVITY (arXiv 2605.01688).**  The finding that reframes the problem: in an oracle
experiment where **all** ground-truth evidence is present in the retrieved set,
accuracy still reaches only 80.9%; scattered among retrieved entries it drops to
75.6%.  The generator fails not because evidence is missing but because the
relational, temporal and thematic connections **between** fragments are never made
explicit.  Their fix is to inject structured anchors — entity profiles, and event
tuples ("who did what, when, where, with what outcome") linked into chronological
traces.

## What we implemented and measured

| approach | source | result | verdict |
|---|---|---|---|
| promote the extreme fact to the front | our own | hit@8 62.2% → 61.3% on the 112 target questions | rejected |
| **additive** date bonus on the score | Mem0 | recall collapsed 61.3% → 36.0% | **implementation bug**, see below |
| chronological context for temporal questions | GRAVITY | F1 31.7% → 30.0%, temporal 4.3% → 3.4% | rejected |
| **more dates extracted (information)** | own diagnosis | **date accuracy 66.3% → 71.1%** | **kept** |

### The bug worth remembering

The additive bonus was applied by re-sorting the candidate list on the `score`
field.  After channel fusion that field **no longer matches the order** — fusion ranks
by RRF, while `score` keeps the pre-fusion heuristic value.  Sorting by it silently
undid the fusion: recall fell from 61.3% to 36.0% on the target subset.  A correct
version must act on the fused order (or inside the fusion), never on that stale
number.  The code is kept, off, with this written down.

### Why both reorderings lost

Same reason as the seven window-widening experiments before them: in a fixed window,
moving one item up pushes another out, and the item pushed out is usually the precise
hit the question matched.  Chronological context has the same effect for a 1.5B
reader — it is fed "oldest first" and loses the strongest fact from the top position.

## What actually worked: date coverage

Diagnosis first: of 1451 facts, only **124 (8.5%)** carried an event date, and of 219
facts containing a time expression only **46.5%** produced a date.  The resolver had a
**fixed table** — "two weeks ago", "last month" — so the general form was silently
unsupported:

| expression | before | after |
|---|---|---|
| `3 days ago` | no date | 2026-09-07 (day) |
| `five weeks ago` | no date | 2026-08-06 (day) |
| `this summer` | no date | Summer 2026 (season) |
| `in the summer of 2022` | **2022-01-01 — a date the text never claimed** | Summer 2022 |
| `recently` | no date | ~7 days back (week granularity) |
| `два месяца назад` | no date | July 2026 (month) |
| `летом 2022` | no date | Summer 2022 |
| `winter 2021` | no date | Winter 2021 |

Coverage after: **179/1451 = 12.3%** (was 8.5%); of facts with a time expression,
**174/219 = 79.5%** now resolve (was 46.5%).  False positives checked explicitly — a
bare number ("3 apples"), a modal verb ("you may want") and ordinary prose still
produce no date.

Measured effect: **date accuracy 66.3% → 71.1%** on the 607-question sample.  Token-F1
did not move (31.7% vs 32.1%, inside noise), which is the known limitation of the word
metric, not evidence against the change.

## The rule this confirms

Every rejected idea in this project has been a **rearrangement or a widening** —
graphs, PMI edges, bridges, role bridge, type routing, edge order, splitting, anaphora
promotion, time ordering, chronological context.  Every kept idea has been
**information or reachability** — the lexical channel, lemmas beside the surface
index, predicate roles, and now more dates.  The temporal category is blocked by
coverage, not by clever ordering.

## Sources

* Mem0, "Introducing Temporal Reasoning in Mem0" (May 2026) — additive temporal
  scoring, 7 memory types, temporal intent without an LLM call.
* Zep / Graphiti, arXiv 2501.13956 — bi-temporal validity windows, invalidation
  instead of deletion.
* GRAVITY, arXiv 2605.01688 — retrieval is not the bottleneck; explicit structure in
  the context is (oracle 80.9% vs scattered 75.6%).
