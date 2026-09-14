# Event-time resolution + type_boost: measured, NOT shipped (2026-09-14)

## Hypothesis

Temporal is the weakest category by a wide margin (F1 ~6-8% against 18-46%
for comparable systems on the same question set), while its retrieval is
already strong and rising (evidence-hit@8 65.9-76.9% across recent versions).
The diagnosis (see `scripts/eval_locomo.py:143-148,211-215` comments and the
version history in this directory) is that **retrieval is not the
bottleneck** — the correct fact is usually already in the window — so the
loss happens at ranking/answer time: `question_types.py`'s existing
`type_boost` mechanism promotes "when"-shaped facts inside the top-16 using
crude regexes (`_YEAR`, `_MONTHS`, `_WEEKDAYS`, `_RELATIVE_TIME`), which
cannot tell a genuine, resolvable calendar date from an unrelated four-digit
number ("2019 dollars") or a bare weekday with nothing to anchor it to a
specific day.

**Hypothesis under test:** replacing that shape-only regex with a real,
resolved event date (using the fact's own record time as the anchor for
relative/partial expressions) would make `type_boost`'s reordering more
precise and raise temporal F1, without touching retrieval or the frozen
reader prompt.

## What was built

`neuro_matrix/temporal.py` — a new, zero-dependency module (pure stdlib
`datetime`/`re`, unlike `morphology.py` it needs no optional package and has
no availability gate). Public surface:

- `resolve(text, anchor_ts=None) -> Resolved(event_ts, granularity) | None`
- `has_event_date(text, anchor_ts=None) -> bool`

Same two design properties as `morphology.py`, deliberately:

- **Narrows, never widens.** `resolve()` either returns a real point on the
  timeline or `None` — a fact with no resolvable date stays exactly as
  unfindable-by-date as before.
- **Anchor discipline.** Relative ("last week") and partial ("on Tuesday",
  "March 5" with no year) expressions are meaningless without a reference
  point; the only honest anchor for a stored fact is the moment it was
  *recorded* (`anchor_ts`), never wall-clock "now" — resolving against "now"
  would make the same fact's date drift between runs.

Coverage: full dates ("March 5, 2019", "5 марта 2019"), month+year ("May
2023", "в мае 2023"), day-without-year using the anchor's year, relative
expressions (the same delta table `scripts/eval_locomo.py` already used
bench-side, kept here as the single source of truth), bare weekdays resolved
to the nearest occurrence around the anchor, and a bare year as the lowest-
precision fallback. Unit-tested in
`tests/test_core.py::test_temporal_resolve_absolute_and_relative_dates`.

**Known, accepted limitation, at parity with the regex it was meant to
replace:** a bare year with no date-like context ("he has 2019 dollars
saved") still resolves as `granularity="year"` — same false-positive class
the old `_YEAR` regex already had. Not a regression; not fixed here either
(see "Next" below).

## Where it was wired (and then unwired)

`question_types.matches_type()`'s `"when"` branch was changed to call
`temporal.resolve(text, ts)` first, falling back to the old regexes only
when no candidate timestamp was available. `question_types.boost()` and
`store._apply_fusion()`'s `type_boost_enabled` branch were changed to thread
each candidate's own `ts` through so the resolver would have an anchor. This
touches nothing else: `type_boost_enabled` already defaults to `False` in
`store.py`, so the change was inert for every caller unless a benchmark run
opted in with `--type-boost`.

## Methodology

Same-sample A/B on the external LoCoMo benchmark (`scripts/eval_locomo.py`,
10 conversations, local `qwen3.5:9b` reader via Ollama, default
`--llm-sample 200` / `--llm-seed 42`, which drew 203 scored questions from
this run). Both arms ingested the same conversations and answered the exact
same 203 questions, so any difference is attributable to the ranking change
alone, not to sampling variance between separate runs.

- Baseline: current shipped defaults (lemma channel on, `type_boost_enabled`
  off).
- Treatment: `--type-boost --type-boost-weight 1`, using the precise
  resolver above instead of the shape regexes.

## Results

Evidence-hit@8 (retrieval; expected to be nearly flat — the metric cannot
see a reorder inside an already-correct top-16, only a change to the top-8
*set*):

| Category | baseline | + precise type_boost (w=1) |
|---|---|---|
| single-hop | 51.7% | 51.7% |
| temporal | 76.9% | 76.9% |
| multi-hop | 12.5% | 12.5% |
| open-domain | 61.5% | 60.3% |
| adversarial | 61.2% | 61.2% |
| **overall** | **61.1%** | **60.6%** |

QA-accuracy (F1), n=203 sampled, same questions both arms:

| Category | n | baseline F1 | + precise type_boost (w=1) |
|---|---|---|---|
| single-hop | 29 | 24.4% | 24.9% |
| **temporal** | 39 | **6.2%** | **5.6%** |
| multi-hop | 8 | 12.5% | 6.7% |
| open-domain | 78 | 40.4% | 40.3% |
| adversarial | 49 | 23.2% | 26.6% |
| **overall** | 203 | **26.3%** | **26.8%** |

## Interpretation

**The hypothesis did not hold.** Temporal F1 — the one number this change
targeted — moved in the wrong direction (6.2% → 5.6%). The apparent overall
gain (26.3% → 26.8%) is not evidence of anything working: it is carried by
adversarial (n=49, +3.4pp) and single-hop (n=29, +0.5pp), and the multi-hop
drop (12.5% → 6.7%, n=8) is literally one question flipping — at that sample
size neither move clears noise. Weight 2/3 were not swept: the direction was
already wrong at the metric this exists for, and the project's own
discipline is not to chase a positive-looking aggregate that contradicts the
targeted category.

**Why, most likely:** `type_boost` is a *reorder-only* mechanism — it moves
facts inside a fixed top-16 window by 1-2 positions. Better date precision
gives it a more trustworthy signal for *which* facts look temporal, but the
underlying problem this project already diagnosed is not "the wrong fact is
in the window" (retrieval is strong and this experiment leaves it exactly
unchanged, as the hit@8 table confirms) — it is what the *reader* does with
an already-correct window. A more precise reorder cannot fix a
generation-time bottleneck, which is consistent with every other reorder
experiment in this project's history (`edge_order`, the original
`type_boost`, the role bridge) landing at best mixed and at worst negative.

## Decision

Not shipped. `question_types.py` and `store.py` are reverted to their
pre-experiment state (`type_boost` still uses the shape regexes, unchanged);
`type_boost_enabled` remains `False` by default either way, so production
behaviour was never at risk. `neuro_matrix/temporal.py` is kept: the
resolver itself is validated (unit-tested, zero-dependency, correctly
anchored) and is infrastructure for the next attempt, not the thing that
failed — what failed was gluing it to a reorder-only lever.

## Next

Per the project's standing diagnosis, the lever that is actually likely to
move temporal F1 is additive, not reorder: a genuine RRF candidate channel
(the same shape as the lemma index — `_temporal_candidates` beside the
existing lists, gated so it only activates for `qtype == "when"`), scored by
proximity between a fact's resolved `event_ts` and a date the *question*
itself names (extracted the same way, via `temporal.resolve()` applied to
the question text — not built yet, since without it there is nothing to
score proximity against). That requires, in order: (1) resolving dates out
of question text, (2) a persistent `event_ts` per fact (a side table +
backfill, mirroring `fact_lem`) so the channel does not have to re-resolve
on every query, (3) the RRF channel itself, weight-swept the same way the
lemma channel was. None of this is built yet — this document closes the
reorder-only branch so the next session does not re-attempt it.
