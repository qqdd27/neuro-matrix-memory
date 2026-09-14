# Model-extracted event dates: implemented, measured, not worth it here

**Idea (from Mem0).** They attach a time signature to *every* memory at write time.
Our pattern resolver only dates what it can match, so the gap should close by asking a
model about the turns it cannot resolve — "right after the Japan trip", "the week I
started the new job".

**Result.** It closes almost nothing on this corpus, and the reason is not the model.

## What was built

* `temporal.resolve_with_llm(llm, text, anchor_ts)` — one JSON question per turn
  ("when did this happen?"), with an explicit licence to answer **null**, and an
  explicit prohibition on falling back to the conversation date.
* `store.enrich_event_times_via_llm(limit, batch)` — background-shaped: bounded per
  call, only over facts never examined, and every examined fact is recorded in
  `fact_time_checked` **whatever the answer was**. Without that record a pass re-asks
  about the same facts forever: expensive, silent, and indistinguishable from a
  working feature.
* A guard against the exact failure the prompt warns about: a day-precision date equal
  to the conversation date, in a turn the pattern resolver found no time in, is
  discarded. A fabricated date would be inherited by every later "when" question.

## Measurements

Whole first conversation (419 facts, 61 already dated by rules — 14.6%):

| | value |
|---|---|
| facts examined by the model | 358 |
| facts that gained a date | **7** (+1.6 pp → 16.2% coverage) |
| time | 84 s |

Then a cleaner probe — 60 turns that rules could *not* date but that contain a time
word ("since", "after", "next month"):

| model | dates found | time | quality |
|---|---|---|---|
| qwen2.5:1.5b | 5 / 60 | 15 s | some invented (a date for a turn naming no time at all) |
| qwen3.5:9b | **2 / 60** | 43 s | careful, and almost always answers "this is not a date" |

## Why — and it is not a model-quality problem

The 9B model is *more* conservative than the 1.5B one and finds fewer dates, which is
the honest answer: most of those turns genuinely do not contain a date. "since we last
chatted", "after the trip", "when I came across it" are not dates without knowing when
the other event happened — and the memory does not know either. Extrapolated, a full
14-conversation pass on the 9B model costs roughly 15-20 minutes of model time to add
dates to a low single-digit percentage of facts.

This is the same ceiling the rule-based work found: **only ~15% of turns in this corpus
contain any time expression at all**, and rules now resolve 79.5% of those. The
remaining turns are not undated because our extraction is weak; they are undated
because they have no date.

## Verdict

**OFF by default** (`temporal_llm_enabled`, `--llm-dates`), kept with its measurements
like every other rejected idea in this project. The infrastructure is real and works —
`fact_time_checked` in particular is what makes such a feature safe to run at all — so
if a corpus with more relative time ("yesterday I…", "last sprint we…") ever arrives,
the switch is one flag away.

## Also corrected here

I suspected the benchmark rendered dates differently from the live agent, which would
mean we were improving the product and measuring something else. Checked:
`eval_locomo._ctx_line` and `provider._format_hit` produce the same rendering —
`[event 2023-10-10 Tue · October 10, 2023]`. No discrepancy; the temporal gap is not
a presentation mismatch.
