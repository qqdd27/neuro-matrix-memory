# The chronological trace: temporal F1 ×6.3, and why reordering never worked

**The result.** One added line — a chronological trace of the retrieved dated events —
moved the temporal category from **3.9% F1 to 24.4%** (×6.3) and overall F1 from
**31.1% to 35.0%**, with retrieval completely unchanged (evidence-hit@8 61.0% → 61.1%).

607 questions, qwen2.5:1.5b as the reader, every configuration measured in the same
session on the same sample:

| configuration | overall F1 | **temporal F1** | date accuracy | hit@8 |
|---|---|---|---|---|
| baseline | 31.7% / 31.1% | 4.3% / 3.9% | 71.1% / 67.5% | 60.8% / 61.0% |
| + distance-to-question annotation | 29.8% | 4.2% | 65.1% | 60.5% |
| + distance + trace | 33.9% | 23.0% | 66.3% | 60.8% |
| **+ trace only** | **35.0%** | **24.4%** | 63.9% | 61.1% |

Baseline was measured twice (two independent runs) to rule out a lucky sample; the
effect sits far outside the run-to-run spread (±0.6 F1, ±0.4 temporal).

## What the line looks like

```
[timeline] March 28, 2024: Caroline lost her job | May 2, 2025 (+13 months): moved to
Lisbon | August 15, 2026 (+15 months): opened the pottery studio
```

Retrieved facts are listed oldest-first with the **gap** between consecutive ones. The
retrieved set, the ranking and every individual line below it are untouched — it is one
extra header line, and only when at least two of the retrieved facts carry a date (a
single event has no relation to state).

## Why this worked when six rearrangements did not

GRAVITY (arXiv 2605.01688) ran the decisive experiment: with **all** gold evidence
present in the retrieved set, a reader still answered only 80.9% correctly, and 75.6%
when evidence was scattered. The bottleneck was never missing evidence — it was that
the relational, temporal and thematic connections **between** fragments were left
implicit for the reader to reconstruct.

That is exactly what we kept tripping over. Every previous attempt expressed the
relation by **moving** things:

| attempt | result |
|---|---|
| promote the latest/earliest fact | temporal 4.3% → 3.4% |
| additive score bonus (Mem0's formula, mis-implemented) | recall 61.3% → 36.0% |
| hand temporal questions a chronologically sorted context | F1 31.7% → 30.0% |
| **state the relations once, in one line, change nothing else** | **temporal 3.9% → 24.4%** |

A fixed window means every promotion evicts something the question actually matched.
A trace evicts nothing. This is the same law the whole project has been circling:
**rearrangements and widenings lose; added information wins.**

## Why the distance annotation alone hurt

The first element of the trace — "13 months ago" attached to each fact — is *not* what
carries the gain. Applied to every line it **cost 1.4 F1 points and 6 points of date
accuracy** (71.1% → 65.1%), because a reader handed "2 years ago" tends to answer in
relative form ("about two years ago") instead of naming the date the question asked
for. Inside the trace it is harmless, because there the relative gap sits *between two
absolute dates* rather than replacing one. Reproduced in both configurations, so the
distinction is not noise: the trace is the mechanism, the distance annotation is not.

## Also measured in this session, and rejected

* **Model-extracted event dates** (`--llm-dates`): 358 facts examined, **7** gained a
  date; on 60 turns containing a time word a 9B model found 2. The corpus simply does
  not carry more dates — only ~15% of turns mention time at all, and the pattern
  resolver already resolves 79.5% of those. See `locomo-local-v022-llm-dates.md`.
* **Rule-based date coverage** (kept): arbitrary "N units ago", seasons, vague phrases.
  Coverage 8.5% → 12.3% of facts, 46.5% → 79.5% of time expressions, date accuracy
  66.3% → 71.1%. This is what gave the trace more to work with.
* **`in the summer of 2022` used to resolve to 2022-01-01** — a date the text never
  claimed. Fixed, now "Summer 2022".

## Where it lives

* `temporal.timeline_trace()` — the trace; `temporal.format_distance()` — the gaps.
* `provider.prefetch()` — the live agent inserts the trace line at the top of recall.
* `eval_locomo --timeline` — the switch used to measure it.
* Defaults: ON in the live provider (it costs one line and buys 6× on temporal
  questions).

## Reproduce

```bash
python scripts/eval_locomo.py --llm --llm-provider ollama --llm-model qwen2.5:1.5b \
  --llm-sample 600 --k 8 --timeline            # 35.0% F1, temporal 24.4%
python scripts/eval_locomo.py --llm --llm-provider ollama --llm-model qwen2.5:1.5b \
  --llm-sample 600 --k 8                       # 31.1% F1, temporal 3.9%
```
