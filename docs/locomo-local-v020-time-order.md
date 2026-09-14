# "Last time" / "first time": rejected — the idea is sound, the data is not there

**Question asked:** when a question says *last time* or *first time*, it is not asking
for the most similar fact but for the **extreme by date**.  The memory now stores the
event date of every fact, so it can find that extreme deterministically, without the
model.  Does that help?

**Answer:** no — and the reason is worth more than the experiment.

## What was built

`temporal.order_intent(query)` detects the ask in both languages ("last", "latest",
"most recent", "first", "earliest", "последний", "впервые", …).  When it fires,
`store._apply_time_order` takes the top of the ranking (never the whole result set),
reads the event time of each candidate (`fact_time`, falling back to the record time),
picks the maximum for *last* and the minimum for *first*, and promotes **that one fact
to the front**.  Every other position stays exactly where the ranking put it.

Not a re-sort, on purpose: measured reorderings (edge order, type routing) lost to
leaving the window alone, and a full sort by date would bury a precise hit under an
old-but-dated one.  Relevance picks the pool; the date only picks the extreme inside
it.

## Measurement

**Full sample (607 questions)** — the mechanism acts on 112 of them.

| metric | without | with | Δ |
|---|---|---|---|
| answer F1 | 30.3% | 31.4% | +1.1 |
| evidence-hit@8 | 62.4% | 62.1% | −0.3 |
| date accuracy | 67.5% | 65.1% | −2.4 |
| single-hop | 20.7% | 24.5% | +3.8 |
| temporal | 3.9% | 2.9% | −1.0 |
| multi-hop | 8.9% | 19.1% | +10.2 (n=22) |
| open-domain | 42.0% | 42.5% | +0.5 |

+1.1 overall is inside the ±2 noise band, and it grew where the mechanism does not act
(temporal *fell*).  So the mechanism was measured on its own territory, the 112
questions that actually ask for an extreme:

| metric (112 target questions) | without | with | Δ |
|---|---|---|---|
| answer F1 | 40.8% | 40.9% | +0.1 |
| evidence-hit@8 | 62.2% | 61.3% | −0.9 |
| temporal (n=23) | 11.3% | 9.9% | −1.4 |
| open-domain (n=56) | 53.6% | 53.6% | 0 |

Zero on its own subset.  The mechanism does fire (verified directly: 1 of 1 intent
questions per conversation got a promotion), so this is a real null result, not a
disconnected feature.

## Why: a data ceiling, and it caps the whole temporal direction

| measurement | value |
|---|---|
| facts with an **event date** | **124 / 1451 = 8.5%** |
| facts containing a time expression | 260 / 1451 = 17.9% |
| …of those, where a date was extracted | 121 / 260 = 46.5% |
| …where it was not | 139 — mostly relative ("recently", "last time") or seasonal ("this summer") |

For the other ~91% of facts the fallback is the **record time**, which is identical
for every fact written in one session.  "Latest" therefore degenerates into "from the
last session", which is not what the question asked.  The order mechanism is only as
good as the dates underneath it, and those cover one fact in twelve.

## Verdict

**Rejected, OFF by default**, kept behind `time_order_enabled` / `--no-time-order`,
same as role-bridge, type-boost, edge-order, and anaphora.  Numbers here.

## What this actually tells us

The temporal direction is not blocked by reasoning but by **date coverage**: 8.5% of
facts are dated, and half of the time expressions in the corpus never become dates at
all.  That is a concrete, measurable place to work — and unlike widening the retrieval
window, extracting more dates adds information rather than candidates.  Until then,
every "when" question is answered from a window where at most one or two facts carry
a real date.

## Reproduce

```bash
# full sample
python scripts/eval_locomo.py --llm --llm-provider ollama --llm-model qwen2.5:1.5b \
  --llm-sample 600 --k 8 --no-time-order
python scripts/eval_locomo.py --llm --llm-provider ollama --llm-model qwen2.5:1.5b \
  --llm-sample 600 --k 8

# only the questions that ask for an extreme
python scripts/eval_locomo.py --llm --llm-provider ollama --llm-model qwen2.5:1.5b \
  --only-intent any --k 8 --no-time-order
python scripts/eval_locomo.py --llm --llm-provider ollama --llm-model qwen2.5:1.5b \
  --only-intent any --k 8
```
