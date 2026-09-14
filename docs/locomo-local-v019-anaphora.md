# Anaphora: measured neutral — the mechanism works, the gain does not

**Question asked:** a fact stored as "She opened a studio there" (or "Caroline: I
just joined a group") is unreachable by the name a question uses.  Resolving the
pronoun should make it reachable.  Does it?

**Answer:** the resolution works; on LoCoMo it buys nothing measurable.

## Why this was worth building anyway

The first design resolved only gendered pronouns ("she"/"he").  Diagnostics on real
conversations showed two things at once:

1. **the widest anaphora is the first person.** Most turns are about oneself, and the
   fact itself names the author — `Caroline: I just joined a new group`.  "I" is then
   *that* name with **no inference at all**, no dictionary and no guessing.
2. **learned genders are dangerous.** The first learner took the nearest capitalised
   word before any pronoun and produced `sweden=f`, `seeing=f` and even
   `caroline=m` (a name in one sentence, a pronoun belonging to somebody else in the
   next).  It was tightened to require the name and the pronoun in the SAME sentence
   within four words; gendered-only resolution also stopped guessing on
   "they"/"there"/"it", where the first candidate is a coin flip.

A regex bug was found by the same diagnostic: the word pattern required two or more
characters, so **"I" (and Russian "я") could never match** — the whole first-person
path was silently dead until it was fixed.

## What the resolution produces

| fact | added to the indexed text |
|---|---|
| `Caroline: I just joined a new group` | `i Caroline` |
| `Melanie: my sister and I moved` | `my Melanie i Melanie` |
| `Каролина: я переехала в Берлин` | `я Каролина` |
| `Evan: we adopted a dog` | `we Evan` |

The stored text is never rewritten — only the index the fact can be found by.

## Measurement (607 questions, qwen2.5:1.5b)

| metric | without | with | Δ |
|---|---|---|---|
| evidence-hit@8 | 62.9% | 62.4% | −0.5 |
| **answer F1** | 32.1% | **32.4%** | **+0.3** |
| date accuracy | 63.9% | 66.3% | +2.4 |
| single-hop | 23.8% | 24.3% | +0.5 |
| temporal | 4.3% | 5.1% | +0.8 |
| multi-hop | 15.0% | 17.5% | +2.5 |
| open-domain | 42.9% | 43.6% | +0.7 |
| adversarial | 40.4% | 39.2% | −1.2 |

Full population (1977 questions, retrieval only): 61.2% → 61.3% — three questions.

## Verdict

**Rejected as an improvement, kept as code.** +0.3 pp F1 at n=607 is inside the ±2
noise band, and it costs 0.5 pp of retrieval.  Recorded the same way as role-bridge,
type-boost and edge-order: the flag exists (`anaphora_enabled`, `--no-anaphora`),
the default is OFF, and the numbers are in this file.

## Why it likely does nothing here

LoCoMo's questions quote words from the answer turn ("what group did Caroline
join?" against a turn about joining a group), so the fact is already reachable
lexically.  Resolution adds a second route to something already found, and in a
fixed window a second route can displace a better hit from another question.

Where it should matter is the opposite case: **the question uses a name, the fact
uses a pronoun, and nothing else overlaps** ("what did the user decide about the
cache?" against "I decided to drop the cache").  That case is not represented in
LoCoMo at all, and it is closer to how this memory is actually used than the
benchmark is.

## Reproduce

```bash
python scripts/eval_locomo.py --k 8 --no-anaphora          # 61.2% full population
python scripts/eval_locomo.py --k 8                        # 61.3% (flag forces on)
python scripts/eval_locomo.py --llm --llm-provider ollama \
  --llm-model qwen2.5:1.5b --llm-sample 600 --k 8 --no-anaphora
```
