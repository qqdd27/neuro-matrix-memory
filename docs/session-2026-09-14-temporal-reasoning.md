# Session report — temporal reasoning research & experiments (2026-09-14)

Full record of one working session: git history cleanup, a literature survey
on temporal reasoning in agent memory, a research-to-architecture mapping,
one implemented-and-measured experiment (reverted), one new permanent
diagnostic tool, and one implemented-and-measured *infrastructure* finding
(reader-model sensitivity) that is informational, not a memory change.
Written so a later session does not re-derive or re-attempt any of this from
scratch.

---

## 0. Git history: Claude co-authorship removed

Unrelated to memory work, done first in this session. `main` had 13/57
commits carrying `Co-Authored-By: Claude ...` / `🤖 Generated with [Claude
Code]` trailers, already pushed to `origin/main` (GitHub). Rewrote all 57
commit messages with `git filter-branch --msg-filter` (stripped the two
trailer patterns, case-insensitive), verified the tree diff between the old
and new history is empty (only commit messages changed, no file content),
force-pushed with `--force-with-lease`, then pruned the `refs/original/`
backup refs and ran `git gc --prune=now`. Confirmed zero remaining matches
for `Co-Authored-By|Generated with` across `git log --all`.

---

## 1. Starting point: where the project actually was

Reviewed the last 5 commits (v0.12.0 lemma index work) and the two existing
"language laws" already shipped:

1. **Roles from case** (`neuro_matrix/propositions.py`, v0.11.0) — predicate
   + semantic roles read from Russian case morphology / English word order,
   +1.1pp retrieval, answer-F1 multiplier ×3.1→×3.6.
2. **Lemma = concept** (`neuro_matrix/morphology.py`, v0.12.0) — WordNet
   lemmatisation (EN) + pymorphy (RU), a SEPARATE RRF channel beside the
   surface FTS index (never replacing it — replacing raised retrieval but
   lowered answer quality), +5.0pp retrieval.

The user supplied an external comparison table (their memory vs. LoCoMo
baselines vs. A-Mem/LangMem/Zep/OpenAI Memory, F1 scale) showing **temporal
is the weakest category by a wide margin**: ~8.5% F1 for this project against
18.4-45.9% for the comparison systems, while every other category is
competitive or better.

## 2. Diagnosis (via an Explore sub-agent, code-grounded)

Full survey of `store.py`, `question_types.py`, `scripts/eval_locomo.py`.
Findings, all still accurate as of this writing:

- `facts.ts` is **record time only** (when a fact was written); there is no
  event-time column and no in-product date extraction anywhere. The only
  date parsing in the whole repo lived in the bench harness
  (`scripts/eval_locomo.py`'s `_parse_dt` and `_RELATIVE_TIME` table),
  never in `store.py`.
- `question_types.matches_type()`'s `"when"` branch is shape-only regex
  (`_YEAR`, `_MONTHS`, `_WEEKDAYS`, `_RELATIVE_TIME`, `_TIME_OF_DAY`) — it
  cannot distinguish a real date from an unrelated four-digit number.
- `question_types.boost()` only **reorders** the existing top-16 window; it
  cannot add or remove candidates.
- `store._apply_fusion()`'s `type_boost_enabled` (the only mechanism that
  calls the above) defaults to `False` and was never shipped on.
- **Retrieval is not the bottleneck.** Temporal evidence-hit@8 rose
  41.4%→62.1%→69.0%→75.9% across the last four versions while temporal F1
  stayed essentially flat (4.7%→7.4%→7.4%→8.5%) and once regressed
  (7.3%→5.1%) when the bench harness tried injecting resolved relative dates
  into every fact's text (`scripts/eval_locomo.py:218-219`, `resolve_relative`
  flag) — a **blanket, window-wide intervention that perturbed every fact**,
  already measured as net negative and never shipped.
- The reader prompt in `scripts/eval_locomo.py::llm_answer()` is explicitly
  documented as the frozen measuring instrument: *"tuning a benchmark's
  reader to raise its score is fitting the test, not improving memory... any
  change to this string invalidates comparisons with every earlier run."*
  This constrains every design considered below: nothing may touch that
  prompt string.

## 3. Literature survey (web research, not yet code)

Searched for external prior art on temporal reasoning in LLM-agent memory.
Most relevant, in order of relevance to this project's architecture:

- **TReMu** (arXiv 2502.01630, ACL Findings 2025) — same benchmark family
  (LoCoMo-derived multi-session temporal QA). Two mechanisms: (a)
  mention-time vs. occurrence-time distinction via per-session timeline
  summarisation, (b) neuro-symbolic reasoning — the LLM writes Python
  (`datetime`/`dateutil`) and a symbolic executor runs it, instead of asking
  the model to do date arithmetic in its head. Reported GPT-4o temporal
  accuracy 29.8%→77.7%. This is the single strongest external result found
  and maps almost directly onto this project's zero-LLM-math philosophy: the
  same effect should be achievable with a plain deterministic function
  instead of LLM-generated code, since the arithmetic itself is trivial once
  dates are structured.
- **Zep / Graphiti** (arXiv 2501.13956) — full bi-temporal schema (4
  timestamps: created/expired system time, valid/invalid world time) and
  edge invalidation on contradiction. Richer than needed here; the
  extraction method (a per-episode LLM call) conflicts with this project's
  zero-mandatory-LLM stance, but the *schema idea* (a side table, not a
  mutation of `facts`) is directly reusable.
- **Mem0 temporal reranking** (blog) — reported top-50→top-1 accuracy
  82.7%→91.8% from a reranker that "distinguishes the right dated instance
  among near-identical alternatives." This is the closest evidence that an
  *additive RRF channel* (not a reorder) is the mechanism actually worth
  building next.
- **MemStrata / Temporal Validity in Retrieval Memory** (arXiv 2606.26511) —
  deterministic `(subject, relation, object)` supersession instead of
  cosine-similarity contradiction detection; notes cosine similarity cannot
  distinguish a contradiction from a rephrasing (AUROC 0.59, near chance).
  Relevant to this project's existing `dossier_conflicts` mechanism, which
  already does structural (not embedding) conflict detection for decisions
  — this would generalise it using `propositions.py`'s existing role triples
  plus a resolved event time.
- **LongMemEval** (arXiv 2410.10813, ICLR 2025) — benchmark reference only;
  confirms temporal reasoning is a distinct, separately-measured axis
  industry-wide (133/500 of its questions), not a NeuroMatrix-specific gap.
- Also surveyed and set aside as less directly applicable: MemoTime (arXiv
  2510.13614, recursive temporal-KG reasoning — heavier machinery than
  warranted yet), general temporal-expression-normalisation survey
  literature (arXiv 2505.20243 et al. — background only).

Full citations are in the conversation history; not re-vendored here since
they are web sources, not project artifacts.

## 4. What was actually built and tested

### 4.1 `neuro_matrix/temporal.py` — KEPT, in the repo now

A new, permanent module. Zero dependencies (stdlib `datetime`/`re` only —
unlike `morphology.py` it needs no optional package, so it has no
availability gate).

Public API:
- `resolve(text, anchor_ts=None) -> Resolved(event_ts, granularity) | None`
- `has_event_date(text, anchor_ts=None) -> bool`

Design properties, deliberately mirroring `morphology.py`:
- **Narrows, never widens** — returns a real timeline point or `None`,
  never a maybe.
- **Anchor discipline** — relative ("last week") and partial ("on Tuesday",
  "March 5" with no year) expressions resolve against the fact's own record
  time (`anchor_ts`), never wall-clock "now" (which would make the same
  fact's date drift between runs — the same determinism requirement
  `morphology.py` has for index vs. query).

Coverage: full dates ("March 5, 2019", "5 марта 2019"), month+year ("May
2023", "в мае 2023"), day-without-year (uses the anchor's year), the
relative-expression delta table (ported verbatim from
`scripts/eval_locomo.py`'s bench-only `_RELATIVE_TIME`, now a single source
of truth), bare weekdays (resolved to the nearest occurrence around the
anchor, within ±3 days), and a bare year as the lowest-precision fallback.

**Known, accepted limitation, at parity with the regex it could replace:** a
bare year with no date-like context ("he has 2019 dollars saved") still
resolves as `granularity="year"` — the same false-positive class the old
`_YEAR` regex already had. Documented, not fixed (candidate for a future
narrow follow-up: require a preposition/context word before accepting a bare
year).

Unit-tested: `tests/test_core.py::test_temporal_resolve_absolute_and_relative_dates`
(full dates in both languages, month+year, relative-with-anchor,
relative-without-anchor correctly returns `None`, the known year-only
false-positive documented inline). Verified passing; full suite 87/88 (the
one failure, `test_structure_backfill_runs_while_the_agent_is_searching`, is
pre-existing and reproduces identically on the pre-session tree — confirmed
via `git stash`).

**Status: not wired into anything.** It is validated, standalone
infrastructure for the next attempt (§6), not a shipped feature.

### 4.2 Experiment: precise date resolution inside `type_boost` — TESTED, REVERTED

**Hypothesis:** replacing `question_types.matches_type()`'s shape-only regex
with `temporal.resolve()` (a real, anchored calendar date) would make the
existing `type_boost` reordering more precise and raise temporal F1.

**What was wired (temporarily):** `matches_type()` gained a `ts` parameter
that tries `temporal.resolve(text, ts)` first and only falls back to the old
regexes when no anchor was supplied; `boost()` gained a `ts_map` parameter to
thread each candidate's own record time through; `store._apply_fusion()`'s
`type_boost_enabled` branch built and passed that map. `type_boost_enabled`
itself stayed `False` by default throughout, so production behaviour was
never at risk during the experiment.

**Methodology:** same-sample A/B on the external LoCoMo benchmark
(`scripts/eval_locomo.py`, 10 conversations, local `qwen3.5:9b` reader via
Ollama, default `--llm-sample 200`/`--llm-seed 42`, which drew 203 scored
questions). Both arms answered the *exact same* 203 questions from the
*exact same* ingested conversations, so any difference is attributable to
the ranking change alone.

**Results — evidence-hit@8** (expected near-flat; hit@8 cannot see a reorder
inside an already-correct top-16, only a change to the top-8 *set*):

| Category | baseline | + precise type_boost (w=1) |
|---|---|---|
| single-hop | 51.7% | 51.7% |
| temporal | 76.9% | 76.9% |
| multi-hop | 12.5% | 12.5% |
| open-domain | 61.5% | 60.3% |
| adversarial | 61.2% | 61.2% |
| **overall** | **61.1%** | **60.6%** |

**Results — QA-accuracy (F1), n=203, same questions both arms:**

| Category | n | baseline F1 | + precise type_boost (w=1) |
|---|---|---|---|
| single-hop | 29 | 24.4% | 24.9% |
| **temporal** | 39 | **6.2%** | **5.6%** |
| multi-hop | 8 | 12.5% | 6.7% |
| open-domain | 78 | 40.4% | 40.3% |
| adversarial | 49 | 23.2% | 26.6% |
| **overall** | 203 | **26.3%** | **26.8%** |

**Interpretation — hypothesis rejected.** Temporal F1, the one number this
targeted, moved the wrong way (6.2%→5.6%). The apparent overall gain
(26.3%→26.8%) is not evidence of anything working: it is carried by
adversarial (n=49, noisy at that size) and single-hop (n=29, +0.5pp,
negligible), and the multi-hop move (12.5%→6.7%, n=8) is one question
flipping. Weight 2/3 were not swept — the direction was already wrong on the
targeted metric, and the project's own discipline is not to chase an
aggregate that contradicts the category the change was built for.

**Why, most likely:** `type_boost` is a *reorder-only* lever — it moves
facts within an already-fixed top-16 by 1-2 positions. Better date precision
gives it a more trustworthy signal for *which* facts look temporal, but the
diagnosed bottleneck (§2) is not "the wrong fact is in the window" — hit@8
confirms retrieval is untouched and already strong — it is what the *reader*
does with an already-correct window. A more precise reorder structurally
cannot fix a generation-time bottleneck. Consistent with every other
reorder-only experiment in this project's history (`edge_order`, the
original shape-based `type_boost`, the role bridge) landing mixed-to-negative.

**Reverted.** `question_types.py` and `store.py` are back to their exact
pre-experiment state (`git diff` against `origin/main` on both files is
empty). `type_boost_enabled` was `False` before, during and after — no
production risk at any point. `neuro_matrix/temporal.py` and its unit test
were kept (§4.1) since the resolver itself was not what failed.

Full writeup with the same tables and a longer discussion:
[`docs/locomo-local-v014-temporal-typeboost.md`](locomo-local-v014-temporal-typeboost.md).

### 4.3 New permanent tool: `--diag-cat` in `scripts/eval_locomo.py` — KEPT

The existing `--diag N` flag only ever printed temporal (category 2)
question/gold/prediction/context triples. Generalised it to
`--diag-cat {1..5}` (default 2, preserving old behaviour exactly) so any
category can be inspected. This is a debug-print-only change — it touches
no scoring logic and no reader prompt, so it does not compromise the frozen
measuring instrument (§2). Used immediately (§5) to inspect category 5
(adversarial) on demand.

### 4.4 Reader-model sensitivity — MEASURED, purely informational (no memory code changed)

Separate line of inquiry, prompted by the product goal of running well on
weak/phone-class local hardware. Pulled a second, much smaller local model
via the already-installed Ollama (`qwen2.5:1.5b`, ~1GB) and re-ran the exact
same benchmark (same 203 questions, same memory configuration — current
shipped defaults, no experimental flags) with it as the reader instead of
`qwen3.5:9b`.

**Caveat up front:** all of this project's recorded F1 baselines are pinned
to `model=qwen3.5:9b` for comparability (same discipline as the frozen
prompt). This sub-experiment does not change that convention or propose
switching the reader-of-record; it answers a different question — "how much
does the *reader's* capability matter given the *same* retrieved evidence."

**Aggregate result**, same 203 questions, memory config identical:

| Category | n | qwen3.5:9b | qwen2.5:1.5b |
|---|---|---|---|
| single-hop | 29 | 24.4% | 20.6% |
| temporal | 39 | 6.2% | 5.6% |
| multi-hop | 8 | 12.5% | 14.7% |
| open-domain | 78 | 40.4% | 39.0% |
| adversarial | 49 | 23.2% | **35.7%** |
| **overall F1** | 203 | 26.3% | **28.2%** |

evidence-hit@8 was identical between the two runs (124/203 = 61.1%), as
expected — retrieval does not depend on the reader.

**Root-caused the adversarial gap with the new `--diag-cat 5` tool.** Ran
`--diag 15 --diag-cat 5` for both models on the same conversations (note:
running both models' Ollama processes *concurrently* caused severe
thrashing under `OLLAMA_MAX_LOADED_MODELS=1` — each request forced a model
swap; the two diag runs had to be executed sequentially instead), captured
all 49 adversarial question/gold/prediction/context transcripts per model,
and matched them by question text with a small ad-hoc script:

| | count / 49 |
|---|---|
| small model strictly better (higher F1) | 16 |
| big model strictly better | 11 |
| both wrong (F1=0) | 17 |
| both partially right | 5 |
| **big model's answer is an explicit refusal** ("None mentioned", "not provided", "facts do not contain...") | **13/49 (27%)** |
| **small model's answer is an explicit refusal** | **0/49 (0%)** |

Two concrete transcript examples (same conversation, same retrieved
context, evidence present and marked in both):

- *"What setback did Caroline face recently?"* — evidence is the literal
  first context line for both models. `qwen3.5:9b` answered **"None
  mentioned"** (F1=0.00) despite the fact being right there; `qwen2.5:1.5b`
  answered "last month" (F1=0.00, a different failure — it grabbed a
  date-shaped fragment instead of the semantic answer, but did not refuse).
- *"What car did Calvin work on in the junkyard?"* — evidence is again the
  first context line. `qwen3.5:9b` answered **"None mentioned"** (F1=0.00);
  `qwen2.5:1.5b` answered **"Ford Mustang"** (F1=1.00, exact).

**Interpretation, stated carefully.** This is *not* evidence that the small
model reasons better. The reader prompt explicitly instructs: *"If the facts
do not contain the answer, still give your single best short guess"* — the
larger model disregards that instruction roughly a quarter of the time on
adversarial questions and refuses anyway, which is a near-guaranteed F1=0
under SQuAD-style token scoring (a refusal's tokens essentially never
overlap with a real gold answer). The smaller model never refuses; it always
copies some literal fragment from context, which under this scoring scheme
carries a non-zero chance of partial token overlap purely as a mechanical
consequence of always guessing rather than of understanding the question
better. **This is a property of the F1 metric's interaction with a
refuse-vs-guess policy, not a claim that the 1.5B model is smarter.** In a
real product, an honest "I don't know" on a fact genuinely absent from
memory is usually *more* useful than a confidently wrong guess — the exact
opposite of what this benchmark rewards.

**What this does confirm, and why it matters for the product goal:**
retrieval (hit@8) was byte-identical regardless of reader size — the memory
system hands both readers the same evidence with the same quality. Where
evidence was present and marked in context, both models could use it (both
produced F1=1.00 answers on multiple questions). The gap between models is
entirely in reader *behaviour* (refuse-or-guess policy), not in what the
memory retrieved. This directly supports running this memory system with a
small, fast, phone/weak-server-class local reader: the memory does not need
a "smart" reader to do its job, since its job (surfacing the right evidence)
measurably does not depend on reader size.

No code in `neuro_matrix/` changed for this sub-experiment. `qwen2.5:1.5b`
is now available locally for fast dev-loop smoke-testing (directional
signal in ~1-2 minutes instead of ~7-13 for `qwen3.5:9b`), while
`qwen3.5:9b` remains the model-of-record for any number meant to be compared
against this project's existing history.

Raw report: [`docs/locomo-local-v015-reader-qwen2.5-1.5b.md`](locomo-local-v015-reader-qwen2.5-1.5b.md).

## 5. Complete inventory: implemented & tested vs. proposed & not attempted

### 5.1 Implemented and tested this session

| # | What | Result | Status in repo |
|---|---|---|---|
| 1 | Strip `Co-Authored-By`/`Generated with` from all 57 `main` commits | Done, force-pushed, tree-identical | Permanent (history) |
| 2 | `neuro_matrix/temporal.py` event-date resolver | Unit-tested, correct | **Kept**, unwired |
| 3 | Wire resolver into `question_types.matches_type`/`boost` + `store._apply_fusion` | Temporal F1 6.2%→5.6% (worse); overall +0.5pp (noise) | **Reverted** |
| 4 | `--diag-cat` flag generalisation in `eval_locomo.py` | Used successfully to extract adversarial transcripts | **Kept** (debug-only, no scoring change) |
| 5 | Pull `qwen2.5:1.5b` via Ollama, benchmark as alternate reader | overall F1 26.3%→28.2%, temporal flat, adversarial +12.5pp explained by refuse-rate (27%→0%), NOT reasoning quality | Informational; model available locally, no code change |
| 6 | Root-cause the reader gap via matched-pair transcript diff (49 adversarial Q&A pairs, both models) | Big model refuses 13/49 times against an explicit "always guess" instruction; small model never refuses | Informational |

### 5.2 Proposed, researched, NOT attempted (still open)

In the priority order discussed, unchanged by this session's negative
result (a failed reorder-only attempt does not invalidate these — they are
a structurally different mechanism):

1. **Date extraction from the question itself**, not just from fact text —
   prerequisite for #2; not built. Without it there is no target date to
   score proximity against for most "when" questions (which typically ask
   for a date rather than supplying one).
2. **A genuine additive RRF channel** (`_temporal_candidates`, shaped like
   the existing `_lemma_candidates`), gated to `qtype == "when"`, scored by
   proximity between a fact's resolved `event_ts` and a date resolved from
   the question — the mechanism the Mem0 external result (§3) suggests is
   actually load-bearing, as opposed to the reorder-only lever just tested
   and rejected. Requires a persistent `event_ts` per fact (side table +
   backfill, mirroring `fact_lem`/`rebuild_lemmas()`) so the channel does
   not re-resolve every candidate on every query.
3. **"Last time / first time X" resolution** — argmax/argmin over a
   grouped entity's resolved event times (grouping via `propositions.py`'s
   existing predicate+role extraction), for the common LoCoMo pattern of
   repeated-activity temporal questions. Depends on #2's persistent
   `event_ts`.
4. **Deterministic duration computation as an added context item** (not a
   prompt change) — for "how long ago/between" question shapes (already
   partially detected by `question_types._WHEN`'s "how long ago" pattern):
   compute `Δ = |event_ts(A) − event_ts(B)|`, format it in LoCoMo's gold
   granularity, and inject it as one additional synthetic candidate fact.
   Deliberately narrower than the already-tried-and-rejected blanket
   `resolve_relative` injection (§2) — this would add one new fact only when
   a duration question is detected and two dated candidates exist, not
   annotate every fact in every window.
5. **Symbolic/code-gen date math** (TReMu-style) — considered and
   deprioritised in favour of #4: the arithmetic needed here is trivial
   subtraction/formatting, so a deterministic Python function achieves the
   same effect as asking an LLM to write and execute code, at zero
   additional LLM calls and zero non-determinism.
6. **Tighten the bare-year false positive** in `temporal.resolve()` (a
   number like "2019" with no date-like context) — cheap, pure narrowing,
   no measured risk, simply not yet done. Candidate: require a preceding
   preposition/context word ("in/since/в ... году") before accepting a bare
   4-digit year.
7. **Granularity-weighted scoring** — `Resolved.granularity` is already
   computed (day/month/year) but nothing downstream uses it as a confidence
   weight yet; relevant once #2's RRF channel exists.
8. **Russian ordinal day forms** ("пятого марта" vs. the currently-supported
   digit form "5 марта") — a real, un-covered gap in `temporal.py`'s
   Russian date coverage, not attempted.
9. **Structural temporal-conflict detection**, generalising the existing
   `dossier_conflicts` mechanism beyond decisions using `propositions.py`
   role triples + a resolved event time (MemStrata-inspired, §3) — not
   attempted, lowest priority of the open items.
10. **Zep-style per-episode LLM date extraction** — evaluated in the
    literature review and explicitly rejected as an approach: conflicts with
    the project's no-mandatory-LLM-call default path. Only the *schema*
    idea (bi-temporal side table) was retained from this source, not the
    extraction method.

## 6. Recommended next step

Per §5.2, item 1 then item 2: resolve a date out of the question text using
the existing `temporal.py` (no new module needed, same function), then build
`_temporal_candidates` as a new RRF list exactly parallel to
`_lemma_candidates` (`store.py`), gated on `question_types.question_type() ==
"when"`, weight-swept the same way the lemma channel was (1/2/3). This is
the first item that was *not* already tested and rejected this session, and
is the mechanism the external literature (Mem0's reranking result
specifically) most directly supports as effective — as opposed to the
reorder-only `type_boost` lever, which this session closed out with a
negative, documented result.
