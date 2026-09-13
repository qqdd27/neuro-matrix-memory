"""eval_locomo.py — evaluate NeuroMatrix retrieval against LoCoMo
(snap-research/locomo, arXiv:2402.17753 "Evaluating Very Long-Term
Conversational Memory of LLM Agents"), the external, non-self-authored
benchmark this project had never been run against.

Honest scope: LoCoMo's published numbers are QA ACCURACY after an LLM reads
retrieved context and answers the question (retrieval + generation). This
script measures the retrieval half only — EVIDENCE HIT@k: for each question,
does the original conversation turn(s) the gold answer depends on
(`evidence`, e.g. "D2:3") show up among NeuroMatrix's top-k `search()` hits?
That is the ceiling on what any downstream reader (LLM or human) could
possibly answer correctly from what this memory handed back. It is NOT the
same number LoCoMo leaderboards report (no LLM reader is invoked here, by
design — zero extra dependency, zero API cost) and this script says so in
its own output rather than implying a like-for-like comparison.

Each of LoCoMo's 10 long conversations (19-32 sessions, both speakers) is
ingested into a FRESH store (speaker name prefixed onto every turn, so a
TitleCase anchor is always present) with timestamps taken from each
session's real date/time. QA pairs use LoCoMo's own category scheme
(1=single-hop, 2=temporal, 3=multi-hop, 4=open-domain, 5=adversarial).

Data license: LoCoMo (snap-research/locomo) is released under CC BY-NC 4.0
(non-commercial). It is fetched on demand into the git-ignored
scripts/bench_data/ directory the first time this script runs and is never
vendored into this (MIT-licensed) repository — only the aggregate NUMBERS
this script prints/writes are ours to keep, not the dataset's copyrighted
dialogue text.

Optional QA-accuracy mode (``--llm``): published LoCoMo leaderboard numbers
(Mem0, ZeroMemory, ByteRover, ...) are QA accuracy AFTER an LLM reads
retrieved context and answers — a composite of exact/F1/LLM-judge scoring,
not retrieval alone. ``--llm`` adds that second half using the SAME optional
LLMClient this project already supports (DeepSeek/OpenAI-compatible; reads
NEUROMATRIX_API_KEY or --llm-api-key) — no new dependency, but real API
calls/cost, so it defaults to a random --llm-sample of questions rather than
all ~2000. Scored with token-level F1 (SQuAD-style, no LLM judge needed for
scoring itself) against the gold answer. This is the number to compare
against other systems' published scores — evidence-hit@k above is not.

Run:
    python scripts/eval_locomo.py
    python scripts/eval_locomo.py --k 8 --out docs/locomo-report.md
    python scripts/eval_locomo.py --llm --llm-sample 200   # QA-accuracy (F1)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuro_matrix.store import NeuroMatrixStore  # noqa: E402
from neuro_matrix.llm import (  # noqa: E402
    DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient, build_llm_client,
)

_LOCOMO_URL = ("https://raw.githubusercontent.com/snap-research/locomo/"
               "main/data/locomo10.json")


def _ensure_dataset(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching LoCoMo dataset (CC BY-NC 4.0, snap-research/locomo) -> {path} ...")
    urllib.request.urlretrieve(_LOCOMO_URL, path)

CATEGORY_NAMES = {
    1: "single-hop", 2: "temporal", 3: "multi-hop",
    4: "open-domain", 5: "adversarial",
}

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
_DT_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s*on\s*(\d{1,2})\s*([A-Za-z]+),\s*(\d{4})",
    re.IGNORECASE)


def _parse_dt(s: str, fallback: float) -> float:
    """'1:56 pm on 8 May, 2023' -> unix ts. Falls back to a monotonic stub
    (fallback + small increment) if the format ever changes upstream --
    ordering matters far more to this eval than absolute calendar accuracy."""
    m = _DT_RE.search(s or "")
    if not m:
        return fallback
    hh, mm, ap, day, mon, year = m.groups()
    hh = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    month = _MONTHS.get(mon.capitalize())
    if month is None:
        return fallback
    try:
        import datetime
        dt = datetime.datetime(int(year), month, int(day), hh, int(mm))
        return dt.timestamp()
    except ValueError:
        return fallback


_ARTICLES = {"a", "an", "the"}


def _normalize_answer_tokens(s: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation, drop
    articles, collapse whitespace, split into tokens."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return [t for t in s.split() if t not in _ARTICLES]


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between a predicted and gold answer (the standard,
    LLM-judge-free QA metric — SQuAD/LoCoMo-style). 1.0 for an exact token
    multiset match, 0.0 for no shared tokens; partial credit otherwise."""
    pred_toks = _normalize_answer_tokens(pred)
    gold_toks = _normalize_answer_tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    from collections import Counter
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# Relative time expressions that a fact's text can contain while its stored
# timestamp gives the anchor to resolve them.  Without this the reader receives
# "[2023-06-09] ...my school event last week..." and answers "last week" — the
# gold answer is "the week before 9 June 2023", i.e. arithmetic the reader
# cannot do from the text alone.  Measured on temporal questions: retrieval was
# already correct (hit@8 62%) while F1 sat at 4.7%.
_RELATIVE_TIME = [
    (re.compile(r"\bthe day before yesterday\b", re.I), -2),
    (re.compile(r"\byesterday\b", re.I), -1),
    (re.compile(r"\blast night\b", re.I), -1),
    (re.compile(r"\b(today|this morning|tonight|right now)\b", re.I), 0),
    (re.compile(r"\btomorrow\b", re.I), 1),
    (re.compile(r"\bnext week\b", re.I), 7),
    (re.compile(r"\blast weekend\b", re.I), -7),
    (re.compile(r"\btwo weekends ago\b", re.I), -14),
    (re.compile(r"\b(a|last) week( ago)?\b", re.I), -7),
    (re.compile(r"\btwo weeks ago\b", re.I), -14),
    (re.compile(r"\b(three|3) weeks ago\b", re.I), -21),
    (re.compile(r"\blast month\b", re.I), -30),
    (re.compile(r"\ba month ago\b", re.I), -30),
    (re.compile(r"\blast year\b", re.I), -365),
    (re.compile(r"\b(позавчера)\b", re.I), -2),
    (re.compile(r"\b(вчера)\b", re.I), -1),
    (re.compile(r"\b(сегодня)\b", re.I), 0),
    (re.compile(r"\b(завтра)\b", re.I), 1),
    (re.compile(r"\b(на прошлой неделе|неделю назад)\b", re.I), -7),
    (re.compile(r"\b(в прошлом месяце|месяц назад)\b", re.I), -30),
    (re.compile(r"\b(два? дня назад)\b", re.I), -2),
]


def absolutize_relative_dates(text: str, ts: float | None) -> str:
    """Append resolved absolute dates for relative time expressions in `text`.

    Memory's job is to hand the reader facts it can actually use.  A fact
    saying "my school event last week" is anchored by its own timestamp, so the
    absolute date is derivable — the reader just cannot derive it reliably.
    Returns the text unchanged when there is nothing to resolve.
    """
    if not text or ts is None:
        return text
    try:
        base = float(ts)
    except (TypeError, ValueError):
        return text
    hits: list[str] = []
    for rx, delta in _RELATIVE_TIME:
        m = rx.search(text)
        if not m:
            continue
        phrase = m.group(0)
        try:
            resolved = time.strftime("%Y-%m-%d", time.localtime(base + delta * 86400))
        except (OSError, ValueError):
            continue
        token = f'"{phrase}" = {resolved}'
        if token not in hits:
            hits.append(token)
        if len(hits) >= 3:
            break
    if not hits:
        return text
    return f"{text}  (relative time resolved: {'; '.join(hits)})"


def _ctx_line(h: dict, *, resolve_relative: bool = False) -> str:
    """Render one hit the way the LIVE agent sees it.

    provider._format_hit has always appended the fact's date, but this reader
    used to receive bare text — so every "when did X happen" question was
    answered with a handicap the benchmark invented, not a limitation of the
    memory (temporal had the best retrieval of all categories, hit@8 41.4%,
    and the worst answer score, F1 4.7%).

    `resolve_relative` is OFF by default: annotating facts with resolved dates
    measured worse on every category (temporal 7.3% -> 5.1% F1, overall 20.8%
    -> 19.1%), because it perturbs every fact in the window.  Kept as a flag so
    the negative result stays reproducible.
    """
    ts = h.get("ts")
    try:
        when = time.strftime("%Y-%m-%d", time.localtime(float(ts))) if ts else ""
    except (TypeError, ValueError, OSError):
        when = ""
    text = str(h.get("text", ""))
    if resolve_relative:
        text = absolutize_relative_dates(text, ts)
    return f"[{when}] {text}" if when else text


def llm_answer(llm: LLMClient, question: str, context_texts: list[str]) -> str:
    """Ask the configured LLM to answer `question` using ONLY the retrieved
    facts as context — the standard retrieve-then-read QA step. Returns '' on
    any failure (network, no key, bad JSON) so callers just score it as wrong
    rather than crash the whole run.

    This prompt is part of the MEASURING INSTRUMENT, not of the product.  It is
    deliberately left alone: tuning a benchmark's reader to raise its score is
    fitting the test, not improving memory.  Product-side findings (what the
    memory itself ranks, stores and hands over) are the only thing that may
    change here, and any change to this string invalidates comparisons with
    every earlier run.
    """
    context = "\n".join(f"- {t}" for t in context_texts[:12])
    content = llm.chat_json([
        {"role": "system",
         "content": ("Answer the question using ONLY the facts below. Extract "
                     "the SHORTEST possible answer span (2-8 words, a name, "
                     "date, or short phrase) — reuse the facts' own wording "
                     "verbatim wherever possible, never a full sentence, "
                     "never restate the question, no filler words like "
                     "'likely' or 'according to the facts'. Each fact may be "
                     "prefixed with [YYYY-MM-DD]: that is the date of the "
                     "event, so for 'when' questions answer with that date. If "
                     "the facts do not contain the answer, still give your "
                     'single best short guess. Return JSON {"answer": "..."}.')},
        {"role": "user", "content": f"FACTS:\n{context}\n\nQUESTION: {question}"},
    ])
    if isinstance(content, dict):
        return str(content.get("answer") or "")
    return ""


def _dia_map(conv: dict) -> dict[str, str]:
    """dia_id ('D3:7') -> original turn text (speaker-prefixed, matching what
    was actually stored) for evidence lookup."""
    out: dict[str, str] = {}
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]))
    for k in session_keys:
        for turn in conv[k]:
            out[turn["dia_id"]] = f"{turn['speaker']}: {turn['text']}"
    return out


def run_sample(sample: dict, k: int, *, llm: "LLMClient | None" = None,
              qa_filter: "set[tuple[str, str]] | None" = None,
              rerank: bool = False, use_reader: bool = True,
              use_memory: bool = True, embedder=None,
              no_bridge: bool = False, bridge_weight: float = 1.0,
              bridge_min_act: float = 0.0, bridge_max_entities: int = 0,
              bridge_adaptive: bool = False, bridge_edge: str = "count",
              bridge_topk: int = 0, fts: "bool | None" = None,
              fts_weight: "float | None" = None,
              fusion_blend: "float | None" = None, diag: int = 0,
              resolve_relative: bool = False, slots: bool = False,
              slot_weight: "float | None" = None, role_bridge: bool = False,
              role_bridge_weight: "float | None" = None,
              type_boost: bool = False,
              type_boost_weight: "float | None" = None,
              edge_order: bool = False, fts_lemmas: "bool | None" = None,
              lemma_weight: "float | None" = None) -> dict:
    conv = sample["conversation"]
    session_keys = sorted(
        (key for key in conv if key.startswith("session_") and not key.endswith("_date_time")),
        key=lambda key: int(key.split("_")[1]))
    path = os.path.join(tempfile.mkdtemp(), f"{sample['sample_id']}.db")
    # llm_daily_budget raised well above the default (20): this is an
    # explicit, opt-in benchmark run, not a live session, and each
    # conversation gets its own fresh store/budget -- the default would
    # silently start no-op'ing rerank/reader calls partway through a single
    # long conversation's questions.
    store = NeuroMatrixStore(path, llm=llm, llm_daily_budget=100_000)
    if no_bridge:
        store.bridge_enabled = False
    store.bridge_weight = float(bridge_weight)
    store.bridge_min_activation = float(bridge_min_act)
    store.bridge_max_entities = int(bridge_max_entities)
    store.bridge_adaptive = bool(bridge_adaptive)
    store.bridge_edge_weight = str(bridge_edge)
    store.bridge_topk = int(bridge_topk)
    store.fts_enabled = True if fts is None else bool(fts)
    if fts_weight is not None:
        store.fts_weight = float(fts_weight)
    if fusion_blend is not None:
        store.fusion_blend = float(fusion_blend)
    store.slots_enabled = bool(slots)
    if slot_weight is not None:
        store.slot_weight = float(slot_weight)
    store.role_bridge_enabled = bool(role_bridge)
    if role_bridge_weight is not None:
        store.role_bridge_weight = float(role_bridge_weight)
    store.type_boost_enabled = bool(type_boost)
    if type_boost_weight is not None:
        store.type_boost_weight = float(type_boost_weight)
    store.edge_order_enabled = bool(edge_order)
    if fts_lemmas is not None:
        store.fts_lemmatize = bool(fts_lemmas)
    if lemma_weight is not None:
        store.lemma_weight = float(lemma_weight)
    if fts_lemmas:
        # Facts are written below through remember(), which indexes the lemma
        # form when the flag is on; nothing else to backfill in a fresh store.
        pass
    base_ts = time.time() - 400 * 86400
    for i, sk in enumerate(session_keys):
        dt_str = conv.get(f"{sk}_date_time", "")
        ts = _parse_dt(dt_str, base_ts + i * 86400)
        sid = f"{sample['sample_id']}_{sk}"
        for turn in conv[sk]:
            store.remember(f"{turn['speaker']}: {turn['text']}", source="turn",
                           session_id=sid, ts=ts, importance=1.0)

    if llm is not None and use_memory:
        # Write-time trait distillation (v0.7.4): targets exactly the
        # inferential/multi-hop questions this benchmark scores worst on
        # ("what field would X pursue?"), which single-shot retrieval
        # structurally cannot answer. Bounded to a handful of batches so one
        # long conversation's cost stays predictable.
        # Skipped in --no-memory runs: the baseline never reads the store, so
        # distilling into it would only burn local compute for nothing.
        for _ in range(15):
            if store.sweep_traits(batch=12) == 0:
                break

    if embedder is not None and use_memory:
        # Index the whole conversation ONCE, up front, so the timed search path
        # is not paying for embedding work mid-measurement (the same reason the
        # provider drains in a bounded slice after a session instead of inside
        # remember()).
        store.embedder = embedder
        indexed = 0
        while True:
            n = store.embed_missing(limit=512, batch=64)
            indexed += n
            if n <= 0:
                break
        print(f"    [{sample['sample_id']}] embedded {indexed} facts "
              f"({embedder.model})", flush=True)

    dmap = _dia_map(conv)
    temporal_cat = 2
    diag_shown = [0]
    per_cat: dict[int, list[bool]] = {}
    per_cat_f1: dict[int, list[float]] = {}
    lat = []
    for qa in sample["qa"]:
        cat = int(qa.get("category") or 0)
        ev_ids = qa.get("evidence") or []
        ev_texts = [dmap[e] for e in ev_ids if e in dmap]
        if not ev_texts:
            continue  # no gold evidence to check against (rare, skip)
        if qa_filter is not None and (sample["sample_id"], qa["question"]) not in qa_filter:
            continue
        if use_memory:
            t0 = time.time()
            hits = store.search(qa["question"], limit=k, rerank=rerank and llm is not None)
            lat.append((time.time() - t0) * 1000)
            hit_blob = "\n".join(h.get("text", "") for h in hits)
            found = any(ev in hit_blob for ev in ev_texts)
            per_cat.setdefault(cat, []).append(found)
        else:
            # Baseline: the SAME reader on the SAME questions with NO retrieved
            # context at all.  Without this the with-memory F1 has nothing to
            # be compared against — "our memory scored 34%" only means
            # something next to "the bare model scored X%".
            hits = []
        if llm is not None and use_reader:
            ctx_lines = [_ctx_line(h, resolve_relative=resolve_relative) for h in hits]
            pred = llm_answer(llm, qa["question"], ctx_lines)
            # Category 5 (adversarial) stores its gold answer under a
            # DIFFERENT key ('adversarial_answer', not 'answer') in the raw
            # dataset -- missing this silently scored every adversarial
            # question against an empty gold string (guaranteed F1=0).
            gold = qa.get("answer")
            if gold is None:
                gold = qa.get("adversarial_answer", "")
            if diag and cat == temporal_cat and diag_shown[0] < diag:
                diag_shown[0] += 1
                print(f"\n--- DIAG temporal #{diag_shown[0]} [{sample['sample_id']}] ---")
                print(f"Q    : {qa['question']}")
                print(f"GOLD : {gold!r}")
                print(f"PRED : {pred!r}   F1={f1_score(pred, gold):.2f}")
                print(f"EVID : {ev_texts[:2]}")
                print("CTX  :")
                for line in ctx_lines[:6]:
                    mark = " <== EVIDENCE" if any(ev in line for ev in ev_texts) else ""
                    print(f"       {line[:160]}{mark}")
            per_cat_f1.setdefault(cat, []).append(f1_score(pred, gold))
    store.close()
    return {"sample_id": sample["sample_id"], "per_cat": per_cat,
            "per_cat_f1": per_cat_f1, "latency_ms": lat}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).parent / "bench_data" / "locomo10.json"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="debug: only run the first N conversations")
    ap.add_argument("--llm", action="store_true",
                    help="also run the retrieve-then-read QA pipeline "
                         "(F1 score) -- the number comparable to published "
                         "LoCoMo leaderboard entries. Costs real API calls.")
    ap.add_argument("--llm-api-key", default=(
        os.environ.get("NEUROMATRIX_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""),
        help="API key for the --llm reader/rerank pass. ANTHROPIC_API_KEY is "
             "deliberately NOT auto-detected here: this benchmark is aimed at "
             "LOCAL models plus the cheap OpenAI-compatible providers, and a "
             "stray sk-ant-* key in the environment used to silently route a "
             "whole run to paid Claude. Pass --llm-provider anthropic AND an "
             "explicit --llm-api-key if you really want Claude.")
    ap.add_argument("--llm-provider", default="",
                    help="'anthropic' routes through the native Messages API "
                         "(must be stated explicitly — never inferred from the "
                         "key); 'ollama' talks to a local server's native API "
                         "with thinking disabled; anything else uses the "
                         "OpenAI-compatible wire format (DeepSeek default).")
    ap.add_argument("--llm-base-url", default="")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--llm-sample", type=int, default=200,
                    help="randomly sample this many questions for the (paid) "
                         "--llm pass instead of all ~2000 (default 200)")
    ap.add_argument("--llm-seed", type=int, default=42)
    ap.add_argument("--rerank", action="store_true",
                    help="use search(rerank=True) -- an LLM reorders the "
                         "heuristic top candidates before reading/scoring. "
                         "Only active together with --llm (needs the same "
                         "key); an extra LLM call per question, on top of "
                         "the reader call.")
    ap.add_argument("--no-memory", action="store_true",
                    help="baseline: run the reader WITHOUT any retrieved facts "
                         "(and skip the retrieval metric) so the with-memory run "
                         "can be reported as a delta over the bare model. "
                         "Pair it with the same --llm-seed/--llm-sample to keep "
                         "both runs on the identical question set.")
    ap.add_argument("--embeddings", action="store_true",
                    help="enable dense/hybrid recall: embed the ingested facts "
                         "with a local Ollama embedding model and fuse their rank "
                         "with the heuristic ranking (RRF). Requires the model "
                         "(e.g. bge-m3) to be present locally.")
    ap.add_argument("--embed-model", default="",
                    help="embedding model name for --embeddings (EMPTY = auto-detect)")
    ap.add_argument("--embed-url", default="http://127.0.0.1:11434")
    ap.add_argument("--no-bridge", action="store_true",
                    help="disable graph-activation bridges (the multi-hop part of "
                         "hybrid recall). Graph bridges need no model and no "
                         "download, so this exists to isolate their contribution "
                         "against the pure heuristic/lexical baseline.")
    ap.add_argument("--bridge-weight", type=float, default=1.0,
                    help="weight of the bridge list inside RRF. Unweighted (1.0) "
                         "measured WORSE overall (28.3%% vs 32.3%%) while doubling "
                         "multi-hop (7.9%% -> 11.2%%): the bridge list is broad and "
                         "out-votes the precise heuristic list. Lower values keep "
                         "the multi-hop gain without the regression.")
    ap.add_argument("--bridge-min-act", type=float, default=0.0,
                    help="drop bridge entities below this activation (0 = keep all). "
                         "Locality knob: the heat-kernel literature localises a "
                         "neighbourhood better than PPR's long random-walk tail.")
    ap.add_argument("--bridge-max-entities", type=int, default=0,
                    help="cap how many activated entities may contribute bridges "
                         "(0 = no cap)")
    ap.add_argument("--bridge-edge", choices=("count", "pmi"), default="count",
                    help="edge weight for graph activation. 'count' = raw "
                         "co-occurrence (the original: the transcript graph is "
                         "nearly complete, so activation spreads everywhere and "
                         "trades hits instead of adding them). 'pmi' = mutual "
                         "information, so only specific associations carry weight.")
    ap.add_argument("--bridge-topk", type=int, default=0,
                    help="keep only each node's K strongest neighbours (0 = all). "
                         "Cheap localisation in place of a heat-kernel solver.")
    ap.add_argument("--fts", action="store_true",
                    help="add the BM25/FTS5 lexical ranking as its own list inside "
                         "RRF. The index always existed but its ranking never "
                         "reached the score: candidates found by BM25 were re-scored "
                         "by importance x recency, so a fact matching every query "
                         "token could lose to an important but unrelated one.")
    ap.add_argument("--no-fts", action="store_true",
                    help="disable the BM25 lexical channel (it is ON by default)")
    ap.add_argument("--rel-time", action="store_true",
                    help="EXPERIMENT, not a product feature: annotate facts with "
                         "resolved absolute dates. Measured WORSE on every "
                         "category (temporal F1 7.3%% -> 5.1%%, overall 20.8%% -> "
                         "19.1%%), so it is off by default and kept only to "
                         "reproduce that result. It is here because the "
                         "hypothesis was about the PRODUCT (should the memory "
                         "hand the agent absolute dates?), not about the "
                         "reader's prompt — tuning the latter to lift the score "
                         "would be fitting the test.")
    ap.add_argument("--slots", action="store_true",
                    help="enable the structural channel (predicate + semantic roles "
                         "from morphology, no model call). It distinguishes "
                         "'Маша дала книгу Пете' from 'Петя дал книгу Маше', which "
                         "share every word.")
    ap.add_argument("--slot-weight", type=float, default=None,
                    help="weight of the structural list inside RRF")
    ap.add_argument("--role-bridge", action="store_true",
                    help="link facts through a SHARED PARTICIPANT (same name in the "
                         "same semantic role) — the structural replacement for the "
                         "co-occurrence graph, which failed because a transcript "
                         "graph is nearly complete")
    ap.add_argument("--role-bridge-weight", type=float, default=None,
                    help="weight of the role-bridge list inside RRF")
    ap.add_argument("--type-boost", action="store_true",
                    help="reorder found facts by QUESTION TYPE (when -> dates, "
                         "who -> names, where -> places, count -> numbers); "
                         "reordering only, the candidate set is unchanged")
    ap.add_argument("--type-boost-weight", type=float, default=None,
                    help="how far a same-shape fact is lifted (default 1)")
    ap.add_argument("--edge-order", action="store_true",
                    help="place the second-strongest fact LAST (readers attend to "
                         "the edges of a long context more than to its middle); "
                         "reordering only")
    ap.add_argument("--fts-lemmas", dest="fts_lemmas", action="store_true", default=None,
                    help="morphological normalisation: index and query by LEMMA so "
                         "'research' finds 'researching' and 'бежал' finds 'бежать' "
                         "(ON by default, mirroring the product)")
    ap.add_argument("--no-lemmas", dest="fts_lemmas", action="store_false",
                    help="turn the lemma channel off (surface index only)")
    ap.add_argument("--lemma-weight", type=float, default=None,
                    help="weight of the lemma channel inside RRF (default 2.0); the "
                         "lemma index always runs BESIDE the surface index, never "
                         "instead of it")
    ap.add_argument("--diag", type=int, default=0,
                    help="print N temporal questions with gold answer, retrieved "
                         "context and the reader's prediction (diagnosis mode)")
    ap.add_argument("--fusion-blend", type=float, default=None,
                    help="1.0 = pure RRF order; lower values let the evidence "
                         "score (kind priority, importance, content relevance) "
                         "share the final ordering")
    ap.add_argument("--fts-weight", type=float, default=None,
                    help="weight of the BM25 list inside RRF")
    ap.add_argument("--bridge-adaptive", action="store_true",
                    help="only add bridges when the entity/lexical path returned "
                         "few confident hits — i.e. answer the questions it cannot, "
                         "instead of displacing the ones it already answers")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    _ensure_dataset(data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit_samples:
        data = data[: args.limit_samples]

    llm = None
    qa_filter = None
    use_reader = args.llm
    use_rerank = args.rerank
    if use_reader or use_rerank:
        provider = args.llm_provider or (
            # NOTE: no sk-ant-* sniffing anymore — Anthropic is opt-in via an
            # explicit --llm-provider anthropic, never inferred from whatever
            # key happens to sit in the environment.
            "anthropic" if args.llm_provider == "anthropic" else "")
        llm = build_llm_client(args.llm_api_key, provider=provider,
                              base_url=args.llm_base_url, model=args.llm_model)
        if not llm.available():
            print("--llm/--rerank requested but no API key found "
                  "(--llm-api-key / NEUROMATRIX_API_KEY / ANTHROPIC_API_KEY / "
                  "DEEPSEEK_API_KEY / OPENAI_API_KEY). Running evidence-hit@k only.")
            llm, use_reader, use_rerank = None, False, False
        else:
            print(f"LLM provider: {provider or 'openai-compatible'}, model: {llm.model}")
            all_qas = [(s["sample_id"], qa["question"]) for s in data
                      for qa in s["qa"] if qa.get("evidence")]
            rnd = random.Random(args.llm_seed)
            n_sample = min(args.llm_sample, len(all_qas))
            qa_filter = set(rnd.sample(all_qas, n_sample))
            print(f"LLM enabled (reader={use_reader}, rerank={use_rerank}): "
                  f"sampling {n_sample}/{len(all_qas)} questions "
                  f"(seed={args.llm_seed}). This costs real API calls.")

    embedder = None
    if args.embeddings or args.embed_model:
        from neuro_matrix.embeddings import OllamaEmbedder
        emb = OllamaEmbedder(model=args.embed_model, base_url=args.embed_url)
        if emb.available():
            embedder = emb
            print(f"Embeddings: {emb.model} via {emb.base_url}")
        else:
            print(f"Embeddings requested but no model found at {args.embed_url} "
                  f"(tried {args.embed_model or 'auto-detect'}) — running without.")

    t0 = time.time()
    all_per_cat: dict[int, list[bool]] = {}
    all_per_cat_f1: dict[int, list[float]] = {}
    all_lat: list[float] = []
    per_sample_summary = []
    for sample in data:
        r = run_sample(sample, args.k, llm=llm, qa_filter=qa_filter,
                       rerank=use_rerank, use_reader=use_reader,
                       use_memory=not args.no_memory, embedder=embedder,
                       no_bridge=args.no_bridge, bridge_weight=args.bridge_weight,
                       bridge_min_act=args.bridge_min_act,
                       bridge_max_entities=args.bridge_max_entities,
                       bridge_adaptive=args.bridge_adaptive,
                       bridge_edge=args.bridge_edge, bridge_topk=args.bridge_topk,
                       fts=(False if args.no_fts else None), fts_weight=args.fts_weight,
                       fusion_blend=args.fusion_blend, diag=args.diag,
                       resolve_relative=bool(args.rel_time),
                       slots=bool(args.slots), slot_weight=args.slot_weight,
                       role_bridge=bool(args.role_bridge),
                       role_bridge_weight=args.role_bridge_weight,
                       type_boost=bool(args.type_boost),
                       type_boost_weight=args.type_boost_weight,
                       edge_order=bool(args.edge_order),
                       fts_lemmas=args.fts_lemmas,
                       lemma_weight=args.lemma_weight)
        for cat, results in r["per_cat"].items():
            all_per_cat.setdefault(cat, []).extend(results)
        for cat, results in r.get("per_cat_f1", {}).items():
            all_per_cat_f1.setdefault(cat, []).extend(results)
        all_lat.extend(r["latency_ms"])
        n = sum(len(v) for v in r["per_cat"].values())
        hits = sum(sum(v) for v in r["per_cat"].values())
        per_sample_summary.append((r["sample_id"], hits, n))
    elapsed = time.time() - t0

    total_n = sum(len(v) for v in all_per_cat.values())
    total_hit = sum(sum(v) for v in all_per_cat.values())

    print(f"=== LoCoMo evidence-hit@{args.k} (retrieval ceiling, NOT a QA-accuracy "
          f"comparison to published LoCoMo leaderboard numbers) ===\n")
    print(f"{len(data)} conversations, {total_n} scored questions "
          f"(of {sum(len(s['qa']) for s in data)} total; unscored ones had no "
          f"resolvable evidence dia_id), evaluated in {elapsed:.1f}s\n")
    lines_md = [
        "# LoCoMo evidence-hit@{} report".format(args.k), "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- conversations: {len(data)}, questions scored: {total_n}",
        f"- overall evidence-hit@{args.k}: **{total_hit}/{total_n} "
        f"({total_hit/total_n:.1%})**" if total_n else "- no questions scored",
        "", "| Category | n | hit@{} |".format(args.k), "|---|---|---|",
    ]
    for cat in sorted(all_per_cat):
        v = all_per_cat[cat]
        rate = sum(v) / len(v) if v else 0.0
        name = CATEGORY_NAMES.get(cat, str(cat))
        print(f"[{name:11s}] n={len(v):4d}  hit@{args.k} = {sum(v):4d}/{len(v):<4d} = {rate:.1%}")
        lines_md.append(f"| {cat} {name} | {len(v)} | {sum(v)}/{len(v)} ({rate:.1%}) |")
    overall_rate = (total_hit / total_n) if total_n else 0.0
    print(f"\nOVERALL evidence-hit@{args.k}: {total_hit}/{total_n} = {overall_rate:.1%}")
    if all_lat:
        sl = sorted(all_lat)
        print(f"search() latency: p50={sl[len(sl)//2]:.2f}ms  "
              f"p95={sl[int(len(sl)*0.95)]:.2f}ms  max={sl[-1]:.2f}ms")
    print("\nPer-conversation:")
    for sid, hits, n in per_sample_summary:
        print(f"  {sid:10s} {hits:3d}/{n:<3d} = {hits/n:.1%}" if n else f"  {sid} (no scored q)")

    if all_per_cat_f1:
        total_f1_n = sum(len(v) for v in all_per_cat_f1.values())
        total_f1 = sum(sum(v) for v in all_per_cat_f1.values())
        avg_f1 = total_f1 / total_f1_n if total_f1_n else 0.0
        print(f"\n=== QA-accuracy (F1), comparable to published leaderboard "
              f"numbers, n={total_f1_n} sampled questions ===")
        lines_md += ["", f"## QA-accuracy (F1), n={total_f1_n} sampled",
                     "", "| Category | n | avg F1 |", "|---|---|---|"]
        for cat in sorted(all_per_cat_f1):
            v = all_per_cat_f1[cat]
            m = sum(v) / len(v) if v else 0.0
            name = CATEGORY_NAMES.get(cat, str(cat))
            print(f"[{name:11s}] n={len(v):4d}  avg F1 = {m:.1%}")
            lines_md.append(f"| {cat} {name} | {len(v)} | {m:.1%} |")
        print(f"\nOVERALL QA-accuracy (F1): {avg_f1:.1%} "
              f"(sampled {total_f1_n}, model={llm.model if llm else args.llm_model})")
        lines_md.append(f"\n**Overall F1: {avg_f1:.1%}** (model={llm.model if llm else args.llm_model})")

    if args.out:
        Path(args.out).write_text("\n".join(lines_md) + "\n", encoding="utf-8")
        print(f"\nreport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
