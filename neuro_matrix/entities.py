"""Entity extraction & alias detection, multilingual (en + ru + crypto identifiers).

Two signal tiers:

* Tier-1 anchors — high-precision identifiers: ``id_777``, ``0x...`` addresses,
  ALL-CAPS token symbols (BTC, USDT, TON), emails, @handles, dotted paths.
* Tier-2 names — Latin TitleCase words (Bitcoin, DeepSeek) that appear next to
  an anchor or inside an explicit alias statement.

Plain lowercase words are ignored on purpose: raw-token co-occurrence graphs
flood with noise edges (a core flaw of naive regex graphs).  Memory is only
worth writing when a durable anchor is present.
"""

from __future__ import annotations

import re

EN_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "have", "was",
    "are", "you", "your", "our", "not", "but", "its", "has", "had", "will",
    "would", "can", "could", "should", "about", "into", "than", "then",
    "there", "these", "those", "which", "what", "when", "where", "who",
    "whom", "how", "why", "also", "very", "just", "only", "please", "thanks",
    "thank", "okay", "hello", "hey", "yes", "no", "does", "did", "done",
    "get", "got", "make", "made", "use", "used", "using", "via", "per",
    "one", "two", "new", "need", "know", "like", "want", "see", "say",
    "said", "tell", "please", "maybe", "well", "let", "us", "im", "ive",
    "dont", "cant", "isnt", "wasnt", "didnt", "youre", "its", "thats",
    # Technical filler that produces false identity merges in raw API/UI text
    # ("none -> profile", "unknown -> url" were real corruptions in the wild).
    "none", "null", "undefined", "unknown", "surface", "self", "default",
    "value", "false", "true", "object", "string", "boolean", "callback",
    # Sentence-continuation fillers (measured, LoCoMo benchmark 2026-09):
    # unlike Russian -- where EVERY sentence capitalizes its first word
    # regardless of content, making POSITION a clean noise signal -- English
    # capitalizes for the same reason AND real proper nouns routinely open a
    # sentence too (every speaker-prefixed fact here literally starts with
    # the person's name), so position-gating Latin TitleCase the way Cyrillic
    # is gated would remove the very entities most needed. These specific
    # words were observed inflating a fact's entity count (and therefore its
    # additive score) purely by starting a sentence, with zero relevance:
    # "Doing research...", "Last month...", "Wow, Caroline!".
    "last", "next", "doing", "going", "coming", "sure", "yeah", "wow",
    "totally", "anyway", "anyways", "honestly", "definitely", "absolutely",
    "exactly", "basically", "literally", "seriously", "meanwhile", "besides",
    "alright", "gonna", "wanna", "gotta", "kinda", "sorta",
    # "Art" observed (LoCoMo, 2026-09) opening a second sentence as a common
    # noun ("Wow, Caroline! Art can be...") and becoming a noise entity.
    # Trade-off, not free: a real "Art" (short for Arthur) would also be
    # filtered. Judged worth it -- the noun sense is far more common in
    # practice than the name, and this denylist is the explicit place to
    # revisit that trade-off if it ever misses a real "Art".
    "art",
}
# Chat/text abbreviations that match the ALL-CAPS anchor pattern (same shape
# as real symbols like BTC/TON) but carry no identity at all. Anchors are
# NOT filtered through STOPWORDS (that would also strip real 2-4 letter
# tickers), so this is a separate, narrow denylist checked only for anchor
# tokens that look like a common chat abbreviation.
_ANCHOR_NOISE = {
    "btw", "lol", "omg", "imo", "imho", "asap", "fyi", "tbh", "idk", "brb",
    "np", "nvm", "wtf", "smh", "lmao", "rofl", "diy", "faq", "aka", "eta",
    "ttyl", "bff", "rn", "irl",
}
RU_STOP = {
    "это", "что", "как", "так", "для", "при", "без", "или", "если", "когда",
    "чтобы", "можно", "нужно", "надо", "будет", "быть", "есть", "был", "была",
    "были", "было", "меня", "тебя", "нас", "вас", "ему", "ей", "их", "его",
    "нее", "еще", "ещё", "уже", "только", "даже", "вот", "же", "бы", "ли",
    "ни", "не", "да", "нет", "ок", "окей", "спасибо", "привет", "пока",
    "сделать", "сделай", "сделал", "сделала", "помоги", "помощь", "хочу",
    "хочешь", "надо", "можно", "просто", "сейчас", "потом", "сегодня",
    "завтра", "вчера", "думаю", "знаю", "знаешь", "скажи", "сказал",
    "говорит", "говорить", "работы", "работает", "работать", "который",
    "которая", "которые", "потому", "поэтому", "должен", "должна", "должны",
}
STOPWORDS = EN_STOP | RU_STOP

# Tier-1 anchors (case-sensitive: ALL-CAPS tokens stay true acronyms/symbols;
# a blanket (?i) here turned every 2+ letter English word into an entity).
_ANCHOR_RE = re.compile(
    r"\b("
    r"(?i:id[-_]?[a-z0-9]{1,20})"              # id_777, id-99, ID777
    r"|0x[a-fA-F0-9]{6,64}"                    # EVM / contract addresses
    r"|[1-9A-HJ-NP-Za-km-z]{25,44}"            # base58-ish wallet keys
    r"|(?:[A-Z]{2,12})(?:[-_/][A-Z0-9]{1,10})*"  # BTC, USDT, USDT-PERP, TON
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|@[A-Za-z0-9_]{1,24}"
    r"|#[A-Za-z0-9_]{1,24}"
    r"|(?:^|[\s/])\d{6,}"                      # long numeric ids
    r")\b",
)
# Tier-2: Latin TitleCase words, 3-14 chars.
_TITLECASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{2,13})\b")
# Tier-2 (RU): Cyrillic capitalized words, 3-14 chars.
_TITLECASE_RU_RE = re.compile(r"\b([А-ЯЁ][а-яё]{2,13})\b")
_WORD_START_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")

# Position-gating: a capitalized word that is NOT the first token of its
# sentence is capitalization the writer chose deliberately (a proper noun /
# brand / name) — every language capitalizes the first word of a sentence
# regardless of part of speech, so that position carries no signal in EITHER
# script. Originally added for Cyrillic only (Russian has no OTHER reason to
# capitalize mid-sentence, so the case for gating was obvious); measured
# evidence (external LoCoMo benchmark, 2026-09) showed the same failure mode
# in English prose just as often: generic sentence-starters inside a longer
# reply ("Wow, Caroline! Art can be in the most unlikely places.") were
# extracted as entities purely for opening their OWN sentence, inflating
# unrelated facts' scores and burying facts that actually answered the
# question under a wall of "Wow, X!" noise (measured across several failing
# multi-hop questions, all sharing this exact shape). A denylist
# (last/next/doing/... in EN_STOP) cannot scale to catch every common noun
# that might open a sentence; position-gating both scripts the same way
# does, uniformly, without needing to enumerate the noise.
def _titlecase_candidates(regex: "re.Pattern[str]", text: str) -> list[str]:
    """Proper-noun candidates for `regex`: capitalized, not sentence-initial."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        first = _WORD_START_RE.search(sent)
        first_start = first.start() if first else -1
        for m in regex.finditer(sent):
            if m.start() == first_start:
                continue  # sentence-initial capitalization is ambiguous
            out.append(m.group(1))
    return out

# Explicit alias/identity statements:  "X aka Y", "X (alias Y)", "X = Y",
# "X — это Y", "X это Y", "X is Y".  Right side must look like an identifier
# (id_/0x/all-caps/TitleCase), left side any TitleCase/anchor-ish word.
_ALIAS_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{1,23})\s*"
    r"(?:aka|a\.k\.a\.|alias|он\s+же|иначе|также\s+известен\s+как|тж\.?|"
    r"—\s*это|–\s*это|\bis\b|=|это)\s*"
    r"([A-Za-z][A-Za-z0-9_\-]{1,23})",
    re.IGNORECASE,
)


def _looks_identifier(word: str) -> bool:
    w = word.lower()
    if w.startswith(("id_", "id-", "0x")):
        return True
    if re.fullmatch(r"[A-Z0-9_\-]{2,16}", word):  # ALL-CAPS token
        return True
    return bool(re.fullmatch(r"[A-Z][a-zA-Z0-9]{2,13}", word))


def extract_entities(text: str, max_entities: int = 16) -> list[str]:
    """Return unique canonical (lowercased, stripped) entity keys in ``text``."""
    if not text:
        return []
    found: list[str] = []
    for m in _ANCHOR_RE.finditer(text):
        tok = m.group(1).strip("#@ ")
        if tok.startswith("/"):
            tok = tok[1:]
        key = tok.lower()
        if key in _ANCHOR_NOISE:
            continue
        if key and len(key) <= 40 and key not in found:
            found.append(key)
    # TitleCase Latin names — NOT position-gated (reverted, measured
    # regression): the "Speaker: message" convention this engine itself uses
    # for every stored turn puts the single most important entity (the
    # speaker's name) as the first word of the whole text every time, with
    # no preceding sentence boundary to distinguish it from "sentence-initial
    # generic word" — position alone cannot tell them apart (proven: full
    # gating dropped the speaker name from virtually every fact and broke
    # five tests). A denylist stays the right tool for Latin; STOPWORDS +
    # _ANCHOR_NOISE catch what has been observed in practice so far.
    for m in _TITLECASE_RE.finditer(text):
        w = m.group(1)
        if w.lower() in STOPWORDS or w.lower() in _ANCHOR_NOISE:
            continue
        key = w.lower()
        if key not in found and len(key) <= 30:
            found.append(key)
    # TitleCase Cyrillic names — same rule, position-gated (see above). NOTE:
    # this has the identical theoretical edge case as Latin (a real entity
    # that is literally the first word of the whole text, before any
    # sentence boundary, is indistinguishable from sentence-initial noise by
    # position alone) — it was never exercised by this project's own tests,
    # which always placed the target entity mid-sentence. Kept as-is because
    # it measurably fixed a real, reproduced recall gap for its original,
    # narrower case (a genuine RU proper noun mid-sentence); the edge case is
    # now documented rather than silently unknown.
    for w in _titlecase_candidates(_TITLECASE_RU_RE, text):
        key = w.lower()
        if key in STOPWORDS or key in found or len(key) > 30:
            continue
        found.append(key)
    return found[:max_entities]


def extract_alias_pairs(text: str) -> list[tuple[str, str]]:
    """Return explicit identity statements as (left, right) lowercased pairs."""
    pairs: list[tuple[str, str]] = []
    for m in _ALIAS_RE.finditer(text):
        a, b = m.group(1).lower().strip(), m.group(2).lower().strip()
        if a == b:
            continue
        if not (_looks_identifier(m.group(1)) or _looks_identifier(m.group(2))):
            continue
        if a in STOPWORDS or b in STOPWORDS:
            continue
        pairs.append((a, b))
    return pairs


def strip_anchor_noise(text: str) -> str:
    """Light text cleanup before storage."""
    return re.sub(r"\s+", " ", text or "").strip()[:2000]
