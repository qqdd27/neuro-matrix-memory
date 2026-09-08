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
# Tier-2: Latin TitleCase words, 3-14 chars, not starting a sentence filler.
_TITLECASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{2,13})\b")

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
        if key and len(key) <= 40 and key not in found:
            found.append(key)
    # TitleCase Latin names — only keep ones that are not pure anchors and
    # not stopwords; they become weak entities (weight handled by caller).
    for m in _TITLECASE_RE.finditer(text):
        w = m.group(1)
        if w.lower() in STOPWORDS:
            continue
        key = w.lower()
        if key not in found and len(key) <= 30:
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
