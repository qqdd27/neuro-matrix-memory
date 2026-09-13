"""Morphological normalisation: collapse inflected forms onto a common base.

The language law being applied here is the one that makes a dictionary possible
at all: an inflected word is a LEMMA plus features (case, number, tense), so
"бежал / бежит / бежать", "research / researching / researched" all denote the
same concept.  A lexical index keyed on surface forms treats them as unrelated
strings, which is why "why did he keep researching" can miss a fact that says
"research" — and why a stem-style prefix hack is only a partial fix.

Two properties matter for how this is used:

* It NARROWS, it does not widen.  No extra candidates are introduced; the same
  fact simply becomes findable from more of its own forms.  That distinction is
  the whole reason this is worth trying here: every experiment that widened the
  candidate window (bridges, type routing, edge reordering) lost to leaving the
  window alone.
* Index and query MUST use the same function, or nothing matches at all — the
  classic silent failure of two-sided normalisation.

Cost is one call per distinct word, memoised, with pymorphy2/3 for Russian and
the Snowball/Porter stemmer for English; both are optional dependencies and a
missing one simply degrades to identity, never to an error.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_WORD = re.compile(r"[0-9a-zа-яё]+")

# Words that carry no topical meaning in either language.  Normalisation makes
# them match MORE often, so they are dropped rather than stemmed.
_STOP = {
    # English
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be", "been",
    "being", "am", "do", "does", "did", "have", "has", "had", "i", "you", "he",
    "she", "it", "we", "they", "my", "your", "his", "her", "its", "our",
    "their", "me", "him", "them", "us", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "not", "no", "yes", "so", "than", "then", "there", "here", "very",
    # Russian
    "и", "а", "но", "или", "если", "что", "чтобы", "это", "этот", "эта", "эти",
    "тот", "та", "те", "в", "во", "на", "из", "с", "со", "к", "ко", "по", "о",
    "об", "обо", "от", "до", "для", "при", "за", "под", "над", "про", "у",
    "не", "ни", "да", "же", "бы", "ли", "как", "так", "там", "тут", "здесь",
    "я", "ты", "он", "она", "оно", "они", "мы", "вы", "меня", "тебя", "его",
    "её", "их", "нас", "вас", "мой", "моя", "твой", "свой", "весь", "всё",
    "был", "была", "было", "были", "быть", "есть", "бы", "может", "можно",
}

_RU = re.compile(r"[а-яё]")

_pymorphy = None
_pymorphy_tried = False
_snowball = None
_snowball_tried = False


def _get_pymorphy():
    global _pymorphy, _pymorphy_tried
    if not _pymorphy_tried:
        _pymorphy_tried = True
        for mod in ("pymorphy3", "pymorphy2"):
            try:
                import importlib

                _pymorphy = importlib.import_module(mod).MorphAnalyzer()
                break
            except Exception:  # noqa: BLE001 - optional dependency
                _pymorphy = None
    return _pymorphy


def _get_snowball():
    global _snowball, _snowball_tried
    if not _snowball_tried:
        _snowball_tried = True
        try:
            from nltk.stem import SnowballStemmer

            _snowball = SnowballStemmer("english")
        except Exception:  # noqa: BLE001
            _snowball = None
    return _snowball


def available(lang: str = "ru") -> bool:
    """Is a real lemmatiser/stemmer present for this language?"""
    if lang.startswith("ru"):
        return _get_pymorphy() is not None
    return _get_snowball() is not None


@lru_cache(maxsize=100_000)
def lemma(word: str) -> str:
    """The normal form of one lowercased word (identity when unsupported)."""
    w = (word or "").strip().lower()
    if not w or w in _STOP:
        return ""
    if _RU.search(w):
        morph = _get_pymorphy()
        if morph is not None:
            try:
                return str(morph.parse(w)[0].normal_form)
            except Exception:  # noqa: BLE001
                return w
        return w
    stemmer = _get_snowball()
    if stemmer is not None and len(w) > 3:
        try:
            return str(stemmer.stem(w))
        except Exception:  # noqa: BLE001
            return w
    return w


def normalize(text: str) -> str:
    """Normalise a passage for indexing or for a query.

    The SAME function must be used on both sides: normalising only one side is
    the classic way this silently matches nothing.
    """
    out: list[str] = []
    for tok in _WORD.findall((text or "").lower()):
        lem = lemma(tok)
        if lem:
            out.append(lem)
    return " ".join(out)


def is_verb(word: str) -> bool:
    """True when the word is a finite verb / participle / gerund (Russian).

    Used to keep sentence-initial capitalisation from becoming a fake entity:
    with the position gate removed (see entities.py) "Бежала" and "Изучала"
    would otherwise be stored as proper nouns, while a real name like "Каролина"
    (which pymorphy does not tag as Name) must survive.
    """
    w = (word or "").strip().lower()
    if not w or not _RU.search(w):
        return False
    morph = _get_pymorphy()
    if morph is None:
        return False
    try:
        tags = str(morph.parse(w)[0].tag)
    except Exception:  # noqa: BLE001
        return False
    return any(t in tags for t in ("VERB", "INFN", "PRTF", "PRTS", "GRND"))


def is_proper_name(word: str) -> bool:
    """A morphologically CONFIRMED proper name (Russian).

    Checks EVERY parse, not just the first: pymorphy returns "каролина" as a
    Geox on the first parse and as a Name on another, so a first-parse-only check
    misses real names.  Conversely an adjective ("Новый") or a common noun
    ("Привет", "Факт") is never tagged Name/Surn/Patr in any parse — which is
    what makes this safe to accept in sentence-initial position, where a plain
    "starts with a capital" test produced fake entities and broke five tests.
    """
    w = (word or "").strip().lower()
    if len(w) < 3 or not _RU.search(w):
        return False
    morph = _get_pymorphy()
    if morph is None:
        return False
    try:
        parses = morph.parse(w)[:6]
    except Exception:  # noqa: BLE001
        return False
    for p in parses:
        tags = str(p.tag)
        if "Name" in tags or "Surn" in tags or "Patr" in tags:
            return True
    return False


def normalize_many(items: Iterable[str]) -> list[str]:
    return [normalize(t) for t in items]
