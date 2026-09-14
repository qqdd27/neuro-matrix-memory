"""Anaphora: turn "she" into the person it actually refers to.

Why this matters here: a fact stored as "She moved to Berlin" is unfindable by the
name a question uses ("where did Caroline move?"). Nothing is wrong with the
ranking — the link between the pronoun and the person lives in the DIALOGUE, not in
the sentence, and this store keeps facts, not dialogues.

Deliberately NOT a new retrieval channel. Eight measured attempts to widen the
candidate window (co-occurrence graph, PMI edges, bridges over BM25, role bridge,
type routing, edge reordering) all lost to leaving the window alone, because a
broad extra list displaces precise hits.  Instead the resolved names are added to
the text that feeds the EXISTING lemma index — the same trick that made "research"
reach "researching".  Two consequences worth stating plainly:

* it NARROWS in effect: an existing fact becomes reachable from a word it did not
  contain; no candidate is added to the window and no fact can be displaced;
* the original text is never rewritten.  Only the indexed form gains the
  resolution, so a wrong guess can only add a key — it cannot remove or corrupt
  what the fact already says.

Candidates come from the preceding facts of the same session.  Gender is read from
pymorphy3 for Russian ("она" needs a feminine name) and from a small table for
English; when the morphology backend is missing, gender is ignored rather than
guessed wrongly.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# pronoun -> required gender ("m" | "f" | None = any)
_PRONOUNS: dict[str, Optional[str]] = {
    # English
    "she": "f", "her": "f", "hers": "f", "herself": "f",
    "he": "m", "him": "m", "his": "m", "himself": "m",
    "they": None, "them": None, "their": None, "themselves": None,
    "it": None, "its": None, "there": None, "that": None, "this": None,
    # Russian
    "она": "f", "её": "f", "ее": "f", "ей": "f", "ней": "f", "неё": "f", "нее": "f",
    "он": "m", "его": "m", "ему": "m", "ним": "m", "нему": "m",
    "они": None, "их": None, "них": None, "им": None, "ними": None,
    "оно": None, "это": None, "этот": None, "эта": None, "эти": None,
    "там": None, "туда": None, "оттуда": None, "здесь": None,
}

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{0,29}")

_RU = re.compile(r"[а-яё]")
_gender_cache: dict[str, Optional[str]] = {}
_morph = None
_morph_tried = False

# A small list of the most common English given names.  It is NOT the mechanism —
# gender is primarily LEARNED from the facts themselves ("Caroline said she…" proves
# Caroline is feminine) and stored per entity.  This list only covers the cold start,
# before any such sentence has been seen.  Getting a gender wrong is cheap by design:
# the resolution is appended to the indexed text and never replaces anything, so a
# bad guess adds one searchable word and can neither hide a fact nor steal a slot.
_EN_F = {
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica",
    "sarah", "karen", "nancy", "lisa", "margaret", "sandra", "ashley", "emily", "kimberly",
    "donna", "michelle", "carol", "amanda", "melissa", "deborah", "stephanie", "dorothy",
    "rebecca", "sharon", "laura", "cynthia", "kathleen", "amy", "angela", "shirley",
    "anna", "brenda", "pamela", "nicole", "samantha", "katherine", "emma", "ruth",
    "christine", "helen", "debra", "rachel", "carolyn", "janet", "maria", "heather",
    "diane", "julie", "joyce", "victoria", "kelly", "christina", "joan", "evelyn",
    "lauren", "judith", "megan", "cheryl", "andrea", "hannah", "martha", "jacqueline",
    "frances", "gloria", "ann", "teresa", "kathryn", "sara", "janice", "jean", "alice",
    "madison", "doris", "abigail", "julia", "judy", "grace", "denise", "amber",
    "marilyn", "beverly", "danielle", "theresa", "sophia", "marie", "diana", "brittany",
    "natalie", "isabella", "charlotte", "rose", "alexis", "kayla", "olivia", "caroline",
    "melanie", "audrey", "claire", "lucy", "stella", "nora", "iris", "jane", "lydia",
    "eva", "ella", "mia", "chloe", "zoe", "lea", "nina", "lily", "ruby", "violet",
    "amelia", "freya", "willow", "jasmine", "paula", "monica", "erica", "tina", "gina",
    "sylvia", "sabrina", "vanessa", "veronica", "josephine", "beatrice", "gwendolyn",
    "harriet", "wilma", "berta", "agnes", "irma", "elsa", "greta", "hedwig", "ingrid",
}
_EN_M = {
    "james", "john", "robert", "michael", "william", "david", "richard", "joseph",
    "thomas", "charles", "christopher", "daniel", "matthew", "anthony", "donald",
    "mark", "paul", "steven", "andrew", "kenneth", "george", "joshua", "kevin",
    "brian", "edward", "ronald", "timothy", "jason", "jeffrey", "ryan", "jacob",
    "gary", "nicholas", "eric", "stephen", "jonathan", "larry", "justin", "scott",
    "brandon", "frank", "benjamin", "gregory", "samuel", "raymond", "patrick",
    "alexander", "jack", "dennis", "jerry", "tyler", "aaron", "henry", "douglas",
    "peter", "adam", "nathan", "zachary", "walter", "kyle", "harold", "carl",
    "jeremy", "gerald", "keith", "roger", "arthur", "terry", "lawrence", "sean",
    "christian", "albert", "joe", "ethan", "austin", "jesse", "willie", "billy",
    "bruce", "bryan", "ralph", "roy", "noah", "eugene", "wayne", "randy", "vincent",
    "russell", "louis", "philip", "bobby", "harry", "alvin", "evan", "calvin",
    "constantine", "dave", "david", "sam", "tom", "mike", "nick", "chris", "alex",
    "max", "leo", "oscar", "felix", "victor", "hugo", "ivan", "boris", "igor",
    "eduardo", "carlos", "juan", "marco", "pablo", "sergio", "andrei", "nikolai",
}
# Endings that reliably betray a feminine given name when the name is unknown.
_FEM_ENDINGS = ("a", "ie", "ine", "elle", "ette", "een", "lyn", "ina", "ita", "ika")
# …except for names that end that way and are not feminine.
_FEM_ENDING_EXCEPTIONS = {"constantine", "augustine", "joshua", "luca", "noa", "andrea"}
# Ending in -a but masculine in many languages that share the Latin script.
_MASC_LIKE_A = {"joshua", "luca", "andrea", "noa", "elia", "nikita"}


def _get_morph():
    global _morph, _morph_tried
    if not _morph_tried:
        _morph_tried = True
        for mod in ("pymorphy3", "pymorphy2"):
            try:
                import importlib

                _morph = importlib.import_module(mod).MorphAnalyzer()
                break
            except Exception:  # noqa: BLE001 - optional dependency
                _morph = None
    return _morph


def _en_gender(name: str) -> Optional[str]:
    key = (name or "").strip().lower()
    if key in _EN_F:
        return "f"
    if key in _EN_M:
        return "m"
    if key in _FEM_ENDING_EXCEPTIONS or key in _MASC_LIKE_A:
        return "m"
    for end in _FEM_ENDINGS:
        if key.endswith(end) and len(key) > 4:
            return "f"
    return None


def gender_of(name: str, hint: Optional[dict[str, str]] = None) -> Optional[str]:
    """'m' / 'f' for a name, or None when it cannot be told.

    Order of trust: (1) what the memory LEARNED from the facts themselves — this is
    the only source that is true for THIS user's people; (2) morphology for Russian
    names; (3) the cold-start list and endings for English names.
    """
    key = (name or "").strip().lower()
    if not key:
        return None
    if hint and key in hint:
        return hint[key]
    if key in _gender_cache:
        return _gender_cache[key]
    result: Optional[str] = None
    if _RU.search(key):
        morph = _get_morph()
        if morph is not None:
            try:
                for p in morph.parse(key)[:5]:
                    g = {str(x) for x in p.tag.grammemes}
                    if "femn" in g:
                        result = "f"
                        break
                    if "masc" in g:
                        result = "m"
                        break
            except Exception:  # noqa: BLE001
                result = None
    if result is None:
        result = _en_gender(key)
    _gender_cache[key] = result
    return result


def learn_genders(text: str, names: Iterable[str]) -> dict[str, str]:
    """Learn "who is she" from one SENTENCE: "Caroline said she…" -> caroline=f.

    The first version of this looked at the whole text and picked the nearest
    capitalised word before any pronoun — and the measurement showed exactly why
    that was wrong: it learned ``sweden=f``, ``seeing=f`` and even ``caroline=m``
    (a name in one sentence, a pronoun belonging to somebody else in the next).
    Bad genders are not harmless here, because they steer resolutions.

    Two rules now bound it: the name and the pronoun must be in the SAME sentence,
    and the pronoun must be within four words of it.  A sentence that does not
    state the connection cannot teach it.
    """
    out: dict[str, str] = {}
    if not text:
        return out
    allowed = {n.lower() for n in names} if names is not None else None
    for sentence in re.split(r"[.!?;:\n]+", text):
        s = sentence.strip()
        if not s:
            continue
        low = s.lower()
        words = [w.lower() for w in _WORD.findall(s)]
        gendered = [(i, w) for i, w in enumerate(words)
                    if w in _PRONOUNS and _PRONOUNS[w]]
        if not gendered:
            continue
        for i, pro in gendered:
            want = _PRONOUNS[pro]
            # nearest preceding capitalised name inside this sentence
            best = None
            for m in re.finditer(r"\b([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]{1,29})\b", s):
                cand = m.group(1)
                if cand.lower() in _PRONOUNS:
                    continue
                if allowed is not None and cand.lower() not in allowed:
                    continue
                # distance in words between the name and this pronoun
                before = s[:m.start()]
                dist = len(_WORD.findall(before))
                if i - dist <= 4:
                    best = cand
            if best:
                out[best.strip().lower()] = want
    return out



def pronouns_in(text: str) -> list[str]:
    """Pronouns present in the text, in order of appearance, lowercased."""
    out: list[str] = []
    for m in _WORD.finditer(text or ""):
        w = m.group(0).lower()
        if w in _PRONOUNS and w not in out:
            out.append(w)
    return out


def resolve_pronouns(text: str, candidates: Iterable[str],
                     hint: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Map each pronoun in ``text`` to one of ``candidates``.

    ``candidates`` should be ordered most-relevant-first (the store passes the
    entities of the most recent facts of the same session) and ``hint`` carries the
    genders this memory has already LEARNED for those names.  Gender agreement is
    used when it is known for both sides; when the pronoun has no gender (they,
    there, it) the first candidate wins.  A pronoun whose gender is required but
    unknown is left unresolved — guessing there would be noise, not signal.  An
    empty result means "nothing to add", which is the safe default.
    """
    pro = pronouns_in(text)
    if not pro:
        return {}
    names: list[str] = []
    for c in candidates:
        c = (c or "").strip()
        if c and c.lower() not in {n.lower() for n in names}:
            names.append(c)
    if not names:
        return {}
    out: dict[str, str] = {}
    for p in pro:
        want = _PRONOUNS.get(p)
        if not want:
            # "they"/"there"/"it" have no gender to agree on, so the first candidate
            # is a coin flip among everything recently mentioned.  Measured, that
            # guess added noise rather than reach, so it is not made at all.
            continue
        pick = None
        for n in names:
            if gender_of(n, hint) == want:
                pick = n
                break
        if pick is not None and pick.lower() != p:
            out[p] = pick
    return out


# First person.  Resolved not by gender but by WHO IS SPEAKING, which the fact
# states for itself ("Caroline: I just joined a group") and which needs no guess at
# all.  This is the widest anaphora there is — most turns are about oneself.
_SELF = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "я", "меня", "мне", "мой", "моя", "моё", "мои", "мы", "нас", "нам", "наш",
    "наша", "наше", "наши",
}
_SPEAKER_RE = re.compile(r"^\s*([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё\-]{1,29})\s*:\s+")


def speaker_of(text: str) -> Optional[str]:
    """The name a turn declares as its author, or None ("Caroline: …" -> Caroline)."""
    m = _SPEAKER_RE.match(text or "")
    return m.group(1) if m else None


def _first_person_in(text: str) -> list[str]:
    out: list[str] = []
    for m in _WORD.finditer(text or ""):
        w = m.group(0).lower()
        if w in _SELF and w not in out:
            out.append(w)
    return out


def resolution_terms(text: str, candidates: Iterable[str],
                     hint: Optional[dict[str, str]] = None,
                     speaker: Optional[str] = None) -> str:
    """The extra indexed terms: "she caroline i caroline" style.

    Two resolutions, in order of certainty:

    1. **First person → the speaker.**  When the fact says who is talking, "I" IS
       that name — no inference, no dictionary.  Most turns in a dialogue are about
       oneself, so this is by far the widest anaphora available.
    2. **Gendered pronoun → a named participant** of the same session (she → Caroline
       when Caroline is the only feminine name around).

    Returned as plain words (pronoun and its referent repeated) so the lemma index,
    which normalises whatever it is given, treats them exactly like the rest.
    """
    out: list[str] = []
    if speaker:
        for p in _first_person_in(text):
            out.extend([p, speaker])
    pairs = resolve_pronouns(text, candidates, hint)
    for p, n in pairs.items():
        out.extend([p, n])
    return " ".join(out)
