"""Propagate meaning as structure, not as a bag of words.

The problem this exists for: "Маша подарила книгу Пете" and "Петя подарил книгу
Маше" contain identical word multisets with opposite meanings.  BM25 and
embeddings score them the same — direction of action is lost.  Structure keeps
it: predicate + roles.

    подарить(agent=маша, patient=книга, recipient=петя)
    подарить(agent=петя, patient=книга, recipient=маша)

Design constraints:
  * No LLM, no training, no network.  Roles come from morphology that is already
    in the text: Russian cases (именительный → agent, винительный → patient,
    дательный → recipient, творительный → instrument, предложный → location) and
    English word order + prepositions.
  * Both parsers are OPTIONAL dependencies in the same spirit as the embedder:
    if pymorphy3 / nltk are absent, `available()` is False and callers fall back
    to the previous word-based behaviour.  Nothing breaks, nothing is required.
  * Deterministic and fast (pure text work; no model call per fact).

Public API:
    available(lang)            -> bool
    parse(text, *, lang=None, ts=None) -> list[Proposition]
    predicate_slot(predicate)  -> str   (normalised key for indexing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------- models


@dataclass
class Proposition:
    """One action with its participants, in normalised (lemma) form."""

    predicate: str = ""
    agent: str = ""
    patient: str = ""
    recipient: str = ""
    instrument: str = ""
    location: str = ""
    time: str = ""
    tense: str = ""
    polarity: int = 1
    lang: str = ""
    raw: str = ""

    def slots(self) -> dict[str, str]:
        return {
            "agent": self.agent,
            "patient": self.patient,
            "recipient": self.recipient,
            "instrument": self.instrument,
            "location": self.location,
        }

    def signature(self) -> str:
        """Index key: predicate plus the roles that are filled."""
        parts = [f"p={self.predicate}"]
        for name, value in sorted(self.slots().items()):
            if value:
                parts.append(f"{name[0]}={value}")
        if self.polarity < 0:
            parts.append("neg")
        return "|".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate, "agent": self.agent, "patient": self.patient,
            "recipient": self.recipient, "instrument": self.instrument,
            "location": self.location, "time": self.time, "tense": self.tense,
            "polarity": self.polarity, "lang": self.lang, "raw": self.raw,
        }


# ------------------------------------------------------------------ resources

_RU_MORPH = None
_RU_TRIED = False
_EN_NLTK: dict[str, Any] = {}
_EN_TRIED = False

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё'_-]*")
_CYR = re.compile(r"[а-яё]", re.I)

_RU_NEG = {"не", "ни", "нет", "никогда", "ничего"}
# Interrogatives are not participants: leaving them in makes "кто" the agent of
# every question, so a strict role match can never fire.
_RU_WH = {"кто", "что", "кого", "кому", "кем", "чем", "чём", "где", "куда", "откуда",
          "когда", "какой", "какая", "какое", "какие", "каких", "каким", "чей", "чья",
          "чьё", "чьи", "сколько", "почему", "зачем", "который", "которая", "которые"}
_EN_WH = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how",
          "whatever", "whoever"}
_EN_NEG = {"not", "no", "never", "n't", "cannot"}
_EN_AUX_FUTURE = {"will", "shall", "'ll"}
_EN_LOC_PREP = {"in", "at", "on", "near", "inside", "outside"}
_EN_INSTR_PREP = {"with", "using", "via"}
_EN_RECIP_PREP = {"to", "for"}
_RU_LOC_PREP = {"в", "во", "на", "около", "возле", "внутри"}
_RU_INSTR_PREP = {"с", "со", "при", "посредством"}
_RU_RECIP_PREP = {"к", "ко", "для", "к"}

_RU_TENSE = {"past": "past", "pres": "present", "futr": "future"}
_EN_TENSE = {"VBD": "past", "VBP": "present", "VBZ": "present", "VBG": "present",
             "VBN": "past", "VB": "present", "MD": "future"}

# Case → role.  Animate/human nouns in the accusative are distinguishable from
# the nominative; inanimate ones are not (книга/книгу is, стол/стол is not), so
# word order decides those (see _parse_ru).
_RU_CASE_ROLE = {"ablt": "instrument", "loct": "location", "datv": "recipient"}


def available(lang: Optional[str] = None) -> bool:
    if lang == "ru":
        return _ru_morph() is not None
    if lang == "en":
        return _en_ready()
    return _ru_morph() is not None or _en_ready()


def _ru_morph():
    global _RU_MORPH, _RU_TRIED
    if _RU_MORPH is None and not _RU_TRIED:
        _RU_TRIED = True
        try:  # optional dependency
            import pymorphy3  # type: ignore

            _RU_MORPH = pymorphy3.MorphAnalyzer()
        except Exception:  # noqa: BLE001 - absence is a supported state
            _RU_MORPH = None
    return _RU_MORPH


def _en_ready() -> bool:
    global _EN_TRIED
    if _EN_TRIED:
        return bool(_EN_NLTK.get("tagger"))
    _EN_TRIED = True
    try:  # optional dependency
        import nltk  # type: ignore

        tagger = nltk.pos_tag
        from nltk.stem import WordNetLemmatizer  # type: ignore

        _EN_NLTK["tagger"] = tagger
        _EN_NLTK["lemma"] = WordNetLemmatizer()
    except Exception:  # noqa: BLE001
        _EN_NLTK.clear()
    return bool(_EN_NLTK.get("tagger"))


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for c in letters if _CYR.match(c))
    return "ru" if cyr / len(letters) > 0.4 else "en"


# ------------------------------------------------------------------ Russian

def _parse_ru(text: str) -> list[Proposition]:
    morph = _ru_morph()
    if morph is None:
        return []
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return []
    parsed: list[tuple[str, Any, str]] = []  # (token, tag, lemma)
    for tok in tokens:
        p = morph.parse(tok)[0]
        parsed.append((tok.lower(), p.tag, str(p.normal_form)))

    # A sentence may hold several actions; every verb gets its own proposition so
    # "купил машину и продал её" does not merge into one blob.  Copulas are not
    # actions ("ты будешь делать X" must yield `делать`, not `быть`), so they are
    # dropped whenever a real verb is present.
    _COPULA = {"быть", "стать", "являться", "становиться"}
    verb_idx = [i for i, (_t, tag, _l) in enumerate(parsed) if tag.POS in ("VERB", "INFN", "PRTF", "PRTS", "GRND")]
    real_verbs = [i for i in verb_idx if parsed[i][2] not in _COPULA]
    if real_verbs:
        verb_idx = real_verbs
    if not verb_idx:
        return []

    props: list[Proposition] = []
    for k, vi in enumerate(verb_idx):
        # Arguments belong to their own clause: without clamping the window,
        # "купил машину в Берлине и продал её брату" gives both verbs all four
        # arguments, inventing relationships that were never stated.
        win_start = (verb_idx[k - 1] + 1) if k > 0 else 0
        win_end = verb_idx[k + 1] if k + 1 < len(verb_idx) else len(parsed)
        prop = Proposition(lang="ru", raw=text[:300])
        _t, vtag, vlemma = parsed[vi]
        prop.predicate = vlemma
        prop.tense = _RU_TENSE.get(str(vtag.tense or ""), "")
        polarity = 1
        # negation: a negator in the two tokens before the verb
        for back in range(max(0, vi - 2), vi):
            if parsed[back][0] in _RU_NEG:
                polarity = -1
        prop.polarity = polarity

        seen_case: dict[str, str] = {}
        for i, (tok, tag, lemma) in enumerate(parsed):
            if i == vi or not (win_start <= i < win_end):
                continue
            if tag.POS not in ("NOUN", "NPRO", "ADJF", "ADJS"):
                continue
            if tok in _RU_NEG or tok in _RU_WH or lemma in _RU_WH:
                continue
            # "домашнее задание" must stay one argument, not "домашний" + "задание".
            if tag.POS in ("ADJF", "ADJS"):
                nxt = parsed[i + 1] if i + 1 < len(parsed) else None
                if nxt and nxt[1].POS in ("NOUN", "NPRO") and nxt[1].case not in (None, "nomn", "accs"):
                    continue
                if nxt and nxt[1].POS in ("NOUN", "NPRO"):
                    lemma = f"{lemma} {nxt[2]}"
                    tag = nxt[1]
                    tok = nxt[0]
            case = str(tag.case or "")
            prep = parsed[i - 1][0] if i > 0 else ""
            role = ""
            explicit = bool(prep in _RU_LOC_PREP | _RU_INSTR_PREP | _RU_RECIP_PREP)
            if case == "ablt":
                role = "instrument"
            elif case == "loct":
                role = "location"
            elif case == "datv":
                role = "recipient"
            elif case in ("nomn", "accs"):
                # Nominative = subject, accusative = object — but inanimate
                # nominative and accusative are identical forms, so position
                # relative to the verb decides.
                if case == "nomn" and i < vi and "agent" not in seen_case:
                    role = "agent"
                elif i > vi:
                    role = "patient" if "patient" not in seen_case else ""
                elif case == "accs":
                    role = "patient"
                else:
                    role = "agent"
            elif case in ("gent", "gen2"):
                role = "patient" if i > vi else ""
            if not role:
                continue
            if prep in _RU_RECIP_PREP and role == "patient":
                role = "recipient"
            elif prep in _RU_LOC_PREP and case in ("loct", "accs"):
                role = "location"
            elif prep in _RU_INSTR_PREP and case == "ablt":
                role = "instrument"
            if role in seen_case:
                continue
            # A role invented purely from position (no case/preposition backing)
            # adds noise: only keep positional guesses for the two core slots.
            if not explicit and role in ("recipient", "instrument", "location") and case in ("nomn", "accs"):
                continue
            seen_case[role] = lemma
            setattr(prop, role, lemma)
        props.append(prop)
    return props


# ------------------------------------------------------------------ English

def _lemmatize_en(words: list[str], tags: list[str]) -> list[str]:
    lemma = _EN_NLTK.get("lemma")
    if lemma is None:
        return words
    out = []
    for w, t in zip(words, tags):
        pos = "v" if t.startswith("VB") else ("n" if t.startswith("NN") else ("a" if t.startswith("JJ") else None))
        if pos is None:
            out.append(w.lower())
        else:
            try:
                out.append(str(lemma.lemmatize(w.lower(), pos)))
            except Exception:  # noqa: BLE001
                out.append(w.lower())
    return out


def _parse_en(text: str) -> list[Proposition]:
    if not _en_ready():
        return []
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return []
    tagger = _EN_NLTK["tagger"]
    try:
        tagged = tagger(tokens)
    except Exception:  # noqa: BLE001
        return []
    words = [w.lower() for w, _t in tagged]
    tags = [t for _w, t in tagged]
    lemmas = _lemmatize_en([w for w, _t in tagged], tags)

    verb_idx = [i for i, t in enumerate(tags) if t.startswith("VB") or t == "MD"]
    if not verb_idx:
        return []
    # Auxiliaries are not actions: "She did not attend the wedding" must yield
    # `attend` alone, not `do` + `attend`.
    _AUX = {"do", "does", "did", "have", "has", "had", "be", "is", "are", "was",
            "were", "am", "been", "being"}
    _MODALS = {"will", "shall", "would", "can", "could", "may", "might", "must", "should"}
    real = [i for i in verb_idx if lemmas[i] not in _AUX and lemmas[i] not in _MODALS]
    if real:
        verb_idx = real

    props: list[Proposition] = []
    for vi in verb_idx:
        if tags[vi] == "MD" and lemmas[vi] in {"will", "shall", "would", "can", "could", "may", "might", "must"}:
            # modal verbs are not the action themselves unless a bare verb follows
            if vi + 1 < len(tags) and tags[vi + 1] == "VB":
                continue
        prop = Proposition(lang="en", raw=text[:300])
        prop.predicate = lemmas[vi]
        prop.tense = _EN_TENSE.get(tags[vi], "")
        prop.polarity = -1 if any(w in _EN_NEG for w in words[max(0, vi - 2): vi + 2]) else 1
        if vi > 0 and words[vi - 1] in _EN_AUX_FUTURE:
            prop.tense = "future"

        # subject: nearest noun-ish token before the verb
        subj = ""
        for i in range(vi - 1, -1, -1):
            if words[i] in _EN_WH:
                break
            if tags[i].startswith(("NN", "PRP", "JJ")):
                subj = lemmas[i]
                break
            if tags[i] in ("IN", "TO"):
                break
        if subj:
            prop.agent = subj

        # objects: nouns after the verb, preposition decides the role
        seen = {"agent"}
        for i in range(vi + 1, len(tags)):
            if not tags[i].startswith(("NN", "PRP", "JJ", "CD")):
                continue
            if words[i] in _EN_WH:
                continue
            prep = ""
            for back in range(i - 1, max(-1, vi), -1):
                if tags[back] in ("IN", "TO", "RP"):
                    prep = words[back]
                    break
            role = "patient"
            if prep in _EN_RECIP_PREP:
                role = "recipient"
            elif prep in _EN_LOC_PREP:
                role = "location"
            elif prep in _EN_INSTR_PREP:
                role = "instrument"
            if role in seen:
                continue
            # Compound nouns are one argument ("the car engine"), and a bare noun
            # with no preposition after a filled object slot is a modifier, not a
            # new participant — inventing a recipient there was measured wrong.
            value = str(lemmas[i])
            if (role == "patient" and not prep and i + 1 < len(tags)
                    and tags[i + 1].startswith("NN")):
                value = f"{value} {lemmas[i + 1]}"
            seen.add(role)
            setattr(prop, role, value)
        props.append(prop)
    return props


# ---------------------------------------------------------------------- API

def parse(text: str, *, lang: Optional[str] = None, ts: Optional[float] = None) -> list[Proposition]:
    """Split `text` into propositions.  Returns [] when nothing can be parsed —
    callers must treat that as 'no structural signal', never as an error."""
    text = (text or "").strip()
    if len(text) < 6:
        return []
    lang = lang or detect_lang(text)
    props = _parse_ru(text) if lang == "ru" else _parse_en(text)
    for p in props:
        if ts is not None:
            p.time = _time_hint(text)
    return props


def _time_hint(text: str) -> str:
    """Coarse temporal marker; refined dates come from the fact's timestamp."""
    low = text.lower()
    for marker in ("today", "yesterday", "tomorrow", "last week", "last month", "last year",
                   "next week", "this week", "two weeks ago", "three weeks ago",
                   "сегодня", "вчера", "завтра", "на прошлой неделе", "неделю назад",
                   "в прошлом месяце", "месяц назад"):
        if marker in low:
            return marker
    return ""


_SPEAKER_RE = re.compile(r"^\s*([A-ZА-ЯЁ][\w'\-]{1,30})\s*:")
# Dialogue facts are written in the first person ("Caroline: I gave a speech"),
# while questions name the person ("When did Caroline give a speech?").  Without
# resolving that, the agent slot can never match and the structural channel stays
# silent on exactly the data it was built for (measured: 1529 propositions
# indexed, yet 0-2 slot candidates per question).
_FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "im", "i've", "ive",
                 "я", "меня", "мне", "мной", "мой", "моя", "моё", "мои", "мною", "себя"}


def speaker_of(text: str) -> str:
    m = _SPEAKER_RE.match(text or "")
    return m.group(1).lower() if m else ""


def resolve_first_person(props: list[Proposition], speaker: str) -> list[Proposition]:
    """Replace first-person role fillers with the speaker's name."""
    if not speaker:
        return props
    for p in props:
        for role, value in p.slots().items():
            if value and value in _FIRST_PERSON:
                setattr(p, role, speaker)
        if p.agent in _FIRST_PERSON:
            p.agent = speaker
    return props


def index_keys(props: Iterable[Proposition]) -> list[str]:
    """Keys a proposition should be findable by (predicate alone, and each role
    paired with the predicate)."""
    keys: list[str] = []
    for p in props:
        if not p.predicate:
            continue
        keys.append(f"pred:{p.predicate}")
        for name, value in p.slots().items():
            if value:
                keys.append(f"{name}:{value}")
                keys.append(f"role:{p.predicate}:{name}={value}")
    return keys
