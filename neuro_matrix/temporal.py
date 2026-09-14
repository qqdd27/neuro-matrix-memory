"""Event-time resolution: turn a date phrase inside fact text into a real
calendar timestamp, anchored to the fact's own record time for anything
relative ("last week") or partial (a bare weekday, "on Tuesday").

The language law here is the temporal counterpart of morphology and roles:
"5 марта 2019" and "in 2019" both name a position on one timeline, and a
channel that only recognises DATE-SHAPED text (question_types.py's regexes)
cannot tell a genuine, resolvable date from an unrelated four-digit number or
an ambiguous weekday — it can only say "looks temporal", not "is this
specific day".

Same two properties as morphology.py:

* NARROWS, does not widen.  resolve() either returns a real point on the
  timeline or None — it never turns a maybe into a match.  A fact with no
  resolvable date stays exactly as unfindable-by-date as before; this can
  only make matching MORE precise than the shape-only regex, never broader.
* Anchor discipline: relative and partial expressions ("last week", "on
  Tuesday") are meaningless without a reference point, and the only honest
  reference point for a stored fact is the moment it was RECORDED
  (``anchor_ts``), never "now" — resolving against wall-clock now would make
  the same fact's date drift between runs, the same determinism requirement
  morphology.py has for index vs query.

Zero dependencies: everything here is stdlib (``datetime``, ``re``), so
unlike morphology.py there is no availability gate — it is always on.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import NamedTuple, Optional


class Resolved(NamedTuple):
    event_ts: float
    granularity: str  # "day" | "month" | "year"


_MONTHS_EN = {
    name: i for i, name in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"], start=1)
}
_MONTHS_EN.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
})
# Russian month names inflect (genitive after a day number "5 марта",
# prepositional after "в" "в мае") -- match by stem rather than enumerate
# every case form.  Sorted longest-stem-first so "март" is tried before the
# shorter "ма" (May) that would otherwise also prefix-match "марта".
_MONTHS_RU = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11,
    "декабр": 12,
}
_MONTHS_RU_SORTED = sorted(_MONTHS_RU.items(), key=lambda kv: -len(kv[0]))

_WEEKDAYS_EN = {name: i for i, name in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
     "sunday"])}
_WEEKDAYS_RU = {
    "понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3, "пятниц": 4,
    "суббот": 5, "воскресень": 6,
}
_WEEKDAYS_RU_SORTED = sorted(_WEEKDAYS_RU.items(), key=lambda kv: -len(kv[0]))

_YEAR = r"(1[89]\d{2}|20\d{2})"
_DAY = r"([0-3]?\d)"

# A bare four-digit number is NOT a date: "2019 dollars saved" and "model 2024"
# are the same shape as "since 2019".  The resolver therefore requires temporal
# CONTEXT around a year-only match — a preposition before it or a year word after
# it — instead of accepting any 4-digit figure (the false positive the old
# shape-only regex had and that this module inherited).
_YEAR_BEFORE_RU = re.compile(
    r"(?:^|[\s,;:(\[])(?:в|во|с|со|до|за|к|ко|около|примерно|уже|начиная\s+с)\s+$", re.I)
_YEAR_BEFORE_EN = re.compile(
    r"(?:^|[\s,;:(\[])(?:in|since|until|till|by|around|circa|from|during|of|year)\s+$", re.I)
_YEAR_AFTER = re.compile(r"^\s*(?:год|году|года|год[а-яё]*|г\.|гг\.|year|years)\b", re.I)


def _year_has_context(text: str, m: "re.Match[str]") -> bool:
    before, after = text[: m.start()], text[m.end():]
    return bool(_YEAR_AFTER.search(after) or _YEAR_BEFORE_RU.search(before)
                or _YEAR_BEFORE_EN.search(before))

# Ordered most-specific-first, the same discipline question_type() documents:
# a full date also contains a bare year, so the coarse pattern must never be
# tried first or it would win on every input.
_FULL_EN = re.compile(rf"\b([A-Za-z]+)\s+{_DAY}(?:st|nd|rd|th)?,?\s+{_YEAR}\b")
_FULL_EN_ALT = re.compile(rf"\b{_DAY}(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+{_YEAR}\b")
_FULL_RU = re.compile(rf"\b{_DAY}\s+([а-яё]+)\s+{_YEAR}\b", re.I)
_MONTH_DAY_EN = re.compile(rf"\b([A-Za-z]+)\s+{_DAY}(?:st|nd|rd|th)?\b")
_MONTH_DAY_RU = re.compile(rf"\b{_DAY}\s+([а-яё]+)\b", re.I)
_MONTH_YEAR_EN = re.compile(rf"\b([A-Za-z]+)\s+{_YEAR}\b")
_MONTH_YEAR_RU = re.compile(rf"\bв\s+([а-яё]+)\s+{_YEAR}\b", re.I)
_YEAR_ONLY = re.compile(_YEAR)

# Relative-time deltas.  This is the single source of truth: it started as a
# duplicate table in scripts/eval_locomo.py (bench-only, used to annotate
# context for the reader) and is reused here unchanged so the two never
# drift apart.
RELATIVE: list[tuple[re.Pattern, int]] = [
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


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1, "couple": 2, "few": 3,
    "один": 1, "одна": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "пару": 2, "несколько": 3,
}
_UNIT_DAYS = {
    "day": 1, "days": 1, "день": 1, "дня": 1, "дней": 1,
    "week": 7, "weeks": 7, "неделю": 7, "недели": 7, "недель": 7,
    "month": 30, "months": 30, "месяц": 30, "месяца": 30, "месяцев": 30,
    "year": 365, "years": 365, "год": 365, "года": 365, "лет": 365,
}
_NUM_ALT = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
_UNIT_ALT = "|".join(sorted(_UNIT_DAYS, key=len, reverse=True))

# Arbitrary counts.  The table above (RELATIVE) covers the expressions someone
# thought of in advance — "two weeks ago", "last month" — and that fixed list was
# the reason 139 time expressions in a 1451-fact corpus never became dates: "3 days
# ago" and "five weeks ago" simply were not in it.  This is the general form.
_AGO = re.compile(rf"\b(?P<n>\d+|{_NUM_ALT})\s*(?P<u>{_UNIT_ALT})\s+(?:ago|назад)\b", re.I)

# Seasons name a real calendar window, and are the case where the old code was not
# merely silent but WRONG: "in the summer of 2022" resolved to 2022-01-01, a date the
# text never claimed.  A season resolves to the middle of that season, at season
# granularity, so the answer can be rendered as "Summer 2022".
_SEASON_FIRST_MONTH = {
    "spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12,
    "весна": 3, "весной": 3, "лето": 6, "летом": 6, "осень": 9, "осенью": 9,
    "зима": 12, "зимой": 12,
}
_SEASON_ALT = "|".join(sorted(_SEASON_FIRST_MONTH, key=len, reverse=True))
_SEASON_PAT = re.compile(
    rf"\b(?:in\s+the\s+)?(?:(this|last|next)\s+)?({_SEASON_ALT})\b"
    rf"(?:\s+(?:of\s+)?(\d{{4}}))?", re.I)

# Vague expressions still place an event in time, coarsely: "recently" is not a
# date, but it is certainly within the last couple of weeks, and saying so at week
# granularity is honest — the alternative is claiming no time at all.
_APPROX = re.compile(
    r"\b(recently|lately|not long ago|a while ago|the other day|недавно|на днях)\b", re.I)
_APPROX_DAYS = {"на днях": -4, "the other day": -4}


# Extended rules (arbitrary "N ago", seasons, vague expressions).  Switchable so the
# improvement can be measured against the fixed-table behaviour it replaces rather
# than argued about.
EXTENDED_RULES = True


def _season_resolved(m: "re.Match[str]", anchor_ts: float) -> Optional[Resolved]:
    rel, name, year_s = m.group(1), m.group(2).lower(), m.group(3)
    first = _SEASON_FIRST_MONTH.get(name)
    if not first:
        return None
    anchor = _dt.datetime.fromtimestamp(float(anchor_ts))
    year = int(year_s) if year_s else anchor.year
    if not year_s and rel:
        rel = rel.lower()
        if rel == "last":
            year -= 1
        elif rel == "next":
            year += 1
    d = _safe_date(year, first, 15)
    return Resolved(d.timestamp(), "season") if d else None


def _ago_resolved(m: "re.Match[str]", anchor_ts: float) -> Optional[Resolved]:
    raw_n, unit = m.group("n").lower(), m.group("u").lower()
    n = int(raw_n) if raw_n.isdigit() else _NUM_WORDS.get(raw_n)
    days = _UNIT_DAYS.get(unit)
    if not n or not days:
        return None
    granularity = "day" if days <= 7 else ("month" if days <= 60 else "year")
    return Resolved(float(anchor_ts) - n * days * 86400.0, granularity)


def _month_num(name: str) -> Optional[int]:
    n = (name or "").lower()
    if n in _MONTHS_EN:
        return _MONTHS_EN[n]
    for stem, num in _MONTHS_RU_SORTED:
        if n.startswith(stem):
            return num
    return None


def _safe_date(year: int, month: int, day: int) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime(year, month, day)
    except ValueError:
        return None


def _nearest_weekday(anchor_ts: float, target_weekday: int) -> float:
    """The calendar day closest to ``anchor_ts`` that falls on
    ``target_weekday`` (Monday=0).  A bare "on Tuesday" in conversational
    text almost always names the Tuesday of that same week, not some
    arbitrary future or distant-past one."""
    anchor = _dt.datetime.fromtimestamp(float(anchor_ts))
    delta = (target_weekday - anchor.weekday()) % 7
    if delta > 3:
        delta -= 7
    return (anchor + _dt.timedelta(days=delta)).timestamp()


def resolve(text: str, anchor_ts: Optional[float] = None) -> Optional[Resolved]:
    """The most specific resolvable calendar date named in ``text``.

    ``anchor_ts`` is the fact's own record time — required to resolve a
    relative phrase ("last week"), a day-without-year ("March 5") or a bare
    weekday ("on Tuesday"), and ignored otherwise.  Returns None when nothing
    in the text names an actual point on the calendar: a bare number, an
    unrelated four-digit figure, "the third option" never match.
    """
    t = text or ""

    m = _FULL_EN.search(t)
    if m:
        month_name, day, year = m.groups()
        month = _month_num(month_name)
        if month:
            d = _safe_date(int(year), month, int(day))
            if d:
                return Resolved(d.timestamp(), "day")
    m = _FULL_EN_ALT.search(t)
    if m:
        day, month_name, year = m.groups()
        month = _month_num(month_name)
        if month:
            d = _safe_date(int(year), month, int(day))
            if d:
                return Resolved(d.timestamp(), "day")
    m = _FULL_RU.search(t)
    if m:
        day, month_name, year = m.groups()
        month = _month_num(month_name)
        if month:
            d = _safe_date(int(year), month, int(day))
            if d:
                return Resolved(d.timestamp(), "day")

    if anchor_ts is not None:
        anchor_year = _dt.datetime.fromtimestamp(float(anchor_ts)).year
        m = _MONTH_DAY_EN.search(t)
        if m and _month_num(m.group(1)):
            d = _safe_date(anchor_year, _month_num(m.group(1)), int(m.group(2)))
            if d:
                return Resolved(d.timestamp(), "day")
        m = _MONTH_DAY_RU.search(t)
        if m and _month_num(m.group(2)):
            d = _safe_date(anchor_year, _month_num(m.group(2)), int(m.group(1)))
            if d:
                return Resolved(d.timestamp(), "day")

    for rx, delta in RELATIVE:
        if rx.search(t):
            if anchor_ts is None:
                continue
            return Resolved(float(anchor_ts) + delta * 86400.0, "day")

    # General "N units ago" — anything the fixed table above did not enumerate.
    if anchor_ts is not None and EXTENDED_RULES:
        m = _AGO.search(t)
        if m:
            r = _ago_resolved(m, anchor_ts)
            if r:
                return r

    # Seasons: a real calendar window, and the case the old code answered WRONGLY
    # ("in the summer of 2022" -> 2022-01-01).  Checked before the bare-year rule,
    # which is what produced that wrong answer.
    if anchor_ts is not None and EXTENDED_RULES:
        m = _SEASON_PAT.search(t)
        if m:
            r = _season_resolved(m, anchor_ts)
            if r:
                return r

    # Coarse but honest: "recently" places the event within the last couple of weeks.
    if anchor_ts is not None and EXTENDED_RULES:
        m = _APPROX.search(t)
        if m:
            days = _APPROX_DAYS.get(m.group(1).lower(), -7)
            return Resolved(float(anchor_ts) + days * 86400.0, "week")

    m = _MONTH_YEAR_EN.search(t)
    if m and _month_num(m.group(1)):
        d = _safe_date(int(m.group(2)), _month_num(m.group(1)), 1)
        if d:
            return Resolved(d.timestamp(), "month")
    m = _MONTH_YEAR_RU.search(t)
    if m and _month_num(m.group(1)):
        d = _safe_date(int(m.group(2)), _month_num(m.group(1)), 1)
        if d:
            return Resolved(d.timestamp(), "month")

    if anchor_ts is not None:
        for stem, wd in _WEEKDAYS_RU_SORTED:
            if re.search(rf"\b{stem}[а-яё]*\b", t, re.I):
                return Resolved(_nearest_weekday(anchor_ts, wd), "day")
        for name, wd in _WEEKDAYS_EN.items():
            if re.search(rf"\b{name}\b", t, re.I):
                return Resolved(_nearest_weekday(anchor_ts, wd), "day")

    m = _YEAR_ONLY.search(t)
    while m:
        if _year_has_context(t, m):
            d = _safe_date(int(m.group(1)), 1, 1)
            if d:
                return Resolved(d.timestamp(), "year")
        m = _YEAR_ONLY.search(t, m.end())

    return None


def has_event_date(text: str, anchor_ts: Optional[float] = None) -> bool:
    return resolve(text, anchor_ts) is not None


_MONTH_NAMES_EN = ["", "January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"]


def format_human(event_ts: float, granularity: str = "day") -> str:
    """The date as a person writes it: "October 10, 2023" / "October 2023" / "2023".

    Why this exists next to the ISO form: a stored fact is dated machine-style
    (2023-10-10), but a question about it ("when did he lose his job?") is
    answered — by a person or by a gold standard — in words ("end of October
    2023").  Measured on the temporal category, a correct answer loses its score
    purely on FORM: "2023-10-10" and "end of October 2023" share no token, so
    token-F1 scores a right answer 0.00.  Handing the reader both forms does not
    change what the memory found or how it ranked anything; it removes a
    translation the reader was silently doing badly.
    """
    try:
        d = _dt.datetime.fromtimestamp(float(event_ts))
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
    g = (granularity or "day").lower()
    if g == "year":
        return f"{d.year}"
    if g == "season":
        # Which season contains this date?  December belongs to winter, which began
        # in the previous calendar year's December — the Northern-hemisphere
        # convention, stated plainly rather than implied.
        if d.month in (12, 1, 2):
            label, year = "Winter", d.year if d.month != 12 else d.year
            return f"{label} {year}"
        for name, first in (("Spring", 3), ("Summer", 6), ("Autumn", 9)):
            if first <= d.month < first + 3:
                return f"{name} {d.year}"
        return f"{d.year}"
    if g == "month":
        return f"{_MONTH_NAMES_EN[d.month]} {d.year}"
    if g == "week":
        return f"around {_MONTH_NAMES_EN[d.month]} {d.day}, {d.year}"
    return f"{_MONTH_NAMES_EN[d.month]} {d.day}, {d.year}"


# --------------------------------------------------------------------------
# Order intent: "last time" / "first time" ask for an EXTREME, not for the
# closest match.  This is the one place where the memory is allowed to reorder
# its answer by time, and it is not a heuristic guess: the question states the
# criterion itself.  Everything else that reordered the window (edge ordering,
# type routing) measured worse than leaving it alone; this is different because
# the order asked for is the order the question is about.
# --------------------------------------------------------------------------
_LAST_PAT = re.compile(
    r"\b(last|latest|most recent|most recently|recently|final|the other day)\b",
    re.IGNORECASE)
_FIRST_PAT = re.compile(
    r"\b(first|earliest|initially|originally|in the beginning|at first)\b",
    re.IGNORECASE)
_LAST_PAT_RU = re.compile(
    r"(последн\w*|недавн\w*|самый свеж\w*|напоследок)", re.IGNORECASE)
_FIRST_PAT_RU = re.compile(
    r"(впервые|перв\w*|изначальн\w*|самый ранн\w*|поначалу)", re.IGNORECASE)


def order_intent(query: str) -> Optional[str]:
    """'last', 'first', or None when the question does not ask for an extreme.

    "first" wins when both appear ("the first time and the last") because the
    earlier event is the narrower, more specific one to surface.
    """
    if not query:
        return None
    if _FIRST_PAT.search(query) or _FIRST_PAT_RU.search(query):
        return "first"
    if _LAST_PAT.search(query) or _LAST_PAT_RU.search(query):
        return "last"
    return None


# --------------------------------------------------------------------------
# Language-model extraction, for what patterns cannot reach.
#
# Rules resolve 79.5% of the turns that CONTAIN a time expression; the rest need
# understanding rather than matching ("right after the Japan trip", "when we
# moved", "the week I started the new job").  Mem0 reports extracting a time
# signature for every memory at write time; this is the local, opt-in equivalent.
#
# Two disciplines keep it honest.  First, the model may answer "no date" — most
# turns genuinely have none, and a fabricated date is worse than an absent one,
# because every downstream "when" question would inherit it.  Second, a fact that
# has been examined is recorded as examined, so a background pass never re-asks
# the same question forever (the failure mode that makes such features expensive
# and silent).
# --------------------------------------------------------------------------
_LLM_SYSTEM = (
    "You extract the calendar date of the EVENT described in one conversational "
    "turn. Reply with JSON only, no prose.\n"
    'Schema: {"date": "YYYY-MM-DD" or null, "precision": "day"|"month"|"year"|"season", '
    '"evidence": "<the words that gave the date, or empty>"}\n'
    "Rules:\n"
    "- If the turn names no time reference for the event, date MUST be null. Never "
    "invent a date and never fall back to the conversation date.\n"
    "- Resolve relative expressions against the conversation date given to you.\n"
    "- For a season or a whole month use the middle of that period and say so in "
    'the precision field ("this summer" -> July, precision "season").\n'
    "- A duration or a recurring habit is not a date: null.\n"
)
_PRECISION_MAP = {"day": "day", "month": "month", "year": "year", "season": "season"}


def resolve_with_llm(llm: Any, text: str, anchor_ts: Optional[float] = None) -> Optional[Resolved]:
    """Ask a language model when the event happened; None when it says it cannot tell."""
    if llm is None or not (text or "").strip():
        return None
    try:
        if hasattr(llm, "available") and not llm.available():
            return None
    except Exception:  # noqa: BLE001
        return None
    anchor_txt = ""
    if anchor_ts is not None:
        try:
            anchor_txt = _dt.datetime.fromtimestamp(float(anchor_ts)).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError, OverflowError):
            anchor_txt = ""
    user = (f"Conversation date: {anchor_txt or 'unknown'}\n"
            f"Turn: {text.strip()[:600]}")
    try:
        out = llm.chat_json([
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user},
        ])
    except Exception:  # noqa: BLE001 - an optional enrichment must not raise
        return None
    if not isinstance(out, dict):
        return None
    raw = out.get("date")
    if not isinstance(raw, str) or not raw.strip():
        return None
    m = _ISO.search(raw)
    if not m:
        return None
    d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if d is None:
        return None
    precision = str(out.get("precision") or "day").strip().lower()
    # Sanity: the model was told not to fall back to the conversation date, so a
    # date identical to the anchor with day precision is suspicious when the turn
    # carried no time words — cheap guard against exactly the failure we warned it
    # about, since a wrong date is inherited by every later "when" question.
    if anchor_ts is not None and precision == "day" and not has_event_date(text, anchor_ts):
        try:
            same = abs(d.timestamp() - float(anchor_ts)) < 86400.0
        except (ValueError, OSError, OverflowError):
            same = False
        if same:
            return None
    return Resolved(d.timestamp(), _PRECISION_MAP.get(precision, "day"))


_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
