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
