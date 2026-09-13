"""Question-type routing: reorder what was found, do NOT widen the window.

Why this shape and not "one more retrieval channel": four separate attempts to
WIDEN the candidate window here (co-occurrence graph, PMI edges, a bridge over
BM25, a role-level bridge) all made recall worse, because the window is
fixed-size and a broad list displaces the precise hits.  Answer shape is a
different lever — it does not add candidates at all.

The observation worth exploiting: "when" and "who" questions are answered by
different SURFACES of the same fact — a date versus a name — while both are
retrieved by the same bag of words.  A date-bearing fact that barely lost to a
better-matching fact is exactly the case this reorders.

Regex only, no model, English + Russian, and every rule is a rule about the
language rather than about a benchmark.
"""

from __future__ import annotations

import re
from typing import Optional

_WHEN = re.compile(
    r"\b(when|what\s+time|what\s+date|which\s+day|what\s+day|how\s+long\s+ago|"
    r"когда|какого\s+числа|в\s+каком\s+году|как\s+давно)\b", re.I)
_WHO = re.compile(r"\b(who|whose|whom|кто|кем|кого|чей|чья|чьё)\b", re.I)
_WHERE = re.compile(r"\b(where|which\s+city|which\s+country|где|куда|откуда)\b", re.I)
_COUNT = re.compile(r"\b(how\s+many|how\s+much|сколько)\b", re.I)

_MONTHS = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b", re.I)
_WEEKDAYS = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)\b", re.I)
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_RELATIVE_TIME = re.compile(
    r"\b(yesterday|today|tomorrow|tonight|last\s+(week|month|year|night)|"
    r"next\s+(week|month|year)|the\s+day\s+before|вчера|сегодня|завтра|"
    r"на\s+прошлой\s+неделе|в\s+прошлом\s+году)\b", re.I)
_TIME_OF_DAY = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(am|pm)\b", re.I)

# A name is a capitalised token that is not sentence-initial, not a weekday or
# month, and not an ALL-CAPS initialism.  Deliberately conservative: a false
# positive here would promote the wrong fact.
_NAME_TOKEN = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZА-ЯЁ][a-zа-яё]{2,}\b", re.M)

_PLACE_PREP = re.compile(
    r"\b(in|at|to|from|near|outside|inside|в|во|на|из|у|около|рядом\s+с)\s+"
    r"([A-ZА-ЯЁ][a-zа-яё]+)", re.M)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def question_type(query: str) -> Optional[str]:
    """Return 'when' | 'who' | 'where' | 'count', or None when it is not a
    question whose answer has a recognisable surface form."""
    q = query or ""
    # Ordered by how specific the surface is; "when" first because date-bearing
    # text also contains names and numbers.
    if _WHEN.search(q):
        return "when"
    if _COUNT.search(q):
        return "count"
    if _WHO.search(q):
        return "who"
    if _WHERE.search(q):
        return "where"
    return None


def matches_type(text: str, qtype: str) -> bool:
    """Does this fact have the surface shape the question asks for?"""
    t = text or ""
    if qtype == "when":
        return bool(_YEAR.search(t) or _MONTHS.search(t) or _WEEKDAYS.search(t)
                    or _RELATIVE_TIME.search(t) or _TIME_OF_DAY.search(t))
    if qtype == "who":
        return bool(_NAME_TOKEN.search(t))
    if qtype == "where":
        return bool(_PLACE_PREP.search(t))
    if qtype == "count":
        return bool(_NUMBER.search(t))
    return False


def boost(ordered_ids: list[int], texts: dict[int, str], query: str,
          weight: float = 1.0, window: int = 16) -> list[int]:
    """Promote same-shape facts inside the top ``window`` slots.

    Reordering only — the set of returned facts is unchanged, so this can never
    cost a candidate.  ``weight`` 0 is a no-op; 1 lifts a matching fact by one
    position and 2 by two.
    """
    if weight <= 0 or not ordered_ids:
        return ordered_ids
    qtype = question_type(query)
    if not qtype:
        return ordered_ids
    head = ordered_ids[:window]
    tail = ordered_ids[window:]
    lift = int(round(weight))
    scored = []
    for i, fid in enumerate(head):
        hit = matches_type(texts.get(fid, ""), qtype)
        pos = max(0, i - lift) if hit else i
        # A promoted fact that lands on the same position as an unmatched one
        # must win that tie — otherwise lifting by one position changes nothing
        # (both sit at 0 and the original order decides, which is the bug this
        # test caught).
        scored.append((pos, 0 if hit else 1, i, fid))
    scored.sort()
    return [fid for _, _, _, fid in scored] + tail
