"""Archive the dead ends that the auto-detector manufactured out of assistant prose.

Why this exists: until the fix in store.add_turn, the outcome detector ran over the
assistant's own replies as well as the user's.  An assistant reply is dense with the
words the marker list looks for ("error", "failed", "не работает", "упал"), so every
technical answer produced dead ends whose subject was the nearest word and whose
reason was a 120-character fragment of a table row.  In one live profile ALL EIGHTEEN
dead ends were of this kind, and they are surfaced to the agent with the highest
recall score — ahead of real knowledge.

This archives them (archived = 1, nothing is deleted) and reports what it did.
Reversible: set archived = 0 again.

Usage:  python scripts/clean_garbage_deadends.py [--db PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import sys
import time

MARKDOWNY = re.compile(r"[|#*`\n]")
# A subject is garbage when it is a number, shorter than 3 characters, or plain
# lowercase non-Latin prose with no term-like shape (the exact failure mode).
NUMBERISH = re.compile(r"^[\d\s.,%]+$")


def find_default_db() -> str:
    base = os.path.expandvars(r"%LOCALAPPDATA%\hermes\profiles")
    hits = [p for p in glob.glob(os.path.join(base, "*", "neuromatrix.db"))]
    if not hits:
        hits = glob.glob(os.path.join(base, "**", "neuromatrix.db"), recursive=True)
    if not hits:
        raise SystemExit("no neuromatrix.db found; pass --db PATH")
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


def is_garbage(subject: str, reason: str) -> str | None:
    s = (subject or "").strip()
    r = (reason or "").strip()
    if not s:
        return "empty subject"
    if NUMBERISH.match(s):
        return "subject is a number"
    if len(s) < 3:
        return "subject shorter than 3 characters"
    if MARKDOWNY.search(s) or MARKDOWNY.search(r):
        return "contains markdown"
    # plain lowercase non-Latin prose (Russian) with no Latin characters: the
    # signature of "the nearest word before the marker"
    if not re.search(r"[A-Za-z]", s) and not s[:1].isupper():
        return "subject is lowercase prose, not a term"
    if len(r) < 15:
        return "reason is a fragment"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = args.db or find_default_db()
    # The live agent holds this database open.  In WAL mode a writer can still get in,
    # but it has to wait for the current write to finish instead of failing outright —
    # "database is locked" here means "try again in a moment", not "cannot".
    con = sqlite3.connect(db, timeout=30.0)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute("PRAGMA journal_mode = WAL")
    except sqlite3.Error:
        pass
    rows = con.execute(
        "SELECT id, ts, json_extract(meta,'$.subject') s, "
        "json_extract(meta,'$.reason') r, json_extract(meta,'$.source') src "
        "FROM facts WHERE kind='deadend' AND archived=0 ORDER BY ts DESC").fetchall()
    print(f"db: {db}")
    print(f"active dead ends: {len(rows)}")
    doomed: list[tuple[int, str]] = []
    for r in rows:
        why = is_garbage(r["s"], r["r"])
        if why:
            doomed.append((int(r["id"]), why))
            print(f"  [{r['id']:>4}] {str(r['s'])[:24]!r:28s} -> {why}")
    if not doomed:
        print("nothing to archive")
        return 0
    if args.dry_run:
        print(f"\ndry run: would archive {len(doomed)}")
        return 0
    ids = [(i,) for i, _ in doomed]
    # The live agent writes to this file continuously, so the write may have to be
    # retried: "database is locked" with a holder that is actively writing is a
    # matter of when, not whether.
    last_err: Exception | None = None
    for attempt in range(1, 7):
        try:
            con.executemany("UPDATE facts SET archived = 1 WHERE id = ?", ids)
            con.commit()
            last_err = None
            break
        except sqlite3.OperationalError as e:
            last_err = e
            print(f"  attempt {attempt}: {e}; retrying in {attempt * 2}s")
            time.sleep(attempt * 2)
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass
            con = sqlite3.connect(db, timeout=30.0)
            con.row_factory = sqlite3.Row
    if last_err is not None:
        print(f"could not archive: {last_err}")
        return 1
    left = con.execute(
        "SELECT COUNT(*) c FROM facts WHERE kind='deadend' AND archived=0").fetchone()["c"]
    print(f"\narchived {len(doomed)}; active dead ends now: {left} "
          f"(reversible: UPDATE facts SET archived=0 WHERE id IN (...))")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
