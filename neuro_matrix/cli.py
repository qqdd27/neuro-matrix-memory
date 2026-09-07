"""NeuroMatrix CLI — ``hermes neuromatrix`` (optional surface).

Registered through ``register_cli(subparsers)`` next to the provider package;
also runnable standalone: ``python -m neuro_matrix.cli <command>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from .store import NeuroMatrixStore


def _resolve_db(args: argparse.Namespace) -> str:
    if getattr(args, "db", None):
        return args.db
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    # Try the provider config written by `hermes memory setup`.
    try:
        import yaml
        cfg_path = os.path.join(hermes_home, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            db = ((cfg.get("plugins") or {}).get("neuromatrix") or {}).get("db_path")
            if db:
                return str(db).replace("$HERMES_HOME", hermes_home)
    except Exception:
        pass
    return os.path.join(hermes_home, "neuromatrix.db")


def _cmd_status(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        for k, v in s.stats().items():
            print(f"{k}: {v}")
        print(f"llm_calls_remaining: {s.llm_budget_remaining()}")
        return 0
    finally:
        s.close()


def _cmd_search(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        hits = s.search(args.query, limit=int(args.limit))
        if not hits:
            print("(no results)")
            return 1
        for h in hits:
            src = h["source"]
            print(f"[{h['score']:.2f} | {src} | {h['kind']}] {h['text']}")
        return 0
    finally:
        s.close()


def _cmd_consolidate(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        rep = s.consolidate(force=True, max_llm_calls=int(args.llm_calls))
        print(rep)
        return 0
    finally:
        s.close()


def _cmd_remind(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        due = s.foresights_due()
        if not due:
            print("(no due foresights)")
            return 0
        for d in due:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(d["trigger_at"]))
            print(f"[#{d['foresight_id']} due {when}] {d['text']}")
        return 0
    finally:
        s.close()


def _cmd_skill_propose(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        rep = s.skill_propose(args.concept, out_dir=args.out)
        print(json.dumps(rep, ensure_ascii=False))
        return 0
    finally:
        s.close()


def _cmd_profile_card(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        rep = s.export_profile_card(out_dir=args.out)
        print(json.dumps(rep, ensure_ascii=False))
        return 0
    finally:
        s.close()


def _cmd_policy(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        print(json.dumps(s.policy_report(), ensure_ascii=False))
        return 0
    finally:
        s.close()


def _cmd_remind_cron(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        due = s.foresights_due()
        if not due:
            print("(none)")
            return 0
        for d in due:
            print(f"⏰ {d.get('text', '')[:300]}")
        return 0
    finally:
        s.close()


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Cold start: replay recent user/assistant rows from a Hermes session DB
    (``state.db``) into the graph.  Heuristic table discovery: any table with a
    text-ish column and (optionally) role/session columns."""
    import sqlite3 as _sq

    src = args.source
    if not src:
        hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        src = os.path.join(hermes_home, "state.db")
    if not os.path.exists(src):
        print(f"source db not found: {src}", file=sys.stderr)
        return 2
    conn = _sq.connect(src)
    conn.row_factory = _sq.Row
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    except _sq.Error as e:
        print(f"cannot read source: {e}", file=sys.stderr)
        return 2
    picked = []
    for t in tables:
        if t in ("sqlite_sequence", "facts_fts"):
            continue
        try:
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
        except _sq.Error:
            continue
        text_col = next((c for c in cols if c.lower() in
                         ("text", "content", "body", "message", "user_content")), None)
        if text_col:
            picked.append((t, text_col, "role" in cols))
    if not picked:
        print("no candidate message tables found — pass --source with a Hermes state.db", file=sys.stderr)
        return 2
    store = NeuroMatrixStore(_resolve_db(args))
    total = 0
    try:
        for t, text_col, has_role in picked:
            try:
                q = f"SELECT * FROM '{t}' ORDER BY rowid DESC LIMIT {int(args.limit)}"
                for row in conn.execute(q):
                    txt = row[text_col]
                    if not isinstance(txt, str) or len(txt) < 10:
                        continue
                    role = str(row["role"]).lower() if has_role and row["role"] else ""
                    if role in ("assistant", "user", "bot", "agent"):
                        fid = store.remember(txt[:1500], source=f"ingest:{t}",
                                             session_id=f"ingest-{t}")
                    elif has_role:
                        continue
                    else:
                        fid = store.remember(txt[:1500], source=f"ingest:{t}")
                    if fid is not None:
                        total += 1
            except _sq.Error:
                continue
        print(f"ingested {total} anchor-bearing messages from {picked[0][0]}" +
              (f" (+{len(picked)-1} more tables)" if len(picked) > 1 else ""))
        return 0
    finally:
        store.close()
        conn.close()


def _cmd_export(args: argparse.Namespace) -> int:
    s = NeuroMatrixStore(_resolve_db(args))
    try:
        s.backup_to(args.out)
        print(f"backup written: {args.out}")
        return 0
    finally:
        s.close()


def _cmd_import_(args: argparse.Namespace) -> int:
    if not os.path.exists(args.infile):
        print(f"backup not found: {args.infile}", file=sys.stderr)
        return 2
    dest = _resolve_db(args)
    if os.path.exists(dest):
        bak = dest + ".pre-import"
        shutil.copyfile(dest, bak)
        print(f"existing store backed up to {bak}")
    shutil.copyfile(args.infile, dest)
    for suffix in ("-wal", "-shm"):
        p = dest + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"store restored from {args.infile} -> {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="SQLite store path (default: from config.yaml / $HERMES_HOME/neuromatrix.db)")
    p = argparse.ArgumentParser(prog="neuromatrix", parents=[common],
                                description="NeuroMatrix memory CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("status", parents=[common], help="store statistics + LLM budget")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser("search", parents=[common], help="associative recall")
    sp.add_argument("query")
    sp.add_argument("--limit", default=8)
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("consolidate", parents=[common], help="run the sleep cycle now")
    sp.add_argument("--llm-calls", default=4)
    sp.set_defaults(func=_cmd_consolidate)

    sp = sub.add_parser("remind", parents=[common], help="list due foresights")
    sp.set_defaults(func=_cmd_remind)

    sp = sub.add_parser("remind-cron", parents=[common],
                        help="cron-friendly: one line per due foresight, exit 0 always")
    sp.set_defaults(func=_cmd_remind_cron)

    sp = sub.add_parser("skill-propose", parents=[common],
                        help="distill a decision concept into a reviewable skill draft")
    sp.add_argument("concept")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=_cmd_skill_propose)

    sp = sub.add_parser("profile-card", parents=[common],
                        help="write the reviewable persona card (PersonaMem mirror)")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=_cmd_profile_card)

    sp = sub.add_parser("policy", parents=[common],
                        help="learned-policy digest from the memory-ops journal")
    sp.set_defaults(func=_cmd_policy)

    sp = sub.add_parser("ingest", parents=[common], help="cold-start from a session DB (state.db)")
    sp.add_argument("source", nargs="?", default=None)
    sp.add_argument("--limit", default=2000)
    sp.set_defaults(func=_cmd_ingest)

    sp = sub.add_parser("export", parents=[common], help="full snapshot to a file")
    sp.add_argument("out")
    sp.set_defaults(func=_cmd_export)

    sp = sub.add_parser("import", parents=[common], help="restore a snapshot (replaces the store)")
    sp.add_argument("infile")
    sp.set_defaults(func=_cmd_import_)
    return p


def register_cli(subparsers) -> None:
    """Hermes CLI integration hook: `hermes neuromatrix ...`."""
    parser = build_parser()
    subparsers.add_parser("neuromatrix", help="NeuroMatrix memory commands",
                          parents=[argparse.ArgumentParser(add_help=False)],
                          conflict_handler="resolve")
    # Hermes dispatches on the subparser's command names; delegate to main.
    _ = parser  # full argparse tree stays available via `python -m neuro_matrix.cli`


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
