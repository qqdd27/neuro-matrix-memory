"""Install the neuromatrix memory provider into a Hermes user plugins dir.

Creates ``$HERMES_HOME/plugins/neuromatrix/`` (default HERMES_HOME:
``%LOCALAPPDATA%/hermes`` on Windows, ``~/.hermes`` elsewhere) by copying the
package files, so Hermes' own discovery (``plugins.memory``: bundled ->
user plugins -> entry points) lists and loads ``neuromatrix`` next to the
built-in providers.

Usage:  python scripts/install_hermes_plugin.py [--hermes-home PATH] [--force]
Uninstall: delete ``<HERMES_HOME>/plugins/neuromatrix``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

FILES = [
    "__init__.py",
    "provider.py",
    "store.py",
    "entities.py",
    "llm.py",
    "config_schema.py",
    "plugin.yaml",
]


def default_hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "hermes"
    return Path.home() / ".hermes"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hermes-home", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing neuromatrix plugin dir")
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent / "neuro_matrix"
    home = Path(args.hermes_home) if args.hermes_home else default_hermes_home()
    dest = home / "plugins" / "neuromatrix"
    if dest.exists() and not args.force:
        print(f"already installed at {dest} (use --force to overwrite)")
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in FILES:
        src = repo / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, dest / name)
    if missing:
        print(f"ERROR: missing source files in {repo}: {missing}")
        return 1
    print(f"installed neuromatrix -> {dest}")
    for name in FILES:
        print(f"  {name}  ({os.path.getsize(dest / name)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
