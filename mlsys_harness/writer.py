"""JSONL writer that puts the run identity on the first line.

A result file that does not name its commit cannot be told apart from a later
rerun. ``provenance()`` records the commit, whether the tree was dirty, the
clock, the interpreter and the command line. Device fields are added by the
caller, because this module does not import torch.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def provenance(cwd: Path | None = None) -> dict:
    root = cwd or Path.cwd()
    commit = _git(["rev-parse", "HEAD"], root)
    dirty = _git(["status", "--porcelain"], root)
    return {
        "kind": "run",
        "git_commit": commit or None,
        "git_dirty": bool(dirty) if commit else None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
    }


class JsonlWriter:
    """Append-only JSONL. The first record is the provenance line."""

    def __init__(self, path: Path, cwd: Path | None = None, fresh: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            self.path.unlink()
        self._f = open(self.path, "a", encoding="utf-8")
        self._n = 0
        self.write(provenance(cwd))

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        self._n += 1

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
