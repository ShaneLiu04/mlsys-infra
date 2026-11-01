"""Build the C++ asst2 task systems into cpp/bin/.

Usage: python scripts/build_cpp.py [--cxx g++]
Flags: -O3 -march=native -std=c++17 -pthread -Wall (+ -static on Windows).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "cpp"
BIN = ROOT / "bin"
EXE = ".exe" if os.name == "nt" else ""

TARGETS = [
    ("tasksys_bench", ROOT / "tasksys" / "main.cpp"),
    ("contention", ROOT / "tasksys" / "contention.cpp"),
]


def build(cxx: str, name: str, src: Path) -> tuple[str, int, str]:
    flags = ["-O3", "-march=native", "-std=c++17", "-pthread", "-Wall"]
    if os.name == "nt":
        flags.append("-static")
    cmd = [cxx, *flags, str(src), "-o", str(BIN / (name + EXE))]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return name, p.returncode, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    args = ap.parse_args()
    BIN.mkdir(parents=True, exist_ok=True)
    fails = 0
    with ThreadPoolExecutor(len(TARGETS)) as ex:
        for name, rc, err in ex.map(lambda t: build(args.cxx, *t), TARGETS):
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}")
            if rc != 0 or err.strip():
                print(err)
            fails += rc != 0
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
