"""Build the C++ versions of asst1 (prog1-prog6) into cpp/bin/.

Usage: python scripts/build_cpp.py [--cxx g++] [--jobs N]
Flags: -O3 -march=native -ffp-contract=off so the SIMD kernels match the scalar
reference bit for bit; prog2 is built once per VECTOR_WIDTH.
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


def targets() -> list[tuple[str, Path, list[str]]]:
    out = [
        ("prog1_mandelbrot_threads", ROOT / "prog1_mandelbrot_threads" / "main.cpp", []),
        ("prog3_mandelbrot_simd", ROOT / "prog3_mandelbrot_simd" / "main.cpp", []),
        ("prog4_sqrt", ROOT / "prog4_sqrt" / "main.cpp", []),
        ("prog5_saxpy", ROOT / "prog5_saxpy" / "main.cpp", []),
        ("prog6_kmeans", ROOT / "prog6_kmeans" / "main.cpp", []),
    ]
    for w in (2, 4, 8, 16):
        out.append((f"prog2_vecintrin_w{w}", ROOT / "prog2_vecintrin" / "main.cpp", [f"-DVECTOR_WIDTH={w}"]))
    if os.name == "nt":
        out.append(("topology", ROOT / "topology" / "topology.cpp", []))
    return out


def build(cxx: str, name: str, src: Path, extra: list[str]) -> tuple[str, int, str]:
    flags = ["-O3", "-march=native", "-ffp-contract=off", "-std=c++17", "-pthread", "-Wall", "-Wno-unused-function"]
    if os.name == "nt":
        flags.append("-static")
    cmd = [cxx, *flags, *extra, str(src), "-o", str(BIN / (name + EXE))]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return name, p.returncode, p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    BIN.mkdir(parents=True, exist_ok=True)
    fails = 0
    with ThreadPoolExecutor(args.jobs) as ex:
        for name, rc, err in ex.map(lambda t: build(args.cxx, *t), targets()):
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}")
            if rc != 0 or err.strip():
                print(err)
            fails += rc != 0
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
