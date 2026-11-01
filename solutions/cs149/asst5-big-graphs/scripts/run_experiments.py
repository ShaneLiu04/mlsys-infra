"""Build graphbench, run both campaigns, and redraw figures.

campaign: five graph families, all BFS variants and PageRank -> artifacts/raw/campaign.jsonl.
extra: edge-range splitting, RMAT scale 16-23 and three graph seeds -> artifacts/raw/extra.jsonl.
Each takes a few minutes on 4 cores; scale 23 needs about 2 GB of memory.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cpp" / "graphbench.cpp"
EXE = ROOT / "cpp" / "bin" / ("graphbench.exe" if sys.platform == "win32" else "graphbench")
RAW = ROOT / "artifacts" / "raw"


def main() -> int:
    EXE.parent.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-O3", "-fopenmp", "-std=c++17", "-o", str(EXE), str(SRC)], check=True)
    subprocess.run([str(EXE), "campaign", "--out", str(RAW / "campaign.jsonl"), "--reps", "3"], check=True)
    subprocess.run([str(EXE), "extra", "--out", str(RAW / "extra.jsonl"), "--reps", "5"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_graphs.py")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
