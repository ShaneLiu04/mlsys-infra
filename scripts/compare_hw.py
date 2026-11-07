"""Build the 5090-vs-4080 comparison from the committed jsonl files. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlsys_harness.compare import main  # noqa: E402

if __name__ == "__main__":
    main()
