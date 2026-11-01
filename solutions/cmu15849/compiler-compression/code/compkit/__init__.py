"""Minimal reproductions of ML-compiler and model-compression papers from CMU 15-849."""

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[5] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
