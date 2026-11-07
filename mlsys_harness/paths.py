"""Where a lab reads and writes raw measurements.

The 5090 tree is ``artifacts/raw/gpu`` and the 4080 SUPER tree is
``artifacts/raw/ada4080``. Those names stay, because the reports link to them.
A run on a new card sets ``MLSYS_RAW_DIR`` and writes a third tree instead of
overwriting either one. Per-lab variables (``P2_OUT``, ``FP8_OUT``) still win,
so the commands that already used them keep working.
"""

from __future__ import annotations

import os
from pathlib import Path

# slug -> directory names to try, in order. Reading falls through; writing does not.
READ_DIRS = {
    "rtx5090": ("rtx5090", "gpu"),
    "rtx4080s": ("rtx4080s", "ada4080"),
}


def raw_dir(lab: Path, *legacy_env: str) -> Path:
    """Output directory for a new run. Explicit env overrides, then the 5090 tree."""
    for key in (*legacy_env, "MLSYS_RAW_DIR"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    return Path(lab) / "artifacts" / "raw" / "gpu"


def read_raw(lab: Path, slug: str) -> Path:
    """Existing result directory for a card. Raises if none of the aliases exist."""
    base = Path(lab) / "artifacts" / "raw"
    for name in READ_DIRS.get(slug, (slug,)):
        path = base / name
        if path.is_dir():
            return path
    tried = ", ".join(str(base / n) for n in READ_DIRS.get(slug, (slug,)))
    raise FileNotFoundError(f"no raw directory for {slug}: {tried}")
