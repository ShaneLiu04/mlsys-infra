"""Check the numbers quoted in the lab write-ups against the raw jsonl.

Usage: python scripts/check_claims.py
Exit 1 if any published number does not match the file it came from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlsys_harness.campaign import CampaignError, check_spec  # noqa: E402


def main() -> int:
    claims = json.loads((ROOT / "claims.json").read_text(encoding="utf-8"))["claims"]
    failed = 0
    for spec in claims:
        try:
            check_spec(ROOT, spec)
        except CampaignError as exc:
            failed += 1
            print(f"  FAIL {exc}")
    if failed:
        print(f"{failed} of {len(claims)} claims failed")
        return 1
    print(f"{len(claims)} claims match the raw files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
