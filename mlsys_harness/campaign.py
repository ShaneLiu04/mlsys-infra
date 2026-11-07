"""Run a list of experiment steps, and refuse to call a step done on exit code alone.

The shell campaign deadlocked when it held ``flock`` and the lab script flocked
the same file again, and it marked the FP8 sweep done when the process exited 0
with a max absolute error of 234. This runner holds the lock itself, exports
``MLSYS_LOCK_HELD=1`` so the lab scripts skip their own flock, and treats a
failed check as a failed step even when the process exits 0.

``verify`` checks result files that are already on disk and runs nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


class CampaignError(RuntimeError):
    pass


class ExclusiveLock:
    """OS lock around one campaign. On POSIX this is flock, so it is the same
    lock the lab shell scripts take. On Windows it locks one byte with msvcrt,
    which is enough to test that two acquirers exclude each other.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._fh = None

    def __enter__(self) -> "ExclusiveLock":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+b")
            if os.name == "posix":
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            else:
                import msvcrt
                self._fh.seek(0, os.SEEK_END)
                if self._fh.tell() == 0:
                    self._fh.write(b"\0")
                    self._fh.flush()
                self._fh.seek(0)
                while True:
                    try:
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
        os.environ["MLSYS_LOCK_HELD"] = "1"
        return self

    def __exit__(self, *exc) -> None:
        os.environ.pop("MLSYS_LOCK_HELD", None)
        if self._fh is None:
            return
        if os.name != "posix":
            import msvcrt
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        self._fh.close()
        self._fh = None


def load_campaign(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "steps" not in data or "name" not in data:
        raise CampaignError(f"{path} needs name and steps")
    return data


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise CampaignError(f"missing result file {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise CampaignError(f"{path} is not an object or a list")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _select(rows: list[dict], where: dict) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def _value(root: Path, spec: dict) -> float:
    rows = _select(_rows(root / spec["file"]), spec.get("where", {}))
    if len(rows) != 1:
        raise CampaignError(f"{spec['file']} where {spec.get('where')} matched {len(rows)} rows, want 1")
    value = rows[0][spec["field"]]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{spec['field']} is not numeric: {value!r}")
    return float(value) * float(spec.get("scale", 1))


def check_spec(root: Path, spec: dict) -> None:
    """Raise CampaignError if a check does not hold."""
    kind = spec.get("type", "field")
    if kind == "field":
        got = _value(root, spec)
    elif kind == "ratio":
        den = _value(root, spec["den"])
        if den == 0:
            raise CampaignError(f"ratio denominator is 0 ({spec.get('id', spec)})")
        got = _value(root, spec["num"]) / den
    else:
        raise CampaignError(f"unknown check type {kind}")
    label = spec.get("id", spec.get("field", "check"))
    if "expect" in spec:
        tol = spec.get("tol_abs")
        if tol is None:
            tol = abs(spec["expect"]) * spec.get("tol_rel", 0.0)
        if abs(got - spec["expect"]) > tol:
            raise CampaignError(f"{label}: got {got:.6g}, expect {spec['expect']} ± {tol}")
    if "min" in spec and got < spec["min"]:
        raise CampaignError(f"{label}: got {got:.6g} < min {spec['min']}")
    if "max" in spec and got > spec["max"]:
        raise CampaignError(f"{label}: got {got:.6g} > max {spec['max']}")


def verify(campaign: dict, root: Path) -> list[str]:
    """Check every step's result constraints. Returns the step ids that passed."""
    passed = []
    for step in campaign["steps"]:
        for spec in step.get("checks", []):
            check_spec(root, spec)
        passed.append(step["id"])
    return passed


def run(campaign: dict, root: Path, only: str | None = None) -> None:
    """Run steps in order. The lock is held for the whole campaign, not per step,
    and it is not taken inside the child. A check failure stops the campaign.
    """
    root = Path(root)
    lock_path = campaign.get("lock")
    with ExclusiveLock(Path(lock_path) if lock_path else None):
        for step in campaign["steps"]:
            if only and step["id"] != only:
                continue
            env = os.environ.copy()
            env.update({k: str(v) for k, v in step.get("env", {}).items()})
            env["MLSYS_LOCK_HELD"] = "1"
            cwd = root / step.get("cwd", ".")
            print(f"=== {step['id']}", flush=True)
            try:
                proc = subprocess.run(step["cmd"], cwd=cwd, env=env,
                                      timeout=step.get("timeout_s", 3600))
            except subprocess.TimeoutExpired as exc:
                raise CampaignError(f"{step['id']} timed out") from exc
            if proc.returncode != 0:
                raise CampaignError(f"{step['id']} exited {proc.returncode}")
            for spec in step.get("checks", []):
                check_spec(root, spec)
            print(f"OK {step['id']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] not in ("verify", "run"):
        print("usage: python -m mlsys_harness.campaign verify|run CAMPAIGN.json", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    campaign = load_campaign(Path(args[1]) if Path(args[1]).is_absolute() else root / args[1])
    try:
        if args[0] == "verify":
            ids = verify(campaign, root)
            print(f"verified {len(ids)} steps")
        else:
            run(campaign, root)
    except CampaignError as exc:
        print(f"FAILED {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
