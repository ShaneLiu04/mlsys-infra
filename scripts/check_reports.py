"""Static checks for the LaTeX reports (no TeX installation needed).

For every reports/*/*.tex with a \\documentclass:
  - the PDF exists next to it;
  - every \\includegraphics resolves through the report's \\graphicspath;
  - every \\bibitem is cited and every \\cite key is defined (bibtex reports: keys exist in the .bib);
  - warning only: the PDF is older than the .tex (meaningless on a fresh clone, useful locally).
Exit code 1 if any hard check fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTS = ("", ".pdf", ".png", ".jpg")


def graphic_dirs(tex: Path, text: str) -> list[Path]:
    dirs = [tex.parent]
    for src in (text, (ROOT / "reports/common/preamble.tex").read_text(encoding="utf-8")):
        m = re.search(r"\\graphicspath\{((?:\{[^}]*\})+)\}", src)
        if m:
            dirs += [tex.parent / d for d in re.findall(r"\{([^}]*)\}", m.group(1))]
            break
    return dirs


def strip(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def expand(tex: Path, text: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if "common/preamble" in name:
            return ""
        p = tex.parent / (name if name.endswith(".tex") else name + ".tex")
        return strip(p.read_text(encoding="utf-8")) if p.exists() else ""
    return re.sub(r"\\input\{([^}]+)\}", sub, text)


def is_report(tex: Path) -> bool:
    return re.search(r"^\s*\\documentclass", strip(tex.read_text(encoding="utf-8")), re.M) is not None


def check(tex: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    text = tex.read_text(encoding="utf-8")
    body = expand(tex, strip(text))
    rel = tex.relative_to(ROOT).as_posix()

    pdf = tex.with_suffix(".pdf")
    if not pdf.exists():
        errors.append(f"{rel}: missing PDF")
    elif pdf.stat().st_mtime + 1 < max(p.stat().st_mtime for p in tex.parent.rglob("*.tex")):
        warnings.append(f"{rel}: PDF older than .tex (rebuild?)")

    dirs = graphic_dirs(tex, body)
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        if not any((d / (name + e)).exists() for d in dirs for e in EXTS):
            errors.append(f"{rel}: figure not found: {name}")

    cited = set()
    for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", body):
        cited.update(k.strip() for k in grp.split(","))
    items = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", body)
    bibs = re.findall(r"\\bibliography\{([^}]+)\}", body)
    if items:
        defined = set(items)
        for k in items:
            if k not in cited:
                errors.append(f"{rel}: \\bibitem{{{k}}} never cited")
    elif bibs:
        defined = set()
        for b in bibs[0].split(","):
            bib = tex.parent / (b.strip() + ("" if b.strip().endswith(".bib") else ".bib"))
            defined |= set(re.findall(r"@\w+\{([^,\s]+),", bib.read_text(encoding="utf-8")))
    else:
        defined = set()
        if cited:
            errors.append(f"{rel}: \\cite used but no bibliography")
    if "*" not in cited:
        for k in sorted(cited - defined):
            errors.append(f"{rel}: \\cite{{{k}}} undefined")
    if not cited:
        errors.append(f"{rel}: no citations")
    return errors, warnings


def main() -> int:
    errors, warnings, n = [], [], 0
    for tex in sorted((ROOT / "reports").glob("*/*.tex")):
        if not is_report(tex):
            continue
        n += 1
        e, w = check(tex)
        errors += e
        warnings += w
    for w in warnings:
        print("  warn:", w)
    for e in errors:
        print("  error:", e)
    print(f"  {n} reports checked" + ("" if errors else ", OK"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
