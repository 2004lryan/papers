#!/usr/bin/env python3
"""
bootstrap_layout.py — create the numbered directory aliases the analysis scripts expect.

Why this exists
---------------
Every script in `code/` resolves its inputs and outputs relative to the repository root
using the numbered directory names of the working tree in which the results were produced:

    ROOT = Path(__file__).resolve().parents[1]
    OUT  = ROOT / "04outputs"
    CSV  = ROOT / "03data" / "processed" / "pmc_real" / "PMC10253207_T16.csv"

The published tree uses shorter names (`code/`, `data/`, `outputs/`, `logs/`, `docs/`), and
`data/SHA256-MANIFEST.tsv` likewise records the numbered paths. The scripts are shipped
byte-identical to the ones that produced every number in the paper — they are deliberately
not rewritten, because a reproducibility mirror whose code differs from the code that ran
is worth less than one that needs a documented bootstrap step.

This script reconciles the two by creating the numbered names as aliases of the shipped
ones. On POSIX systems the aliases are symbolic links and are already committed, so a
fresh `git clone` needs nothing; running this script is then a no-op that only verifies.
On Windows, git checks symlinks out as plain text stubs unless `core.symlinks=true`, so
this script replaces the stubs with directory junctions.

Usage
-----
    python tools/bootstrap_layout.py            # create/repair, then verify
    python tools/bootstrap_layout.py --check    # verify only, non-zero exit if incomplete

Exit code 0 means every alias resolves and the manifest paths are reachable as written.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# alias -> shipped directory it stands for
ALIASES: dict[str, str] = {
    "02code": "code",
    "03data": "data",
    "04outputs": "outputs",
    "05logs": "logs",
    "06doc": "docs",
    # code/120datacard_sha256.py writes the Chinese data card to refine-logs/, the working
    # tree's name for its editorial-artefact directory; here that is docs/, so the script
    # regenerates the shipped docs/0NDATACARD_C5.md in place rather than failing on a
    # directory that does not exist.
    "refine-logs": "docs",
}

# directories the generators write into but do not create themselves
ENSURE_DIRS: tuple[str, ...] = ("docs/01manuscript",)


def _is_good_alias(alias: Path, target: Path) -> bool:
    """True when `alias` already resolves to `target` (symlink, junction, or bind)."""
    if not alias.exists():
        return False
    try:
        return alias.resolve() == target.resolve()
    except OSError:
        return False


def _make_alias(alias: Path, target: Path) -> str:
    """Create one alias. Returns a short word describing what was done."""
    if _is_good_alias(alias, target):
        return "ok"

    # A Windows checkout without symlink support leaves a text stub holding the link text.
    if alias.is_symlink() or (alias.is_file() and alias.stat().st_size < 256):
        alias.unlink()
    elif alias.exists():
        raise SystemExit(
            f"refusing to touch {alias}: it exists and is not an alias of {target}.\n"
            f"Move it aside and re-run."
        )

    rel = os.path.relpath(target, alias.parent)
    try:
        os.symlink(rel, alias, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass

    if os.name == "nt":  # directory junction needs no privilege
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=True, capture_output=True,
        )
        return "junction"

    raise SystemExit(f"could not create an alias {alias} -> {target} on this platform")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify only; do not create anything")
    args = ap.parse_args()

    problems: list[str] = []

    for name in ENSURE_DIRS:
        d = ROOT / name
        if d.is_dir():
            continue
        if args.check:
            problems.append(f"missing directory {name}")
        else:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  mkdir    {name}")

    for alias_name, target_name in ALIASES.items():
        alias, target = ROOT / alias_name, ROOT / target_name
        if not target.is_dir():
            problems.append(f"shipped directory {target_name}/ is absent")
            continue
        if args.check:
            if not _is_good_alias(alias, target):
                problems.append(f"{alias_name} does not resolve to {target_name}/")
            continue
        action = _make_alias(alias, target)
        print(f"  {action:<8} {alias_name} -> {target_name}/")

    # The point of the aliases is that the shipped manifest resolves as written.
    manifest = ROOT / "data" / "SHA256-MANIFEST.tsv"
    if manifest.is_file():
        unreachable = 0
        checked = 0
        for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            path = parts[4]
            if path.startswith("<"):        # external, reached by DOI
                continue
            checked += 1
            if not (ROOT / path).exists():
                unreachable += 1
        if unreachable:
            problems.append(f"{unreachable}/{checked} manifest paths still unreachable")
        else:
            print(f"  verified {checked}/{checked} manifest paths resolve as written")

    if problems:
        print("\nINCOMPLETE:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("\nOK: layout aliases in place; scripts in code/ can be run from the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
