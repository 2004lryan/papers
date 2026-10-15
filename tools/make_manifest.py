#!/usr/bin/env python3
"""
make_manifest.py — regenerate `data/SHA256-MANIFEST.tsv` from the shipped data card.

Scope
-----
The manifest covers exactly what the paper's Data & Code Availability statement promises:
every source file and every derived file on which a number in the paper depends. That set
is enumerated by `outputs/120datacard_sha256.json`, the data card emitted by
`code/120datacard_sha256.py`; this script is a projection of it onto the published tree,
not an independent list. Keeping the two one-to-one is deliberate: a manifest assembled by
hand is a manifest that drifts.

The one row that is not a file in this repository is the raw Hass avocado image set: it is
large, third-party, and reached through its Mendeley DOI rather than re-hosted. Its path is
redacted to `<EXTERNAL_DATA_ROOT>/...` so that no local absolute path is published, and its
SHA-256 is still recorded so the copy a reader downloads can be checked.

Integrity rule
--------------
Every row is verified against the file on disk before it is written. A path that cannot be
resolved, or whose bytes hash to something other than the data card says, aborts the run
instead of being written out. The previous manifest was assembled by hand and listed 60
paths that did not exist as written; that failure mode is what this script exists to remove.

Usage
-----
    python tools/make_manifest.py             # verify, then rewrite the manifest
    python tools/make_manifest.py --check     # verify only, non-zero exit on any mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "outputs" / "120datacard_sha256.json"
MANIFEST = ROOT / "data" / "SHA256-MANIFEST.tsv"

# data-card path prefix -> published directory holding those files
SHIPPED: dict[str, str] = {
    "03data/": "data/",
    "04outputs/": "outputs/",
    "05logs/": "logs/",
    # The dependency lock is the one card row that already lives in this repository rather
    # than in the working tree's numbered directories: the card records it by the path it
    # has there (`08github/requirements.lock.txt`), and here it is simply the repository
    # root. Added 2026-08-21 (RYA-569) when the lock was registered in the data card so a
    # reader can check that the lock shipped is the lock the paper's numbers were produced
    # under; without this row the generator reported "no published location" and refused
    # to write, which is the correct failure — a row it cannot place must not be dropped.
    "08github/": "",
}
RAW_TOKEN = "${ARIS_RAW_ROOT}"
RAW_REDACTED = "<EXTERNAL_DATA_ROOT>"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shipped_path(card_path: str) -> Path | None:
    """Where a data-card path lives in the published tree, or None if not shipped.

    The card is emitted in whichever form the generator saw. Run in the working tree it
    records the numbered names (`04outputs/...`); run here, where those names are aliases,
    `Path.resolve()` reports the shipped ones (`outputs/...`). Both are accepted so that
    regenerating the card in this repository does not strand every row.
    """
    for prefix, shipped in SHIPPED.items():
        if card_path.startswith(prefix):
            return ROOT / shipped / card_path[len(prefix):]
    direct = ROOT / card_path
    if direct.is_file():
        return direct
    return None


def is_tracked(path: Path) -> bool:
    """Whether git tracks the file. A manifest row pointing at an untracked or ignored file
    passes every local check and is then missing from every clone — the same failure the
    published manifest already had once, in a different form."""
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False         # outside the repository: a clone certainly would not have it
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(rel)],
                           cwd=ROOT, capture_output=True)
    except OSError:
        return True          # git unavailable, or not a checkout: nothing to assert
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; do not rewrite")
    args = ap.parse_args()

    if not CARD.is_file():
        print(f"data card absent: {CARD}", file=sys.stderr)
        return 1
    card = json.loads(CARD.read_text(encoding="utf-8"))

    rows: list[tuple[str, str, str, str, str]] = []
    problems: list[str] = []
    external = 0

    for entry in card["rows"]:
        card_path = entry["path"]

        if card_path.startswith(RAW_TOKEN):
            external += 1
            rows.append((entry["kind"], entry["id"], entry["sha256"], str(entry["bytes"]),
                         card_path.replace(RAW_TOKEN, RAW_REDACTED, 1)))
            continue

        target = shipped_path(card_path)
        if target is None:
            problems.append(f"no published location for {card_path}")
            continue
        if not target.is_file():
            problems.append(f"absent from the published tree: {card_path}")
            continue
        actual = sha256(target)
        if actual != entry["sha256"]:
            problems.append(f"hash differs from the data card: {card_path}\n"
                            f"      card {entry['sha256']}\n      disk {actual}")
            continue
        if not is_tracked(target):
            problems.append(f"present but not tracked by git, so a fresh clone would not "
                            f"have it: {card_path}")
            continue
        # The path column must be one a reader of THIS tree can follow. For the numbered
        # prefixes that is the card path itself, because `tools/bootstrap_layout.py` ships
        # `04outputs/` and friends as aliases. The dependency lock has no such alias — it
        # sits at this repository's root — so emit its published path instead of the
        # working-tree path, which would resolve nowhere here. (RYA-569.)
        listed = card_path
        if card_path.startswith("08github/"):
            listed = card_path[len("08github/"):]
        rows.append((entry["kind"], entry["id"], entry["sha256"],
                     str(target.stat().st_size), listed))

    verified = len(rows) - external
    print(f"data card rows      : {len(card['rows'])}")
    print(f"verified on disk    : {verified}")
    print(f"external (by DOI)   : {external}")
    print(f"problems            : {len(problems)}")
    for p in problems:
        print(f"    - {p}")
    if problems:
        print("\nrefusing to write a manifest with unverified rows", file=sys.stderr)
        return 1

    body = "kind\tid\tsha256\tbytes\tpath\n" + "".join("\t".join(r) + "\n" for r in rows)

    if args.check:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.is_file() else ""
        if current != body:
            print("\nmanifest is out of date with respect to the data card", file=sys.stderr)
            return 1
        print(f"\nOK: {MANIFEST.relative_to(ROOT)} matches the data card, all rows verified.")
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(body, encoding="utf-8")
    print(f"\nwrote {MANIFEST.relative_to(ROOT)} — {len(rows)} rows "
          f"({verified} verified byte-for-byte, {external} external).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
