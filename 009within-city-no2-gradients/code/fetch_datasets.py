"""List and verify the cleaned tables that the analysis reads.

The tables are not redistributed in this repository (see data/DATA.md for sources and licences); they are
available from the corresponding author on reasonable request. Place each file at the path listed here,
relative to the repository root, then run

    python code/fetch_datasets.py --list      # paths and SHA-256 checksums
    python code/fetch_datasets.py --verify    # check the SHA-256 of every file that is present
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# id -> (title, path relative to the repository root, SHA-256)
TABLES: dict[str, tuple[str, str, str]] = {
    "station-annual-panel": (
        "Station annual panel, 2015-2024",
        "data/站点年面板-2015_2024.csv",
        "732a01f4a452bb55e613ef24550c280b1243deed24b67f9ab4dc54bb87ce3f36",
    ),
    "station-register": (
        "Station register with coordinates",
        "data/国控站点元数据.csv",
        "29964754dfafcac06390223bff667d2edb847b0ea49bc3843ccbab25bef2eec1",
    ),
    "builtup-exposure": (
        "Built-up exposure at 500 m and 1000 m",
        "data/站点建成区暴露.csv",
        "3acbc9f46a30fe6cc61e30516c1fd02387d07028b0109695a7afc53c27b52c00",
    ),
    "provincial-nev-stock": (
        "Provincial new energy vehicle stock, 2017-2023",
        "data/省级新能源汽车保有量-公安部口径-2017_2023.csv",
        "36f46632fc39ec649be3536cc1b4194f28cff87815912be6a6a818c64d4fa4d9",
    ),
    "city-vehicle-stock": (
        "City panel with vehicle stock, 2017-2023",
        "data/城市面板-2017_2023.csv",
        "10876f2b75a9d40492fe04ad1ea44d70e04408e2dfd013e7767771c08fb937d7",
    ),
    "city-covariates": (
        "City covariates, 2017-2023",
        "data/国家统计局-城市级协变量-2017_2023.csv",
        "c48c50578c424e0a6965b1d7c466095b4e09eef82324f550bd24a6bdb7ed610f",
    ),
    "provincial-covariates": (
        "Provincial covariates, 2017-2023",
        "data/建模面板-2017_2023.csv",
        "e2c47e5c9643c2e2ec83265a3004e11383de8b05a5b70253f8e99fc10bcb6edf",
    ),
    "city-meteorology": (
        "City meteorology, 2015-2024 (workbook)",
        "outputs/29_气象构建过程.xlsx",
        "cadf7cd91a6f2d841a565f6d130db46c23d6b549b5c4c37fdc60c8279e921688",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the tables, their paths and checksums")
    parser.add_argument("--verify", action="store_true", help="verify the checksum of every table that is present")
    args = parser.parse_args()
    if not (args.list or args.verify):
        parser.print_help()
        return
    bad = 0
    for tid, (title, rel, digest) in TABLES.items():
        path = ROOT / rel
        if args.list:
            print(f"[{tid}] {title}\n    path   : {rel}\n    SHA-256: {digest}")
        if args.verify:
            if not path.exists():
                print(f"MISSING  {rel}")
                bad += 1
            elif sha256(path) != digest:
                print(f"MISMATCH {rel}")
                bad += 1
            else:
                print(f"OK       {rel}")
    if args.verify:
        print(f"{len(TABLES) - bad}/{len(TABLES)} tables present with the expected checksum")


if __name__ == "__main__":
    main()
