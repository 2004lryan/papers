"""
fetch_datasets.py: pointers and SHA-256 checksums for the public external
datasets used in the cross-domain leakage analysis.

We do NOT re-host third-party datasets. This script documents where to obtain
each dataset and the expected SHA-256 of the preprocessed file, so that a
reproduction can verify integrity against the values reported in the manuscript
data card. Download each dataset from its official source (respecting its own
license), place the files under 05data/<id>/, then run the analysis scripts.

The apple near-infrared benchmark is NOT public (XJAU restricted access) and is
therefore not listed here; request it from the corresponding author.

Usage:
    python scripts/fetch_datasets.py --list
"""

from __future__ import annotations

import argparse

# id -> (name, modality, official source, preprocessed-file SHA-256 [first 16 hex])
EXTERNAL_DATASETS = {
    "oliveoil": (
        "OliveOil (FT-IR, 4 countries)",
        "FT-IR spectra",
        "UCR Time Series Classification Archive: "
        "https://www.timeseriesclassification.com/description.php?Dataset=OliveOil",
        "472c5b73157b2e0f",
    ),
    "corn": (
        "Cargill corn (NIR, 3 instruments m5/mp5/mp6)",
        "NIR point spectra",
        "Eigenvector Research data sets: https://eigenvector.com/resources/data-sets/",
        "e28fd4be274a54ca",
    ),
    "tablet": (
        "IDRC 2002 'shootout' tablet (NIR, 2 instruments)",
        "NIR point spectra",
        "Eigenvector Research data sets (Software Shootout): "
        "https://eigenvector.com/resources/data-sets/",
        "129a32ec9e194e56",
    ),
    "mango": (
        "Anderson 2020 mango dry matter (NIR, 4 seasons)",
        "NIR point spectra",
        "Anderson et al. 2020, Mendeley Data: "
        "https://data.mendeley.com/datasets/46htwnp833",
        "86159e2df9244fa5",
    ),
    "kiwifruit": (
        "Wohlers kiwifruit (NIR, 2017-2019 seasons; the analysis keeps unit KK1)",
        "NIR point spectra",
        "Wohlers 2025, figshare: https://doi.org/10.6084/m9.figshare.21747983",
        "37157f2c42a5d441",
    ),
    "indianpines": (
        "Indian Pines (AVIRIS hyperspectral)",
        "Hyperspectral imaging",
        "Purdue / Univ. of the Basque Country (EHU): "
        "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        "cube ec2f8808710919d5 / gt 65c4687a8ab04f6d",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list dataset sources and checksums")
    parser.parse_args()

    print("External public datasets (download from official source; do not re-host):\n")
    for did, (name, modality, source, sha) in EXTERNAL_DATASETS.items():
        print(f"[{did}] {name}")
        print(f"    modality : {modality}")
        print(f"    source   : {source}")
        print(f"    SHA-256  : {sha}\n")
    print("Apple NIR benchmark: not public (XJAU restricted access); request from the corresponding author.")


if __name__ == "__main__":
    main()
