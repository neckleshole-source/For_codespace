#!/usr/bin/env python3
"""Download the FREUID Challenge 2026 dataset from Kaggle.

Usage
-----
    pip install kaggle
    # Place kaggle.json at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME /
    # KAGGLE_KEY env vars). The competition slug is on the Kaggle page.
    python code/scripts/download_data.py --competition freuid-challenge-2026

The dataset is extracted into ``dataset/`` next to this repository root:

    dataset/
    ├── train/
    │   └── 000001.jpg
    ├── test/
    │   └── 000001.jpg
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv

If Kaggle credentials are missing or the network is unavailable, this
script exits with a clear instruction rather than failing silently. A
synthetic dataset can be generated locally with ``--synthetic`` so the
rest of the pipeline can be exercised end-to-end without the real data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def download_via_kaggle(competition: str) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(DATASET_DIR),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # The download is a zip — extract anything that landed in DATASET_DIR.
    for z in DATASET_DIR.glob("*.zip"):
        print(f"extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATASET_DIR)
        z.unlink()


def generate_synthetic(n_train: int = 256, n_test: int = 64) -> None:
    """Create a tiny synthetic dataset for pipeline testing only.

    Each "document" is a noisy random image; labels, types and digital
    flags are sampled so the stratified split has variety. This is NOT
    a model of the real data and will not produce meaningful predictions.
    """
    import numpy as np
    import pandas as pd

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "train").mkdir(exist_ok=True)
    (DATASET_DIR / "test").mkdir(exist_ok=True)

    from PIL import Image

    rng = np.random.default_rng(0)
    types = ["USA/DL", "CROATIA/ID", "SWITZERLAND/ID", "GERMANY/PP"]
    rows = []
    for i in range(1, n_train + 1):
        label = int(rng.integers(0, 2))
        is_digital = int(rng.integers(0, 2))
        doc_type = types[rng.integers(0, len(types))]
        rows.append({
            "id": f"{i:06d}",
            "image_path": f"train/{i:06d}.jpg",
            "label": label,
            "is_digital": is_digital,
            "type": doc_type,
        })
        img = (rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
        Image.fromarray(img).save(DATASET_DIR / "train" / f"{i:06d}.jpg", quality=85)
    pd.DataFrame(rows).to_csv(DATASET_DIR / "train.csv", index=False)

    test_rows = []
    for i in range(1, n_test + 1):
        test_rows.append({"id": f"{i:06d}", "image_path": f"test/{i:06d}.jpg"})
        img = (rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))
        Image.fromarray(img).save(DATASET_DIR / "test" / f"{i:06d}.jpg", quality=85)
    pd.DataFrame(test_rows).to_csv(DATASET_DIR / "test.csv", index=False)

    pd.DataFrame({"id": [r["id"] for r in test_rows], "label": [0.0] * n_test}).to_csv(
        DATASET_DIR / "sample_submission.csv", index=False
    )
    print(f"wrote synthetic dataset to {DATASET_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="freuid-challenge-2026")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate a tiny synthetic dataset for pipeline testing",
    )
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=64)
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic(args.n_train, args.n_test)
        return 0

    if not _has_kaggle_credentials():
        print(
            "ERROR: Kaggle credentials not found. Either:\n"
            "  - place kaggle.json at ~/.kaggle/kaggle.json, or\n"
            "  - set KAGGLE_USERNAME and KAGGLE_KEY env vars, or\n"
            "  - run with --synthetic to generate a tiny synthetic dataset\n"
            "    for pipeline testing only.",
            file=sys.stderr,
        )
        return 1

    download_via_kaggle(args.competition)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
