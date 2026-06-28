"""Self-test for code.src.data — runs end-to-end on the synthetic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from code.src.data import (  # noqa: E402
    FREUIDDataset,
    build_transforms,
    class_weights,
    load_metadata,
    make_dataloaders,
    split_train_val,
    stratified_kfold_indices,
    TransformSpec,
)


def main() -> int:
    csv = REPO_ROOT / "dataset" / "train.csv"
    if not csv.exists():
        print(f"missing {csv} — run `python code/scripts/download_data.py --synthetic` first")
        return 1
    df = load_metadata(csv)
    print(f"loaded {len(df)} rows; columns={list(df.columns)}")
    print(f"label counts: {df['label'].value_counts().to_dict()}")
    print(f"type counts : {df['type'].value_counts().to_dict()}")

    train_df, val_df = split_train_val(df, val_fraction=0.25, seed=42)
    print(f"train={len(train_df)} val={len(val_df)}")
    tr_pos = (train_df["label"] == 1).mean()
    va_pos = (val_df["label"] == 1).mean()
    print(f"positive ratio: train={tr_pos:.3f} val={va_pos:.3f}")

    # 5-fold check.
    splits = list(stratified_kfold_indices(df, n_splits=5, seed=0))
    print(f"got {len(splits)} stratified folds")
    ratios = [df.iloc[va]['label'].mean() for _, va in splits]
    print(f"fold positive ratios: {[round(r, 3) for r in ratios]}")

    ds = FREUIDDataset(train_df, transform=build_transforms(TransformSpec(96, train=True)))
    img, label, meta = ds[0]
    print(f"sample shape={tuple(img.shape)} label={label} meta={meta}")
    assert img.shape == (3, 96, 96), img.shape
    assert 0 <= label <= 1

    cw = class_weights(train_df)
    print(f"class weights: {cw.tolist()}")

    tl, vl, pw = make_dataloaders(csv, image_size=96, batch_size=8, val_fraction=0.25, num_workers=0)
    print(f"train batches={len(tl)} val batches={len(vl)} pos_weight={pw.item():.3f}")
    x, y, m = next(iter(tl))
    print(f"first batch: x={tuple(x.shape)} y={y.shape} meta_ids={m['id'][:3]} meta_types={m.get('type', [])[:3]}")
    assert x.shape[0] == 8 and x.shape[1] == 3
    assert len(m["id"]) == x.shape[0]  # collate returns list-of-id per batch

    print("data self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
