"""Inference and submission generation for the FREUID pipeline.

Loads one or more checkpoints, runs inference with horizontal-flip TTA,
and writes a submission CSV in the exact row order of
``dataset/sample_submission.csv``.

Usage::

    python -m code.src.infer \
        --ckpt runs/baseline/best.pt \
        --test-csv dataset/test.csv \
        --sample-submission dataset/sample_submission.csv \
        --out submission.csv

Multiple checkpoints can be averaged by passing ``--ckpt`` multiple
times. Image size defaults to the value saved in the checkpoint config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.src.data import FREUIDDataset, TransformSpec, build_transforms, load_metadata  # noqa: E402
from code.src.model import build_model  # noqa: E402


def _flip_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_hflip: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Return (ids, scores) where scores is shape ``(N,)`` float32 in [0, 1]."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_ids: list[str] = []
    for x, _y, meta in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
            logits = model(x)
            if use_hflip:
                logits = logits + model(_flip_batch(x))
                logits = logits / 2.0
        scores = torch.sigmoid(logits.float()).cpu().numpy()
        all_scores.append(scores)
        all_ids.extend(list(meta["id"]))
    return all_ids, np.concatenate(all_scores)


def write_submission(
    ids: Sequence[str],
    scores: np.ndarray,
    sample_submission_csv: str | Path,
    out_csv: str | Path,
) -> None:
    """Write ``id,label`` aligned to ``sample_submission.csv`` row order.

    The Kaggle leaderboard joins on the row id, so the submission MUST
    have exactly the same id order as ``sample_submission.csv``. Any
    test ids missing from our predictions are filled with 0.5 (neutral);
    any extra predictions are appended at the end as a safety net.
    """
    sample = pd.read_csv(sample_submission_csv)
    sample["id"] = sample["id"].astype(str)
    pred_df = pd.DataFrame({"id": [str(i) for i in ids], "label": scores.astype(float)})
    out = sample[["id"]].merge(pred_df, on="id", how="left")
    if out["label"].isna().any():
        missing = out["label"].isna().sum()
        print(f"warning: {missing} ids missing from predictions, filling with 0.5")
        out["label"] = out["label"].fillna(0.5)
    if len(pred_df) > len(out):
        extras = pred_df[~pred_df["id"].isin(out["id"])]
        print(f"warning: {len(extras)} extra predictions not in sample_submission, appending")
        out = pd.concat([out, extras], ignore_index=True)
    out["label"] = out["label"].clip(0.0, 1.0)
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} rows to {out_csv}")
    print(out.head().to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", action="append", required=True, help="checkpoint path (repeat for ensemble)")
    parser.add_argument("--test-csv", default="dataset/test.csv")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--sample-submission", default="dataset/sample_submission.csv")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None, help="override checkpoint image size")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-hflip", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = load_metadata(args.test_csv, root=args.dataset_root)
    print(f"loaded {len(test_df)} test rows")

    all_scores = []
    all_ids = None
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        backbone = cfg["backbone"]
        pretrained = cfg.get("pretrained", True) and not args.no_pretrained
        drop_rate = cfg.get("drop_rate", 0.1)
        image_size = int(args.image_size or cfg["image_size"])
        print(f"loading {ckpt_path}: backbone={backbone} image_size={image_size}")

        model = build_model(backbone, pretrained=pretrained, drop_rate=drop_rate).to(device)
        model.load_state_dict(ckpt["model_state"])

        ds = FREUIDDataset(
            test_df,
            transform=build_transforms(TransformSpec(image_size=image_size, train=False)),
            has_labels=False,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        ids, scores = predict_with_tta(model, loader, device, use_hflip=not args.no_hflip)
        if all_ids is None:
            all_ids = ids
        else:
            assert ids == all_ids, "checkpoint predictions out of order"
        all_scores.append(scores)

    ensemble = np.mean(np.stack(all_scores, axis=0), axis=0)
    write_submission(all_ids, ensemble, args.sample_submission, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
