"""Data loading and stratified splits for FREUID.

Reads the CSV metadata shipped with the dataset and exposes a PyTorch
``Dataset`` plus a stratified train/val split keyed on ``(label, type)``
so local validation mirrors the cross-document distribution that the
private test set is drawn from.

Expected CSV columns (``train.csv``):

    id, image_path, label, is_digital, type

``label`` is 0 for bona-fide and 1 for fraudulent. ``type`` has the
form ``<COUNTRY>/<DOCUMENT-TYPE>`` (e.g. ``USA/DL``).

The dataset ships with ``test.csv`` containing ``id`` and ``image_path``
only — labels are withheld by Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# Image reading is single-threaded by default — these workers parallelize it.
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_metadata(csv_path: str | Path, root: str | Path | None = None) -> pd.DataFrame:
    """Load a metadata CSV and resolve image paths relative to ``root``.

    If ``root`` is None, the directory containing the CSV is used. The
    returned frame has a new ``abs_path`` column with the absolute path
    that the dataset should open.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    root = Path(root) if root is not None else csv_path.parent
    df["abs_path"] = df["image_path"].apply(lambda p: str((root / p).resolve()))
    # Some Kaggle datasets ship images with different capitalisation;
    # normalise for the sanity check below.
    df["abs_path"] = df["abs_path"].apply(lambda p: str(Path(p)))
    return df


def stratified_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) pairs stratified on (label, type).

    Combines the two columns into a single categorical key for sklearn's
    StratifiedKFold; falls back to label-only stratification if the
    combined key has too few samples per class for the requested fold
    count.
    """
    df = df.reset_index(drop=True)
    if "type" in df.columns:
        # Combine label and type into one string key so the fold keeps
        # both the class balance and the document-type mix.
        combo = df["label"].astype(str) + "|" + df["type"].astype(str)
        counts = combo.value_counts()
        min_per_class = int(counts.min()) if len(counts) else 0
        if min_per_class >= n_splits:
            y = combo.values
        else:
            y = df["label"].values
    else:
        y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        yield tr, va


def split_train_val(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single train/val split stratified on (label, type)."""
    df = df.reset_index(drop=True)
    n_splits = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
    for tr, va in stratified_kfold_indices(df, n_splits=n_splits, seed=seed):
        return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    raise RuntimeError("stratified split produced no folds")


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights for the binary fraud task.

    Use as ``pos_weight`` in ``BCEWithLogitsLoss`` or to rebalance a
    weighted sampler.
    """
    counts = df["label"].value_counts().to_dict()
    n0, n1 = counts.get(0, 1), counts.get(1, 1)
    # weight[i] = N / (K * count_i); here K=2 so each weight is
    # N / (2 * count_i) which normalises to sum=2.
    n = n0 + n1
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32)


@dataclass
class TransformSpec:
    image_size: int = 224
    train: bool = True


def build_transforms(spec: TransformSpec):
    """Albumentations transforms for train (heavy) and eval (light).

    The heavy augmentations are tuned for ID documents: mild geometric,
    mild colour, JPEG/gaussian noise. They are deliberately NOT extreme
    because the training set already contains physical and GenAI
    artefacts that would be destroyed by overly aggressive warps.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    size = spec.image_size
    if spec.train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=int(size * 1.15)),
                A.PadIfNeeded(min_height=int(size * 1.15), min_width=int(size * 1.15), border_mode=0, fill=0),
                A.RandomCrop(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.05), translate_percent=(0.02, 0.02), rotate=(-3, 3), p=0.4, fill=0),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=5, p=1.0),
                    ],
                    p=0.2,
                ),
                A.ImageCompression(quality_range=(60, 95), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, fill=0),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _pil_loader(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


class FREUIDDataset(torch.utils.data.Dataset):
    """PyTorch dataset for FREUID train/val/test images.

    Each item returns ``(image_tensor, label, meta)`` where ``meta`` is a
    dict with ``id``, ``type`` (if available) and ``is_digital`` (if
    available). For test data ``label`` is 0 (a placeholder; callers
    should not use it).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable | None = None,
        has_labels: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _pil_loader(row["abs_path"])
        if self.transform is not None:
            out = self.transform(image=np.asarray(img))
            img = out["image"]
        else:
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0

        label = int(row["label"]) if self.has_labels and "label" in row else 0
        meta = {"id": str(row["id"])}
        if "type" in row and not pd.isna(row["type"]):
            meta["type"] = str(row["type"])
        if "is_digital" in row and not pd.isna(row["is_digital"]):
            meta["is_digital"] = int(row["is_digital"])
        return img, label, meta


def make_dataloaders(
    train_csv: str | Path,
    dataset_root: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor]:
    """Build train/val dataloaders and the pos_weight tensor for BCE."""
    df = load_metadata(train_csv, root=dataset_root)
    train_df, val_df = split_train_val(df, val_fraction=val_fraction, seed=seed)
    train_ds = FREUIDDataset(train_df, transform=build_transforms(TransformSpec(image_size, train=True)), has_labels=True)
    val_ds = FREUIDDataset(val_df, transform=build_transforms(TransformSpec(image_size, train=False)), has_labels=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    # pos_weight for BCEWithLogitsLoss — ratio of negatives to positives.
    counts = train_df["label"].value_counts().to_dict()
    n0 = max(1, counts.get(0, 1))
    n1 = max(1, counts.get(1, 1))
    pos_weight = torch.tensor([n0 / n1], dtype=torch.float32)
    return train_loader, val_loader, pos_weight


__all__ = [
    "FREUIDDataset",
    "build_transforms",
    "class_weights",
    "load_metadata",
    "make_dataloaders",
    "split_train_val",
    "stratified_kfold_indices",
    "TransformSpec",
]
