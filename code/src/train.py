"""Training entry point for the FREUID pipeline.

Usage::

    python -m code.src.train --config code/configs/baseline.yaml

The script:

1. Reads the YAML config and overrides from CLI flags.
2. Builds stratified train/val dataloaders.
3. Constructs the model (timm backbone + single-logit head).
4. Runs AdamW with cosine LR, AMP autocast, gradient clipping, EMA.
5. Validates each epoch with the FREUID Score and saves the best
   checkpoint to ``ckpt.out_dir/best.pt``.

Run ``python -m code.src.train --help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.src.data import make_dataloaders  # noqa: E402
from code.src.metric import freuid_score  # noqa: E402
from code.src.model import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    train_csv: str
    dataset_root: str
    image_size: int
    batch_size: int
    val_fraction: float
    seed: int
    num_workers: int
    backbone: str
    pretrained: bool
    drop_rate: float
    lr: float
    weight_decay: float
    warmup_epochs: int
    epochs: int
    grad_clip: float
    ema_decay: float
    amp: bool
    amp_dtype: str
    out_dir: str
    save_best: bool
    log_every: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls(
            train_csv=raw["data"]["train_csv"],
            dataset_root=raw["data"]["dataset_root"],
            image_size=int(raw["data"]["image_size"]),
            batch_size=int(raw["data"]["batch_size"]),
            val_fraction=float(raw["data"]["val_fraction"]),
            seed=int(raw["data"]["seed"]),
            num_workers=int(raw["data"]["num_workers"]),
            backbone=raw["model"]["backbone"],
            pretrained=bool(raw["model"]["pretrained"]),
            drop_rate=float(raw["model"]["drop_rate"]),
            lr=float(raw["optim"]["lr"]),
            weight_decay=float(raw["optim"]["weight_decay"]),
            warmup_epochs=int(raw["optim"]["warmup_epochs"]),
            epochs=int(raw["optim"]["epochs"]),
            grad_clip=float(raw["optim"]["grad_clip"]),
            ema_decay=float(raw["optim"]["ema_decay"]),
            amp=bool(raw["optim"]["amp"]),
            amp_dtype=str(raw["optim"]["amp_dtype"]),
            out_dir=raw["ckpt"]["out_dir"],
            save_best=bool(raw["ckpt"]["save_best"]),
            log_every=int(raw["ckpt"]["log_every"]),
        )


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# EMA helper — keeps a copy of the model weights for evaluation.
# ---------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
                logits = model(x)
            all_scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            all_labels.append(np.asarray(y))
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(np.int64)
    if labels.min() == labels.max():
        # A single-class validation set yields undefined ROC; skip metric.
        return {"freuid": float("nan"), "apcer_at_1pct": float("nan"), "audet": float("nan"), "n": int(len(labels))}
    return {
        "freuid": float(freuid_score(labels, scores)),
        "n": int(len(labels)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code/configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.no_pretrained:
        cfg.pretrained = False

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, config={asdict(cfg)}")

    train_loader, val_loader, pos_weight = make_dataloaders(
        train_csv=cfg.train_csv,
        dataset_root=cfg.dataset_root,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    print(f"train_batches={len(train_loader)} val_batches={len(val_loader)} pos_weight={pos_weight.item():.3f}")

    model = build_model(cfg.backbone, pretrained=cfg.pretrained, drop_rate=cfg.drop_rate).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    best_freuid = float("inf")
    best_path = out_dir / "best.pt"
    log_path = out_dir / "train.log"
    history = []
    global_step = 0

    log_path.write_text(f"start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_n = 0
        t0 = time.time()
        for it, (x, y, _meta) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            lr = cosine_lr(global_step, total_steps, warmup_steps, cfg.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type != "cpu"):
                logits = model(x)
                loss = bce(logits, y)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            ema.update(model)

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            running_n += bs
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg = running_loss / max(1, running_n)
                print(f"epoch {epoch} step {global_step}/{total_steps} lr={lr:.2e} loss={avg:.4f}")

        train_loss = running_loss / max(1, running_n)
        metrics = evaluate(ema.module, val_loader, device)
        elapsed = time.time() - t0
        line = f"epoch {epoch} train_loss={train_loss:.4f} val_freuid={metrics['freuid']:.4f} val_n={metrics['n']} elapsed={elapsed:.1f}s"
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics, "lr": lr, "elapsed_s": elapsed})

        if cfg.save_best and metrics["freuid"] == metrics["freuid"] and metrics["freuid"] < best_freuid:
            best_freuid = metrics["freuid"]
            torch.save({
                "model_state": ema.module.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "metrics": metrics,
                "best_freuid": best_freuid,
            }, best_path)
            with log_path.open("a") as f:
                f.write(f"  saved new best to {best_path}\n")
            print(f"  saved new best to {best_path} (freuid={best_freuid:.4f})")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best_val_freuid={best_freuid:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
