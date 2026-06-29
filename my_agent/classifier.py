"""Loads the trained FREUID checkpoint and runs inference on demand.

Kept separate from ``agent.py`` so the heavy model is only loaded once
on first use (lazy singleton). The agent wraps a single function
``classify_document(image_path)`` that ADK can call as a tool.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Repo root on sys.path so we can import the pipeline modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.src.data import build_transforms, TransformSpec  # noqa: E402
from code.src.model import build_model  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "runs" / "freuid_real" / "best.pt"

# Verdict thresholds chosen so the agent's text mirrors the Kaggle
# operating point: high-recall "review" zone and high-precision "reject"
# zone, with a neutral middle band.
REJECT_THRESHOLD = 0.65
REVIEW_THRESHOLD = 0.35


@lru_cache(maxsize=1)
def _load_runtime(ckpt_path: str) -> dict[str, Any]:
    """Load checkpoint + build model + cache transform. Runs once."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found at {ckpt_path}. "
            "Run `python -m code.src.train --config code/configs/baseline.yaml` first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(
        backbone_name=cfg["backbone"],
        pretrained=False,  # weights come from the checkpoint, no download
        drop_rate=cfg.get("drop_rate", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    transform = build_transforms(
        TransformSpec(image_size=int(cfg["image_size"]), train=False)
    )
    return {"model": model, "transform": transform, "config": cfg}


def _preprocess(image_path: str, transform) -> torch.Tensor:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = transform(image=np.asarray(img))["image"]
    return arr.unsqueeze(0)


def _predict_with_tta(model: torch.Tensor, x: torch.Tensor) -> float:
    """Average sigmoid(logit) over the original and horizontal flip."""
    with torch.no_grad():
        logit_orig = model(x)
        logit_flip = model(torch.flip(x, dims=[3]))
        score = torch.sigmoid((logit_orig + logit_flip) / 2.0).item()
    return float(score)


def classify_document(image_path: str, ckpt_path: str | None = None) -> dict[str, Any]:
    """Classify an identity-document image as bona-fide or fraudulent.

    Args:
        image_path: absolute or repo-relative path to a JPG/PNG image.
        ckpt_path: optional override; defaults to runs/freuid_real/best.pt.

    Returns:
        dict with keys: ``image_path``, ``fraud_score`` (0..1),
        ``verdict`` (one of ``"bona_fide"``, ``"review"``, ``"fraudulent"``),
        ``confidence`` (how far the score is from 0.5), and ``model``.
    """
    path = Path(image_path)
    if not path.is_absolute():
        path = (REPO_ROOT / image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    ckpt = str(Path(ckpt_path) if ckpt_path else DEFAULT_CKPT)
    rt = _load_runtime(ckpt)
    x = _preprocess(str(path), rt["transform"])
    score = _predict_with_tta(rt["model"], x)

    if score >= REJECT_THRESHOLD:
        verdict = "fraudulent"
    elif score <= REVIEW_THRESHOLD:
        verdict = "bona_fide"
    else:
        verdict = "review"

    return {
        "image_path": str(path),
        "fraud_score": round(score, 4),
        "verdict": verdict,
        "confidence": round(abs(score - 0.5) * 2, 4),  # 0..1
        "model": rt["config"]["backbone"],
    }