"""Model factory for FREUID fraud detection.

Wraps a ``timm`` backbone with a single-logit binary head. The output
is a logit; ``torch.sigmoid(logit)`` is the calibrated fraud score in
``[0, 1]`` expected by the FREUID Score metric and the Kaggle submission
format.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class FREUIDClassifier(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, drop_rate: float = 0.1) -> None:
        super().__init__()
        # ``num_classes=0`` strips the classification head so we can
        # attach our own single-logit head. ``global_pool`` defaults to
        # the backbone's recommended pooling (often ``avg``).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(-1)


def build_model(
    backbone_name: str = "convnext_tiny.fb_in1k",
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> FREUIDClassifier:
    """Convenience constructor."""
    return FREUIDClassifier(backbone_name=backbone_name, pretrained=pretrained, drop_rate=drop_rate)


__all__ = ["FREUIDClassifier", "build_model"]
