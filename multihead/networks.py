"""
networks.py
===========
Neural network modules for the multi-head clock recognition model.

Architecture:
  - MultiHeadClockNet: Shared EfficientNet-B0 backbone + 3 classification heads
      head_rot:    4-class  (dial rotation)
      head_hour:   72-class (short-hand angle, 5-deg bins, image coords)
      head_minute: 12-class (minute-hand angle, 30-deg bins, image coords)
  - WarmupNet: Temporary 144-class classifier for Stage 1 backbone pretraining
  - RotAdapter / HourAdapter / MinuteAdapter: Thin wrappers for DeepProbLog Network
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiHeadClockNet(nn.Module):
    """Shared EfficientNet-B0 backbone with three classification heads.

    Heads:
      - head_rot:    4-class rotation (0/90/180/270 degrees)
      - head_hour:   72-class short-hand angle (5-degree bins, image coords)
      - head_minute: 12-class minute-hand angle (30-degree bins, image coords)

    All heads output softmax probabilities (required by DeepProbLog).
    """

    def __init__(self, dropout: float = 0.2, hour_classes: int = 72):
        super().__init__()
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Extract feature layers (everything except the classifier)
        self.features = eff.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 1280  # EfficientNet-B0 output channels
        self.hour_classes = hour_classes

        self.head_rot = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 4),
            nn.Softmax(dim=1),
        )
        self.head_hour = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hour_classes),
            nn.Softmax(dim=1),
        )
        self.head_minute = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 12),
            nn.Softmax(dim=1),
        )

        # Feature cache to avoid redundant backbone passes within a batch
        self._cache_id = None
        self._cache_feat = None

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with caching for repeated calls on same input."""
        key = id(x)
        if self._cache_id == key and self._cache_feat is not None:
            return self._cache_feat
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        self._cache_id = key
        self._cache_feat = feat
        return feat

    def clear_cache(self):
        self._cache_id = None
        self._cache_feat = None

    def forward_rot(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_rot(self._extract(x))

    def forward_hour(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_hour(self._extract(x))

    def forward_minute(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_minute(self._extract(x))


class WarmupNet(nn.Module):
    """EfficientNet-B0 backbone + 144-class head for Stage 1 warmup.

    Uses the same backbone as MultiHeadClockNet.
    After warmup training, backbone weights are transferred to MultiHeadClockNet.
    """

    def __init__(self, num_classes: int = 144, dropout: float = 0.2):
        super().__init__()
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = eff.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# DeepProbLog adapters
# ---------------------------------------------------------------------------
# Each adapter wraps the shared MultiHeadClockNet and exposes a single head
# as a standalone nn.Module. DeepProbLog's Network class requires one module
# per neural predicate, but we want all three to share the backbone.

def _unpack_dpb_input(x):
    """Handle DeepProbLog's Constant(list) input format."""
    if isinstance(x, list):
        x = torch.stack([item.value if hasattr(item, 'value') else item for item in x])
    return x


class RotAdapter(nn.Module):
    def __init__(self, shared: MultiHeadClockNet):
        super().__init__()
        self.shared = shared

    def forward(self, x):
        x = _unpack_dpb_input(x)
        x = x.to(next(self.shared.parameters()).device)
        return self.shared.forward_rot(x)


class HourAdapter(nn.Module):
    def __init__(self, shared: MultiHeadClockNet):
        super().__init__()
        self.shared = shared

    def forward(self, x):
        x = _unpack_dpb_input(x)
        x = x.to(next(self.shared.parameters()).device)
        return self.shared.forward_hour(x)


class MinuteAdapter(nn.Module):
    def __init__(self, shared: MultiHeadClockNet):
        super().__init__()
        self.shared = shared

    def forward(self, x):
        x = _unpack_dpb_input(x)
        x = x.to(next(self.shared.parameters()).device)
        return self.shared.forward_minute(x)


def transfer_backbone_weights(warmup_net: WarmupNet, multi_head: MultiHeadClockNet):
    """Transfer backbone (features) weights from WarmupNet to MultiHeadClockNet."""
    multi_head.features.load_state_dict(warmup_net.features.state_dict())
    print("Backbone weights transferred from WarmupNet to MultiHeadClockNet.")
