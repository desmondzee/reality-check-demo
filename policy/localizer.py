"""Spatial CNN: a heatmap peak on the cup, then a pixel-to-table map.

Global pooling threw position away (about 10 cm error). Soft-argmax keeps it.
The three maps are initialised from the calibrated affines, so the network
only has to find the coloured blob.
"""
from __future__ import annotations

import numpy as np

from policy.act_net import IMG
from policy.vision import BLUE, GREEN, RED

HPARAMS = {"image": IMG}


def kind_from_task(task_id: str | None) -> str:
    text = str(task_id or "")
    for key in ("stack", "pick", "place", "tip", "lift"):
        if key in text:
            return key
    return "lift"


def _linear_from_coef(layer, coef: np.ndarray, width: int = 640, height: int = 480) -> None:
    """nn.Linear(3, 2) matching xy = [u*(W-1), v*(H-1), 1] @ coef, u,v in [0, 1]."""
    import torch

    scale_u = float(width - 1)
    scale_v = float(height - 1)
    weight = torch.tensor(
        [
            [scale_u * coef[0, 0], scale_v * coef[1, 0], coef[2, 0]],
            [scale_u * coef[0, 1], scale_v * coef[1, 1], coef[2, 1]],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.zero_()


def build_localizer(image: int = IMG):
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),  # 64
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 32
            )
            self.cup_heat = nn.Conv2d(16, 1, 1)
            self.blue_heat = nn.Conv2d(16, 1, 1)
            self.green_heat = nn.Conv2d(16, 1, 1)
            self.cup_map = nn.Linear(3, 2)
            self.blue_map = nn.Linear(3, 2)
            self.green_map = nn.Linear(3, 2)
            _linear_from_coef(self.cup_map, RED)
            _linear_from_coef(self.blue_map, BLUE)
            _linear_from_coef(self.green_map, GREEN)

        def _uv(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            b, _, h, w = logits.shape
            prob = torch.softmax((logits * 8).flatten(1), dim=-1).view(b, 1, h, w)
            xs = torch.linspace(0, 1, w, device=logits.device)
            ys = torch.linspace(0, 1, h, device=logits.device)
            u = (prob.sum(dim=2) * xs).sum(-1).squeeze(1)
            v = (prob.sum(dim=3) * ys).sum(-1).squeeze(1)
            return u, v

        def _xy(self, heat: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
            u, v = self._uv(heat)
            ones = torch.ones_like(u)
            return layer(torch.stack([u, v, ones], dim=-1))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            feat = self.cnn(image)
            cup = self._xy(self.cup_heat(feat), self.cup_map)
            blue = self._xy(self.blue_heat(feat), self.blue_map)
            green = self._xy(self.green_heat(feat), self.green_map)
            return torch.cat([cup, blue, green], dim=-1)

    net = _Net()
    net.hparams = {"image": image}
    return net
