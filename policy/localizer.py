"""CNN that reads the overhead frame and predicts cup / target xy in metres."""
from __future__ import annotations

from policy.act_net import IMG, N_TASKS, task_index

LOCALIZER_HPARAMS = {"d_model": 128, "n_tasks": N_TASKS, "image": IMG}


def build_localizer(d_model: int = 128, n_tasks: int = N_TASKS, image: int = IMG):
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, d_model, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.task_emb = nn.Embedding(n_tasks, 32)
            self.head = nn.Sequential(
                nn.Linear(d_model + 32, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 4),
            )

        def forward(self, image, task):
            feat = self.cnn(image).flatten(1)
            extra = self.task_emb(task)
            return self.head(torch.cat([feat, extra], dim=-1))

    net = _Net()
    net.hparams = {"d_model": d_model, "n_tasks": n_tasks, "image": image}
    return net


def kind_from_task(task_id: str | None) -> str:
    text = str(task_id or "")
    for key in ("stack", "pick", "place", "tip", "lift"):
        if key in text:
            return key
    return "lift"


__all__ = ["IMG", "build_localizer", "kind_from_task", "task_index"]
