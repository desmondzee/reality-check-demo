"""Small action-chunking transformer for the cup cell.

Predicts CHUNK normalized actions from one 128px overhead frame, a 20-D
proprio vector, and a task index. Deployment executes the first action only
so /act stays closed-loop and a pure function of the envelope.
"""
from __future__ import annotations

IMG = 128
STATE_DIM = 20
CHUNK = 8
N_TASKS = 8
MAX_DELTA = 0.049

TASKS = (
    "cup_lift_v0",
    "cup_pick_red_v0",
    "cup_place_v0",
    "cup_tip_v0",
    "cup_stack_v0",
)
TASK_INDEX = {name: i for i, name in enumerate(TASKS)}


def task_index(task_id: str | None) -> int:
    text = str(task_id or "")
    if text in TASK_INDEX:
        return TASK_INDEX[text]
    for key in ("stack", "pick", "place", "tip", "lift"):
        if key in text:
            return TASK_INDEX[f"cup_{key}_v0"] if key != "pick" else TASK_INDEX["cup_pick_red_v0"]
    return 5  # unknown / hidden


def state_from_proprio(proprio: dict) -> list[float]:
    joint_pos = list(proprio.get("joint_pos") or [0.0] * 6)
    joint_vel = list(proprio.get("joint_vel") or [0.0] * 6)
    ee_pos = list(proprio.get("ee_pos") or [0.0] * 3)
    ee_quat = list(proprio.get("ee_quat") or [1.0, 0.0, 0.0, 0.0])
    grip = float(proprio.get("gripper") or 0.0)
    vec = [*joint_pos[:6], *joint_vel[:6], grip, *ee_pos[:3], *ee_quat[:4]]
    if len(vec) != STATE_DIM:
        raise ValueError(f"state dim {len(vec)} != {STATE_DIM}")
    return [float(x) for x in vec]


def normalize_action(values):
    """Map joint deltas and gripper into about [-1, 1]. `values` is (..., 7)."""
    import torch

    out = values.clone()
    out[..., :6] = values[..., :6] / 0.05
    out[..., 6] = values[..., 6] * 2.0 - 1.0
    return out


def denormalize_action(values):
    """Inverse of normalize_action, clamped to the schema v2 limits."""
    import torch

    arm = (values[..., :6] * 0.05).clamp(-MAX_DELTA, MAX_DELTA)
    grip = ((values[..., 6] + 1.0) * 0.5).clamp(0.0, 1.0)
    return torch.cat([arm, grip.unsqueeze(-1)], dim=-1)


class ACTNet:
    """Constructed lazily so importing this module does not require torch at selftest time."""

    def __new__(cls, *args, **kwargs):
        return build_act_net(*args, **kwargs)


def build_act_net(d_model: int = 128, nhead: int = 4, layers: int = 2, chunk: int = CHUNK,
                  n_tasks: int = N_TASKS, state_dim: int = STATE_DIM):
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.chunk = chunk
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, d_model, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            )
            self.state_proj = nn.Linear(state_dim, d_model)
            self.task_emb = nn.Embedding(n_tasks, d_model)
            enc = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.0, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.query = nn.Parameter(torch.randn(chunk, d_model) * 0.02)
            dec = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2, batch_first=True, dropout=0.0, norm_first=True)
            self.decoder = nn.TransformerDecoder(dec, layers)
            self.head = nn.Linear(d_model, 7)

        def forward(self, image, state, task):
            feat = self.cnn(image).flatten(2).transpose(1, 2)
            extra = torch.stack([self.state_proj(state), self.task_emb(task)], dim=1)
            memory = self.encoder(torch.cat([feat, extra], dim=1))
            query = self.query.unsqueeze(0).expand(image.shape[0], -1, -1)
            return self.head(self.decoder(query, memory))

    net = _Net()
    net.hparams = {
        "d_model": d_model,
        "nhead": nhead,
        "layers": layers,
        "chunk": chunk,
        "n_tasks": n_tasks,
        "state_dim": state_dim,
    }
    return net
