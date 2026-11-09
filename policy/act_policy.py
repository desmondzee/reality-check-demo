"""Schema v2 policy: load an ACT checkpoint from /weights and return joint_delta."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from policy.act_net import IMG, build_act_net, denormalize_action, state_from_proprio, task_index
from rc_sdk.policy import Policy

WEIGHTS = Path(os.environ.get("RC_WEIGHTS", "/weights"))


def frame_from_jpeg(raw: str | bytes) -> np.ndarray:
    if isinstance(raw, str):
        blob = base64.b64decode(raw)
    else:
        blob = raw
    image = Image.open(io.BytesIO(blob)).convert("RGB")
    image = image.resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def tensor_from_uint8(frame: np.ndarray):
    import torch

    x = torch.from_numpy(frame.astype(np.float32) / 255.0)
    x = (x - 0.5) / 0.5
    return x.permute(2, 0, 1).unsqueeze(0)


class ActPolicy(Policy):
    def load(self) -> None:
        import torch

        cfg_path = WEIGHTS / "config.json"
        ckpt_path = WEIGHTS / "act.pt"
        cfg = json.loads(cfg_path.read_text())
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_act_net(**cfg["model"])
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        frame = frame_from_jpeg(obs["images"]["top"])
        image = tensor_from_uint8(frame).to(self.device)
        state = torch.tensor([state_from_proprio(obs.get("proprio") or {})], dtype=torch.float32, device=self.device)
        task = torch.tensor([task_index(obs.get("task_id"))], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = self.model(image, state, task)
        action = denormalize_action(pred[0, 0]).detach().cpu().tolist()
        values = [float(v) for v in action[:6]]
        # The oracle gripper is bang-bang. Snap so a 0.1 regression does not
        # leave the fingers half open and drop the cup.
        grip = 0.0 if float(action[6]) < 0.5 else 1.0
        return {"type": "joint_delta", "values": values, "gripper": grip}
