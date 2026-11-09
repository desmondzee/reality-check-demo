"""Schema v2 policy: localizer weights from /weights, IK tracker for the arm.

act() memoises (episode_id, step) so a byte-identical retry does not advance
the tracker. A new episode (or step 0) re-estimates the cup and replans.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from policy.act_policy import frame_from_jpeg, tensor_from_uint8
from policy.localizer import build_localizer, kind_from_task, task_index
from policy.tracker import Tracker
from rc_sdk.policy import Policy

WEIGHTS = Path(os.environ.get("RC_WEIGHTS", "/weights"))


class CupPolicy(Policy):
    def load(self) -> None:
        import torch

        cfg = json.loads((WEIGHTS / "config.json").read_text())
        torch.manual_seed(0)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_localizer(**cfg["model"])
        state = torch.load(WEIGHTS / "localizer.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.tracker = Tracker()
        self.episode: Any = None
        self.memo: tuple[tuple[Any, Any], dict] | None = None
        self.refined = False

    def _predict(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        frame = frame_from_jpeg(obs["images"]["top"])
        image = tensor_from_uint8(frame).to(self.device)
        task = torch.tensor([task_index(obs.get("task_id"))], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = self.model(image, task)[0].detach().cpu().numpy()
        return pred[:2].astype(float), pred[2:].astype(float)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        key = (obs["episode_id"], obs["step"])
        if self.memo is not None and self.memo[0] == key:
            return self.memo[1]
        proprio = obs.get("proprio") or {}
        q = np.array(proprio.get("joint_pos") or [0.0] * 6, dtype=float)
        ee = np.array(proprio.get("ee_pos") or [0.0, 0.0, 0.0], dtype=float)
        kind = kind_from_task(obs.get("task_id"))
        if obs["episode_id"] != self.episode or obs["step"] == 0:
            self.episode = obs["episode_id"]
            self.refined = False
            self.tracker.reset(q, kind)
            cup, other = self._predict(obs)
            self.tracker.set_goals(cup, None if kind in ("lift", "tip") else other)
        elif not self.refined and self.tracker.phase >= 1:
            # Home view is often partly blocked by the arm. Re-estimate at hover.
            cup, other = self._predict(obs)
            self.tracker.set_goals(cup, None if kind in ("lift", "tip") else other)
            self.refined = True
        values, grip = self.tracker.act(ee, q)
        action = {"type": "joint_delta", "values": values, "gripper": grip}
        self.memo = (key, action)
        return action
