"""Schema v2 policy: pull the arm clear, read the cup from the overhead
camera, then track the oracle waypoints.

(episode_id, step) is memoised so a byte-identical retry does not advance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from policy.act_net import IMG
from policy.act_policy import tensor_from_uint8
from policy.localizer import build_localizer, kind_from_task
from policy.tracker import OPEN, Tracker
from policy.vision import COEFS, MIN_PIXELS, REVEAL_XYZ, decode_rgb, mask_count, table_xy
from rc_sdk.policy import Policy

OTHER = {"place": "green", "stack": "blue", "pick": "blue"}
REVEAL_STEPS = 0


class CupPolicy(Policy):
    def load(self) -> None:
        weights = Path(os.environ.get("RC_WEIGHTS", "/weights"))
        self.nn = (weights / "model.safetensors").is_file()
        if self.nn:
            import torch
            from safetensors.torch import load_file

            cfg = json.loads((weights / "config.json").read_text())
            torch.manual_seed(0)
            self.torch = torch
            self.device = torch.device("cpu")
            self.model = build_localizer(**cfg.get("model", {}))
            self.model.load_state_dict(load_file(str(weights / "model.safetensors")))
            self.model.eval()
        else:
            raw = json.loads((weights / "affine.json").read_text())
            self.coefs = {name: np.asarray(raw[name], dtype=float) for name in ("red", "blue", "green")}
        self.tracker = Tracker()
        self._reset_episode()
        self.memo: tuple[tuple[Any, Any], dict] | None = None

    def _reset_episode(self) -> None:
        self.episode = None
        self.kind = "lift"
        self.cup: np.ndarray | None = None
        self.other: np.ndarray | None = None
        self.locked = False
        self.reveal_steps = 0

    def _predict(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        small = np.asarray(Image.fromarray(img).resize((IMG, IMG), Image.BILINEAR))
        image = tensor_from_uint8(small)
        with self.torch.no_grad():
            return self.model(image)[0].detach().cpu().numpy()

    def _see(self, img: np.ndarray, kind: str) -> np.ndarray | None:
        uv, n = mask_count(img, kind)
        if uv is None or n < MIN_PIXELS:
            return None
        return table_xy(uv, kind, self.coefs)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        key = (obs["episode_id"], obs["step"])
        if self.memo is not None and self.memo[0] == key:
            return self.memo[1]
        proprio = obs.get("proprio") or {}
        q = np.asarray(proprio.get("joint_pos") or [0.0] * 6, dtype=float)
        ee = np.asarray(proprio.get("ee_pos") or [0.0, 0.0, 0.0], dtype=float)
        kind = kind_from_task(obs.get("task_id"))
        if obs["episode_id"] != self.episode or obs["step"] == 0:
            self._reset_episode()
            self.episode = obs["episode_id"]
            self.kind = kind
            self.tracker.reset(q, kind)
        img = decode_rgb(obs["images"]["top"])
        if not self.locked:
            if self.nn and self.reveal_steps < REVEAL_STEPS:
                self.tracker.waypoints = [(REVEAL_XYZ.copy(), OPEN, 2)]
                self.tracker.phase = 0
                self.tracker.mode = "track"
                self.reveal_steps += 1
            elif self.nn:
                pred = self._predict(img)
                cup = pred[:2]
                other = pred[2:4] if kind in ("pick", "stack") else pred[4:6] if kind == "place" else None
                q_cmd = self.tracker.q_cmd.copy()
                self.tracker.reset(q_cmd, kind)
                self.tracker.set_goals(cup, other)
                self.locked = True
            else:
                cup = self._see(img, "red")
                need = OTHER.get(kind)
                other = self._see(img, need) if need else None
                if cup is not None and (need is None or other is not None):
                    q_cmd = self.tracker.q_cmd.copy()
                    self.tracker.reset(q_cmd, kind)
                    self.tracker.set_goals(cup, other)
                    self.cup, self.other, self.locked = cup, other, True
                else:
                    self.tracker.waypoints = [(REVEAL_XYZ.copy(), OPEN, 2)]
                    self.tracker.phase = 0
                    self.tracker.mode = "track"
        values, grip = self.tracker.act(ee, q)
        action = {"type": "joint_delta", "values": values, "gripper": float(grip)}
        self.memo = (key, action)
        return action
