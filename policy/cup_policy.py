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

from policy.localizer import kind_from_task
from policy.tracker import OPEN, Tracker
from policy.vision import COEFS, MIN_PIXELS, REVEAL_XYZ, decode_rgb, mask_count, table_xy
from rc_sdk.policy import Policy

OTHER = {"place": "green", "stack": "blue", "pick": "blue"}


class CupPolicy(Policy):
    def load(self) -> None:
        weights = Path(os.environ.get("RC_WEIGHTS", "/weights"))
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
        self.revealing = False

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
