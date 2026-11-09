"""Joint-delta tracker. Same waypoints as the privileged oracle, but the cup
xy comes from the localizer and the servo state comes from proprioception.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

KIN_XML = Path(__file__).resolve().parents[1] / "kin" / "ur12e_kin.xml"
MAX_DELTA = 0.049
OPEN, CLOSED = 1.0, 0.0
DOWN = np.array([0.0, 0.0, -1.0])
HOVER_Z, GRASP_Z, LIFT_Z = 0.22, 0.125, 0.22
CUP_H = 0.06


class Kinematics:
    def __init__(self, xml: Path = KIN_XML) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.tip = self.m.site("gripperframe").id
        self.lo = self.m.jnt_range[:6, 0].copy()
        self.hi = self.m.jnt_range[:6, 1].copy()

    def solve(self, q0: np.ndarray, pos: np.ndarray, iters: int = 40) -> np.ndarray:
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            self.d.qpos[:6] = q
            mujoco.mj_forward(self.m, self.d)
            p = self.d.site_xpos[self.tip]
            z = self.d.site_xmat[self.tip].reshape(3, 3)[:, 2]
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.tip)
            err = pos - p
            err_r = 0.4 * np.cross(z, DOWN)
            jac = np.vstack([jp[:, :6], jr[:, :6]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), np.r_[err, err_r])
            q = np.clip(q + 0.8 * dq, self.lo, self.hi)
            if np.linalg.norm(err) < 1.5e-3:
                break
        return q


class Tracker:
    """Stateful across steps of one episode. Retries must not call act() twice."""

    def __init__(self) -> None:
        self.kin = Kinematics()
        self.waypoints: list[tuple[np.ndarray, float, int]] = []
        self.phase = 0
        self.held = 0
        self.mode = "track"
        self.kind = "lift"
        self.q_cmd = np.zeros(6)

    def reset(self, q, kind: str) -> None:
        self.q_cmd = np.array(q, dtype=float)
        self.kind = kind
        self.phase = 0
        self.held = 0
        self.mode = "track"
        self.waypoints = []

    def set_goals(self, cup_xy: np.ndarray, other_xy: np.ndarray | None) -> None:
        phase, held, mode, q_cmd = self.phase, self.held, self.mode, self.q_cmd
        cup = np.asarray(cup_xy, dtype=float)
        other = None if other_xy is None else np.asarray(other_xy, dtype=float)
        wps: list[tuple[np.ndarray, float, int]] = []
        add = wps.append
        if self.kind == "tip":
            add((np.array([cup[0], cup[1] + 0.10, 0.09]), CLOSED, 2))
            add((np.array([cup[0], cup[1] - 0.08, 0.07]), CLOSED, 10))
        else:
            add((np.array([cup[0], cup[1], 0.30]), OPEN, 2))
            add((np.array([cup[0], cup[1], HOVER_Z]), OPEN, 3))
            add((np.array([cup[0], cup[1], GRASP_Z]), OPEN, 4))
            add((np.array([cup[0], cup[1], GRASP_Z]), CLOSED, 25))
            add((np.array([cup[0], cup[1], LIFT_Z]), CLOSED, 15))
            if self.kind in ("place", "stack") and other is not None:
                extra = CUP_H + 0.01 if self.kind == "stack" else 0.0
                add((np.array([other[0], other[1], extra + LIFT_Z]), CLOSED, 4))
                add((np.array([other[0], other[1], extra + GRASP_Z]), CLOSED, 6))
                add((np.array([other[0], other[1], extra + GRASP_Z + 0.01]), OPEN, 8))
                add((np.array([other[0], other[1], extra + 0.28]), OPEN, 6))
        self.waypoints = wps
        self.phase = min(phase, max(0, len(wps) - 1))
        self.held = held
        self.mode = mode
        self.q_cmd = q_cmd

    def act(self, ee: np.ndarray, q_meas: np.ndarray) -> tuple[list[float], float]:
        if self.mode == "freeze" or not self.waypoints:
            grip = CLOSED if self.waypoints and self.waypoints[-1][1] == CLOSED else OPEN
            return [0.0] * 6, grip
        xyz, grip, hold = self.waypoints[self.phase]
        q_cmd = self.q_cmd
        goal = np.clip(self.kin.solve(q_cmd, xyz), self.kin.lo, self.kin.hi)
        dq = np.clip(goal - q_cmd, -MAX_DELTA, MAX_DELTA)
        if np.max(np.abs(dq)) < 0.008:
            self.kin.d.qpos[:6] = q_meas
            mujoco.mj_forward(self.kin.m, self.kin.d)
            jp = np.zeros((3, self.kin.m.nv))
            mujoco.mj_jacSite(self.kin.m, self.kin.d, jp, None, self.kin.tip)
            err = xyz - np.asarray(ee, dtype=float)
            j = jp[:, :6]
            corr = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            dq = np.clip(corr, -MAX_DELTA, MAX_DELTA)
        self.q_cmd = q_cmd + dq
        arrived = np.linalg.norm(ee[:2] - xyz[:2]) < 0.02 and abs(float(ee[2] - xyz[2])) < 0.03
        if arrived:
            self.held += 1
            if self.held >= max(hold, 1):
                if grip == OPEN and self.phase == len(self.waypoints) - 1:
                    self.mode = "freeze"
                    return [0.0] * 6, OPEN
                if self.phase < len(self.waypoints) - 1:
                    self.phase += 1
                    self.held = 0
        else:
            self.held = 0
        return [float(x) for x in dq], float(grip)
