"""Privileged scripted oracle for the five public cup tasks.

Uses true cup / target poses (the learned policy never sees these) and
damped-least-squares IK on kin/ur12e_kin.xml, which matches the engine palm TCP.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from rc_engine.schema import JointDelta
from rc_engine.tasks.cups import PUBLIC, CupsEnv

KIN_XML = Path(__file__).resolve().parents[1] / "kin" / "ur12e_kin.xml"
MAX_DELTA = 0.049
OPEN, CLOSED = 1.0, 0.0
DOWN = np.array([0.0, 0.0, -1.0])
HOVER_Z, GRASP_Z, LIFT_Z = 0.22, 0.125, 0.22
CUP_H = 0.06
TASKS = (
    "cup_lift_v0",
    "cup_pick_red_v0",
    "cup_place_v0",
    "cup_tip_v0",
    "cup_stack_v0",
)


class Kinematics:
    """Damped least squares: 6 arm joints, tool z aligned with world -z."""

    def __init__(self, xml: Path = KIN_XML) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.tip = self.m.site("gripperframe").id
        self.lo = self.m.jnt_range[:6, 0].copy()
        self.hi = self.m.jnt_range[:6, 1].copy()

    def solve(self, q0: np.ndarray, pos: np.ndarray, iters: int = 60) -> np.ndarray:
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
            if np.linalg.norm(err) < 1e-3 and np.linalg.norm(err_r) < 2e-2:
                break
        return q


class ScriptedOracle:
    """Closed-loop waypoint tracker. Call reset() after env.reset()."""

    def __init__(self, kin: Kinematics | None = None) -> None:
        self.kin = kin or Kinematics()
        self.waypoints: list[tuple[np.ndarray, float, int]] = []
        self.phase = 0
        self.held = 0

    def reset(self, env: CupsEnv) -> None:
        kind = env.params.kind
        cup = env.data.xpos[env.cup_body][:2].copy()
        other = None
        if kind == "place":
            other = env.data.mocap_pos[env.target_mocap][:2].copy()
        elif kind in ("stack", "pick"):
            other = env.data.xpos[env.cup2_body][:2].copy()
        self.waypoints = []
        self.phase = 0
        self.held = 0
        self.mode = "track"  # track | freeze
        add = self.waypoints.append
        if kind == "tip":
            # Come in from the side and shove through the cup.
            add((np.array([cup[0], cup[1] + 0.10, 0.09]), CLOSED, 2))
            add((np.array([cup[0], cup[1] - 0.08, 0.07]), CLOSED, 10))
            return
        # Clear the table, then descend on the true cup centre.
        add((np.array([cup[0], cup[1], 0.30]), OPEN, 2))
        add((np.array([cup[0], cup[1], HOVER_Z]), OPEN, 3))
        add((np.array([cup[0], cup[1], GRASP_Z]), OPEN, 4))
        add((np.array([cup[0], cup[1], GRASP_Z]), CLOSED, 25))
        add((np.array([cup[0], cup[1], LIFT_Z]), CLOSED, 15))
        if kind in ("place", "stack") and other is not None:
            extra = CUP_H + 0.01 if kind == "stack" else 0.0
            add((np.array([other[0], other[1], extra + LIFT_Z]), CLOSED, 4))
            add((np.array([other[0], other[1], extra + GRASP_Z]), CLOSED, 6))
            add((np.array([other[0], other[1], extra + GRASP_Z + 0.01]), OPEN, 8))
            add((np.array([other[0], other[1], extra + 0.28]), OPEN, 6))

    def act(self, env: CupsEnv) -> JointDelta:
        if self.mode == "freeze" or not self.waypoints or self.phase >= len(self.waypoints):
            grip = CLOSED if self.waypoints and self.waypoints[-1][1] == CLOSED else OPEN
            return JointDelta((0.0,) * 6, grip)
        xyz, grip, hold = self.waypoints[self.phase]
        dq = self._delta(env, xyz)
        ee = env.data.site_xpos[env.tip]
        arrived = np.linalg.norm(ee[:2] - xyz[:2]) < 0.02 and abs(float(ee[2] - xyz[2])) < 0.03
        if arrived:
            self.held += 1
            if self.held >= max(hold, 1):
                if grip == OPEN and self.phase == len(self.waypoints) - 1:
                    self.mode = "freeze"
                    return JointDelta((0.0,) * 6, OPEN)
                if self.phase < len(self.waypoints) - 1:
                    self.phase += 1
                    self.held = 0
        else:
            self.held = 0
        return JointDelta(tuple(float(x) for x in dq), float(grip))

    def _delta(self, env: CupsEnv, xyz: np.ndarray) -> np.ndarray:
        q_cmd = np.array(env.target, dtype=float)
        goal = np.clip(self.kin.solve(q_cmd, xyz), env.ctrl_range[:6, 0], env.ctrl_range[:6, 1])
        dq = np.clip(goal - q_cmd, -MAX_DELTA, MAX_DELTA)
        ee = env.data.site_xpos[env.tip]
        if np.max(np.abs(dq)) < 0.008:
            self.kin.d.qpos[:6] = env.data.qpos[:6]
            mujoco.mj_forward(self.kin.m, self.kin.d)
            jp = np.zeros((3, self.kin.m.nv))
            mujoco.mj_jacSite(self.kin.m, self.kin.d, jp, None, self.kin.tip)
            err = xyz - ee
            j = jp[:, :6]
            corr = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            nxt = np.clip(q_cmd + np.clip(corr, -MAX_DELTA, MAX_DELTA), env.ctrl_range[:6, 0], env.ctrl_range[:6, 1])
            dq = nxt - q_cmd
        return dq


def rollout(task_id: str, seed: int, render: bool = False) -> dict:
    """Run one episode. Returns success and, when render=True, frames/states/actions."""
    env = CupsEnv(task_id, PUBLIC[task_id])
    env.render = render
    obs = env.reset(seed)
    oracle = ScriptedOracle()
    oracle.reset(env)
    images, states, actions = [], [], []
    success = False
    for _ in range(env.spec.episode_length):
        action = oracle.act(env)
        if render:
            images.append(obs.images["top"].copy())
            states.append(_state(obs))
            actions.append([*action.values, action.gripper])
        result = env.step(action)
        obs = result.obs
        success = bool(result.success)
        if result.done:
            break
    out = {"success": success, "steps": env.step_count, "task_id": task_id, "seed": seed}
    if render:
        out["images"] = images
        out["states"] = states
        out["actions"] = actions
    return out


def _state(obs) -> list[float]:
    return [
        *np.asarray(obs.joint_pos, dtype=float).tolist(),
        *np.asarray(obs.joint_vel, dtype=float).tolist(),
        float(obs.gripper),
        *np.asarray(obs.ee_pos, dtype=float).tolist(),
        *np.asarray(obs.ee_quat, dtype=float).tolist(),
    ]
