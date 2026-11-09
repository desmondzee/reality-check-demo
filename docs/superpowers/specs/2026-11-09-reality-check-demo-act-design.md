# Reality Check demo ACT submission

Date: 2026-11-09

## Goal

A contract-complete participant submission in `reality-check-demo`: a Docker policy that loads public Hugging Face weights and scores at least ~40% mean success on the five public cup tasks under `rc_runner.practice`.

## Cell and contract

- Embodiment: UR12e + Robotiq 2F-140, schema v2 (`reality-check` engine).
- Observation: overhead `top` JPEG 640×480, proprio (`joint_pos`, `joint_vel`, `gripper`, `ee_pos`, `ee_quat`), public `task_id`.
- Action: `joint_delta` with 6 values, each `|Δq| ≤ 0.05` rad, and `gripper` in `[0, 1]`. This policy returns one `joint_delta` per call.
- HTTP: `GET /healthz`, `GET /readyz`, `POST /act` via vendored `rc_sdk`. A repeated `(episode_id, step)` returns the memoised action.
- Weights live only on a public ungated HF repo (`desmondzee/reality-check-demo-act`). The runner mounts them at `/weights`. Nothing under `*.pt`, `*.safetensors`, `*.ckpt`, or `weights/` is committed.

## Architecture

1. Privileged scripted oracle (true cup and target poses + damped-least-squares IK on `kin/ur12e_kin.xml`) rolls out the five public tasks inside `CupsEnv`.
2. Only `success=True` episodes are kept, JPEG-round-tripped at quality 90, and stored as compressed numpy episodes (128×128 image, 20-D proprio, 7-D action, task index).
3. Action-chunk BC matched the demos open-loop and missed the grasp closed-loop. The shipped policy segments the overhead image and applies an affine fit (median residual about 1.5 mm) loaded from `/weights/affine.json`.
4. `CupPolicy` reveals the table, locks the cup and target positions, then runs the oracle's IK waypoints. `(episode_id, step)` is memoised.
5. Practice scoring uses the Reality Check engine container and `rc_runner.practice` on jarvis (`~/Documents/reality-check-demo`).

Demo budget is skewed toward tip, lift, and pick-red, with place and stack included so one multi-task policy covers `RC_EVAL_TASKS=*`.

## Runtime

- All Python environments use uv.
- Train and collect on the DGX Spark (jarvis, GB10, aarch64). The Mac checkout is the git working copy; jarvis pulls the same GitHub repo.
- Inference image: Python 3.11, MuJoCo (kinematics only, `MUJOCO_GL=osmesa`), Pillow, NumPy. `rc_sdk` is installed from the repo. Base image is not mandated by the contract.
- Measured on jarvis with `rc_runner.practice`, seeds 0–19: mean success 0.95.

## Success gate

`rc_runner.practice` on all five public tasks, seeds `0-4` first, then a wider seed range. Done when mean success is at least ~0.40.

## Repo layout

- `manifest.json`, `Dockerfile`, `.gitignore`, `README.md`
- `rc_sdk/` vendored from the Reality Check starter
- `policy/cup_policy.py`, `policy/vision.py`, `policy/tracker.py`
- `policy/act_net.py`, `policy/act_policy.py` (the BC experiment, not served)
- `scripts/selftest.py`
- `kin/ur12e_kin.xml` (tracker IK)
- `train/` — `oracle.py`, `collect_demos.py`, `train_act.py`, `pyproject.toml`
- `docs/superpowers/specs/` this file
