# Reality Check demo ACT submission

Date: 2026-11-09

## Goal

A contract-complete participant submission in `reality-check-demo`: a Docker policy that loads public Hugging Face weights and scores at least ~40% mean success on the five public cup tasks under `rc_runner.practice`.

## Cell and contract

- Embodiment: UR12e + Robotiq 2F-140, schema v2 (`reality-check` engine).
- Observation: overhead `top` JPEG 640×480, proprio (`joint_pos`, `joint_vel`, `gripper`, `ee_pos`, `ee_quat`), public `task_id`.
- Action: `joint_delta` with 6 values, each `|Δq| ≤ 0.05` rad, and `gripper` in `[0, 1]`. The server may also return `action_chunk` of length 1–16. This policy returns a single `joint_delta` (the first step of an ACT chunk) so control stays closed-loop.
- HTTP: `GET /healthz`, `GET /readyz`, `POST /act` via vendored `rc_sdk`. `act()` is a pure function of the envelope.
- Weights live only on a public ungated HF repo (`desmondzee/reality-check-demo-act`). The runner mounts them at `/weights`. Nothing under `*.pt`, `*.safetensors`, `*.ckpt`, or `weights/` is committed.

## Architecture

1. Privileged scripted oracle (true cup and target poses + damped-least-squares IK on `kin/ur12e_kin.xml`) rolls out the five public tasks inside `CupsEnv`.
2. Only `success=True` episodes are kept, JPEG-round-tripped at quality 90, and stored as compressed numpy episodes (128×128 image, 20-D proprio, 7-D action, task index).
3. Action-chunk BC did not hold a grasp once the arm left the demonstrated states. The shipped network is a CNN localizer: it predicts cup xy (and the place/stack target) from the overhead image. The same IK waypoint tracker used by the oracle converts that prediction into `joint_delta`.
4. `CupPolicy.load()` reads `/weights/config.json` and `/weights/localizer.pt`. `act()` memoises `(episode_id, step)` so a runner retry does not advance the tracker.
5. Practice scoring uses the Reality Check engine container and `rc_runner.practice` on jarvis (`~/Documents/reality-check-demo`).

Demo budget is skewed toward tip, lift, and pick-red, with place and stack included so one multi-task policy covers `RC_EVAL_TASKS=*`.

## Runtime

- All Python environments use uv.
- Train and collect on the DGX Spark (jarvis, GB10, aarch64). The Mac checkout is the git working copy; jarvis pulls the same GitHub repo.
- Inference image: Python 3.11 + PyTorch (CUDA 12.8 index on the Spark, so the wheel matches GB10). `rc_sdk` is installed from the repo. Base image is not mandated by the contract.
- RL fine-tune is out of scope unless behavior cloning stays under ~40% after more demos and a longer train.

## Success gate

`rc_runner.practice` on all five public tasks, seeds `0-4` first, then a wider seed range. Done when mean success is at least ~0.40.

## Repo layout

- `manifest.json`, `Dockerfile`, `.gitignore`, `README.md`
- `rc_sdk/` vendored from the Reality Check starter
- `policy/act_net.py`, `policy/act_policy.py`
- `scripts/selftest.py`
- `kin/ur12e_kin.xml` (oracle IK only; not used at inference)
- `train/` — `oracle.py`, `collect_demos.py`, `train_act.py`, `pyproject.toml`
- `docs/superpowers/specs/` this file
