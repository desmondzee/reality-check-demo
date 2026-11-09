# Reality Check demo

Participant submission for the UR12e + Robotiq 2F-140 cup cell (contract v0, schema v2).

The container serves `GET /healthz`, `GET /readyz`, and `POST /act` through `rc_sdk`. `manifest.json` names the public Hugging Face repo whose snapshot the runner mounts at `/weights`. `policy/cup_policy.py` reads `affine.json` from that mount.

The policy pulls the arm aside until the overhead camera can see the cups, maps red / blue / green pixel centroids to the table with that affine, and tracks the same IK waypoints as the privileged oracle in `train/oracle.py`. `(episode_id, step)` is memoised so a retry does not advance the tracker.

On jarvis, `rc_runner.practice` against the engine and this image scored **0.95** mean success on the five public tasks, seeds 0–19 (lift 0.90, pick 0.95, place 0.95, stack 0.95, tip 1.00).

## Layout

| Path | Role |
|---|---|
| `Dockerfile`, `manifest.json` | Submission image and weight pin |
| `rc_sdk/` | Vendored contract server |
| `policy/cup_policy.py`, `policy/vision.py`, `policy/tracker.py` | Camera, affine, IK |
| `kin/ur12e_kin.xml` | Kinematic model used by the tracker |
| `train/` | Oracle, ACT experiment, `eval_cup.py`, `push_weights.py` (uv) |

Action-chunk cloning is still in `train/train_act.py`. It matched the demos open-loop and lost the grasp as soon as the arm left those states, so it is not what the container serves.

## Contract check

```sh
python3 scripts/selftest.py
```

Prints `ALL PASS`.

## Weights

`uv run python eval_cup.py --seeds 0` writes `runs/localizer/affine.json`. Push it with a Hub token that is allowed to create models:

```sh
cd train
uv run python push_weights.py
```

The token currently stored on this machine can read the `akoniti` account and cannot create a model repo, so that upload still needs a write-capable login.

## Image and practice

Build on the Spark (linux/aarch64). From a `reality-check` checkout:

```sh
docker build -t rc-engine:dev engine/
docker build -t rc-demo-act /path/to/reality-check-demo
docker run -d --name rc-engine --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -p 127.0.0.1:8090:8090 rc-engine:dev
docker run -d --name rc-policy -p 127.0.0.1:8080:8080 \
  -v /path/to/runs/localizer:/weights:ro rc-demo-act
cd runner && uv run python -m rc_runner.practice \
  --engine http://127.0.0.1:8090 --policy http://127.0.0.1:8080 --seeds 0-19
```
