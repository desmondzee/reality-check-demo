# Reality Check demo — multi-task ACT

Participant-style submission for the UR12e + Robotiq 2F-140 cup cell (contract v0, schema v2).

The container serves `GET /healthz`, `GET /readyz`, and `POST /act` via `rc_sdk`. Weights are not in git. `manifest.json` points at the public Hugging Face repo `desmondzee/reality-check-demo-act`, and the runner mounts that snapshot at `/weights`.

`policy/cup_policy.py` loads `affine.json` from `/weights`. It moves the arm to a reveal pose, segments the red cup (and the blue cup or green target) in the overhead JPEG, maps pixels to the table with a fitted affine, then follows the same IK waypoints as the oracle. `(episode_id, step)` is memoised so a retry does not advance the tracker.

## Layout

| Path | Role |
|---|---|
| `Dockerfile`, `manifest.json` | Submission image and weight pin |
| `rc_sdk/` | Vendored contract server |
| `policy/act_net.py`, `policy/act_policy.py` | Model and `/act` wrapper |
| `kin/ur12e_kin.xml` | IK model used only by the oracle |
| `train/` | Privileged oracle, demo collection, ACT training (uv) |

## Contract check

```sh
python3 scripts/selftest.py
```

Prints `ALL PASS`.

## Data and training

Needs a checkout of [reality-check](https://github.com/w3joe/reality-check) next to this repo (or set `RC_ENGINE` to its `engine/` directory). Python environments use uv.

```sh
cd train
uv sync
uv run python eval_oracle.py --seeds 0-9          # privileged IK oracle
uv run python eval_cup.py --seeds 0-4
# writes runs/localizer/affine.json — push that directory to the public HF repo
# named in manifest.json (desmondzee/reality-check-demo-act)
```

On the DGX Spark, install the CUDA wheel before training so it is not replaced by a CPU build:

```sh
uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python train_act.py --steps 4000 --push desmondzee/reality-check-demo-act
```

Demos are JPEG-round-tripped at quality 90 and resized to 128×128, matching runtime preprocessing. Episodes that the oracle does not solve are dropped.

## Image

Build on linux/aarch64 (the Spark). The default pip index is the CUDA 12.8 wheel, which covers GB10.

```sh
docker build -t rc-demo-act .
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v /path/to/act-export:/weights:ro rc-demo-act
```

Practice against the engine (from a `reality-check` checkout):

```sh
docker build -t rc-engine:dev engine/
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -p 127.0.0.1:8090:8090 rc-engine:dev
cd runner && uv run python -m rc_runner.practice \
  --engine http://127.0.0.1:8090 --policy http://127.0.0.1:8080 --seeds 0-4
```
