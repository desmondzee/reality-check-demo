"""Closed-loop score of CupPolicy inside CupsEnv."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("RC_ENGINE", ROOT.parent / "reality-check" / "engine"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rc_sdk"))
sys.path.insert(0, str(ROOT / "train"))

from collect_demos import parse_seeds  # noqa: E402
from oracle import TASKS  # noqa: E402
from policy.cup_policy import CupPolicy  # noqa: E402
from policy.vision import BLUE, GREEN, RED  # noqa: E402
from rc_engine.schema import JointDelta  # noqa: E402
from rc_engine.tasks.cups import PUBLIC, CupsEnv  # noqa: E402


def jpeg_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=90, subsampling=0, optimize=False, progressive=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_weights(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"red": RED.tolist(), "blue": BLUE.tolist(), "green": GREEN.tolist()}
    (folder / "affine.json").write_text(json.dumps(payload) + "\n")
    (folder / "config.json").write_text(json.dumps({"kind": "affine-blobs"}) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--weights", type=Path, default=ROOT / "runs" / "localizer")
    args = p.parse_args()
    if not (args.weights / "model.safetensors").is_file():
        write_weights(args.weights)
    os.environ["RC_WEIGHTS"] = str(args.weights)
    policy = CupPolicy()
    policy.load()
    tasks = [t for t in args.tasks.split(",") if t]
    seeds = parse_seeds(args.seeds)
    rates = []
    for task in tasks:
        wins = 0
        for seed in seeds:
            env = CupsEnv(task, PUBLIC[task])
            env.render = True
            obs = env.reset(seed)
            success = False
            episode = f"{task}-{seed}"
            for step in range(env.spec.episode_length):
                envelope = {
                    "episode_id": episode,
                    "step": step,
                    "task_id": task,
                    "images": {"top": jpeg_b64(obs.images["top"])},
                    "proprio": {
                        "joint_pos": obs.joint_pos.tolist(),
                        "joint_vel": obs.joint_vel.tolist(),
                        "gripper": float(obs.gripper),
                        "ee_pos": obs.ee_pos.tolist(),
                        "ee_quat": obs.ee_quat.tolist(),
                    },
                }
                action = policy.act(envelope)
                result = env.step(JointDelta(tuple(action["values"]), float(action["gripper"])))
                obs = result.obs
                success = bool(result.success)
                if result.done:
                    break
            wins += int(success)
            print(f"{task} seed={seed} success={success} steps={env.step_count}", flush=True)
        rate = wins / len(seeds)
        rates.append(rate)
        print(f"  {task}: {wins}/{len(seeds)} = {rate:.2f}")
    print(f"MEAN {sum(rates) / len(rates):.3f}")


if __name__ == "__main__":
    main()
