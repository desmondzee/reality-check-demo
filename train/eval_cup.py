"""Closed-loop score of the localizer + IK tracker inside CupsEnv."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("RC_ENGINE", ROOT.parent / "reality-check" / "engine"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rc_sdk"))
sys.path.insert(0, str(ROOT / "train"))

from collect_demos import jpeg_resize, parse_seeds  # noqa: E402
from oracle import TASKS  # noqa: E402
from policy.act_policy import tensor_from_uint8  # noqa: E402
from policy.localizer import build_localizer, kind_from_task, task_index  # noqa: E402
from policy.tracker import Tracker  # noqa: E402
from rc_engine.schema import JointDelta  # noqa: E402
from rc_engine.tasks.cups import PUBLIC, CupsEnv  # noqa: E402


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "localizer")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.ckpt / "config.json").read_text())
    model = build_localizer(**cfg["model"])
    model.load_state_dict(torch.load(args.ckpt / "localizer.pt", map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    def predict(frame: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        image = tensor_from_uint8(frame).to(device)
        task_t = torch.tensor([task_index(task)], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(image, task_t)[0].detach().cpu().numpy()
        return pred[:2].astype(float), pred[2:].astype(float)

    tasks = [t for t in args.tasks.split(",") if t]
    seeds = parse_seeds(args.seeds)
    rates = []
    for task in tasks:
        wins = 0
        kind = kind_from_task(task)
        for seed in seeds:
            env = CupsEnv(task, PUBLIC[task])
            env.render = True
            obs = env.reset(seed)
            tracker = Tracker()
            tracker.reset(obs.joint_pos, kind)
            cup, other = predict(jpeg_resize(obs.images["top"]), task)
            tracker.set_goals(cup, None if kind in ("lift", "tip") else other)
            refined = False
            success = False
            for _ in range(env.spec.episode_length):
                if not refined and tracker.phase >= 1:
                    cup, other = predict(jpeg_resize(obs.images["top"]), task)
                    tracker.set_goals(cup, None if kind in ("lift", "tip") else other)
                    refined = True
                values, grip = tracker.act(obs.ee_pos, obs.joint_pos)
                result = env.step(JointDelta(tuple(values), grip))
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
