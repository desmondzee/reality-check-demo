"""Closed-loop score of a trained ACT checkpoint inside CupsEnv."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("RC_ENGINE", ROOT.parent / "reality-check" / "engine"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rc_sdk"))
sys.path.insert(0, str(ROOT / "train"))

from collect_demos import jpeg_resize, parse_seeds  # noqa: E402
from oracle import TASKS  # noqa: E402
from policy.act_net import build_act_net, denormalize_action, state_from_proprio, task_index  # noqa: E402
from policy.act_policy import tensor_from_uint8  # noqa: E402
from rc_engine.schema import JointDelta  # noqa: E402
from rc_engine.tasks.cups import PUBLIC, CupsEnv  # noqa: E402


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "act")
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.ckpt / "config.json").read_text())
    model = build_act_net(**cfg["model"])
    state = torch.load(args.ckpt / "act.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

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
            for _ in range(env.spec.episode_length):
                frame = jpeg_resize(obs.images["top"])
                image = tensor_from_uint8(frame).to(device)
                proprio = {
                    "joint_pos": obs.joint_pos.tolist(),
                    "joint_vel": obs.joint_vel.tolist(),
                    "gripper": float(obs.gripper),
                    "ee_pos": obs.ee_pos.tolist(),
                    "ee_quat": obs.ee_quat.tolist(),
                }
                st = torch.tensor([state_from_proprio(proprio)], dtype=torch.float32, device=device)
                task_t = torch.tensor([task_index(task)], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(image, st, task_t)
                action = denormalize_action(pred[0, 0]).detach().cpu().tolist()
                grip = 0.0 if float(action[6]) < 0.5 else 1.0
                delta = JointDelta(tuple(float(v) for v in action[:6]), grip)
                result = env.step(delta)
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
