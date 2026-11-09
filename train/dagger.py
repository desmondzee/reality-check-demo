"""One DAgger round: label on-policy states with the privileged oracle.

The behavior-cloned checkpoint is accurate on its own demos and drifts once
the arm leaves that trajectory. Rolling the policy (mixed with the oracle)
and storing the oracle action at every visited state teaches the correction.
"""
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
from oracle import TASKS, ScriptedOracle  # noqa: E402
from policy.act_net import TASK_INDEX, build_act_net, denormalize_action, state_from_proprio, task_index  # noqa: E402
from policy.act_policy import tensor_from_uint8  # noqa: E402
from rc_engine.schema import JointDelta  # noqa: E402
from rc_engine.tasks.cups import PUBLIC, CupsEnv  # noqa: E402


def _state(obs) -> list[float]:
    return state_from_proprio({
        "joint_pos": obs.joint_pos.tolist(),
        "joint_vel": obs.joint_vel.tolist(),
        "gripper": float(obs.gripper),
        "ee_pos": obs.ee_pos.tolist(),
        "ee_quat": obs.ee_quat.tolist(),
    })


def policy_action(model, obs, task: str, device, torch) -> JointDelta:
    frame = jpeg_resize(obs.images["top"])
    image = tensor_from_uint8(frame).to(device)
    state = torch.tensor([_state(obs)], dtype=torch.float32, device=device)
    task_t = torch.tensor([task_index(task)], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = denormalize_action(model(image, state, task_t)[0, 0]).detach().cpu().tolist()
    grip = 0.0 if float(pred[6]) < 0.5 else 1.0
    return JointDelta(tuple(float(v) for v in pred[:6]), grip)


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "act")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "dagger")
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--beta", type=float, default=0.5, help="probability of executing the oracle action")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((args.ckpt / "config.json").read_text())
    model = build_act_net(**cfg["model"])
    model.load_state_dict(torch.load(args.ckpt / "act.pt", map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in args.tasks.split(",") if t]
    seeds = parse_seeds(args.seeds)
    kept = 0
    for task in tasks:
        solved = 0
        for seed in seeds:
            env = CupsEnv(task, PUBLIC[task])
            env.render = True
            obs = env.reset(seed)
            oracle = ScriptedOracle()
            oracle.reset(env)
            images, states, actions = [], [], []
            success = False
            for _ in range(env.spec.episode_length):
                expert = oracle.act(env)
                images.append(jpeg_resize(obs.images["top"]))
                states.append(_state(obs))
                actions.append([*expert.values, expert.gripper])
                if rng.random() < args.beta:
                    action = expert
                else:
                    action = policy_action(model, obs, task, device, torch)
                result = env.step(action)
                obs = result.obs
                success = bool(result.success)
                if result.done:
                    break
            path = args.out / f"{task}_seed{seed:03d}.npz"
            np.savez_compressed(
                path,
                images=np.stack(images),
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                task_index=np.int64(TASK_INDEX[task]),
                task_id=np.str_(task),
                seed=np.int64(seed),
                success=np.bool_(True),
            )
            solved += int(success)
            kept += 1
            print(f"dagger {task} seed={seed} rollout_success={success} steps={len(actions)}", flush=True)
        print(f"{task} rollout {solved}/{len(seeds)}")
    print(f"wrote {kept} episodes -> {args.out}")


if __name__ == "__main__":
    main()
