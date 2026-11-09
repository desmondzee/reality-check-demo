"""Score the privileged oracle. Usage: uv run python eval_oracle.py --seeds 0-3"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("RC_ENGINE", ROOT.parent / "reality-check" / "engine"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT / "train"))

from oracle import TASKS, rollout  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--seeds", default="0-2")
    args = p.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]
    seeds = parse_seeds(args.seeds)
    rates = []
    for task in tasks:
        wins = 0
        for seed in seeds:
            ep = rollout(task, seed, render=False)
            wins += int(ep["success"])
            print(f"{task} seed={seed} success={ep['success']} steps={ep['steps']}", flush=True)
        rate = wins / len(seeds)
        rates.append(rate)
        print(f"  {task}: {wins}/{len(seeds)} = {rate:.2f}")
    print(f"MEAN {sum(rates) / len(rates):.3f}")


if __name__ == "__main__":
    main()
