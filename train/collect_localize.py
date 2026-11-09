"""Save JPEG frames plus privileged cup / target xy for the localizer."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("RC_ENGINE", ROOT.parent / "reality-check" / "engine"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from collect_demos import jpeg_resize, parse_seeds  # noqa: E402
from oracle import TASKS, rollout  # noqa: E402
from policy.act_net import TASK_INDEX  # noqa: E402
from policy.localizer import kind_from_task  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "localize")
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in args.tasks.split(",") if t]
    for task in tasks:
        for seed in parse_seeds(args.seeds):
            ep = rollout(task, seed, render=True)
            if not ep["success"]:
                print(f"skip {task} seed={seed}", flush=True)
                continue
            kind = kind_from_task(task)
            images = np.stack([jpeg_resize(f) for f in ep["images"]])
            path = args.out / f"{task}_seed{seed:03d}.npz"
            np.savez_compressed(
                path,
                images=images,
                cup_xy=np.asarray(ep["cup_xy"], dtype=np.float32),
                other_xy=np.asarray(ep["other_xy"], dtype=np.float32),
                has_other=np.bool_(kind in ("place", "stack", "pick")),
                task_index=np.int64(TASK_INDEX[task]),
                success=np.bool_(True),
            )
            print(f"kept {path.name} steps={len(ep['images'])}", flush=True)


if __name__ == "__main__":
    main()
