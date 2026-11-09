"""Roll the privileged oracle and keep successful episodes.

Images are JPEG-round-tripped at quality 90 (same as the engine default) then
resized to 128, so training pixels match what ActPolicy sees from /act.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("RC_ENGINE", ROOT.parent / "reality-check" / "engine"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from oracle import TASKS, rollout  # noqa: E402
from policy.act_net import IMG, TASK_INDEX  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x]


def jpeg_resize(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90, subsampling=0, optimize=False, progressive=False)
    image = Image.open(buf).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "demos")
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in args.tasks.split(",") if t]
    seeds = parse_seeds(args.seeds)
    kept = 0
    tried = 0
    for task in tasks:
        wins = 0
        for seed in seeds:
            tried += 1
            ep = rollout(task, seed, render=True)
            if not ep["success"]:
                print(f"skip {task} seed={seed}", flush=True)
                continue
            images = np.stack([jpeg_resize(f) for f in ep["images"]])
            path = args.out / f"{task}_seed{seed:03d}.npz"
            np.savez_compressed(
                path,
                images=images,
                states=np.asarray(ep["states"], dtype=np.float32),
                actions=np.asarray(ep["actions"], dtype=np.float32),
                task_index=np.int64(TASK_INDEX[task]),
                task_id=np.str_(task),
                seed=np.int64(seed),
                success=np.bool_(True),
            )
            wins += 1
            kept += 1
            print(f"kept {path.name} steps={len(ep['actions'])}", flush=True)
        print(f"{task}: {wins}/{len(seeds)}")
    print(f"kept {kept}/{tried} -> {args.out}")


if __name__ == "__main__":
    main()
