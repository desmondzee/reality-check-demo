"""Save the frame the policy queries, labelled with the true cup position."""
from __future__ import annotations

import argparse
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

from collect_demos import parse_seeds  # noqa: E402
from oracle import TASKS, _other_xy  # noqa: E402
from policy.act_net import IMG, TASK_INDEX  # noqa: E402
from policy.localizer import kind_from_task  # noqa: E402
from policy.tracker import OPEN, Tracker  # noqa: E402
from policy.vision import MIN_PIXELS, REVEAL_XYZ, mask_count  # noqa: E402
from rc_engine.schema import JointDelta  # noqa: E402
from rc_engine.tasks.cups import PUBLIC, CupsEnv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "query")
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--tasks", default=",".join(TASKS))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task in [t for t in args.tasks.split(",") if t]:
        kind = kind_from_task(task)
        for seed in parse_seeds(args.seeds):
            env = CupsEnv(task, PUBLIC[task])
            env.render = True
            obs = env.reset(seed)
            tracker = Tracker()
            tracker.reset(obs.joint_pos, kind)
            frame = obs.images["top"]
            for _ in range(40):
                uv, n = mask_count(frame, "red")
                if uv is not None and n >= MIN_PIXELS:
                    break
                tracker.waypoints = [(REVEAL_XYZ.copy(), OPEN, 2)]
                tracker.phase = 0
                tracker.mode = "track"
                values, grip = tracker.act(obs.ee_pos, obs.joint_pos)
                result = env.step(JointDelta(tuple(values), grip))
                obs = result.obs
                frame = obs.images["top"]
            small = np.asarray(Image.fromarray(frame).resize((IMG, IMG), Image.BILINEAR))
            path = args.out / f"{task}_seed{seed:03d}.npz"
            np.savez_compressed(
                path,
                images=small[None],
                cup_xy=env.data.xpos[env.cup_body][:2].astype(np.float32)[None],
                other_xy=_other_xy(env).astype(np.float32)[None],
                has_other=np.bool_(kind in ("place", "stack", "pick")),
                task_index=np.int64(TASK_INDEX[task]),
                success=np.bool_(True),
            )
            print(f"kept {path.name} red_px={mask_count(frame, 'red')[1]}", flush=True)


if __name__ == "__main__":
    main()
