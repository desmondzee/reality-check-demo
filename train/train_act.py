"""Train the cup-cell ACT on oracle demos and optionally push the checkpoint."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.act_net import CHUNK, build_act_net, normalize_action  # noqa: E402


def load_episodes(folder: Path) -> list[dict]:
    episodes = []
    folders = [Path(p) for p in str(folder).split(",") if p]
    paths = []
    for one in folders:
        paths.extend(sorted(one.glob("*.npz")))
    for path in paths:
        data = np.load(path)
        if not bool(data["success"]):
            continue
        episodes.append({
            "images": data["images"],
            "states": data["states"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
            "task": int(data["task_index"]),
        })
    if not episodes:
        raise SystemExit(f"no successful demos in {folder}")
    return episodes


def sample_batch(episodes: list[dict], batch: int, torch):
    images, states, tasks, actions, mask = [], [], [], [], []
    for _ in range(batch):
        ep = random.choice(episodes)
        t = random.randrange(len(ep["actions"]))
        frame = ep["images"][t].astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5
        images.append(torch.from_numpy(frame).permute(2, 0, 1))
        states.append(torch.from_numpy(ep["states"][t]))
        tasks.append(ep["task"])
        chunk = []
        valid = []
        n = len(ep["actions"])
        for k in range(CHUNK):
            j = min(t + k, n - 1)
            chunk.append(ep["actions"][j])
            valid.append(1.0 if t + k < n else 0.0)
        actions.append(torch.from_numpy(np.stack(chunk)))
        mask.append(torch.tensor(valid))
    return (
        torch.stack(images),
        torch.stack(states),
        torch.tensor(tasks, dtype=torch.long),
        normalize_action(torch.stack(actions)),
        torch.stack(mask),
    )


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "demos")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "act")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--push", default="", help="public HF repo id, e.g. akoniti/reality-check-demo-act")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    episodes = load_episodes(args.data)
    print(f"episodes={len(episodes)}", flush=True)

    model = build_act_net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    for step in range(1, args.steps + 1):
        image, state, task, target, mask = sample_batch(episodes, args.batch, torch)
        image, state, task = image.to(device), state.to(device), task.to(device)
        target, mask = target.to(device), mask.to(device)
        pred = model(image, state, task)
        err = ((pred - target) ** 2).mean(dim=-1)
        loss = (err * mask).sum() / mask.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            print(f"step {step} loss {float(loss):.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.eval()
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, args.out / "act.pt")
    cfg = {"model": model.hparams, "chunk": CHUNK, "image": 128, "tasks": list(__import__('policy.act_net', fromlist=['TASKS']).TASKS)}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(repo_id=args.push, folder_path=str(args.out), repo_type="model", commit_message="ACT cup-cell checkpoint")
        print(f"pushed {args.push}", flush=True)


if __name__ == "__main__":
    main()
