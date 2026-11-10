"""Train the overhead cup localizer and optionally push it to Hugging Face."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.localizer import build_localizer  # noqa: E402

# pick=1 and stack=4 label the blue cup; place=2 labels the green marker.
BLUE_TASKS = {1, 4}
GREEN_TASKS = {2}


def _visible(frame: np.ndarray) -> bool:
    """Loose red-pixel test on the stored 128 image. Occluded frames are dropped."""
    red = frame[..., 0].astype(np.int16)
    green = frame[..., 1].astype(np.int16)
    blue = frame[..., 2].astype(np.int16)
    return int(((red > 120) & (red > green + 30) & (red > blue + 30)).sum()) > 12


def load_episodes(folder: Path) -> list[dict]:
    episodes = []
    for path in sorted(folder.glob("*.npz")):
        data = np.load(path)
        images = data["images"]
        visible = [t for t in range(len(images)) if _visible(images[t])]
        if not visible:
            visible = [0]
        episodes.append({
            "images": images,
            "cup_xy": data["cup_xy"].astype(np.float32),
            "other_xy": data["other_xy"].astype(np.float32),
            "task": int(data["task_index"]),
            "visible": visible,
        })
    if not episodes:
        raise SystemExit(f"no episodes in {folder}")
    return episodes


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "localize")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "localizer")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--push", default="")
    args = p.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = load_episodes(args.data)
    print(f"device={device} episodes={len(episodes)}", flush=True)
    model = build_localizer().to(device)
    # Keep the calibrated pixel-to-table maps fixed. The net only learns where the blob is.
    for name in ("cup_map", "blue_map", "green_map"):
        for param in getattr(model, name).parameters():
            param.requires_grad = False
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    for step in range(1, args.steps + 1):
        images, cup, other = [], [], []
        blue_rows, green_rows = [], []
        for i in range(args.batch):
            ep = random.choice(episodes)
            t = random.choice(ep["visible"])
            frame = ep["images"][t].astype(np.float32) / 255.0
            images.append(torch.from_numpy(((frame - 0.5) / 0.5)).permute(2, 0, 1))
            cup.append(torch.from_numpy(ep["cup_xy"][t]))
            other.append(torch.from_numpy(ep["other_xy"][t]))
            if ep["task"] in BLUE_TASKS:
                blue_rows.append(i)
            elif ep["task"] in GREEN_TASKS:
                green_rows.append(i)
        image = torch.stack(images).to(device)
        cup_t = torch.stack(cup).to(device)
        other_t = torch.stack(other).to(device)
        pred = model(image)
        loss = ((pred[:, :2] - cup_t) ** 2).mean()
        if blue_rows:
            idx = torch.tensor(blue_rows, device=device)
            loss = loss + ((pred[idx, 2:4] - other_t[idx]) ** 2).mean()
        if green_rows:
            idx = torch.tensor(green_rows, device=device)
            loss = loss + ((pred[idx, 4:6] - other_t[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            err = (pred[:, :2] - cup_t).norm(dim=-1).mean()
            print(f"step {step} loss {float(loss.detach()):.5f} cup_m {float(err.detach()):.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.eval()
    # Report error on the first frame of every episode (the home view).
    errs = []
    with torch.no_grad():
        for ep in episodes:
            frame = ep["images"][0].astype(np.float32) / 255.0
            image = torch.from_numpy(((frame - 0.5) / 0.5)).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(image)[0, :2].cpu().numpy()
            errs.append(float(np.linalg.norm(pred - ep["cup_xy"][0])))
    print(f"home-view cup error median {np.median(errs):.4f} m  p90 {np.quantile(errs, 0.9):.4f} m", flush=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(args.out / "model.safetensors"))
    cfg = {"kind": "spatial-localizer", "model": model.hparams}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(repo_id=args.push, folder_path=str(args.out), repo_type="model", commit_message="cup localizer")
        print(f"pushed {args.push}", flush=True)


if __name__ == "__main__":
    main()
