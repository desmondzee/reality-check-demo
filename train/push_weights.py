"""Upload runs/localizer to the public Hub repo named in manifest.json.

The token on this machine must be allowed to create models. A fine-grained
token with an empty permission list cannot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    repo = manifest["weights"]["repo_id"]
    folder = ROOT / "runs" / "localizer"
    if not (folder / "affine.json").is_file():
        raise SystemExit(f"missing {folder / 'affine.json'}; run: uv run python eval_cup.py --seeds 0")
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo,
        folder_path=str(folder),
        repo_type="model",
        commit_message="Calibrated overhead affine for the cup cell",
    )
    info = api.model_info(repo)
    print(f"pushed {info.id} private={info.private} sha={info.sha}")


if __name__ == "__main__":
    sys.exit(main())
