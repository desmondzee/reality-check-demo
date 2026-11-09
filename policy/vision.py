"""Overhead colour blobs to table metres.

Coefficients are a least-squares fit of pixel centroids to true cup / marker
positions on the engine's `top` camera (640x480). Median residual is about
1–2 mm. The policy loads the same numbers from /weights/affine.json.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# 3x2: [u, v, 1] @ coef -> (x, y) metres. Fit on reveal frames, seeds 0-19.
RED = np.array([
    [-5.58616236e-07, -2.57937773e-03],
    [-2.54471911e-03, 2.15848799e-05],
    [1.45794045e00, 8.18070379e-01],
])
BLUE = np.array([
    [1.03842843e-07, -2.58644874e-03],
    [-2.58336476e-03, -2.25668073e-06],
    [1.46860471e00, 8.26716134e-01],
])
GREEN = np.array([
    [-1.36568845e-06, -2.71175037e-03],
    [-2.71350690e-03, -6.43326690e-07],
    [1.50025229e00, 8.66392962e-01],
])
COEFS = {"red": RED, "blue": BLUE, "green": GREEN}

REVEAL_XYZ = np.array([0.45, 0.0, 0.45])
MIN_PIXELS = 150


def decode_rgb(raw: str | bytes) -> np.ndarray:
    blob = base64.b64decode(raw) if isinstance(raw, str) else raw
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def mask_count(img: np.ndarray, kind: str) -> tuple[np.ndarray | None, int]:
    r, g, b = (img[..., i].astype(np.int16) for i in range(3))
    mask = {
        "red": (r > 150) & (g < 90) & (b < 90),
        "blue": (b > 150) & (r < 90) & (g < 120),
        "green": (g > 140) & (r < 90) & (b < 110),
    }[kind]
    n = int(mask.sum())
    if n < 30:
        return None, n
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()], dtype=float), n


def table_xy(uv: np.ndarray, kind: str, coefs: dict[str, np.ndarray] | None = None) -> np.ndarray:
    coef = (coefs or COEFS)[kind]
    return np.array([uv[0], uv[1], 1.0], dtype=float) @ coef
