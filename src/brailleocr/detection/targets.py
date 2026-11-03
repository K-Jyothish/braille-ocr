"""Training targets rendered from dot centres."""

from __future__ import annotations

import numpy as np


def gaussian_heatmap(shape: tuple[int, int], points: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Heatmap with a Gaussian (peak 1) per dot, combined by maximum.

    The pixel nearest each centre is set to exactly 1 so the focal loss has a
    positive there; neighbouring pixels follow the sub-pixel Gaussian.
    """
    h, w = shape
    heat = np.zeros((h, w), np.float32)
    for (x, y), s in zip(np.asarray(points, float).reshape(-1, 2), np.broadcast_to(sigmas, len(points))):
        r = int(np.ceil(3 * s))
        x0, x1 = max(int(x) - r, 0), min(int(x) + r + 2, w)
        y0, y1 = max(int(y) - r, 0), min(int(y) + r + 2, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32)
        np.maximum(heat[y0:y1, x0:x1], g, out=heat[y0:y1, x0:x1])
        cx, cy = int(round(x)), int(round(y))
        if 0 <= cx < w and 0 <= cy < h:
            heat[cy, cx] = 1.0
    return heat


def disk_mask(shape: tuple[int, int], points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Binary mask with a filled disk per dot (the original project's target)."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    for (x, y), r in zip(np.asarray(points, float).reshape(-1, 2), np.broadcast_to(radii, len(points))):
        x0, x1 = max(int(x - r) - 1, 0), min(int(x + r) + 2, w)
        y0, y1 = max(int(y - r) - 1, 0), min(int(y + r) + 2, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = 1.0
    return mask
