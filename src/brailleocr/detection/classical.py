"""Classical threshold + contour dot detector (baseline).

Embossed dots appear as a highlight next to a shadow, not as dark blobs, so
the original project's fixed dark threshold does not apply to photos. This
baseline keeps its structure (threshold -> morphology -> contours -> area and
circularity filters -> centroids) on a local-contrast image instead:

1. local contrast = |gray - Gaussian background|, smoothed at about the dot
   radius (side-lit dots are a bright half and a dark half with little
   contrast at the centre; smoothing fills each ring into one blob);
2. threshold at k times Otsu's threshold of the contrast image (paper vs
   dots), so it adapts to each page's dot contrast;
3. morphological closing joins each dot's highlight and shadow lobes;
4. contours filtered by area and circularity; centroid = dot centre,
   confidence = mean contrast in the contour relative to the threshold.

Parameters are tuned on the validation split (:func:`tune`), like the
networks' thresholds, so the comparison is fair.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class ClassicalParams:
    bg_sigma: float = 10.0
    smooth: float = 1.5
    k: float = 1.0
    close: int = 3
    min_area: float = 6.0
    max_area: float = 400.0
    min_circularity: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


def detect(image_rgb: np.ndarray, p: ClassicalParams = ClassicalParams()) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    contrast = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), p.bg_sigma))
    contrast = cv2.GaussianBlur(contrast, (0, 0), p.smooth)
    c8 = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu, _ = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = p.k * (otsu / 255.0 * (contrast.max() - contrast.min()) + contrast.min())
    binary = (contrast > thr).astype(np.uint8)
    if p.close > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close, p.close)))

    # CCOMP, not EXTERNAL: a textured background around the page can form one
    # component enclosing the whole page, and dots inside its hole must still
    # be found. Contours without a parent are outer boundaries at any depth.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    xy, conf = [], []
    for c, h in zip(contours, hierarchy[0] if hierarchy is not None else []):
        if h[3] != -1:
            continue
        area = cv2.contourArea(c)
        if not p.min_area <= area <= p.max_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0 or 4 * np.pi * area / perim**2 < p.min_circularity:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
        bx, by, bw, bh = cv2.boundingRect(c)
        patch = contrast[by:by + bh, bx:bx + bw]
        xy.append((x, y))
        conf.append(min(1.0, float(patch.mean()) / (2 * thr)))
    return np.array(xy, float).reshape(-1, 2), np.array(conf, float)


GRID = {
    "bg_sigma": [8.0, 14.0],
    "smooth": [1.0, 1.5, 2.5],
    "k": [0.8, 1.0, 1.3, 1.7],
    "close": [1, 3],
}


def tune(pages: list[tuple[np.ndarray, np.ndarray]], tol_px: float = 3.0, grid: dict = GRID) -> tuple[ClassicalParams, float]:
    """Grid search maximising micro dot F1 on ``pages`` = [(image_rgb, gt_xy), ...]."""
    from ..evaluation import match_dots

    best, best_f1 = ClassicalParams(), -1.0
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        p = ClassicalParams(**dict(zip(keys, values)))
        tp = fp = fn = 0
        for img, gt in pages:
            xy, _ = detect(img, p)
            a, b, c, _ = match_dots(xy, gt, tol_px)
            tp, fp, fn = tp + a, fp + b, fn + c
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best, best_f1 = p, f1
    return best, best_f1
