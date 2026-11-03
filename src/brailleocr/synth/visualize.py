"""Ground-truth overlays for visual inspection of synthetic pages."""

from __future__ import annotations

import cv2
import numpy as np

from .dataset import cell_slots


def draw_ground_truth(image_bgr: np.ndarray, label: dict) -> np.ndarray:
    """Copy of ``image_bgr`` with the label drawn on it.

    Blue points = cell slot grid, green circles = raised dots, red circles =
    verso ghost dots (which a detector must ignore).
    """
    img = image_bgr.copy()
    r = max(2, int(label["px_per_mm"] * 0.4))
    for slots in cell_slots(label).values():
        for x, y in slots:
            cv2.circle(img, (int(round(x)), int(round(y))), 1, (255, 128, 0), -1)
    for x, y in label["ghost_dots"]:
        cv2.circle(img, (int(round(x)), int(round(y))), r, (0, 0, 255), 1)
    for x, y, *_ in label["dots"]:
        cv2.circle(img, (int(round(x)), int(round(y))), r, (0, 200, 0), 1)
    return img
