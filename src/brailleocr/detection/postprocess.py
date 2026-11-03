"""Turning probability maps into dot centres with confidences."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def heatmap_peaks(prob: np.ndarray, threshold: float = 0.3, nms: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of a heatmap above ``threshold``.

    Non-maximum suppression keeps pixels equal to the maximum of their
    ``nms`` x ``nms`` neighbourhood (5 px is below the smallest dot pitch in
    the data, ~7 px). Each peak is refined to sub-pixel precision by the
    weighted centroid of its 3x3 neighbourhood. Confidence is the peak value.
    """
    prob = np.ascontiguousarray(prob, np.float32)
    local_max = cv2.dilate(prob, np.ones((nms, nms), np.uint8))
    peaks = (prob >= local_max) & (prob >= threshold)
    # Plateaus (equal neighbouring maxima) count once.
    labels, n = ndimage.label(peaks)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    ys, xs = np.array(ndimage.center_of_mass(peaks, labels, range(1, n + 1))).T
    ys, xs = np.rint(ys).astype(int), np.rint(xs).astype(int)

    h, w = prob.shape
    pad = np.pad(prob, 1)
    win = np.stack([pad[ys + dy, xs + dx] for dy in range(3) for dx in range(3)], 1)  # (n, 9)
    offs = np.array([(dx - 1, dy - 1) for dy in range(3) for dx in range(3)], float)
    wsum = win.sum(1, keepdims=True)
    refined = np.column_stack([xs, ys]) + (win @ offs) / np.maximum(wsum, 1e-6)
    return refined, prob[ys, xs]


def mask_centroids(prob: np.ndarray, threshold: float = 0.5, min_area: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Centroids of connected components of ``prob >= threshold`` (the original project's
    post-processing, without its hand-tuned circularity filter). Confidence is the
    component's maximum probability."""
    binary = (prob >= threshold).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return np.zeros((0, 2)), np.zeros(0)
    conf = np.asarray(ndimage.maximum(prob, labels, keep), float)
    return centroids[keep], conf


POSTPROCESS = {"heatmap": heatmap_peaks, "mask": mask_centroids}
