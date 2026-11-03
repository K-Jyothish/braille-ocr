"""Full-page inference and a common detector interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import classical
from .data import to_tensor
from .models import build_model
from .postprocess import POSTPROCESS


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_prob(
    model: nn.Module, image_rgb: np.ndarray, device: torch.device, tile: int = 1024, overlap: int = 64, batch: int = 1
) -> np.ndarray:
    """Sigmoid output for a whole page, computed in overlapping tiles.

    Each tile contributes only its centre (``overlap`` px from every side is
    context), so the result has no seams. ``tile`` must be a multiple of the
    model's required input multiple.
    """
    model.eval()
    h, w = image_rgb.shape[:2]
    step = tile - 2 * overlap
    ny, nx = -(-h // step), -(-w // step)
    ph, pw = ny * step + 2 * overlap, nx * step + 2 * overlap
    padded = np.pad(image_rgb, ((overlap, ph - h - overlap), (overlap, pw - w - overlap), (0, 0)), mode="reflect")
    x = to_tensor(padded)

    out = np.zeros((ny * step, nx * step), np.float32)
    coords = [(iy * step, ix * step) for iy in range(ny) for ix in range(nx)]
    for k in range(0, len(coords), batch):
        chunk = coords[k:k + batch]
        tiles = torch.stack([x[:, y0:y0 + tile, x0:x0 + tile] for y0, x0 in chunk]).to(device)
        prob = torch.sigmoid(model(tiles)).float().cpu().numpy()[:, 0]
        for (y0, x0), p in zip(chunk, prob):
            out[y0:y0 + step, x0:x0 + step] = p[overlap:overlap + step, overlap:overlap + step]
    return out[:h, :w]


class NeuralDetector:
    def __init__(self, model: nn.Module, kind: str, threshold: float, device: torch.device, tile: int = 1024, batch: int = 1):
        self.model = model.to(device).eval()
        self.kind = kind
        self.threshold = threshold
        self.device = device
        self.tile = tile
        self.batch = batch

    def prob(self, image_rgb: np.ndarray) -> np.ndarray:
        return predict_prob(self.model, image_rgb, self.device, self.tile, batch=self.batch)

    def detect(self, image_rgb: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.points(self.prob(image_rgb), threshold)

    def points(self, prob: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        return POSTPROCESS[self.kind](prob, threshold=self.threshold if threshold is None else threshold)


class ClassicalDetector:
    kind = "classical"

    def __init__(self, params: classical.ClassicalParams):
        self.params = params

    def detect(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return classical.detect(image_rgb, self.params)


def save_checkpoint(path: str | Path, model: nn.Module, name: str, kind: str, threshold: float, **extra) -> None:
    torch.save({"model": name, "kind": kind, "threshold": threshold, "state_dict": model.state_dict(), **extra}, path)


def load_detector(path: str | Path, device: torch.device | str = "auto", tile: int = 1024, batch: int = 1) -> NeuralDetector:
    device = pick_device(device) if isinstance(device, str) else device
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kwargs = dict(ckpt.get("model_kwargs", {}))
    if ckpt["model"] == "resnet_heatmap":
        kwargs["pretrained"] = False  # weights come from the checkpoint
    model, kind = build_model(ckpt["model"], **kwargs)
    model.load_state_dict(ckpt["state_dict"])
    return NeuralDetector(model, kind, ckpt["threshold"], device, tile, batch)
