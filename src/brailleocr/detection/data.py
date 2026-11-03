"""Page loading and training crops for dot detectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import disk_mask, gaussian_heatmap


@dataclass
class PageRecord:
    id: str
    image: Path
    label: Path
    page_index: int
    severity: float


def load_index(split_dir: str | Path, pages_per_level: int | None = None, max_pages: int | None = None) -> list[PageRecord]:
    """Pages listed in ``index.jsonl``.

    ``pages_per_level`` keeps page indices below that number (for paired test
    splits this gives the same pages at every severity); ``max_pages`` keeps
    the first pages in index order.
    """
    root = Path(split_dir)
    recs = []
    for line in (root / "index.jsonl").read_text().splitlines():
        r = json.loads(line)
        if pages_per_level is not None and r["page_index"] >= pages_per_level:
            continue
        recs.append(PageRecord(r["id"], root / r["image"], root / r["label"], r["page_index"], r["severity"]))
    return recs[:max_pages] if max_pages else recs


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)


class CropDataset(Dataset):
    """Random crops with dot targets. Item ``i`` yields ``crops_per_page`` crops of page ``i``.

    Augmentation: random scale (dot size varies with camera distance), flips
    and 90-degree rotations. Crops are centred near a dot most of the time and
    anywhere on the image otherwise, so background, page edges and ghost dots
    are seen as negatives.

    ``target``: ``"heatmap"`` (Gaussian, sigma = 1/4 dot diameter) or ``"mask"``
    (filled disks of the dot radius, as in the original project).
    """

    def __init__(
        self,
        records: list[PageRecord],
        target: str = "heatmap",
        crop: int = 384,
        crops_per_page: int = 4,
        scale: tuple[float, float] = (0.7, 1.4),
        p_near_dot: float = 0.8,
    ):
        if target not in ("heatmap", "mask"):
            raise ValueError(target)
        self.records = records
        self.target = target
        self.crop = crop
        self.crops_per_page = crops_per_page
        self.scale = scale
        self.p_near_dot = p_near_dot

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        label = json.loads(rec.label.read_text())
        img = read_rgb(rec.image)
        dots = np.array([d[:2] for d in label["dots"]], float).reshape(-1, 2)
        diameter = label["geometry"]["dot_diameter_mm"] * label["px_per_mm"]
        rng = np.random.default_rng()
        xs, ys = [], []
        for _ in range(self.crops_per_page):
            x, y = self._crop(img, dots, diameter, rng)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    def _crop(self, img, dots, diameter, rng):
        S = self.crop
        h, w = img.shape[:2]
        f = float(np.exp(rng.uniform(*np.log(self.scale))))
        if len(dots) and rng.uniform() < self.p_near_dot:
            cx, cy = dots[rng.integers(len(dots))] + rng.uniform(-0.4, 0.4, 2) * S / f
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)

        # Source -> crop: scale by f (with optional flips), rotate by k*90 deg, centre.
        k = int(rng.integers(4))
        rot = np.array([[1, 0], [0, 1]], float)
        for _ in range(k):
            rot = np.array([[0, -1], [1, 0]], float) @ rot
        flip = np.diag([rng.choice([-1, 1]), 1.0])
        A2 = f * rot @ flip
        A = np.column_stack([A2, np.array([S / 2, S / 2]) - A2 @ np.array([cx, cy])])
        border = tuple(int(v) for v in img.reshape(-1, 3)[:: max(1, img.size // 3000)].mean(0))
        crop = cv2.warpAffine(img, A, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

        pts = dots @ A2.T + A[:, 2] if len(dots) else dots
        inside = (pts[:, 0] > -diameter) & (pts[:, 0] < S + diameter) & (pts[:, 1] > -diameter) & (pts[:, 1] < S + diameter)
        pts = pts[inside]
        if self.target == "heatmap":
            t = gaussian_heatmap((S, S), pts, max(1.0, 0.25 * diameter * f))
        else:
            t = disk_mask((S, S), pts, max(1.5, 0.5 * diameter * f))
        return to_tensor(crop), torch.from_numpy(t)[None]


def collate_crops(batch):
    xs, ys = zip(*batch)
    return torch.cat(xs), torch.cat(ys)
