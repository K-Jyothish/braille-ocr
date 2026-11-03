"""Build train/val/test splits of synthetic pages with full ground truth.

Each page gets a JPEG image and a JSON label holding the print text, the UEB
grade, the Braille lines, every raised dot (image px, with its line, column and
dot number), the verso ghost dots, and the page-to-image homography from which
the ideal position of any cell slot can be recovered (:func:`cell_slots`).

Train and val pages draw a random severity per page. Test pages are paired:
the same text, layout, geometry and nuisance factors (paper, light direction,
shadow placement, ...) are rendered at every level in ``TEST_LEVELS``, and only
the strength of each degradation changes, so robustness curves compare like
with like. Text comes from
disjoint books per split (see :mod:`brailleocr.corpus`).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import translation
from ..encoding import char_to_mask
from .layout import Geometry, PageLayout, paginate
from .render import render_page

TEST_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
DOT_FIELDS = ["x", "y", "line", "col", "dot", "amplitude"]


@dataclass
class SplitText:
    """Sentences of one split with their UEB Grade 1 and Grade 2 translations."""

    text: list[str]
    braille: dict[int, list[str]]

    @classmethod
    def from_sentences(cls, sentences: list[str]) -> "SplitText":
        g1 = translation.to_braille(sentences, grade=1)
        g2 = translation.to_braille(sentences, grade=2)
        # Keep sentences Liblouis translates fully into 6-dot cells in both grades.
        keep = [
            i for i in range(len(sentences))
            if g1[i] and g2[i] and translation.is_six_dot(g1[i]) and translation.is_six_dot(g2[i])
        ]
        return cls(
            text=[sentences[i] for i in keep],
            braille={1: [g1[i] for i in keep], 2: [g2[i] for i in keep]},
        )


def make_page(
    split_text: SplitText, split: str, index: int, seed: int, severity: float | None = None
) -> tuple[bytes, dict]:
    """Render page ``index`` of ``split``. Returns (jpeg bytes, label dict).

    Content (text, grade, layout, geometry) depends only on (seed, split, index);
    the rendering also depends on severity. ``severity=None`` draws one at random.
    """
    content_rng = np.random.default_rng([seed, _SPLIT_CODE[split], index])
    grade = int(content_rng.choice([1, 2]))
    cells_per_line = int(content_rng.integers(20, 37))
    max_lines = int(content_rng.integers(6, 19))
    geometry = Geometry.sample(content_rng)
    margins = tuple(content_rng.uniform(8, 20, size=2))
    px_per_mm = content_rng.uniform(5.0, 8.0)
    n = len(split_text.text)
    start = int(content_rng.integers(0, n))
    ghost_start = int(content_rng.integers(0, n))
    if severity is None:
        severity = float(content_rng.uniform(0, 1))

    sentences = split_text.braille[grade]
    lines, used, truncated = paginate(sentences, start, cells_per_line, max_lines)
    for _ in range(50):  # avoid pages whose print text would be cut off
        if not truncated:
            break
        start = int(content_rng.integers(0, n))
        lines, used, truncated = paginate(sentences, start, cells_per_line, max_lines)
    layout = PageLayout(lines, cells_per_line, geometry, margins)
    ghost_lines, _, _ = paginate(sentences, ghost_start, cells_per_line, max_lines + 2)
    ghost = PageLayout(ghost_lines, cells_per_line, Geometry.sample(content_rng), margins)

    # Same render stream at every severity: paired pages differ only in degradation strength.
    render_rng = np.random.default_rng([seed, _SPLIT_CODE[split], index, 1])
    r = render_page(layout, render_rng, severity, px_per_mm, ghost_layout=ghost)

    label = {
        "split": split,
        "page_index": index,
        "seed": seed,
        "severity": severity,
        "grade": grade,
        "text": " ".join(split_text.text[start : start + used]),
        "text_truncated": truncated,
        "braille_lines": lines,
        "image_size": [int(r.image.shape[1]), int(r.image.shape[0])],
        "px_per_mm": r.px_per_mm,
        "geometry": geometry.to_dict(),
        "cells_per_line": cells_per_line,
        "margin_mm": list(margins),
        "page_to_image": r.page_to_image.tolist(),
        "dot_fields": DOT_FIELDS,
        "dots": [[round(x, 2), round(y, 2), int(l), int(c), int(d), round(a, 3)] for x, y, l, c, d, a in r.dots],
        "ghost_dots": np.round(r.ghost_dots, 2).tolist(),
        "render_params": r.params,
    }
    return r.jpeg, label


def cell_slots(label: dict) -> dict[tuple[int, int], np.ndarray]:
    """Ideal image positions of the 6 dot slots of every cell, keyed by (line, col).

    Row ``d - 1`` of each (6, 2) array is the slot of dot ``d``. Blank cells are
    included, so each line has ``len(line)`` entries.
    """
    layout = PageLayout(
        label["braille_lines"], label["cells_per_line"], Geometry(**label["geometry"]), tuple(label["margin_mm"])
    )
    H = np.asarray(label["page_to_image"])
    out = {}
    for i, line in enumerate(label["braille_lines"]):
        for j in range(len(line)):
            mm = np.array([layout.slot_mm(i, j, d) for d in range(1, 7)])
            p = np.column_stack([mm, np.ones(6)]) @ H.T
            out[(i, j)] = p[:, :2] / p[:, 2:3]
    return out


def label_masks(label: dict) -> dict[tuple[int, int], int]:
    """Ground-truth mask of every cell, keyed by (line, col)."""
    return {
        (i, j): char_to_mask(ch)
        for i, line in enumerate(label["braille_lines"])
        for j, ch in enumerate(line)
    }


# ---- split generation -------------------------------------------------------

_WORKER_TEXT: SplitText | None = None


def _init_worker(split_text: SplitText) -> None:
    global _WORKER_TEXT
    _WORKER_TEXT = split_text


def _write_page(job: tuple[str, str, int, int, float | None, str]) -> dict:
    page_id, split, index, seed, severity, out_dir = job
    jpeg, label = make_page(_WORKER_TEXT, split, index, seed, severity)
    label["id"] = page_id
    root = Path(out_dir)
    (root / "images" / f"{page_id}.jpg").write_bytes(jpeg)
    (root / "labels" / f"{page_id}.json").write_text(json.dumps(label, separators=(",", ":")))
    return {
        "id": page_id,
        "image": f"images/{page_id}.jpg",
        "label": f"labels/{page_id}.json",
        "page_index": index,
        "severity": label["severity"],
        "grade": label["grade"],
        "n_dots": len(label["dots"]),
    }


def generate_split(
    sentences: list[str],
    split: str,
    out_dir: str | Path,
    n_pages: int,
    seed: int = 0,
    levels: tuple[float, ...] | None = None,
    workers: int = 1,
) -> Path:
    """Write ``n_pages`` pages (times ``len(levels)`` if given) to ``out_dir/split``.

    With ``levels``, every page index is rendered once per severity level
    (paired design, used for the test split). Returns the split directory,
    which holds ``images/``, ``labels/`` and ``index.jsonl``.
    """
    root = Path(out_dir) / split
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    split_text = SplitText.from_sentences(sentences)

    jobs = []
    for i in range(n_pages):
        if levels is None:
            jobs.append((f"{split}_{i:05d}", split, i, seed, None, str(root)))
        else:
            for lv in levels:
                jobs.append((f"{split}_{i:05d}_s{round(lv * 100):03d}", split, i, seed, lv, str(root)))

    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(split_text,)) as ex:
            rows = list(ex.map(_write_page, jobs, chunksize=8))
    else:
        _init_worker(split_text)
        rows = [_write_page(j) for j in jobs]

    with open(root / "index.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return root
