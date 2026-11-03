"""Six-dot Braille cell encoding.

A cell is a 6-bit mask in Unicode Braille order: bit ``n-1`` is dot ``n``.
Dots 1-2-3 run down the left column and dots 4-5-6 down the right column,
so the dot at grid slot (col, row) is ``col * 3 + row + 1`` (column-major).

The character for a mask is ``chr(0x2800 + mask)``; mask 0 (U+2800) is the
blank cell used as the word separator.

Dot positions are turned into a mask with :func:`assign_dots`, which snaps
each point to a fixed 2x3 grid given the cell origin and dot pitch. It never
normalises by the cell's own dots: that was the bug in the original project,
where cells with an empty bottom row or right column decoded as other letters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

BRAILLE_BASE = 0x2800
BLANK = chr(BRAILLE_BASE)
NUM_DOTS = 6
NUM_PATTERNS = 1 << NUM_DOTS


def slot_to_dot(col: int, row: int) -> int:
    """Dot number (1-6) at grid slot ``col`` in {0, 1}, ``row`` in {0, 1, 2}."""
    if col not in (0, 1) or row not in (0, 1, 2):
        raise ValueError(f"invalid slot (col={col}, row={row})")
    return col * 3 + row + 1


def dot_to_slot(dot: int) -> tuple[int, int]:
    """Grid slot ``(col, row)`` of dot number 1-6."""
    if not 1 <= dot <= NUM_DOTS:
        raise ValueError(f"invalid dot number {dot}")
    return (dot - 1) // 3, (dot - 1) % 3


def dots_to_mask(dots: Iterable[int]) -> int:
    mask = 0
    for d in dots:
        if not 1 <= d <= NUM_DOTS:
            raise ValueError(f"invalid dot number {d}")
        mask |= 1 << (d - 1)
    return mask


def mask_to_dots(mask: int) -> tuple[int, ...]:
    _check_mask(mask)
    return tuple(d for d in range(1, NUM_DOTS + 1) if mask & (1 << (d - 1)))


def mask_to_char(mask: int) -> str:
    _check_mask(mask)
    return chr(BRAILLE_BASE + mask)


def char_to_mask(ch: str) -> int:
    """Mask of a Unicode Braille character. An ASCII space is the blank cell."""
    if ch == " ":
        return 0
    code = ord(ch) - BRAILLE_BASE
    if not 0 <= code < NUM_PATTERNS:
        raise ValueError(f"{ch!r} is not a 6-dot Unicode Braille character")
    return code


def masks_to_text(masks: Iterable[int]) -> str:
    return "".join(mask_to_char(m) for m in masks)


def text_to_masks(text: str) -> list[int]:
    return [char_to_mask(ch) for ch in text]


def dot_distance(a: int, b: int) -> int:
    """Number of dot flips (Hamming distance) between two cells."""
    _check_mask(a)
    _check_mask(b)
    return bin(a ^ b).count("1")


def neighbors(mask: int, max_flips: int = 1) -> list[int]:
    """All masks reachable from ``mask`` by 1..``max_flips`` dot flips."""
    _check_mask(mask)
    out = []
    for k in range(1, max_flips + 1):
        for bits in combinations(range(NUM_DOTS), k):
            flip = 0
            for b in bits:
                flip |= 1 << b
            out.append(mask ^ flip)
    return out


def _check_mask(mask: int) -> None:
    if not 0 <= mask < NUM_PATTERNS:
        raise ValueError(f"mask {mask} out of range 0..63")


@dataclass
class CellReading:
    """Result of snapping detected dots to one cell's grid."""

    mask: int
    # Highest confidence among points assigned to each dot slot (index = dot - 1);
    # 0 where no point was assigned.
    slot_confidence: np.ndarray = field(default_factory=lambda: np.zeros(NUM_DOTS))
    # Indices of input points that did not fall near any slot.
    rejected: list[int] = field(default_factory=list)

    @property
    def char(self) -> str:
        return mask_to_char(self.mask)


def assign_dots(
    points: Sequence[Sequence[float]] | np.ndarray,
    origin: Sequence[float],
    col_step: float | Sequence[float],
    row_step: float | Sequence[float] | None = None,
    confidences: Sequence[float] | None = None,
    tolerance: float = 0.4,
) -> CellReading:
    """Snap dot centres to a cell's fixed 2x3 grid.

    Args:
        points: (N, 2) dot centres ``(x, y)`` in image coordinates.
        origin: ``(x, y)`` of the dot-1 slot (top-left) of the cell.
        col_step: offset from the dot-1 slot to the dot-4 slot. A number means
            ``(col_step, 0)``; a 2-vector allows a rotated or sheared grid.
        row_step: offset from the dot-1 slot to the dot-2 slot. A number means
            ``(0, row_step)``; ``None`` means the same length as ``col_step``,
            perpendicular to it.
        confidences: optional per-point detector confidence (defaults to 1).
        tolerance: maximum offset from a slot centre, in grid units, for a
            point to be accepted. Must be below 0.5.

    The grid must come from page-level spacing estimates, never from the
    cell's own dots.

    Returns:
        A :class:`CellReading` with the mask, per-slot confidence and the
        indices of points that matched no slot.
    """
    if not 0 < tolerance < 0.5:
        raise ValueError("tolerance must be in (0, 0.5)")
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    conf = np.ones(len(pts)) if confidences is None else np.asarray(confidences, dtype=float)
    if len(conf) != len(pts):
        raise ValueError("confidences must have one value per point")

    c = np.array([col_step, 0.0]) if np.isscalar(col_step) else np.asarray(col_step, dtype=float)
    if row_step is None:
        r = np.array([-c[1], c[0]])
    elif np.isscalar(row_step):
        r = np.array([0.0, row_step])
    else:
        r = np.asarray(row_step, dtype=float)
    basis = np.column_stack([c, r])
    if abs(np.linalg.det(basis)) < 1e-9:
        raise ValueError("col_step and row_step must not be parallel")

    # Grid coordinates (u = column, v = row) relative to the dot-1 slot.
    u, v = np.linalg.solve(basis, (pts - np.asarray(origin, dtype=float)).T)
    col = np.rint(u).astype(int)
    row = np.rint(v).astype(int)
    ok = (
        np.isin(col, (0, 1))
        & np.isin(row, (0, 1, 2))
        & (np.abs(u - col) <= tolerance)
        & (np.abs(v - row) <= tolerance)
    )

    reading = CellReading(mask=0, slot_confidence=np.zeros(NUM_DOTS))
    for i in range(len(pts)):
        if not ok[i]:
            reading.rejected.append(i)
            continue
        idx = slot_to_dot(int(col[i]), int(row[i])) - 1
        reading.mask |= 1 << idx
        reading.slot_confidence[idx] = max(reading.slot_confidence[idx], conf[i])
    return reading
