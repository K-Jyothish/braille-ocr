"""Physical page layout: Braille cell geometry, line wrapping and pagination.

All lengths are in millimetres. Default spacing follows common embossing
standards (dot pitch 2.5 mm, cell pitch 6.2 mm, line pitch 10 mm, dot base
diameter 1.5 mm); :meth:`Geometry.sample` varies them within the range seen
across real embossers and slates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ..encoding import BLANK, char_to_mask, dot_to_slot


@dataclass(frozen=True)
class Geometry:
    dot_pitch_mm: float = 2.5      # dot 1 -> dot 2 and dot 1 -> dot 4
    cell_pitch_mm: float = 6.2     # dot 1 of a cell -> dot 1 of the next cell
    line_pitch_mm: float = 10.0    # dot 1 of a line -> dot 1 of the next line
    dot_diameter_mm: float = 1.5

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "Geometry":
        return cls(
            dot_pitch_mm=rng.uniform(2.3, 2.6),
            cell_pitch_mm=rng.uniform(6.0, 7.0),
            line_pitch_mm=rng.uniform(10.0, 11.0),
            dot_diameter_mm=rng.uniform(1.3, 1.7),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageLayout:
    """Braille lines placed on a page. ``lines[i][j]`` is the cell at line i, column j."""

    lines: list[str]
    cells_per_line: int
    geometry: Geometry
    margin_mm: tuple[float, float] = (15.0, 15.0)

    @property
    def size_mm(self) -> tuple[float, float]:
        g = self.geometry
        w = 2 * self.margin_mm[0] + (self.cells_per_line - 1) * g.cell_pitch_mm + g.dot_pitch_mm
        h = 2 * self.margin_mm[1] + (max(len(self.lines), 1) - 1) * g.line_pitch_mm + 2 * g.dot_pitch_mm
        return w, h

    def slot_mm(self, line: int, col: int, dot: int) -> tuple[float, float]:
        """Ideal position of a dot slot in page millimetres."""
        g = self.geometry
        c, r = dot_to_slot(dot)
        return (
            self.margin_mm[0] + col * g.cell_pitch_mm + c * g.dot_pitch_mm,
            self.margin_mm[1] + line * g.line_pitch_mm + r * g.dot_pitch_mm,
        )

    def dots_mm(self) -> np.ndarray:
        """(N, 5) array of raised dots: x_mm, y_mm, line, col, dot."""
        rows = []
        for i, line in enumerate(self.lines):
            for j, ch in enumerate(line):
                mask = char_to_mask(ch)
                for d in range(1, 7):
                    if mask & (1 << (d - 1)):
                        rows.append((*self.slot_mm(i, j, d), i, j, d))
        return np.array(rows, dtype=float).reshape(-1, 5)


def wrap_braille(braille: str, width: int) -> list[str]:
    """Greedy word wrap at blank cells. Words longer than ``width`` are hard-split."""
    if width < 1:
        raise ValueError("width must be positive")
    lines: list[str] = []
    current = ""
    for word in braille.replace(" ", BLANK).split(BLANK):
        if not word:
            continue
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += BLANK + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paginate(
    braille_sentences: Sequence[str], start: int, cells_per_line: int, max_lines: int
) -> tuple[list[str], int, bool]:
    """Fill one page with whole sentences starting at index ``start``.

    Returns the wrapped lines, the number of sentences used and whether the
    page was truncated. Pages end at a sentence boundary so the page's print
    text is known exactly; only a first sentence too long for a page on its own
    is cut off (``truncated=True``), in which case the print text is inexact.
    """
    lines: list[str] = []
    used = 0
    for i in range(start, len(braille_sentences)):
        candidate = wrap_braille(BLANK.join(braille_sentences[start : i + 1]), cells_per_line)
        if len(candidate) > max_lines:
            break
        lines, used = candidate, i - start + 1
    if used == 0 and start < len(braille_sentences):
        return wrap_braille(braille_sentences[start], cells_per_line)[:max_lines], 1, True
    return lines, used, False
