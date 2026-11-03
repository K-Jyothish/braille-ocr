"""Liblouis wrappers for Unified English Braille (UEB) Grade 1 and Grade 2.

Uses the ``lou_translate`` command-line tool so the same code runs in a
virtualenv on macOS (``brew install liblouis``) and on Colab
(``apt-get install liblouis-bin``). Each call translates a whole batch of
lines in one subprocess; Liblouis handles input line by line, so line ``i`` of
the output always belongs to item ``i`` of the input.

Braille is exchanged as Unicode Braille (U+2800 block). Liblouis writes word
spaces as the blank cell U+2800.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from typing import Sequence

from .encoding import BLANK, BRAILLE_BASE, NUM_PATTERNS

TABLES = {
    1: "unicode.dis,en-ueb-g1.ctb",
    2: "unicode.dis,en-ueb-g2.ctb",
}

# Liblouis back-translates a cell it cannot interpret as "\<dots>/".
_UNKNOWN_CELL = re.compile(r"\\[1-8]+/")


class LiblouisError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _binary() -> str:
    path = shutil.which("lou_translate")
    if path is None:
        raise LiblouisError(
            "lou_translate not found. Install Liblouis: `brew install liblouis` "
            "(macOS) or `apt-get install liblouis-bin` (Colab/Ubuntu)."
        )
    return path


def _run(direction: str, grade: int, lines: Sequence[str]) -> list[str]:
    if grade not in TABLES:
        raise ValueError(f"grade must be 1 or 2, got {grade}")
    if not lines:
        return []
    for line in lines:
        if "\n" in line or "\r" in line:
            raise ValueError("input items must not contain line breaks")
    proc = subprocess.run(
        [_binary(), direction, TABLES[grade]],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise LiblouisError(proc.stderr.strip() or f"lou_translate exited {proc.returncode}")
    out = proc.stdout.split("\n")
    if out and out[-1] == "":
        out.pop()
    if len(out) != len(lines):
        raise LiblouisError(f"expected {len(lines)} output lines, got {len(out)}")
    return out


def to_braille(text: str | Sequence[str], grade: int = 2) -> str | list[str]:
    """Translate print text to Unicode Braille (UEB). Accepts one string or a list."""
    if isinstance(text, str):
        return _run("--forward", grade, [text])[0]
    return _run("--forward", grade, list(text))


def to_text(braille: str | Sequence[str], grade: int = 2) -> str | list[str]:
    """Back-translate Unicode Braille (UEB) to print text. ASCII spaces are read as blank cells."""
    if isinstance(braille, str):
        return _run("--backward", grade, [braille.replace(" ", BLANK)])[0]
    return _run("--backward", grade, [b.replace(" ", BLANK) for b in braille])


def is_six_dot(braille: str) -> bool:
    """True if every character is a 6-dot Unicode Braille cell.

    Forward translation of characters Liblouis does not know produces 8-dot
    cells (e.g. U+2873), so this doubles as a check that the input was fully
    translatable.
    """
    return all(0 <= ord(ch) - BRAILLE_BASE < NUM_PATTERNS for ch in braille)


def has_unknown_cells(text: str) -> bool:
    """True if back-translated ``text`` contains Liblouis's escape for an uninterpretable cell."""
    return _UNKNOWN_CELL.search(text) is not None
