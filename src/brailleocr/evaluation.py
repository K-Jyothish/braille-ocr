"""Evaluation metrics, reported as separate quantities at three levels.

1. Dots: precision / recall / F1, matching each predicted dot to at most one
   ground-truth dot within ``tol_px`` pixels.
2. Cells: accuracy of the 6-dot pattern of every cell. Until Phase 3 adds
   grid estimation, predicted dots are grouped with the *oracle* grid (the
   ground-truth page-to-image homography), so this isolates detection errors
   from grouping errors. Reported over all cells and over non-blank cells.
3. Text: CER and WER of the Liblouis back-translation against the page's
   print text, alongside the oracle CER/WER of back-translating the
   ground-truth Braille, the floor set by Grade 2 ambiguity.

Cell accuracy and CER/WER are deliberately kept apart: one cell error can
cost several characters (a contraction) or none (a blank between words).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from . import translation
from .encoding import BLANK, assign_dots, char_to_mask, mask_to_char


# ---- edit distance --------------------------------------------------------------


def levenshtein(a: Iterable, b: Iterable) -> int:
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) in characters."""
    return levenshtein(ref, hyp), len(ref)


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance, reference length) in whitespace-separated words."""
    r = ref.split()
    return levenshtein(r, hyp.split()), len(r)


# ---- dots -----------------------------------------------------------------------


def match_dots(pred: np.ndarray, gt: np.ndarray, tol_px: float) -> tuple[int, int, int, np.ndarray]:
    """Greedy one-to-one matching by increasing distance within ``tol_px``.

    Returns (tp, fp, fn, pairs) where ``pairs`` is (tp, 2) indices into pred, gt.
    With tolerance below half the dot pitch, greedy matching equals optimal matching.
    """
    pred = np.asarray(pred, float).reshape(-1, 2)
    gt = np.asarray(gt, float).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt), np.zeros((0, 2), int)
    dist = cKDTree(pred).sparse_distance_matrix(cKDTree(gt), tol_px, output_type="ndarray")
    order = np.argsort(dist["v"], kind="stable")
    used_p, used_g, pairs = set(), set(), []
    for k in order:
        i, j = int(dist["i"][k]), int(dist["j"][k])
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            pairs.append((i, j))
    tp = len(pairs)
    return tp, len(pred) - tp, len(gt) - tp, np.array(pairs, int).reshape(-1, 2)


# ---- cells (oracle grid) ---------------------------------------------------------


def oracle_cell_masks(
    points: np.ndarray, confidences: np.ndarray | None, label: dict
) -> dict[tuple[int, int], int]:
    """Group predicted dots into the label's cells using the true page geometry.

    Points are mapped back to page millimetres with the inverse of the label's
    page-to-image homography, assigned to the (line, col) whose cell box
    contains them, and snapped to the 2x3 slot grid. Returns a mask for every
    ground-truth cell (0 where no dot was found).
    """
    g = label["geometry"]
    dp, cp, lp = g["dot_pitch_mm"], g["cell_pitch_mm"], g["line_pitch_mm"]
    mx, my = label["margin_mm"]
    lines = label["braille_lines"]
    out = {(i, j): 0 for i, line in enumerate(lines) for j in range(len(line))}
    pts = np.asarray(points, float).reshape(-1, 2)
    if len(pts) == 0:
        return out
    conf = np.ones(len(pts)) if confidences is None else np.asarray(confidences, float)

    Hinv = np.linalg.inv(np.asarray(label["page_to_image"]))
    p = np.column_stack([pts, np.ones(len(pts))]) @ Hinv.T
    mm = p[:, :2] / p[:, 2:3]
    xr, yr = mm[:, 0] - mx, mm[:, 1] - my
    # Each cell's box extends half the inter-cell / inter-line gap beyond its slots.
    col = np.floor((xr + (cp - dp) / 2) / cp).astype(int)
    line = np.floor((yr + (lp - 2 * dp) / 2) / lp).astype(int)

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, key in enumerate(zip(line.tolist(), col.tolist())):
        if key in out:
            groups[key].append(k)
    for (i, j), idx in groups.items():
        origin = (j * cp, i * lp)
        local = np.column_stack([xr[idx], yr[idx]])
        out[(i, j)] = assign_dots(local, origin, dp, dp, confidences=conf[idx]).mask
    return out


def masks_to_lines(masks: dict[tuple[int, int], int], lines: list[str]) -> list[str]:
    """Rebuild Braille lines with the ground-truth line lengths from per-cell masks."""
    return ["".join(mask_to_char(masks[(i, j)]) for j in range(len(line))) for i, line in enumerate(lines)]


# ---- page and summary --------------------------------------------------------------


def score_page(
    points: np.ndarray,
    confidences: np.ndarray | None,
    label: dict,
    tol_px: float = 3.0,
    with_text: bool = True,
) -> dict:
    """All metrics for one page, as raw counts so pages can be summed."""
    gt_xy = np.array([d[:2] for d in label["dots"]], float).reshape(-1, 2)
    tp, fp, fn, _ = match_dots(points, gt_xy, tol_px)

    pred = oracle_cell_masks(points, confidences, label)
    truth = {(i, j): char_to_mask(ch) for i, line in enumerate(label["braille_lines"]) for j, ch in enumerate(line)}
    nonblank = [k for k, m in truth.items() if m]
    row = {
        "id": label.get("id"),
        "severity": label["severity"],
        "grade": label["grade"],
        "dot_tp": tp,
        "dot_fp": fp,
        "dot_fn": fn,
        "cells": len(truth),
        "cells_correct": sum(pred[k] == truth[k] for k in truth),
        "cells_nonblank": len(nonblank),
        "cells_nonblank_correct": sum(pred[k] == truth[k] for k in nonblank),
    }
    if with_text:
        grade = label["grade"]
        hyp_braille = BLANK.join(masks_to_lines(pred, label["braille_lines"]))
        ref_braille = BLANK.join(label["braille_lines"])
        hyp_text, oracle_text = translation.to_text([hyp_braille, ref_braille], grade=grade)
        text = label["text"]
        row["char_edits"], row["chars"] = char_errors(text, hyp_text)
        row["word_edits"], row["words"] = word_errors(text, hyp_text)
        row["oracle_char_edits"], _ = char_errors(text, oracle_text)
        row["oracle_word_edits"], _ = word_errors(text, oracle_text)
    return row


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def summarize(rows: list[dict], by: str | None = "severity") -> list[dict]:
    """Micro-averaged metrics per value of ``by`` (and an ``all`` row)."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[by] if by else "all"].append(r)
    if by:
        groups["all"] = list(rows)

    out = []
    for key, rs in groups.items():
        s = defaultdict(float)
        for r in rs:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    s[k] += v
        p = _ratio(s["dot_tp"], s["dot_tp"] + s["dot_fp"])
        rc = _ratio(s["dot_tp"], s["dot_tp"] + s["dot_fn"])
        summary = {
            by or "group": key,
            "pages": len(rs),
            "dot_precision": p,
            "dot_recall": rc,
            "dot_f1": _ratio(2 * p * rc, p + rc),
            "cell_acc": _ratio(s["cells_correct"], s["cells"]),
            "cell_acc_nonblank": _ratio(s["cells_nonblank_correct"], s["cells_nonblank"]),
        }
        if "chars" in s:
            summary.update(
                cer=_ratio(s["char_edits"], s["chars"]),
                wer=_ratio(s["word_edits"], s["words"]),
                oracle_cer=_ratio(s["oracle_char_edits"], s["chars"]),
                oracle_wer=_ratio(s["oracle_word_edits"], s["words"]),
            )
        out.append(summary)
    return sorted(out, key=lambda d: (d[by or "group"] == "all", str(d[by or "group"])))


def format_table(summary: list[dict]) -> str:
    """Markdown table of a :func:`summarize` result."""
    cols = list(summary[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in summary:
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
