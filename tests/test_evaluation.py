import shutil

import numpy as np
import pytest

from brailleocr.encoding import char_to_mask
from brailleocr.evaluation import (
    char_errors,
    levenshtein,
    match_dots,
    oracle_cell_masks,
    score_page,
    summarize,
    word_errors,
)
from brailleocr.synth.dataset import SplitText, label_masks, make_page

needs_louis = pytest.mark.skipif(shutil.which("lou_translate") is None, reason="Liblouis not installed")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She had never seen anything like it in all her life.",
    "It was the best of times, it was the worst of times.",
    "Behind the house there was a small garden full of roses.",
]


@pytest.fixture(scope="module")
def page():
    _, label = make_page(SplitText.from_sentences(SENTENCES), "test", 0, seed=0, severity=0.5)
    return label


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b"], ["a", "b"]) == 0
    assert char_errors("abcd", "abd") == (1, 4)
    assert word_errors("the cat sat", "the hat sat down") == (2, 3)


def test_match_dots_one_to_one_within_tolerance():
    gt = np.array([[0, 0], [10, 0], [20, 0]])
    pred = np.array([[1, 0], [1.5, 0], [10, 2.5], [30, 0]])
    tp, fp, fn, pairs = match_dots(pred, gt, tol_px=3)
    assert (tp, fp, fn) == (2, 2, 1)
    assert sorted(map(tuple, pairs)) == [(0, 0), (2, 1)]
    assert match_dots(np.zeros((0, 2)), gt, 3)[:3] == (0, 0, 3)


def _gt_points(label):
    return np.array([d[:2] for d in label["dots"]])


@needs_louis
def test_oracle_cells_from_perfect_dots(page):
    assert oracle_cell_masks(_gt_points(page), None, page) == label_masks(page)


@needs_louis
def test_perfect_prediction_scores(page):
    row = score_page(_gt_points(page), None, page)
    assert row["dot_fp"] == row["dot_fn"] == 0
    assert row["cells_correct"] == row["cells"]
    assert row["char_edits"] == row["oracle_char_edits"]


@needs_louis
def test_missing_dot_costs_one_cell(page):
    pts = _gt_points(page)[1:]  # drop one dot
    row = score_page(pts, None, page)
    assert (row["dot_tp"], row["dot_fn"]) == (len(pts), 1)
    assert row["cells"] - row["cells_correct"] == 1
    assert row["cells_nonblank"] - row["cells_nonblank_correct"] == 1


@needs_louis
def test_ghost_false_positives_hurt_precision_not_recall():
    text = SplitText.from_sentences(SENTENCES)
    pages = (make_page(text, "test", i, seed=0, severity=1.0)[1] for i in range(20))
    page = next((p for p in pages if p["ghost_dots"]), None)
    assert page is not None, "ghosts appear with probability 0.9 at severity 1"
    ghosts = np.array(page["ghost_dots"]).reshape(-1, 2)
    row = score_page(np.vstack([_gt_points(page), ghosts]), None, page, with_text=False)
    assert row["dot_fn"] == 0
    # A ghost can land within tolerance of a raised dot only if the dot was already matched.
    assert row["dot_fp"] == len(ghosts)


def test_summarize_micro_average():
    rows = [
        {"severity": 0.0, "dot_tp": 9, "dot_fp": 1, "dot_fn": 1, "cells": 10, "cells_correct": 9, "cells_nonblank": 8, "cells_nonblank_correct": 7,
         "char_edits": 1, "chars": 10, "word_edits": 1, "words": 2, "oracle_char_edits": 0, "oracle_word_edits": 0},
        {"severity": 1.0, "dot_tp": 5, "dot_fp": 5, "dot_fn": 5, "cells": 10, "cells_correct": 5, "cells_nonblank": 8, "cells_nonblank_correct": 4,
         "char_edits": 5, "chars": 10, "word_edits": 2, "words": 2, "oracle_char_edits": 1, "oracle_word_edits": 1},
    ]
    s = {r["severity"]: r for r in summarize(rows)}
    assert s[0.0]["dot_f1"] == pytest.approx(0.9)
    assert s["all"]["cell_acc"] == pytest.approx(0.7)
    assert s["all"]["cer"] == pytest.approx(0.3)
    assert s[1.0]["oracle_wer"] == pytest.approx(0.5)
    assert list(s)[-1] == "all"
