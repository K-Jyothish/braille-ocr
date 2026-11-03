import json
import shutil
from collections import defaultdict

import cv2
import numpy as np
import pytest

from brailleocr.encoding import BLANK, assign_dots, char_to_mask
from brailleocr.synth.dataset import SplitText, cell_slots, generate_split, label_masks, make_page
from brailleocr.synth.layout import Geometry, PageLayout, paginate, wrap_braille

needs_louis = pytest.mark.skipif(shutil.which("lou_translate") is None, reason="Liblouis not installed")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Bright vixens jump; dozy fowl quack.",
    "She had never seen anything like it in all her life.",
    "\"Where are you going?\" asked the old man at the gate.",
    "It was the best of times, it was the worst of times.",
    "They walked for 12 miles before they found the river.",
    "Behind the house there was a small garden full of roses.",
    "Nobody knew why the bells had stopped ringing that night.",
]


@pytest.fixture(scope="module")
def split_text():
    return SplitText.from_sentences(SENTENCES)


# ---- layout ---------------------------------------------------------------------


def test_wrap_respects_width_and_keeps_words():
    braille = BLANK.join(["⠁" * n for n in (3, 5, 2, 7, 4, 1, 6)])
    lines = wrap_braille(braille, 10)
    assert all(len(l) <= 10 for l in lines)
    assert BLANK.join(lines).split(BLANK) == braille.split(BLANK)


def test_wrap_hard_splits_long_word():
    assert wrap_braille("⠁" * 25, 10) == ["⠁" * 10, "⠁" * 10, "⠁" * 5]


def test_paginate_fits_whole_sentences():
    sents = ["⠁" * 8, "⠃" * 8, "⠉" * 8, "⠙" * 8]
    lines, used, truncated = paginate(sents, 0, cells_per_line=17, max_lines=1)
    assert lines == [sents[0] + BLANK + sents[1]] and used == 2 and not truncated
    lines, used, truncated = paginate(["⠁" * 40], 0, cells_per_line=10, max_lines=2)
    assert len(lines) == 2 and used == 1 and truncated


def test_layout_dots_match_masks():
    layout = PageLayout(["⠃⠀⠿", "⠤"], 10, Geometry())
    dots = layout.dots_mm()
    assert len(dots) == 2 + 6 + 2
    assert sorted(dots[dots[:, 2] == 1][:, 4].astype(int)) == [3, 6]


# ---- rendering and labels --------------------------------------------------------


def _decode_with_slots(label):
    """Re-read every cell from the ground-truth dots using the ideal slot grid."""
    slots = cell_slots(label)
    by_cell = defaultdict(list)
    for x, y, line, col, *_ in label["dots"]:
        by_cell[(int(line), int(col))].append((x, y))
    out = {}
    for key, s in slots.items():
        pts = np.array(by_cell.get(key, []), dtype=float).reshape(-1, 2)
        out[key] = assign_dots(pts, s[0], s[3] - s[0], (s[2] - s[0]) / 2).mask
    return out


@needs_louis
@pytest.mark.parametrize("severity", [0.0, 0.5, 1.0])
def test_ground_truth_is_consistent(split_text, severity):
    for index in range(3):
        _, label = make_page(split_text, "train", index, seed=1, severity=severity)
        assert _decode_with_slots(label) == label_masks(label)
        w, h = label["image_size"]
        xy = np.array([d[:2] for d in label["dots"]])
        assert (xy[:, 0] >= 0).all() and (xy[:, 0] < w).all() and (xy[:, 1] >= 0).all() and (xy[:, 1] < h).all()
        assert sum(bin(char_to_mask(c)).count("1") for l in label["braille_lines"] for c in l) == len(label["dots"])


@needs_louis
def test_page_text_matches_braille(split_text):
    for index in range(4):
        _, label = make_page(split_text, "val", index, seed=0, severity=0.0)
        assert not label["text_truncated"]
        sentences = label["text"]
        grade = label["grade"]
        expected = BLANK.join(split_text.braille[grade][i] for i, s in enumerate(split_text.text) if s in sentences)
        assert BLANK.join(label["braille_lines"]).split(BLANK) == expected.split(BLANK)


@needs_louis
def test_deterministic(split_text):
    a = make_page(split_text, "test", 2, seed=7, severity=0.6)
    b = make_page(split_text, "test", 2, seed=7, severity=0.6)
    assert a[0] == b[0] and a[1] == b[1]


@needs_louis
def test_paired_severities_share_content(split_text):
    _, clean = make_page(split_text, "test", 0, seed=0, severity=0.0)
    jpeg_hard, hard = make_page(split_text, "test", 0, seed=0, severity=1.0)
    for key in ("text", "grade", "braille_lines", "geometry", "cells_per_line"):
        assert clean[key] == hard[key]
    assert clean["image_size"] != hard["image_size"] or clean["dots"] != hard["dots"]


@needs_louis
def test_paired_severities_share_nuisance_factors(split_text):
    """Common random numbers: levels differ in strength only, and effects are monotone in severity."""
    for index in range(6):
        params = [make_page(split_text, "test", index, seed=0, severity=s)[1]["render_params"] for s in (0.0, 0.5, 1.0)]
        assert len({p["light_azimuth_deg"] for p in params}) == 1
        elevations = [p["light_elevation_deg"] for p in params]
        assert elevations == sorted(elevations)
        for flag in ("cast_shadow", "motion_blur"):
            present = [bool(p.get(flag)) for p in params]
            assert present == sorted(present)
        ghosts = [p["ghost_dots"] > 0 for p in params]
        assert ghosts == sorted(ghosts)
        assert [p["jpeg_quality"] for p in params] == sorted((p["jpeg_quality"] for p in params), reverse=True)


@needs_louis
def test_dots_are_visible_on_clean_page(split_text):
    jpeg, label = make_page(split_text, "train", 0, seed=3, severity=0.0)
    gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1))
    r = max(2, int(label["px_per_mm"] * 0.8))

    def energy(points):
        return np.mean([grad[int(y) - r:int(y) + r, int(x) - r:int(x) + r].mean() for x, y in points])

    raised = [(d[0], d[1]) for d in label["dots"]]
    masks = label_masks(label)
    empty = [tuple(s[k]) for key, s in cell_slots(label).items() for k in range(6) if not masks[key] & (1 << k)]
    assert energy(raised) > 3 * energy(empty)


@needs_louis
def test_generate_split_writes_files(tmp_path):
    root = generate_split(SENTENCES, "test", tmp_path, n_pages=2, seed=0, levels=(0.0, 1.0))
    rows = [json.loads(l) for l in (root / "index.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["test_00000_s000", "test_00000_s100", "test_00001_s000", "test_00001_s100"]
    for r in rows:
        assert (root / r["image"]).stat().st_size > 0
        label = json.loads((root / r["label"]).read_text())
        assert label["id"] == r["id"] and label["severity"] == r["severity"]
