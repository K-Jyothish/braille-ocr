import numpy as np
import pytest

from brailleocr.encoding import (
    BLANK,
    NUM_PATTERNS,
    assign_dots,
    char_to_mask,
    dot_distance,
    dot_to_slot,
    dots_to_mask,
    mask_to_char,
    mask_to_dots,
    masks_to_text,
    neighbors,
    slot_to_dot,
    text_to_masks,
)

LETTER_DOTS = {
    "a": [1], "b": [1, 2], "c": [1, 4], "d": [1, 4, 5], "e": [1, 5],
    "f": [1, 2, 4], "g": [1, 2, 4, 5], "h": [1, 2, 5], "i": [2, 4], "j": [2, 4, 5],
    "k": [1, 3], "l": [1, 2, 3], "m": [1, 3, 4], "n": [1, 3, 4, 5], "o": [1, 3, 5],
    "p": [1, 2, 3, 4], "q": [1, 2, 3, 4, 5], "r": [1, 2, 3, 5], "s": [2, 3, 4], "t": [2, 3, 4, 5],
    "u": [1, 3, 6], "v": [1, 2, 3, 6], "w": [2, 4, 5, 6], "x": [1, 3, 4, 6], "y": [1, 3, 4, 5, 6],
    "z": [1, 3, 5, 6],
}
LETTER_UNICODE = "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵"

PITCH = 10.0
ORIGIN = (137.0, 52.0)


def ideal_points(mask, origin=ORIGIN, col_step=(PITCH, 0.0), row_step=(0.0, PITCH)):
    pts = []
    for d in mask_to_dots(mask):
        c, r = dot_to_slot(d)
        pts.append(np.asarray(origin) + c * np.asarray(col_step) + r * np.asarray(row_step))
    return np.array(pts).reshape(-1, 2)


def old_bbox_extract(points):
    """The original project's extract_patterns.py logic, kept as a regression reference."""
    cell = np.asarray(points)
    x_min, y_min = cell.min(0)
    x_max, y_max = cell.max(0)
    w, h = x_max - x_min + 1, y_max - y_min + 1
    mask = 0
    for x, y in cell:
        rx, ry = (x - x_min) / w, (y - y_min) / h
        col = 0 if rx < 0.5 else 1
        row = 0 if ry < 0.3 else (1 if ry < 0.7 else 2)
        mask |= 1 << [[0, 3], [1, 4], [2, 5]][row][col]
    return mask


# ---- dot numbering and Unicode ------------------------------------------------


def test_slots_are_column_major():
    assert [slot_to_dot(c, r) for c in (0, 1) for r in (0, 1, 2)] == [1, 2, 3, 4, 5, 6]
    for d in range(1, 7):
        assert slot_to_dot(*dot_to_slot(d)) == d


def test_letters_match_unicode():
    got = "".join(mask_to_char(dots_to_mask(LETTER_DOTS[ch])) for ch in sorted(LETTER_DOTS))
    assert got == LETTER_UNICODE


def test_all_masks_round_trip():
    for m in range(NUM_PATTERNS):
        assert dots_to_mask(mask_to_dots(m)) == m
        assert char_to_mask(mask_to_char(m)) == m
    assert text_to_masks(masks_to_text(range(64))) == list(range(64))


def test_blank_cell_and_ascii_space():
    assert mask_to_char(0) == BLANK == "⠀"
    assert char_to_mask(" ") == 0


@pytest.mark.parametrize("bad", ["a", "⡀", "⣿"])
def test_rejects_non_six_dot_characters(bad):
    with pytest.raises(ValueError):
        char_to_mask(bad)


@pytest.mark.parametrize("fn,arg", [(mask_to_char, 64), (mask_to_char, -1), (dots_to_mask, [7]), (dot_to_slot, 0)])
def test_rejects_out_of_range(fn, arg):
    with pytest.raises(ValueError):
        fn(arg)


# ---- grid assignment ------------------------------------------------------------


def test_assign_all_patterns_ideal():
    for m in range(NUM_PATTERNS):
        reading = assign_dots(ideal_points(m), ORIGIN, PITCH, PITCH)
        assert reading.mask == m
        assert reading.rejected == []


MISSING_ROW_OR_COLUMN = {
    "a: dot 1 only": [1],
    "b: no right column, no bottom row": [1, 2],
    "c: top row only": [1, 4],
    "e: diagonal, no bottom row": [1, 5],
    "i: no top-left, no bottom row": [2, 4],
    "k: left column, no middle": [1, 3],
    "apostrophe: dot 3 only": [3],
    "capital sign: dot 6 only": [6],
    "dot 5 only": [5],
    "right column only": [4, 5, 6],
    "middle row only": [2, 5],
    "bottom row only": [3, 6],
    "no left column, rows 2-3": [5, 6],
    "full cell": [1, 2, 3, 4, 5, 6],
    "blank cell": [],
}


@pytest.mark.parametrize("dots", MISSING_ROW_OR_COLUMN.values(), ids=MISSING_ROW_OR_COLUMN.keys())
def test_assign_missing_rows_and_columns(dots):
    mask = dots_to_mask(dots)
    assert assign_dots(ideal_points(mask), ORIGIN, PITCH, PITCH).mask == mask


def test_regression_old_bbox_bug():
    """The original pipeline misread b, d-j; grid assignment must read every letter."""
    old_wrong = set()
    for ch, dots in LETTER_DOTS.items():
        mask = dots_to_mask(dots)
        pts = ideal_points(mask)
        assert assign_dots(pts, ORIGIN, PITCH, PITCH).mask == mask, ch
        if old_bbox_extract(pts) != mask:
            old_wrong.add(ch)
    assert old_wrong == set("bdefghij")


def test_assign_tolerates_jitter():
    rng = np.random.default_rng(0)
    for _ in range(500):
        m = int(rng.integers(64))
        pts = ideal_points(m) + rng.uniform(-0.3, 0.3, size=(bin(m).count("1"), 2)) * PITCH
        assert assign_dots(pts, ORIGIN, PITCH, PITCH, tolerance=0.4).mask == m


def test_assign_rotated_and_anisotropic_grid():
    for angle in (-15, -4, 7, 20):
        t = np.deg2rad(angle)
        col_step = 9.0 * np.array([np.cos(t), np.sin(t)])
        row_step = 11.0 * np.array([-np.sin(t), np.cos(t)])
        for m in range(NUM_PATTERNS):
            pts = ideal_points(m, col_step=col_step, row_step=row_step)
            assert assign_dots(pts, ORIGIN, col_step, row_step).mask == m


def test_default_row_step_is_perpendicular():
    t = np.deg2rad(10)
    col_step = PITCH * np.array([np.cos(t), np.sin(t)])
    pts = ideal_points(0b111111, col_step=col_step, row_step=PITCH * np.array([-np.sin(t), np.cos(t)]))
    assert assign_dots(pts, ORIGIN, col_step).mask == 0b111111


def test_assign_rejects_points_off_grid():
    pts = np.vstack([
        ideal_points(dots_to_mask([1, 5])),
        [ORIGIN[0] + 2 * PITCH, ORIGIN[1]],          # next cell's column
        [ORIGIN[0], ORIGIN[1] + 3 * PITCH],          # below the cell
        [ORIGIN[0] + 0.5 * PITCH, ORIGIN[1]],        # between columns
    ])
    reading = assign_dots(pts, ORIGIN, PITCH, PITCH)
    assert reading.mask == dots_to_mask([1, 5])
    assert reading.rejected == [2, 3, 4]


def test_assign_keeps_max_confidence_per_slot():
    pts = ideal_points(dots_to_mask([1, 6]))
    pts = np.vstack([pts, pts[0] + 0.5])  # duplicate detection of dot 1
    reading = assign_dots(pts, ORIGIN, PITCH, PITCH, confidences=[0.4, 0.7, 0.9])
    assert reading.mask == dots_to_mask([1, 6])
    np.testing.assert_allclose(reading.slot_confidence, [0.9, 0, 0, 0, 0, 0.7])


def test_assign_argument_validation():
    with pytest.raises(ValueError):
        assign_dots([[0, 0]], ORIGIN, PITCH, PITCH, tolerance=0.5)
    with pytest.raises(ValueError):
        assign_dots([[0, 0]], ORIGIN, (1, 0), (2, 0))
    with pytest.raises(ValueError):
        assign_dots([[0, 0]], ORIGIN, PITCH, PITCH, confidences=[1, 1])


# ---- dot distance ---------------------------------------------------------------


def test_dot_distance():
    b, k = dots_to_mask([1, 2]), dots_to_mask([1, 3])
    assert dot_distance(b, k) == 2
    assert dot_distance(b, b) == 0
    assert dot_distance(0, 63) == 6


def test_neighbors():
    one = neighbors(dots_to_mask([1, 2]), 1)
    assert len(one) == len(set(one)) == 6
    assert all(dot_distance(n, dots_to_mask([1, 2])) == 1 for n in one)
    two = neighbors(0, 2)
    assert len(set(two)) == 6 + 15 and 0 not in two
