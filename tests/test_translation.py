import shutil

import pytest

from brailleocr import translation as T

pytestmark = pytest.mark.skipif(shutil.which("lou_translate") is None, reason="Liblouis not installed")


def test_grade1_letters_and_space():
    assert T.to_braille("hello world", grade=1) == "⠓⠑⠇⠇⠕⠀⠺⠕⠗⠇⠙"


def test_grade1_capital_and_number():
    assert T.to_braille("James", grade=1) == "⠠⠚⠁⠍⠑⠎"
    assert T.to_braille("42", grade=1) == "⠼⠙⠃"


def test_grade2_uses_contractions():
    # UEB: "the" = ⠮, "and" = ⠯, "children" = ⠡⠝ (wordsign)
    assert T.to_braille("the children and", grade=2) == "⠮⠀⠡⠝⠀⠯"
    assert len(T.to_braille("the children and", grade=2)) < len(T.to_braille("the children and", grade=1))


@pytest.mark.parametrize("grade", [1, 2])
def test_round_trip(grade):
    sentences = [
        "Hello world.",
        "The children ran into the garden and played with the dog.",
        "She asked, \"Where is it?\" and he said it was 42 miles away.",
    ]
    assert T.to_text(T.to_braille(sentences, grade=grade), grade=grade) == sentences


def test_batch_keeps_order_and_empty_items():
    assert T.to_braille(["a", "", "b"], grade=1) == ["⠁", "", "⠃"]
    # In Grade 2 a lone "b" would read as the wordsign "but", so it needs the grade-1 indicator.
    assert T.to_braille(["a", "", "b"], grade=2) == ["⠁", "", "⠰⠃"]


def test_to_text_accepts_ascii_spaces():
    assert T.to_text("⠓⠊ ⠞⠓⠑⠗⠑", grade=1) == "hi there"


def test_six_dot_check():
    assert T.is_six_dot("⠓⠑⠀⠇")
    assert not T.is_six_dot(T.to_braille("x\u0001y", grade=1))  # untranslatable -> 8-dot escape
    assert not T.is_six_dot("abc")


def test_unknown_cell_escape():
    assert T.has_unknown_cells(T.to_text("⠿⠿⠿", grade=1))
    assert not T.has_unknown_cells("hello")


def test_rejects_line_breaks_and_bad_grade():
    with pytest.raises(ValueError):
        T.to_braille("a\nb", grade=1)
    with pytest.raises(ValueError):
        T.to_braille("a", grade=3)
