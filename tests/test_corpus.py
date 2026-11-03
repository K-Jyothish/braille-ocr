import json
from pathlib import Path

import pytest

from brailleocr.corpus import SPLITS, build_splits, load_config, load_split, normalize, split_sentences, strip_gutenberg

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"


def _book(body: str) -> str:
    return (
        "Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + body
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicence junk"
    )


def test_strip_gutenberg():
    out = strip_gutenberg(_book("Body text."))
    assert "Body text." in out and "junk" not in out


def test_normalize_punctuation():
    assert normalize("“It’s _fine_” — café") == "\"It's fine\" -- cafe"


def test_split_sentences_filters_headings_and_joins_lines():
    text = "CHAPTER I.\n\nThe cat sat on the\nmat all day long. Then it left the room quietly!\n\nOk."
    assert split_sentences(text) == ["The cat sat on the mat all day long.", "Then it left the room quietly!"]


def test_build_splits_removes_cross_split_duplicates(tmp_path):
    shared = "This sentence appears in two different books."
    (tmp_path / "1.txt").write_text(_book(f"{shared} Only the training book has this one."))
    (tmp_path / "2.txt").write_text(_book(f"{shared} Only the test book has this one here."))
    books = [{"id": 1, "split": "train"}, {"id": 2, "split": "test"}]
    splits = build_splits(books, tmp_path)
    assert shared in splits["train"]
    assert splits["test"] == ["Only the test book has this one here."]


def test_config_splits_share_no_author():
    books = load_config(ROOT / "configs" / "corpus.json")
    authors = {s: {b["author"] for b in books if b["split"] == s} for s in SPLITS}
    assert all(authors[s] for s in SPLITS)
    assert not (authors["train"] & authors["val"] or authors["train"] & authors["test"] or authors["val"] & authors["test"])


@pytest.mark.skipif(not (CORPUS_DIR / "test.txt").exists(), reason="run scripts/fetch_corpus.py first")
def test_built_corpus_is_disjoint():
    sets = {s: set(load_split(CORPUS_DIR, s)) for s in SPLITS}
    assert not (sets["train"] & sets["test"])
    assert not (sets["train"] & sets["val"])
    assert not (sets["val"] & sets["test"])
