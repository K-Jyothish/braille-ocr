"""English text corpus for synthetic pages, split by source book.

Raw Project Gutenberg texts are cleaned, cut into sentences and assigned to
train/val/test by book (see ``configs/corpus.json``). Sentences that also occur
in an earlier split (train before val before test) are dropped, so evaluation
text is never seen during training.

A second test source, ``test_unseen``, holds recent sentences written for this
project (``texts/``), so that LLM correction cannot benefit from having
memorized public-domain books.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

SPLITS = ("train", "val", "test")
UNSEEN_SPLIT = "test_unseen"

_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')]?\s+(?=["\'(]?[A-Z])')

_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "--", "…": "...", "_": "",
}

MIN_CHARS = 20
MAX_CHARS = 300


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg header and licence footer."""
    start = _START.search(text)
    if start:
        text = text[start.end():]
    end = _END.search(text)
    if end:
        text = text[: end.start()]
    return text


def normalize(text: str) -> str:
    """Map typographic punctuation to ASCII and drop anything outside printable ASCII."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if 32 <= ord(ch) < 127 or ch == "\n")


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences, keeping those of usable length.

    Paragraphs are separated by blank lines; line breaks inside a paragraph are
    joined. Very short fragments (headings, "CHAPTER I.") and very long run-ons
    are dropped.
    """
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        for s in _SENTENCE_END.split(para):
            s = s.strip()
            letters = sum(ch.isalpha() for ch in s)
            if MIN_CHARS <= len(s) <= MAX_CHARS and letters >= 0.6 * len(s) and not s.isupper():
                sentences.append(s)
    return sentences


def load_config(path: str | Path) -> list[dict]:
    books = json.loads(Path(path).read_text())["books"]
    for b in books:
        if b["split"] not in SPLITS:
            raise ValueError(f"book {b['id']}: unknown split {b['split']!r}")
    return books


def build_splits(books: list[dict], raw_dir: str | Path) -> dict[str, list[str]]:
    """Sentences per split from raw book files ``<raw_dir>/<id>.txt``, de-duplicated across splits."""
    raw_dir = Path(raw_dir)
    per_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for b in books:
        raw = (raw_dir / f"{b['id']}.txt").read_text(encoding="utf-8", errors="replace")
        per_split[b["split"]].extend(split_sentences(normalize(strip_gutenberg(raw))))

    seen: set[str] = set()
    out = {}
    for split in SPLITS:
        unique = list(dict.fromkeys(s for s in per_split[split] if s not in seen))
        seen.update(unique)
        out[split] = unique
    return out


def save_splits(splits: dict[str, list[str]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, sentences in splits.items():
        (out_dir / f"{split}.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")


def load_split(corpus_dir: str | Path, split: str) -> list[str]:
    path = Path(corpus_dir) / f"{split}.txt"
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_unseen(paths: list[str | Path]) -> list[str]:
    """Sentences from plain-text files, one per line; ``#`` comments and missing files are skipped."""
    out = []
    for path in map(Path, paths):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = " ".join(normalize(line).split())
            if line and not line.startswith("#"):
                out.append(line)
    return list(dict.fromkeys(out))


def shared_ngrams(sentences: list[str], reference: list[str], n: int = 8) -> dict[str, str]:
    """For each sentence sharing an ``n``-word sequence with ``reference``, one such sequence.

    Used to check that unseen test text is not quoted from the training books.
    """
    def grams(s: str) -> list[str]:
        words = re.findall(r"[a-z0-9']+", s.lower())
        return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]

    ref = {g for s in reference for g in grams(s)}
    hits = {}
    for s in sentences:
        common = [g for g in grams(s) if g in ref]
        if common:
            hits[s] = common[0]
    return hits
