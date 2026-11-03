"""Download the Project Gutenberg books in configs/corpus.json and build sentence splits.

    python scripts/fetch_corpus.py            # -> data/corpus/{train,val,test,test_unseen}.txt

test_unseen comes from the local files listed under "unseen_sources" in the
config, minus any sentence that also occurs in the Gutenberg splits.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.corpus import UNSEEN_SPLIT, build_splits, load_config, load_unseen, save_splits, shared_ngrams  # noqa: E402

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "corpus.json", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "corpus", type=Path)
    args = ap.parse_args()

    books = load_config(args.config)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for b in books:
        path = raw_dir / f"{b['id']}.txt"
        if path.exists():
            continue
        print(f"downloading {b['id']} {b['title']}")
        req = urllib.request.Request(URL.format(id=b["id"]), headers={"User-Agent": "brailleocr-research"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())

    splits = build_splits(books, raw_dir)
    gutenberg = [s for sentences in splits.values() for s in sentences]
    sources = json.loads(args.config.read_text()).get("unseen_sources", [])
    unseen = load_unseen([ROOT / p for p in sources])
    seen = set(gutenberg)
    splits[UNSEEN_SPLIT] = [s for s in unseen if s not in seen]
    overlap = shared_ngrams(splits[UNSEEN_SPLIT], gutenberg, n=8)
    for sentence, gram in overlap.items():
        print(f"warning: unseen sentence shares 8-gram {gram!r} with Gutenberg: {sentence}")
    save_splits(splits, args.out)
    for split, sentences in splits.items():
        print(f"{split:11s} {len(sentences):6d} sentences")


if __name__ == "__main__":
    main()
