"""Download the Project Gutenberg books in configs/corpus.json and build sentence splits.

    python scripts/fetch_corpus.py            # -> data/corpus/{train,val,test}.txt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.corpus import build_splits, load_config, save_splits  # noqa: E402

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
    save_splits(splits, args.out)
    for split, sentences in splits.items():
        print(f"{split:5s} {len(sentences):6d} sentences")


if __name__ == "__main__":
    main()
