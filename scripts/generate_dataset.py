"""Generate the synthetic train/val/test page dataset.

    python scripts/fetch_corpus.py                       # once
    python scripts/generate_dataset.py --workers 8       # -> data/synth/{train,val,test,test_unseen}

A split given 0 pages is skipped and left as it is.

Test and test_unseen pages are rendered at every severity level in TEST_LEVELS (paired).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.corpus import load_split  # noqa: E402
from brailleocr.synth.dataset import TEST_LEVELS, generate_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=ROOT / "data" / "corpus", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "synth", type=Path)
    ap.add_argument("--train", type=int, default=2000, help="train pages")
    ap.add_argument("--val", type=int, default=200, help="val pages")
    ap.add_argument("--test", type=int, default=100, help="test pages per severity level")
    ap.add_argument("--test-unseen", type=int, default=40, help="test_unseen pages per severity level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plan = [
        ("train", args.train, None),
        ("val", args.val, None),
        ("test", args.test, TEST_LEVELS),
        ("test_unseen", args.test_unseen, TEST_LEVELS),
    ]
    for split, n, levels in plan:
        if n == 0:
            continue
        t0 = time.time()
        root = generate_split(load_split(args.corpus, split), split, args.out, n, args.seed, levels, args.workers)
        count = sum(1 for _ in open(root / "index.jsonl"))
        print(f"{split:11s} {count:6d} pages  {time.time() - t0:6.1f}s  -> {root}")

    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(seed=args.seed, test_levels=TEST_LEVELS)
    meta.setdefault("pages", {}).update({split: n for split, n, _ in plan if n})
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
