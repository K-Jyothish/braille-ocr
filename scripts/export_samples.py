"""Copy one generated test page at every severity level to results/samples/ for inspection.

    python scripts/export_samples.py --split test --index 4

Writes, per level, the image as generated (<id>.jpg) and with the ground truth
drawn on it (<id>_gt.jpg), plus a side-by-side strip of the same central crop
at every level (<split>_<index>_levels.jpg) and the page text (<split>_<index>.txt).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.synth.visualize import draw_ground_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT / "data" / "synth", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=4)
    ap.add_argument("--out", default=ROOT / "results" / "samples", type=Path)
    args = ap.parse_args()

    split_dir = args.data / args.split
    rows = [json.loads(l) for l in (split_dir / "index.jsonl").read_text().splitlines()]
    rows = sorted((r for r in rows if r["page_index"] == args.index), key=lambda r: r["severity"])
    if not rows:
        sys.exit(f"no page {args.index} in {split_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    tiles = []
    for r in rows:
        label = json.loads((split_dir / r["label"]).read_text())
        shutil.copy(split_dir / r["image"], args.out / f"{r['id']}.jpg")
        img = cv2.imread(str(split_dir / r["image"]))
        cv2.imwrite(str(args.out / f"{r['id']}_gt.jpg"), draw_ground_truth(img, label))

        # ~50 mm square around the centre of the text block
        x, y = np.array([d[:2] for d in label["dots"]]).mean(0).astype(int)
        k = int(label["px_per_mm"] * 25)
        tile = cv2.resize(img[max(0, y - k):y + k, max(0, x - k):x + k], (400, 400))
        cv2.putText(tile, f"severity {r['severity']:.2f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        tiles.append(tile)

    stem = f"{args.split}_{args.index:05d}"
    cv2.imwrite(str(args.out / f"{stem}_levels.jpg"), cv2.hconcat(tiles))
    (args.out / f"{stem}.txt").write_text(
        f"grade {label['grade']}\n\n{label['text']}\n\n" + "\n".join(label["braille_lines"]) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} levels of {stem} to {args.out}")


if __name__ == "__main__":
    main()
