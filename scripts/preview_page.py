"""Render one synthetic page and save it with and without a ground-truth overlay.

    python scripts/preview_page.py --split val --index 3 --severity 0.5 --out results/preview

Green = raised dots, red = verso ghost dots, blue = cell slot grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.corpus import load_split  # noqa: E402
from brailleocr.synth.dataset import SplitText, cell_slots, make_page  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=ROOT / "data" / "corpus", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--severity", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=ROOT / "results" / "preview", type=Path)
    args = ap.parse_args()

    text = SplitText.from_sentences(load_split(args.corpus, args.split)[:3000])
    jpeg, label = make_page(text, args.split, args.index, args.seed, args.severity)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.index}_s{round(label['severity'] * 100):03d}"
    (args.out / f"{stem}.jpg").write_bytes(jpeg)

    img = cv2.imdecode(__import__("numpy").frombuffer(jpeg, "uint8"), cv2.IMREAD_COLOR)
    r = max(2, int(label["px_per_mm"] * 0.4))
    for slots in cell_slots(label).values():
        for x, y in slots:
            cv2.circle(img, (int(round(x)), int(round(y))), 1, (255, 128, 0), -1)
    for x, y in label["ghost_dots"]:
        cv2.circle(img, (int(round(x)), int(round(y))), r, (0, 0, 255), 1)
    for x, y, *_ in label["dots"]:
        cv2.circle(img, (int(round(x)), int(round(y))), r, (0, 200, 0), 1)
    cv2.imwrite(str(args.out / f"{stem}_gt.jpg"), img)
    print(json.dumps({k: label[k] for k in ("grade", "severity", "text", "image_size", "render_params")}, indent=1))
    print("\n".join(label["braille_lines"]))


if __name__ == "__main__":
    main()
