"""Evaluate a dot detector on paired test splits, broken down by severity.

    python scripts/evaluate_detector.py --checkpoint runs/heatmap_quick/best.pt --name heatmap_quick
    python scripts/evaluate_detector.py --classical --name classical          # tunes on val first

Writes results/phase2/<name>/<split>_{pages.csv,summary.json,summary.md}.
Columns kept separate: dot P/R/F1 (within --tol px), cell accuracy (oracle
grid; all cells and non-blank cells), and CER/WER with the oracle CER/WER of
the ground-truth Braille.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brailleocr.detection import classical  # noqa: E402
from brailleocr.detection.data import load_index, read_rgb  # noqa: E402
from brailleocr.detection.inference import ClassicalDetector, load_detector  # noqa: E402
from brailleocr.evaluation import format_table, score_page, summarize  # noqa: E402


def gt_points(label: dict) -> np.ndarray:
    return np.array([d[:2] for d in label["dots"]], float).reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path)
    src.add_argument("--classical", action="store_true")
    ap.add_argument("--classical-params", type=Path, help="JSON params; skips tuning")
    ap.add_argument("--tune-pages", type=int, default=16, help="val pages for classical tuning")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=ROOT / "data" / "synth", type=Path)
    ap.add_argument("--splits", nargs="+", default=["test", "test_unseen"])
    ap.add_argument("--pages-per-level", type=int, default=None, help="first N pages of each level (default all)")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--no-text", action="store_true", help="skip Liblouis CER/WER")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--infer-tile", type=int, default=1024)
    ap.add_argument("--out", default=ROOT / "results" / "phase2", type=Path)
    args = ap.parse_args()

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"tol_px": args.tol, "pages_per_level": args.pages_per_level}

    if args.classical:
        if args.classical_params:
            params = classical.ClassicalParams(**json.loads(args.classical_params.read_text()))
        else:
            val = []
            for rec in load_index(args.data / "val", max_pages=args.tune_pages):
                val.append((read_rgb(rec.image), gt_points(json.loads(rec.label.read_text()))))
            t0 = time.time()
            params, f1 = classical.tune(val, args.tol)
            print(f"tuned classical on {len(val)} val pages in {time.time() - t0:.0f}s: F1 {f1:.4f} {params.to_dict()}")
            meta["val_f1"] = f1
        (out / "classical_params.json").write_text(json.dumps(params.to_dict(), indent=2))
        detector = ClassicalDetector(params)
        meta.update(method="classical", params=params.to_dict())
    else:
        detector = load_detector(args.checkpoint, args.device, tile=args.infer_tile)
        meta.update(method=str(args.checkpoint), kind=detector.kind, threshold=detector.threshold)
        print(f"loaded {args.checkpoint} ({detector.kind}, threshold {detector.threshold}) on {detector.device}")

    for split in args.splits:
        records = load_index(args.data / split, pages_per_level=args.pages_per_level)
        rows, t0 = [], time.time()
        for k, rec in enumerate(records, 1):
            label = json.loads(rec.label.read_text())
            label["id"] = rec.id
            xy, conf = detector.detect(read_rgb(rec.image))
            rows.append(score_page(xy, conf, label, tol_px=args.tol, with_text=not args.no_text))
            if k % 25 == 0:
                print(f"  {split}: {k}/{len(records)} pages ({(time.time() - t0) / k:.2f}s/page)", flush=True)

        with open(out / f"{split}_pages.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary = summarize(rows, by="severity")
        (out / f"{split}_summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
        table = format_table(summary)
        (out / f"{split}_summary.md").write_text(f"# {args.name} on {split}\n\n{table}\n")
        print(f"\n{args.name} on {split} ({len(records)} pages)\n{table}\n")


if __name__ == "__main__":
    main()
