"""Side-by-side comparison of evaluated detectors, per severity level.

    python scripts/compare_detectors.py classical plainunet_quick heatmap_quick --split test

Prints one markdown table per metric (rows = severity, columns = method) and
writes them to results/phase2/comparison_<split>.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["dot_f1", "dot_precision", "dot_recall", "cell_acc", "cell_acc_nonblank", "cer", "wer", "oracle_cer"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--results", default=ROOT / "results" / "phase2", type=Path)
    args = ap.parse_args()

    data = {}
    for n in args.names:
        path = args.results / n / f"{args.split}_summary.json"
        data[n] = {str(r["severity"]): r for r in json.loads(path.read_text())["summary"]}
    levels = list(next(iter(data.values())))

    parts = [f"# Detector comparison on {args.split}\n"]
    for m in METRICS:
        if not all(m in rows[levels[0]] for rows in data.values()):
            continue
        lines = [f"## {m}\n", "| severity | " + " | ".join(args.names) + " |", "|---|" + "---|" * len(args.names)]
        for lv in levels:
            lines.append(f"| {lv} | " + " | ".join(f"{data[n][lv][m]:.4f}" for n in args.names) + " |")
        parts.append("\n".join(lines) + "\n")
    text = "\n".join(parts)
    (args.results / f"comparison_{args.split}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
