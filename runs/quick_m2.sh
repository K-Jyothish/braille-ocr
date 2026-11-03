#!/bin/zsh
# Quick M2 sanity runs (8 GB): small batches/crops, short schedules.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/train_detector.py --model resnet_heatmap --steps 2000 --batch 8 --crop 256 --val-every 500 --val-pages 12 --infer-tile 512 --workers 4 --name heatmap_quick
.venv/bin/python scripts/train_detector.py --model plain_unet --steps 600 --batch 8 --crop 256 --val-every 300 --val-pages 12 --infer-tile 512 --workers 4 --name plainunet_quick
