# brailleocr

Optical Braille recognition with confidence-guided LLM correction. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the research plan and current status, and [SETUP_LAB.md](SETUP_LAB.md) to set up a machine with an NVIDIA GPU.

## Setup (MacBook M2)

```bash
brew install liblouis                # provides lou_translate and the UEB tables
python3.11 -m venv .venv
.venv/bin/pip install -e '.[train,dev]'
cp .env.example .env                 # then set GEMINI_API_KEY (never commit it)
.venv/bin/python -m pytest
```

On Colab: `apt-get install -y liblouis-bin` and `pip install -e '.[train]'`.

## Data

```bash
.venv/bin/python scripts/fetch_corpus.py           # Gutenberg books -> data/corpus/{train,val,test}.txt
.venv/bin/python scripts/generate_dataset.py       # -> data/synth/{train,val,test}
.venv/bin/python scripts/preview_page.py --split val --index 0 --severity 0.5   # visual check
```

- **Corpus split** is by book, with no author shared between splits (`configs/corpus.json`); sentences repeated across splits are removed.
- **Pages**: 2000 train and 200 val pages with random severity; 100 test pages, each rendered at severities 0, 0.25, 0.5, 0.75 and 1.0. Test levels share text, layout and nuisance factors (common random numbers), so only degradation strength differs.
- **test_unseen**: 40 pages x 5 levels from sentences written for this project (`texts/`), never published, to rule out LLM memorization in the correction experiments. Add human-written sentences as `texts/unseen_human.txt` and re-run `fetch_corpus.py`.
- `scripts/export_samples.py` copies one page at every severity, with ground-truth overlays, to `results/samples/`.
- Generation is deterministic given `--seed`; `data/` is gitignored and can be rebuilt.

Each page has `images/<id>.jpg` and `labels/<id>.json`:

| field | meaning |
|---|---|
| `text`, `grade` | print text of the page and UEB grade (1 or 2) |
| `braille_lines` | Unicode Braille per line; U+2800 is a blank cell |
| `dots` | raised dots as `[x, y, line, col, dot, amplitude]` in image px (`dot_fields`) |
| `ghost_dots` | verso dots showing through, `[x, y]`; negatives for the detector |
| `page_to_image`, `geometry`, `margin_mm` | recover any cell's ideal slot grid with `synth.dataset.cell_slots` |
| `severity`, `render_params` | degradation level and the sampled parameters |

## Phase 2: dot detection

```bash
# quick sanity runs on the M2 (8 GB): see runs/quick_m2.sh
.venv/bin/python scripts/train_detector.py --model resnet_heatmap --steps 2000 --batch 8 --crop 256 --infer-tile 512 --name heatmap_quick
.venv/bin/python scripts/evaluate_detector.py --checkpoint runs/heatmap_quick/best.pt --name heatmap_quick --pages-per-level 20 --infer-tile 512
.venv/bin/python scripts/evaluate_detector.py --classical --name classical --pages-per-level 20
.venv/bin/python scripts/compare_detectors.py classical heatmap_quick --split test
```

Full runs: `notebooks/phase2_detection_colab.ipynb`.

| method | model | target / loss | points |
|---|---|---|---|
| `resnet_heatmap` | U-Net, ImageNet ResNet-34 encoder, full-res output | Gaussian heatmap, focal loss | NMS peaks + sub-pixel refinement; confidence = peak value |
| `plain_unet` | original project's U-Net, retrained on native-resolution crops | disk masks, BCE (pos_weight 5) | component centroids; confidence = max prob |
| classical | – | – | local contrast, Otsu, closing, contour area/circularity |

Every method's threshold/parameters are chosen on val; the evaluation reports, per severity level, separately:
dot precision/recall/F1 (within `--tol` px, default 3), cell accuracy with the oracle grid (all cells and non-blank cells),
and CER/WER after Liblouis next to the oracle CER/WER of perfect Braille.

## Layout

```
src/brailleocr/
  encoding.py      6-dot masks <-> Unicode, grid-based dot assignment, dot distance
  translation.py   Liblouis UEB Grade 1/2 wrappers (lou_translate)
  corpus.py        Gutenberg cleaning, sentence split, book-level splits
  synth/           page layout, photorealistic renderer, dataset builder
  detection/       models, targets, losses, post-processing, tiled inference, classical baseline
  evaluation.py    dot P/R/F1, oracle-grid cell accuracy, CER/WER, per-severity summaries
scripts/           corpus, dataset, samples, training, evaluation, comparison
notebooks/         Colab notebooks for full runs
texts/             unseen test sentences
tests/             pytest suite
configs/           corpus book list
```
