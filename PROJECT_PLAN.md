# Braille-to-Text — Rebuild Plan for Journal Publication

Author: Jyothish K (M.Tech CSE, TKMCE, Reg TKM25CSCE08)
Original work: Mini project report "Braille-to-Text using U-Net and AI" (April 2026, guide Dr. Shyna A.)

## Context
The original pipeline (Image → U-Net dot segmentation → cell grouping → 6-dot pattern → Unicode → Liblouis → LLM correction → Flask UI) worked, but is not publishable as-is:
- Evaluated only on simple synthetic data, and on **training images**: `evaluate_unet.py` scored the first 5 files of `dataset/pages`, the same folder used for training (no train/test split). The GT masks were also made by thresholding those same images. Dice/IoU = 1.00 is therefore leakage, not performance. The data itself is trivial too: 500 pages built from ~13 repeated phrases.
- No baselines (Angelina Braille Reader, YOLO-style detectors, classical methods).
- End-to-end accuracy (60–70%) was estimated, not measured — need CER/WER.
- Systematic raw-output errors (b→*, f→m, j→+) despite near-perfect detection ⇒ deterministic bug (confirmed by re-running the old code, Nov 2026):
  - **Per-cell bounding-box row/column assignment** in `extract_patterns.py`. Each dot's row/column was computed relative to the bounding box of *that cell's own detected dots*, not a fixed 2×3 grid. A cell with no bottom-row dot gets its middle-row dots stretched into row 3, and a single-column cell gets its column misjudged. With ideal dot positions, 8 of 26 letters always decode wrong: b d e f g h i j → k % * m x u / + (b=⠃ becomes ⠅, e=⠑ becomes ⠡, …).
  - The bit order itself was correct (column-major: dots 1-2-3 = bits 0-2, dots 4-5-6 = bits 3-5). Row-major indexing was suspected but is *not* what the code did.
  - Fix: assign dots to slots of a 2×3 grid whose origin and pitch come from page-level spacing estimates, never from the cell's own dots. Unit tests cover missing-row and missing-column cells.
  - Also: the Grade 2 table `en-us-g2.ctb` was used on uncontracted (Grade 1) data; the rebuild uses UEB tables.
  - Separately, the cell-classifier dataset (`create_cell_dataset.py`) cropped at a 70 px step while the generator placed glyphs 76 px apart, so crop labels drift (explains e.g. precision 0.52 for "b").
- Free LLM correction can hallucinate (e.g. "Map room in castle annex") — no hallucination measurement.
- Report hygiene: leftover `:contentReference[oaicite...]` text, duplicate sections 3.2/3.3, loss 24.47 odd for mean BCE, verify all references (e.g. "IEEE Access 9:123456–123465" looks placeholder).

## Core contribution (paper hook)
**Confidence-guided, Braille-aware LLM correction with a hallucination guard.**
1. Detector outputs per-dot confidence.
2. Uncertain cells → set of plausible alternative characters (by dot flips).
3. LLM chooses only among candidates using context.
4. Guard rejects corrections too far (in dot-edit distance) from what's physically detected.
Claim: higher accuracy + measurably lower hallucination than free LLM correction.

Working title: *Confidence-Guided LLM Correction for Robust Optical Braille Recognition*

## Improved system
| Stage | Old | New |
|---|---|---|
| Data | Simple synthetic | Realistic generator: paper texture, shadows, blur, perspective, double-sided ghost dots, controlled severity |
| Detection | Plain U-Net, binary mask, thr 0.35 | U-Net w/ pretrained ResNet encoder, dot-center heatmap; peaks → positions + confidence |
| Grouping | Fixed grid | Deskew + row/column clustering using Braille spacing ratios; robust to missing dots |
| Pattern→Unicode | Per-cell bbox row/col assignment (bug) | Fixed-grid slot assignment + unit tests |
| Translation | en-us-g2 | Liblouis UEB Grade 1 & 2 |
| Correction | Free LLM | Constrained LLM + guard; spell-checker baseline |
| Evaluation | Dice/IoU on training images | Held-out test split (by source text); Dot F1, cell acc, CER, WER, hallucination rate, robustness curves |

## Experiments
1. Dot detection: classical vs plain U-Net vs improved U-Net (synthetic + real).
2. Real data: DSBI and Angelina datasets (dot/cell level; may not be English). Optional: 30–50 self-collected English Braille phone photos with transcriptions.
3. Correction: none / spell-checker / free LLM / constrained LLM (ours) / constrained with open-source LLM.
4. Robustness: CER vs noise/blur severity per method (main figure).
5. Ablations: heatmap vs mask, deskew on/off, confidences on/off, Grade 1 vs 2.

Evaluation notes from Phase 1:
- **Grade 2 has an oracle error floor.** Liblouis back-translation of *perfect* UEB Grade 2 Braille reproduces the source exactly for only ~96.5% of sentences (e.g. `toast,)` → `toastea)`); Grade 1 is exact. Report an "oracle" CER/WER (perfect cells → Liblouis) next to every Grade 2 result, so translation ambiguity isn't blamed on the detector.
- **Test set is paired across severities with common random numbers** (same text, layout, paper, lighting and shadow placement; only degradation strength changes), so the robustness curves (experiment 4) isolate severity.
- **The held-out split is by book with disjoint authors** (train: Austen, Shelley, Dickens, Melville, Twain, Doyle, Fitzgerald; val: Carroll; test: Stoker, Wilde).
- **Second test source `test_unseen`** (152 present-day sentences written for this project on 2026-11-03, none sharing an 8-word sequence with the Gutenberg corpus). It rules out verbatim LLM memorization in experiment 3. Caveat: they were drafted with an AI assistant, so their style may be easy for an LLM to predict. Before final experiments, add human-written sentences (`new/texts/unseen_human.txt`) and report results on both sources.
- **Cell accuracy is reported separately from CER/WER**; one cell error can cost several characters under Grade 2. Until Phase 3, cell accuracy uses the oracle (ground-truth) grid, so it measures detection only.
- Hardware: the M2 has 8 GB of unified memory, so local runs use batch 8, 256 px crops and 512 px inference tiles. Full runs (batch 16, 384 px crops) go on Colab.

## Phases (~6–8 weeks)
1. Synthetic generator + Braille encoding + unit tests (2–3 d) — **done 2026-11-03** (see `README.md`)
2. U-Net detector + training (2–3 d) — **in progress**: code, baselines and quick M2 runs done; full runs pending on the lab GPU
3. Grouping, Unicode, Liblouis, CER/WER eval (2–3 d)
4. Real datasets + baselines (1 wk)
5. Correction methods incl. constrained LLM (1 wk)
6. Full experiments, figures, tables (1 wk)
7. Paper writing + venue selection (1–2 wk)

## Decisions (confirmed 2026-11-03)
1. Training: develop and debug on the MacBook M2 (PyTorch MPS, 8 GB). Full runs move to a **lab PC with an NVIDIA GPU** (Linux, or Windows with WSL2; see `SETUP_LAB.md`). The Colab notebook stays as a fallback.
2. LLM: Gemini, key read from the `GEMINI_API_KEY` environment variable (never hard-coded), plus one small open-source model (TBD in Phase 5).
3. Real English Braille photos (~30–50): **pending, still to be confirmed.**
4. Target: conference paper first, then an extended journal version.

## Folder layout
```
braille-project/
  old_files/        # everything recovered from the original project (read-only, not in git)
  new/              # rebuilt project = the git repository
    PROJECT_PLAN.md # this file
    SETUP_LAB.md    # lab GPU machine setup
    src/brailleocr/ # encoding, translation, corpus, synth/, detection/, evaluation
    scripts/        # corpus, dataset, samples, train, evaluate, compare
    notebooks/      # Colab fallback for full runs
    texts/          # unseen test sentences
    tests/
    data/           # generated, gitignored (rebuild from seed 0)
    runs/           # checkpoints, gitignored except the M2 quick-run best.pt files
    results/        # samples, evaluation summaries
```

## Current status (2026-11-03, end of session)

**Repository:** `new/` is a git repo on branch `main`. No remote yet. Add the private GitHub remote and push: `git remote add origin git@github.com:<user>/<repo>.git && git push -u origin main`.

**Done**
- Phase 1: encoding with fixed-grid dot assignment, Liblouis UEB 1/2 wrapper, book-level corpus, photorealistic generator, paired test set (common random numbers). 81 tests pass.
- Data (seed 0): train 2000, val 200, test 100x5 severities, test_unseen 40x5 severities. Rebuild with `scripts/fetch_corpus.py` and `scripts/generate_dataset.py`.
- Phase 2 code: `resnet_heatmap` (ResNet-34 U-Net, Gaussian heatmap, focal loss, NMS peaks + confidence); `plain_unet` (the original architecture retrained on native-resolution crops, disk masks, BCE); classical detector (local contrast, Otsu, contours). Every method's threshold/parameters are selected on val only.
- Evaluation (`src/brailleocr/evaluation.py`): per severity, dot P/R/F1 (3 px tolerance), cell accuracy on the oracle grid (all cells and non-blank cells) and CER/WER with the oracle CER/WER, all as separate columns.
- Trainer: `--resume` (optimizer, scheduler, AMP scaler), CUDA mixed precision by default, device order CUDA → MPS → CPU.

**Results so far (quick runs; not paper numbers)**

| method | data | val dot F1 | test (20 pages/level) |
|---|---|---|---|
| classical (tuned on 16 val pages) | – | – | dot F1 0.932 / 0.867 / 0.696 at severity 0 / 0.5 / 1; cell acc 0.922 / 0.822 / 0.540; CER 0.139 / 0.286 / 0.616 |
| resnet_heatmap, M2 quick | 2000 steps, batch 8, 256 px crops | 0.959 (thr 0.25), still rising | not evaluated yet |
| plain_unet, M2 quick | 300 of 600 steps (stopped for the move) | 0.912 (thr 0.98) | not evaluated yet |

Checkpoints: `runs/heatmap_quick/best.pt`, `runs/plainunet_quick/best.pt` (in git; weights only).

**Next steps**
1. Set up the lab PC with `SETUP_LAB.md`; check `pytest` passes and `torch.cuda.is_available()` is True. The CUDA mixed-precision path has not run yet (it couldn't be tested on the M2); if it misbehaves, use `--no-amp`.
2. Regenerate the data from seed 0; check sentence/page counts and compare `export_samples.py` output with `results/samples/`.
3. Full training, same budget for both networks: `--steps 20000 --batch 16 --crop 384`. Optional: heatmap model `--no-pretrained` (encoder ablation).
4. Evaluate classical, plain_unet and heatmap_r34 on all test and test_unseen pages; run `compare_detectors.py`; commit `results/phase2/*` summaries.
5. Check where the heatmap model fails (severity 1: low resolution, flat light, ghosts) and decide whether the generator or the model needs work before Phase 3.
6. Phase 3: estimate the cell grid from detected dots (deskew + row/column clustering using spacing ratios) to replace the oracle grid; report grid-estimation errors separately.
7. Open items: the photo-collection decision (still pending); a human-written `texts/unseen_human.txt`; Liblouis version on the lab PC (3.20 on Ubuntu 22.04 vs 3.37 on the M2), so re-check the Grade 2 oracle CER there.
