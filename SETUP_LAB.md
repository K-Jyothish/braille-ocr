# Lab machine setup (NVIDIA GPU)

Step-by-step setup for continuing this project on a lab PC with an NVIDIA GPU.

- **Linux:** Ubuntu 22.04 or 24.04.
- **Windows:** Windows 10 21H2+ or 11. Everything runs inside **WSL2 (Ubuntu)**, because Liblouis (`lou_translate`) is packaged for Ubuntu and the code calls it as a command-line tool.

After step 1, Windows (WSL2) and Linux use the same commands. Run every command after step 1 in the Ubuntu terminal (on Windows, the WSL terminal, not PowerShell).

---

## 1. GPU driver

### Linux
```bash
nvidia-smi            # should list the GPU and a "CUDA Version"
```
If the command is missing, install the recommended driver and reboot:
```bash
sudo ubuntu-drivers install
sudo reboot
```

### Windows + WSL2
1. Install the latest **NVIDIA Windows driver** for the GPU (GeForce/Studio or RTX/Quadro) from nvidia.com. It includes WSL2 GPU support. **Do not install any NVIDIA/CUDA driver inside WSL.**
2. In **PowerShell as Administrator**:
   ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --update
   ```
   Reboot if asked. Open "Ubuntu 22.04" from the Start menu and create a Linux user.
3. Optional but recommended: give WSL enough memory for data generation and training. Create `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=16GB
   processors=8
   ```
   Then `wsl --shutdown` in PowerShell and reopen Ubuntu.
4. Inside Ubuntu: `nvidia-smi` should list the GPU.

> Keep the project in the Linux filesystem (e.g. `~/brailleocr`), **not** under `/mnt/c/...`. File access across the Windows boundary is many times slower, and the dataset is ~3,000 images.

---

## 2. System packages: Python 3.11, Liblouis, git

```bash
sudo apt update
sudo apt install -y software-properties-common git build-essential tmux
sudo add-apt-repository -y ppa:deadsnakes/ppa      # provides Python 3.11 on 22.04 and 24.04
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y liblouis-bin liblouis-data     # lou_translate + UEB tables
```

Check:
```bash
python3.11 --version
lou_translate --version
echo "hello world" | lou_translate --forward unicode.dis,en-ueb-g2.ctb     # expect ⠓⠑⠇⠇⠕⠀⠸⠺
```

> The M2 used Liblouis 3.37; Ubuntu 22.04 ships 3.20. Translation of the plain English in this project should be the same, but if `tests/test_translation.py` fails, build a current release from source (https://github.com/liblouis/liblouis/releases): `./configure && make && sudo make install && sudo ldconfig`.

---

## 3. Get the code

```bash
cd ~
git clone git@github.com:<user>/<repo>.git brailleocr     # or https://github.com/<user>/<repo>.git with a personal access token
cd brailleocr
```

---

## 4. Python environment with CUDA PyTorch

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[train,dev]'
```

On Linux x86-64 (WSL2 included), the PyPI `torch` wheel is already a CUDA build. Check that it sees the GPU:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It must print `True` and the GPU name. If it prints `False`, the driver is usually older than the CUDA version the wheel was built for. Install a wheel for an older CUDA from the selector at https://pytorch.org/get-started/locally/, for example:
```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Run the test suite (about 10 s):
```bash
python -m pytest -q
```

The code picks the device automatically: **CUDA, then Apple MPS, then CPU** (`pick_device` in `src/brailleocr/detection/inference.py`). Every script also accepts `--device cuda:0` (or `cuda:1`, ...) to choose a GPU. On CUDA, training uses fp16 mixed precision by default (`--no-amp` turns it off).

---

## 5. Regenerate the dataset from the seed

`data/` is not in git. It is rebuilt deterministically from the books (downloaded from Project Gutenberg), the unseen test sentences in `texts/` and seed 0:

```bash
python scripts/fetch_corpus.py                       # ~1 min, needs internet
python scripts/generate_dataset.py --workers 8       # ~4-6 min with 8 cores; ~390 MB
cat data/synth/meta.json
```

Expected:
- `fetch_corpus.py` prints train 36797, val 1295, test 14052, test_unseen 152 sentences.
- The generator writes train 2000, val 200, test 500 (100 x 5 severities) and test_unseen 200 (40 x 5) pages.

Labels (text, Braille, dot positions) depend only on the seed. Image pixels can differ very slightly from the M2's because of JPEG encoder differences between OpenCV builds. To check by eye, compare against the committed samples:
```bash
python scripts/export_samples.py --out results/samples_lab
# compare results/samples_lab/ with results/samples/
```

---

## 6. Training

Use `tmux` so training survives an SSH disconnect (`tmux new -s train`; detach with `Ctrl-b d`; reattach with `tmux attach -t train`).

Full runs: same budget for both networks. Validation picks the checkpoint and the detection threshold on val pages only.
```bash
COMMON="--steps 20000 --batch 16 --crop 384 --val-every 1000 --val-pages 24 --workers 8 --resume"
python scripts/train_detector.py --model resnet_heatmap --name heatmap_r34 $COMMON
python scripts/train_detector.py --model plain_unet     --name plainunet   $COMMON
```
- Checkpoints go to `runs/<name>/best.pt` (best val F1) and `runs/<name>/last.pt` (for `--resume`).
- If the GPU runs out of memory, lower `--batch` (e.g. 8). With more than 12 GB, `--batch 32` is fine.
- Optional ablation (encoder without ImageNet pretraining):
  `python scripts/train_detector.py --model resnet_heatmap --no-pretrained --name heatmap_r34_scratch $COMMON`

To continue the M2 quick runs instead of starting fresh, copy `runs/<name>/` from the repo and use the same `--name` with `--resume`. Those checkpoints hold weights only, so the optimizer restarts.

---

## 7. Evaluation and comparison

```bash
python scripts/evaluate_detector.py --classical                             --name classical
python scripts/evaluate_detector.py --checkpoint runs/plainunet/best.pt     --name plainunet
python scripts/evaluate_detector.py --checkpoint runs/heatmap_r34/best.pt   --name heatmap_r34
python scripts/compare_detectors.py classical plainunet heatmap_r34 --split test
python scripts/compare_detectors.py classical plainunet heatmap_r34 --split test_unseen
```
Results go to `results/phase2/<name>/` (per-page CSV and per-severity summary JSON/Markdown) and `results/phase2/comparison_<split>.md`. Commit the summaries (not the checkpoints, unless under 100 MB).

Rough timing on a recent GPU (e.g. RTX 3090/4090): heatmap model 30-60 min, plain U-Net 1-2 h, evaluation ~15 min per model.

---

## 8. Secrets

Phase 5 needs `GEMINI_API_KEY`. Put it in `.env` (gitignored; template in `.env.example`) or export it in the shell. Never commit keys.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `lou_translate not found` | `sudo apt install liblouis-bin liblouis-data` |
| `torch.cuda.is_available()` is False in WSL | update the **Windows** NVIDIA driver; `wsl --update`; never install a Linux NVIDIA driver inside WSL |
| `CUDA out of memory` | lower `--batch`; for evaluation, lower `--infer-tile` (e.g. 512) |
| DataLoader worker errors / shared memory in WSL | lower `--workers`, or raise `memory=` in `.wslconfig` |
| very slow data loading on Windows | the repo is under `/mnt/c`; move it to `~/` |
