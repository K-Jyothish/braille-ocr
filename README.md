# brailleocr

Optical Braille recognition with confidence-guided LLM correction. See `../PROJECT_PLAN.md` for the research plan.

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

## Layout

```
src/brailleocr/
  encoding.py      6-dot masks <-> Unicode, grid-based dot assignment, dot distance
  translation.py   Liblouis UEB Grade 1/2 wrappers (lou_translate)
  corpus.py        Gutenberg cleaning, sentence split, book-level splits
  synth/           page layout, photorealistic renderer, dataset builder
scripts/           corpus download, dataset generation, preview
tests/             pytest suite
configs/           corpus book list
```
