# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceStick: a speech-to-drone-command classification pipeline. Audio recordings of French speakers issuing drone piloting commands are parsed from Praat TextGrid annotations, converted into wav2vec2 embeddings, and classified using SVM or MLP models.

The 9 command classes are: `forward`, `backward`, `left`, `right`, `up`, `down`, `yawleft`, `yawright`, `none`.

## Running the Pipeline

```bash
# Parse only (TextGrid annotations → CSV dataset)
python parser_simple_final.py

# Full pipeline (parse → segment → embed → train → evaluate)
python pipeline_corrected.py
```

Both scripts require editing hardcoded paths in the `__main__` block before running. Paths point to directories containing `.TextGrid` and `.wav` files.

`pipeline_corrected.py` is the improved version of `pipeline.py` with caching, class balancing, embedding/label alignment via `segment_id`, and StandardScaler normalization.

## Architecture

### Two-file core

- **`parser_simple_final.py`** — `SimpleCommandParser` class that manually parses Praat TextGrid files (handles both UTF-8 and UTF-16 BOM encoding). Extracts time-aligned command intervals from a tier named `"commands"` and produces a CSV with columns: `audio_file, start, end, duration, command`.

- **`pipeline_corrected.py`** — 6-step pipeline orchestrated by `pipeline_annotation_vers_modele()`:
  1. Parse TextGrid annotations (via `SimpleCommandParser`)
  2. Train/test split (85/15, stratified)
  3. Class balancing (subsample `none` class, controlled by `none_ratio`)
  4. Audio segmentation (resample to 16kHz via librosa)
  5. Embedding extraction using `LeBenchmark/wav2vec2-FR-7K-large` (French wav2vec2, mean-pooled)
  6. Train SVM (`kernel='rbf', C=10, class_weight='balanced'`) or MLP (`512-256, early_stopping`) and evaluate (F1-macro, confusion matrix)

`PipelineConfig` centralizes all output paths. The pipeline supports `skip_if_cached=True` to reuse intermediate artifacts.

### Legacy files

- `pipeline.py` / `parser_annotation_simple.py` — earlier versions without caching, alignment fixes, or class balancing. Kept for reference but superseded by the corrected versions.

### Test directories

`test_1/`, `test_2/`, `test_3/` contain copies of scripts run against different data subsets, with outputs in `output/` subdirectories. These are experiment snapshots, not automated tests.

## Key Dependencies

- `transformers` (Hugging Face) — wav2vec2 model loading
- `torch` — inference
- `librosa`, `soundfile` — audio I/O and resampling
- `scikit-learn` — SVM, MLP, evaluation metrics, StandardScaler
- `pandas`, `numpy` — data handling
- `tqdm` — progress bars
- `matplotlib`, `seaborn` — confusion matrix visualization

## Important Details

- Audio files are recorded at **48kHz** but resampled to **16kHz** for wav2vec2 processing.
- TextGrid files may be UTF-16 encoded with BOM (common from Praat on Windows). The parser in `parser_simple_final.py` auto-detects encoding.
- The `"commands"` tier name is lowercase (was previously `"Commands"` — older parser versions are case-sensitive).
- `data/` contains the raw TextGrid annotations in `data/textgrid/` and `data/desorder/train_A/`.
