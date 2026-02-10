# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceStick: a speech-to-drone-command classification pipeline. Audio recordings of French speakers issuing drone piloting commands are parsed from Praat TextGrid annotations, converted into wav2vec2 embeddings, and classified using SVM or MLP models.

The 9 command classes are: `forward`, `backward`, `left`, `right`, `up`, `down`, `yawleft`, `yawright`, `none`.

## Running the Pipeline

```bash
# Step 1: Prepare data (parse TextGrids → segment audio → extract embeddings)
python prepare_data.py

# Step 2: Train SVM with 5-fold speaker-independent cross-validation
python train_svm.py
```

Both scripts read paths and settings from `config.yaml` (in the same directory by default). A custom config path can be passed as CLI argument: `python prepare_data.py path/to/config.yaml`.

## Configuration

All settings are in `config.yaml`:

```yaml
paths:
  textgrid_dir: "..."   # Directory containing .TextGrid files
  audio_dir: "..."       # Directory containing .wav files
  output_dir: "..."      # Output directory for all artifacts

data_preparation:
  tier_name: "commands"  # TextGrid tier name
  skip_if_cached: true   # Reuse existing intermediate files

training:
  balance_classes: true   # Subsample "none" class
  none_ratio: 1.5         # Max ratio of "none" vs largest other class
  n_folds: 5              # Number of cross-validation folds
```

## Architecture

### Active files

- **`prepare_data.py`** — Shared data preparation (used by both SVM and future MLP training). Contains `SimpleCommandParser` class (TextGrid parsing with UTF-8/UTF-16 BOM auto-detection) and a 3-step pipeline:
  1. Parse TextGrid annotations → `dataset.csv` (columns: `audio_file, participant_id, attempt, start, end, duration, command, segment_id`)
  2. Segment audio to 16 kHz WAV clips in `audio_segments/` (filename-based segment IDs preserve original recording names)
  3. Extract embeddings via `LeBenchmark/wav2vec2-FR-7K-large` (mean-pooled) → `all_embeddings.npz`

- **`train_svm.py`** — SVM training with speaker-independent 5-fold `GroupKFold` cross-validation (grouped by `participant_id`). Per-fold: class balancing, `StandardScaler`, `SVC(kernel='rbf', C=10, class_weight='balanced')`. Reports per-fold and average F1 metrics, trains final model on all data. Saves all artifacts to `output_dir/SVM_model/`: `model_svm.pkl`, `scaler.pkl`, `label_encoder.pkl`, `cv_results_svm.json`, `confusion_matrix_svm.png`, `svm_output.txt`.

- **`config.yaml`** — Centralized configuration for paths, data preparation, and training settings.

### Data format

Audio filenames follow `DD_MM_YY_HH_MM_SS_NNN` where:
- `DD_MM_YY_HH_MM_SS` = participant session identifier (unique per participant)
- `NNN` = attempt number (`000`–`005`, 6 attempts per participant, one has only 5)

Segment IDs are `{audio_stem}_{counter:04d}` (e.g., `01_04_25_10_20_56_000_0012`), preserving the link to the original recording for cross-validation grouping.

### Legacy files

- **`pipeline_v1.py`** — Original proof-of-concept pipeline (parse → split → balance → segment → embed → train/evaluate). Kept for reference. Imports `SimpleCommandParser` from `prepare_data.py`.

### Test directories

`test_0/`, `test_1/`, `test_2/`, `test_3/` contain outputs from earlier experiment runs. These are snapshots, not automated tests.

## Key Dependencies

- `transformers` (Hugging Face) — wav2vec2 model loading
- `torch` — inference
- `librosa`, `soundfile` — audio I/O and resampling
- `scikit-learn` — SVM, MLP, GroupKFold, evaluation metrics, StandardScaler
- `pandas`, `numpy` — data handling
- `pyyaml` — configuration
- `tqdm` — progress bars
- `joblib` — model serialization
- `matplotlib`, `seaborn` — confusion matrix visualization

## Important Details

- Audio files are recorded at **48 kHz** but resampled to **16 kHz** for wav2vec2 processing.
- TextGrid files may be UTF-16 encoded with BOM (common from Praat on Windows). The parser auto-detects encoding.
- The `"commands"` tier name is lowercase and matched case-insensitively.
- `data/textgrid/` contains TextGrid annotations, `data/audio/` contains WAV files.
- Cross-validation uses `GroupKFold` grouped by participant to prevent speaker leakage between train and test folds.

## Long-term Roadmap

1. `train_mlp.py` — MLP training script (teammate's responsibility), reusing `prepare_data.py` output
2. `predict.py` — WAV file → command list with timestamps (select SVM or MLP)
3. `evaluate.py` — Comprehensive evaluation: accuracy, F1-macro, F1 per class, confusion matrices, SVM vs MLP comparison
4. Technical and user documentation (architecture, user manual, `requirements.txt`)
5. Detailed `README.md` for GitHub
