# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoiceStick: a speech-to-drone-command classification pipeline. Audio recordings of French speakers issuing drone piloting commands are parsed from Praat TextGrid annotations, converted into wav2vec2 embeddings, and classified using SVM or MLP models.

The 9 command classes are: `forward`, `backward`, `left`, `right`, `up`, `down`, `yawleft`, `yawright`, `none`.

## Running the Pipeline

```bash
# Step 1: Prepare data (parse → split → segment → embed + copy test audio)
python prepare_data.py

# Step 2: Train models with 5-fold speaker-independent cross-validation
python train_svm.py
python train_mlp.py

# Step 3: Predict commands on raw audio (VAD-based segmentation)
python predict.py                          # VAD on test/audio/, both models
python predict.py --model svm              # SVM only
python predict.py --audio path/to/file.wav # any raw audio file
python predict.py --use-ground-truth       # use dataset_test.csv boundaries (for eval)

# Step 4: Evaluate predictions against ground truth
python evaluate.py
```

All scripts read paths and settings from `config.yaml` (in the same directory by default). A custom config path can be passed as CLI argument: `python prepare_data.py path/to/config.yaml`.

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
  test_size: 0.15        # Fraction of participants held out for testing
  random_seed: 42        # Reproducible train/test split

training:
  balance_classes: true   # Subsample "none" class
  none_ratio: 1.0         # Max ratio of "none" vs largest other class
  n_folds: 5              # Number of cross-validation folds
```

## Architecture

### Active files

- **`prepare_data.py`** — Shared data preparation (used by both SVM and MLP training). Contains `SimpleCommandParser` class (TextGrid parsing with UTF-8/UTF-16 BOM auto-detection) and a 5-step pipeline:
  1. Parse TextGrid annotations → `dataset.csv` (columns: `audio_file, participant_id, attempt, start, end, duration, command, segment_id`)
  2. Split participants 85/15 into train/test sets (speaker-independent, deterministic via `random_seed`)
  3. Segment train audio to 16 kHz WAV clips in `train/audio_segments/`
  4. Extract embeddings for train via `LeBenchmark/wav2vec2-FR-7K-large` (mean-pooled) → `train/all_embeddings.npz`
  5. Copy raw (unsegmented) audio files for test participants → `test/audio/`

- **`train_svm.py`** — SVM training with speaker-independent 5-fold `GroupKFold` cross-validation (grouped by `participant_id`). Reads data from `output_dir/train/`. Per-fold: class balancing, `StandardScaler`, `SVC(kernel='rbf', C=10, class_weight='balanced')`. Reports per-fold and average F1 metrics, trains final model on all data. Saves all artifacts to `output_dir/SVM_model/`: `model_svm.pkl`, `scaler.pkl`, `label_encoder.pkl`, `cv_results_svm.json`, `confusion_matrix_svm.png`, `svm_output.txt`.

- **`train_mlp.py`** — MLP training with speaker-independent `GroupKFold` cross-validation. Reads data from `output_dir/train/`. Saves artifacts to `output_dir/MLP_model/`.

- **`predict.py`** — Predicts commands from raw audio. Two modes: **default** uses energy-based VAD (`librosa.effects.split`) to auto-detect speech segments; **`--use-ground-truth`** uses boundaries from `dataset_test.csv` (for fair evaluation). Extracts wav2vec2 embeddings and classifies with SVM and/or MLP. Accepts `--audio path` for any WAV file/directory. VAD tunable via `--top-db`, `--min-dur`, `--max-dur`. Saves to `output_dir/predictions/predictions_{svm,mlp}.csv`.

- **`evaluate.py`** — Evaluates predictions against ground truth. Per-model: accuracy, F1-macro, F1-weighted, per-class report, confusion matrix. Side-by-side comparison table when both models available. Saves to `output_dir/evaluation/`.

- **`config.yaml`** — Centralized configuration for paths, data preparation, and training settings.

### Output structure

```
output_dir/
├── dataset.csv              (full dataset — all participants, kept for reference)
├── train/
│   ├── dataset_train.csv    (train-only rows)
│   ├── audio_segments/      (segmented 16 kHz WAV clips)
│   └── all_embeddings.npz   (wav2vec2 embeddings, train only)
├── test/
│   ├── dataset_test.csv     (ground-truth labels for test participants)
│   └── audio/               (raw unsegmented WAV files, copied from data/audio/)
├── predictions/
│   ├── predictions_svm.csv  (predicted commands from predict.py)
│   └── predictions_mlp.csv
├── evaluation/
│   ├── eval_svm.json        (per-model metrics)
│   ├── eval_mlp.json
│   ├── comparison.json      (side-by-side summary)
│   ├── confusion_matrix_eval_svm.png
│   ├── confusion_matrix_eval_mlp.png
│   └── evaluation_output.txt
├── SVM_model/               (SVM artifacts from train_svm.py)
└── MLP_model/               (MLP artifacts from train_mlp.py)
```

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
- The 85/15 train/test split is by participant (not by segment), ensuring complete speaker independence. Training scripts only see train participants; test audio is kept raw for `predict.py`.

## Model Performance (test set, 9 test participants)

| Metric | SVM | MLP |
|---|---|---|
| Accuracy | 0.836 | **0.866** |
| F1-macro | 0.735 | **0.774** |
| F1-weighted | 0.843 | **0.868** |

Weak classes (both models): `left`, `yawleft`, `yawright`, `backward` — low support (28-76 test samples).

SVM v2 (PCA 1024→256 + GridSearchCV) was tested and performed worse (F1-macro 0.747 vs 0.759). GridSearchCV confirmed C=10/gamma='scale' was already optimal. PCA hurt RBF kernel distances. Experiment reverted.

## Long-term Roadmap

1. ~~`train_mlp.py`~~ — Done
2. ~~`predict.py`~~ — Done
3. ~~`evaluate.py`~~ — Done
4. Technical and user documentation (architecture, user manual, `requirements.txt`)
5. Detailed `README.md` for GitHub
