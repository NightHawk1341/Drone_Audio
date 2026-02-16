# =============================================================================
# PREDICT: Command prediction from raw audio using trained SVM / MLP models
# =============================================================================
"""
Predicts drone commands from raw audio files.

Default mode: detects speech segments automatically via energy-based VAD
(librosa.effects.split), extracts wav2vec2 embeddings, and classifies each
segment with the selected model(s).

Evaluation mode (--use-ground-truth): uses segment boundaries from
dataset_test.csv for fair comparison with evaluate.py.

Usage:
    python predict.py                              # VAD on test/audio/, both models
    python predict.py --audio path/to/file.wav     # VAD on a single file
    python predict.py --audio path/to/folder/      # VAD on all WAVs in folder
    python predict.py --model svm                  # SVM only
    python predict.py --model mlp                  # MLP only
    python predict.py --use-ground-truth           # use dataset_test.csv boundaries
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import json
import yaml
import librosa
import joblib
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# VAD SEGMENTATION
# =============================================================================

def detect_segments(audio_dir: Path, top_db: float = 30.0,
                    min_duration: float = 0.2, max_duration: float = 2.0):
    """Detect speech segments in all WAV files using energy-based VAD.

    Uses librosa.effects.split to find non-silent intervals, then filters
    by duration. Returns a DataFrame with the same columns as dataset_test.csv
    (audio_file, segment_id, start, end, duration) so downstream code is
    identical for both modes.

    Parameters
    ----------
    top_db : float
        Threshold (in dB below peak) to consider as silence. Lower = stricter.
    min_duration : float
        Discard segments shorter than this (seconds).
    max_duration : float
        Discard segments longer than this (seconds).
    """
    print(f"\n  VAD parameters: top_db={top_db}, "
          f"min_dur={min_duration}s, max_dur={max_duration}s")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    rows = []
    for wav_path in tqdm(wav_files, desc="  Detecting segments"):
        audio, sr = librosa.load(wav_path, sr=16000)
        intervals = librosa.effects.split(audio, top_db=top_db)

        counter = 0
        for start_sample, end_sample in intervals:
            start_sec = start_sample / sr
            end_sec = end_sample / sr
            duration = end_sec - start_sec

            if duration < min_duration or duration > max_duration:
                continue

            counter += 1
            seg_id = f"{wav_path.stem}_{counter:04d}"
            rows.append({
                'audio_file': wav_path.name,
                'segment_id': seg_id,
                'start': round(start_sec, 4),
                'end': round(end_sec, 4),
                'duration': round(duration, 4),
            })

    df = pd.DataFrame(rows)
    print(f"  Detected {len(df)} segments in {len(wav_files)} files")
    return df


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embeddings(df: pd.DataFrame, audio_dir: Path):
    """Extract wav2vec2 embeddings for segments defined in df.

    Segments are cut from the full-length recordings in *audio_dir* using
    the start/end boundaries in *df*. Mean-pools the last hidden state of
    wav2vec2-FR-7K-large — identical to prepare_data.py.

    Returns (X, valid_segment_ids) where X has shape (n_valid, 1024).
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    print("\n  Loading wav2vec2-FR-7K-large...")
    model_name = "LeBenchmark/wav2vec2-FR-7K-large"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    embeddings = []
    segment_ids = []
    errors = []
    audio_cache: dict[str, np.ndarray] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Extracting embeddings"):
        seg_id = row['segment_id']
        audio_file = row['audio_file']

        try:
            if audio_file not in audio_cache:
                audio_path = audio_dir / audio_file
                if not audio_path.exists():
                    errors.append(f"{seg_id}: file not found {audio_path}")
                    continue
                audio, _ = librosa.load(audio_path, sr=16000)
                audio_cache[audio_file] = audio

            audio = audio_cache[audio_file]

            start_sample = int(row['start'] * 16000)
            end_sample = int(row['end'] * 16000)
            segment = audio[start_sample:end_sample]

            if len(segment) < 160:          # < 10 ms
                errors.append(f"{seg_id}: too short ({len(segment)} samples)")
                continue

            inputs = feature_extractor(
                segment, sampling_rate=16000, return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            embeddings.append(emb)
            segment_ids.append(seg_id)

        except Exception as e:
            errors.append(f"{seg_id}: {e}")

    if errors:
        print(f"  Warning: {len(errors)} errors:")
        for err in errors[:5]:
            print(f"    - {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    X = np.vstack(embeddings)
    print(f"  Embeddings: {X.shape}")
    return X, segment_ids


# =============================================================================
# PREDICTION
# =============================================================================

def predict_with_model(X: np.ndarray, model_dir: Path, model_type: str):
    """Load model artifacts and return predicted labels."""
    model_file = "model_svm.pkl" if model_type == "svm" else "model_mlp.pkl"
    model = joblib.load(model_dir / model_file)
    scaler = joblib.load(model_dir / "scaler.pkl")
    le = joblib.load(model_dir / "label_encoder.pkl")

    X_scaled = scaler.transform(X)
    y_pred_enc = model.predict(X_scaled)
    y_pred = le.inverse_transform(y_pred_enc)
    return y_pred


def save_predictions(df: pd.DataFrame, segment_ids, y_pred, output_path: Path):
    """Save predictions CSV aligned with segments."""
    pred_map = dict(zip(segment_ids, y_pred))

    rows = []
    for _, row in df.iterrows():
        sid = row['segment_id']
        if sid in pred_map:
            rows.append({
                'audio_file': row['audio_file'],
                'segment_id': sid,
                'start': row['start'],
                'end': row['end'],
                'duration': row['duration'],
                'predicted_command': pred_map[sid],
            })

    pred_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved: {output_path} ({len(pred_df)} segments)")
    return pred_df


def save_commands_json(df: pd.DataFrame, segment_ids, y_pred, output_path: Path):
    """Save client-facing JSON with drone commands grouped by audio file.

    Excludes 'none' predictions (silence/noise). Timestamps rounded to
    centisecond precision. Segments ordered by start time within each file.
    """
    pred_map = dict(zip(segment_ids, y_pred))

    commands_by_file: dict[str, list] = {}
    for _, row in df.iterrows():
        sid = row['segment_id']
        if sid not in pred_map:
            continue
        cmd = pred_map[sid]
        if cmd == 'none':
            continue
        audio_file = row['audio_file']
        commands_by_file.setdefault(audio_file, []).append({
            'start': round(float(row['start']), 2),
            'end': round(float(row['end']), 2),
            'command': cmd,
        })

    for segs in commands_by_file.values():
        segs.sort(key=lambda s: s['start'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(commands_by_file, f, indent=2, ensure_ascii=False)

    n_cmds = sum(len(v) for v in commands_by_file.values())
    print(f"  Saved: {output_path} ({n_cmds} commands, {len(commands_by_file)} files)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict drone commands from raw audio",
    )
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to config.yaml")
    parser.add_argument("--model", choices=["svm", "mlp", "both"], default="both",
                        help="Which model(s) to use (default: both)")
    parser.add_argument("--audio", default=None,
                        help="Path to a WAV file or directory of WAVs "
                             "(default: output_dir/test/audio/)")
    parser.add_argument("--use-ground-truth", action="store_true",
                        help="Use segment boundaries from dataset_test.csv "
                             "instead of VAD (for evaluation)")
    parser.add_argument("--top-db", type=float, default=30.0,
                        help="VAD silence threshold in dB below peak (default: 30)")
    parser.add_argument("--min-dur", type=float, default=0.2,
                        help="Min segment duration in seconds (default: 0.2)")
    parser.add_argument("--max-dur", type=float, default=2.0,
                        help="Max segment duration in seconds (default: 2.0)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])

    # Resolve audio directory
    if args.audio:
        audio_path = Path(args.audio)
        if audio_path.is_file():
            # Single file — put it in a temp-like context
            audio_dir = audio_path.parent
            single_file = audio_path.name
        else:
            audio_dir = audio_path
            single_file = None
    else:
        audio_dir = output_dir / "test" / "audio"
        single_file = None

    if not audio_dir.exists():
        print(f"ERREUR: {audio_dir} introuvable.")
        sys.exit(1)

    pred_dir = output_dir / "predictions"

    # Discover models
    models_to_run = []
    if args.model in ("svm", "both"):
        svm_dir = output_dir / "SVM_model"
        if (svm_dir / "model_svm.pkl").exists():
            models_to_run.append(("svm", svm_dir))
        else:
            print(f"Warning: SVM model not found in {svm_dir}, skipping")
    if args.model in ("mlp", "both"):
        mlp_dir = output_dir / "MLP_model"
        if (mlp_dir / "model_mlp.pkl").exists():
            models_to_run.append(("mlp", mlp_dir))
        else:
            print(f"Warning: MLP model not found in {mlp_dir}, skipping")

    if not models_to_run:
        print("ERREUR: No trained models found. Run train_svm.py / train_mlp.py first.")
        sys.exit(1)

    print("=" * 60)
    print("PREDICT — Command prediction on raw audio")
    print("=" * 60)

    # ---- Segmentation -------------------------------------------------------
    if args.use_ground_truth:
        test_csv = output_dir / "test" / "dataset_test.csv"
        if not test_csv.exists():
            print(f"ERREUR: {test_csv} introuvable. Lancez d'abord: python prepare_data.py")
            sys.exit(1)
        print("\nMode: ground-truth boundaries (dataset_test.csv)")
        df_segments = pd.read_csv(test_csv)
    else:
        print(f"\nMode: automatic VAD segmentation")
        print(f"Audio: {audio_dir}")
        df_segments = detect_segments(
            audio_dir,
            top_db=args.top_db,
            min_duration=args.min_dur,
            max_duration=args.max_dur,
        )
        # Filter to single file if specified
        if single_file:
            df_segments = df_segments[
                df_segments['audio_file'] == single_file
            ].copy()

    print(f"\n  Segments:    {len(df_segments)}")
    print(f"  Audio files: {df_segments['audio_file'].nunique()}")

    # ---- Embedding extraction ------------------------------------------------
    print("\nExtracting embeddings from raw audio...")
    X_test, segment_ids = extract_embeddings(df_segments, audio_dir)

    # ---- Prediction ----------------------------------------------------------
    for model_type, model_dir in models_to_run:
        print(f"\nPredicting with {model_type.upper()}...")
        y_pred = predict_with_model(X_test, model_dir, model_type)

        if args.use_ground_truth:
            pred_path = pred_dir / f"predictions_{model_type}.csv"
            save_predictions(df_segments, segment_ids, y_pred, pred_path)
        else:
            json_path = pred_dir / f"commands_{model_type}.json"
            save_commands_json(df_segments, segment_ids, y_pred, json_path)

    print("\n" + "=" * 60)
    print("PREDICTION TERMINEE")
    print("=" * 60)
    print(f"Predictions: {pred_dir}")
