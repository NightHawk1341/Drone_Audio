# =============================================================================
# PREPARE_DATA: Shared data preparation for SVM / MLP training
# Parse TextGrids → segment audio → extract wav2vec2 embeddings
# =============================================================================
"""
Prepares the full dataset (no train/test split) for model training scripts.

Usage:
    python prepare_data.py                   # uses config.yaml in same directory
    python prepare_data.py path/to/config.yaml
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import yaml
import soundfile as sf
import librosa
from tqdm import tqdm

from parser_simple_v1 import SimpleCommandParser


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
# STEP 1: PARSE
# =============================================================================

def parse_annotations(textgrid_dir: Path, audio_dir: Path, output_csv: Path,
                      tier_name: str = "commands",
                      skip_if_cached: bool = True) -> pd.DataFrame:
    """Parse TextGrid files and produce a dataset CSV with participant info."""
    print("\n[1/3] Parsing des annotations TextGrid...")

    if skip_if_cached and output_csv.exists():
        print(f"  -> Cache trouve: {output_csv}")
        return pd.read_csv(output_csv)

    parser = SimpleCommandParser()
    df = parser.create_dataset_from_annotations(
        textgrid_dir=textgrid_dir,
        audio_dir=audio_dir,
        output_csv=output_csv,
        tier_name=tier_name,
    )

    # Generate filename-based segment_id: {audio_stem}_{per_file_counter}
    segment_ids = []
    counters: dict[str, int] = {}
    for _, row in df.iterrows():
        stem = Path(row['audio_file']).stem
        counters[stem] = counters.get(stem, 0) + 1
        segment_ids.append(f"{stem}_{counters[stem]:04d}")
    df['segment_id'] = segment_ids

    # Re-save CSV with segment_id
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"  Total segments: {len(df)}")
    print(f"  Participants:   {df['participant_id'].nunique()}")
    return df


# =============================================================================
# STEP 2: SEGMENT AUDIO
# =============================================================================

def segment_audio(df: pd.DataFrame, audio_dir: Path, output_dir: Path,
                  skip_if_cached: bool = True):
    """Segment audio files to 16 kHz WAV clips named by segment_id."""
    print("\n[2/3] Segmentation des fichiers audio...")

    output_dir.mkdir(exist_ok=True, parents=True)

    existing = set(p.stem for p in output_dir.glob("*.wav"))
    if skip_if_cached and len(existing) >= len(df):
        print(f"  -> Cache trouve ({len(existing)} segments)")
        return

    errors = []
    # Cache loaded audio per source file to avoid reloading for each segment
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmentation"):
        segment_id = row['segment_id']
        if segment_id in existing:
            continue

        audio_filename = row['audio_file']
        try:
            if audio_filename not in audio_cache:
                audio, sr = librosa.load(audio_dir / audio_filename, sr=16000)
                audio_cache[audio_filename] = (audio, sr)
            audio, sr = audio_cache[audio_filename]

            start_sample = int(row['start'] * sr)
            end_sample = int(row['end'] * sr)
            segment = audio[start_sample:end_sample]

            if len(segment) < 160:  # < 10 ms at 16 kHz
                errors.append(f"{segment_id}: segment trop court ({len(segment)} samples)")
                continue

            sf.write(output_dir / f"{segment_id}.wav", segment, sr)
        except Exception as e:
            errors.append(f"{segment_id}: {e}")

    if errors:
        print(f"  Warning: {len(errors)} erreurs de segmentation:")
        for err in errors[:5]:
            print(f"    - {err}")
        if len(errors) > 5:
            print(f"    ... et {len(errors) - 5} autres")

    total = len(list(output_dir.glob("*.wav")))
    print(f"  Total segments sur disque: {total}")


# =============================================================================
# STEP 3: EXTRACT EMBEDDINGS
# =============================================================================

def extract_embeddings(df: pd.DataFrame, segments_dir: Path, output_file: Path,
                       skip_if_cached: bool = True):
    """Extract wav2vec2-FR-7K-large embeddings for all segments."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    print("\n[3/3] Extraction des embeddings wav2vec2...")

    if skip_if_cached and output_file.exists():
        print(f"  -> Cache trouve: {output_file}")
        data = np.load(output_file, allow_pickle=True)
        print(f"  Embeddings: {data['embeddings'].shape}")
        return

    # Load wav2vec2 model
    print("  Chargement du modele wav2vec2-FR-7K-large...")
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
    labels = []
    segment_ids = []
    participant_ids = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extraction embeddings"):
        seg_id = row['segment_id']
        audio_path = segments_dir / f"{seg_id}.wav"

        if not audio_path.exists():
            errors.append(f"{seg_id}: fichier non trouve")
            continue

        try:
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            inputs = feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            embeddings.append(embedding)
            labels.append(row['command'])
            segment_ids.append(seg_id)
            participant_ids.append(row['participant_id'])
        except Exception as e:
            errors.append(f"{seg_id}: {e}")

    if errors:
        print(f"  Warning: {len(errors)} erreurs d'extraction:")
        for err in errors[:5]:
            print(f"    - {err}")

    X = np.vstack(embeddings)
    y = np.array(labels)

    np.savez(
        output_file,
        embeddings=X,
        labels=y,
        segment_ids=np.array(segment_ids),
        participant_ids=np.array(participant_ids),
    )

    print(f"  Embeddings shape: {X.shape}")
    print(f"  Labels:           {len(y)}")
    print(f"  Participants:     {len(set(participant_ids))}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)

    paths = cfg["paths"]
    prep = cfg.get("data_preparation", {})

    textgrid_dir = Path(paths["textgrid_dir"])
    audio_dir = Path(paths["audio_dir"])
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    skip_cached = prep.get("skip_if_cached", True)
    tier_name = prep.get("tier_name", "commands")

    dataset_csv = output_dir / "dataset.csv"
    segments_dir = output_dir / "audio_segments"
    embeddings_file = output_dir / "all_embeddings.npz"

    print("=" * 70)
    print("PREPARE DATA: TextGrid -> Segments -> Embeddings")
    print("=" * 70)

    # Step 1: Parse
    df = parse_annotations(
        textgrid_dir, audio_dir, dataset_csv,
        tier_name=tier_name, skip_if_cached=skip_cached,
    )

    # Step 2: Segment audio
    segment_audio(df, audio_dir, segments_dir, skip_if_cached=skip_cached)

    # Step 3: Extract embeddings
    extract_embeddings(df, segments_dir, embeddings_file, skip_if_cached=skip_cached)

    print("\n" + "=" * 70)
    print("PREPARATION TERMINEE")
    print("=" * 70)
    print(f"Sortie: {output_dir}")
    print(f"  - dataset.csv          ({len(df)} segments)")
    print(f"  - audio_segments/      (WAV 16 kHz)")
    print(f"  - all_embeddings.npz   (wav2vec2 embeddings)")
