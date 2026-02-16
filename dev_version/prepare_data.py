# =============================================================================
# PREPARE_DATA: Shared data preparation for SVM / MLP training
# Parse TextGrids → segment audio → extract wav2vec2 embeddings
# =============================================================================
"""
Prepares the dataset for model training and evaluation.

Steps:
    1. Parse TextGrid annotations → full dataset.csv
    2. Split participants 85/15 into train/test sets (speaker-independent)
    3. Segment audio for train participants → train/audio_segments/
    4. Extract wav2vec2 embeddings for train → train/all_embeddings.npz
    5. Copy raw audio for test participants → test/audio/

Usage:
    python prepare_data.py                   # uses config.yaml in same directory
    python prepare_data.py path/to/config.yaml
"""

from pathlib import Path
import shutil
import sys
import warnings
from typing import List, Dict
import pandas as pd
import numpy as np
import yaml
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =============================================================================
# TEXTGRID PARSER
# =============================================================================

class SimpleCommandParser:
    """
    Parser pour TextGrid avec annotations directes de commandes.
    Gere l'encodage UTF-16 avec BOM (courant sous Praat/Windows).
    """

    VALID_COMMANDS = [
        'forward', 'backward', 'left', 'right',
        'up', 'down', 'yawleft', 'yawright', 'none'
    ]

    def _detect_encoding(self, file_path: Path) -> str:
        """Detecte l'encodage du fichier (UTF-8 ou UTF-16)."""
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(4)
            if first_bytes[:2] in (b'\xff\xfe', b'\xfe\xff'):
                return 'utf-16'
            return 'utf-8'
        except Exception:
            return 'utf-8'

    def _parse_textgrid_manual(self, tg_path: Path) -> Dict:
        """Parse manuellement un fichier TextGrid (UTF-8 / UTF-16)."""
        encoding = self._detect_encoding(tg_path)

        try:
            with open(tg_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
        except Exception as e:
            try:
                with open(tg_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception:
                raise ValueError(f"Impossible de lire {tg_path}: {e}")

        tiers: List[Dict] = []
        current_tier = None
        current_interval = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if 'class = "IntervalTier"' in line or 'class = " I n t e r v a l T i e r "' in line:
                if current_tier:
                    tiers.append(current_tier)
                current_tier = {'name': None, 'intervals': []}

            elif 'name =' in line:
                parts = line.split('=')
                if len(parts) > 1:
                    name = parts[1].strip().strip('"').replace(' ', '')
                    if current_tier is not None:
                        current_tier['name'] = name.lower()

            elif line.startswith('xmin') and 'intervals:' not in ''.join(lines[max(0, i-5):i]):
                if current_interval and current_tier:
                    current_tier['intervals'].append(current_interval)
                try:
                    xmin = float(line.split('=')[1].strip())
                    current_interval = {'xmin': xmin, 'xmax': None, 'text': ''}
                except (IndexError, ValueError):
                    current_interval = None

            elif line.startswith('xmax') and current_interval is not None:
                try:
                    current_interval['xmax'] = float(line.split('=')[1].strip())
                except (IndexError, ValueError):
                    pass

            elif line.startswith('text') and current_interval is not None:
                parts = line.split('=', 1)
                if len(parts) > 1:
                    current_interval['text'] = parts[1].strip().strip('"').replace(' ', '')

            i += 1

        if current_interval and current_tier:
            current_tier['intervals'].append(current_interval)
        if current_tier:
            tiers.append(current_tier)

        return {'tiers': tiers}

    def parse_annotated_textgrid(self, tg_path: Path, tier_name: str = "commands") -> List[Dict]:
        """Parse un TextGrid et retourne les segments de commandes."""
        try:
            tg = self._parse_textgrid_manual(tg_path)
        except Exception as e:
            warnings.warn(f"Erreur lors du parsing de {tg_path}: {e}")
            return []

        command_tier = None
        tier_name_lower = tier_name.lower()
        for tier in tg['tiers']:
            if tier['name'] and tier['name'].lower() == tier_name_lower:
                command_tier = tier
                break

        if not command_tier:
            available = [t['name'] for t in tg['tiers'] if t['name']]
            warnings.warn(f"Tier '{tier_name}' non trouve dans {tg_path}. Disponibles: {available}")
            return []

        segments = []
        for interval in command_tier['intervals']:
            command = interval['text'].strip().lower()
            if not command:
                continue
            if command not in self.VALID_COMMANDS:
                warnings.warn(f"Commande invalide '{command}' a {interval['xmin']:.2f}s dans {tg_path.name}")
                continue
            segments.append({
                'start': interval['xmin'],
                'end': interval['xmax'],
                'duration': interval['xmax'] - interval['xmin'],
                'command': command,
            })
        return segments

    def create_dataset_from_annotations(
        self,
        textgrid_dir: Path,
        audio_dir: Path,
        output_csv: Path,
        tier_name: str = "commands",
        audio_extension: str = '.wav',
    ) -> pd.DataFrame:
        """Cree un dataset CSV a partir d'annotations TextGrid."""
        all_segments = []

        if not textgrid_dir.exists():
            raise FileNotFoundError(f"Le repertoire n'existe pas: {textgrid_dir}")

        textgrids = list(textgrid_dir.glob('*.TextGrid'))
        if not textgrids:
            textgrids = list(textgrid_dir.glob('*.textgrid'))
        if not textgrids:
            textgrids = [f for f in textgrid_dir.iterdir() if 'textgrid' in f.name.lower()]
        if not textgrids:
            raise FileNotFoundError(f"Aucun fichier TextGrid dans {textgrid_dir}")

        print(f"  Traitement de {len(textgrids)} fichier(s) TextGrid...")

        for tg_file in sorted(textgrids):
            audio_file = tg_file.stem + audio_extension
            audio_path = audio_dir / audio_file

            if not audio_path.exists():
                warnings.warn(f"Fichier audio manquant: {audio_path}")

            segments = self.parse_annotated_textgrid(tg_file, tier_name)
            if not segments:
                warnings.warn(f"Aucun segment valide dans {tg_file.name}")
                continue

            for seg in segments:
                seg['audio_file'] = audio_file
                all_segments.append(seg)

            print(f"    {tg_file.name}: {len(segments)} segments")

        if not all_segments:
            raise ValueError("Aucun segment trouve dans les fichiers TextGrid")

        df = pd.DataFrame(all_segments)

        # Extraire participant_id et attempt depuis le nom de fichier
        stems = df['audio_file'].apply(lambda f: Path(f).stem)
        df['participant_id'] = stems.apply(lambda s: '_'.join(s.split('_')[:6]))
        df['attempt'] = stems.apply(lambda s: int(s.split('_')[6]))

        df = df[['audio_file', 'participant_id', 'attempt', 'start', 'end', 'duration', 'command']]

        output_csv.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')

        print(f"\n  Dataset: {len(df)} segments, {df['audio_file'].nunique()} fichiers")
        print(f"  Distribution:")
        for cmd, count in df['command'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"    {cmd:12s}: {count:4d} ({pct:5.1f}%)")

        return df


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
    print("\n[1/5] Parsing des annotations TextGrid...")

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
    print("\n[3/5] Segmentation des fichiers audio...")

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

    print("\n[4/5] Extraction des embeddings wav2vec2...")

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
# STEP: SPLIT PARTICIPANTS
# =============================================================================

def split_participants(df: pd.DataFrame, test_size: float = 0.15,
                       random_seed: int = 42):
    """Split unique participant IDs into train/test sets."""
    print(f"\n[2/5] Split participants (test_size={test_size}, seed={random_seed})...")

    participants = df['participant_id'].unique()
    train_pids, test_pids = train_test_split(
        participants, test_size=test_size, random_state=random_seed,
    )

    df_train = df[df['participant_id'].isin(train_pids)].copy()
    df_test = df[df['participant_id'].isin(test_pids)].copy()

    print(f"  Train: {len(train_pids)} participants, {len(df_train)} segments")
    print(f"  Test:  {len(test_pids)} participants, {len(df_test)} segments")

    return df_train, df_test, sorted(train_pids), sorted(test_pids)


# =============================================================================
# STEP: COPY TEST AUDIO
# =============================================================================

def copy_test_audio(df_test: pd.DataFrame, audio_dir: Path, output_dir: Path,
                    skip_if_cached: bool = True):
    """Copy raw (unsegmented) audio files for test participants."""
    print("\n[5/5] Copie des fichiers audio test (bruts)...")

    output_dir.mkdir(exist_ok=True, parents=True)

    audio_files = df_test['audio_file'].unique()
    copied, skipped = 0, 0

    for audio_file in sorted(audio_files):
        src = audio_dir / audio_file
        dst = output_dir / audio_file

        if skip_if_cached and dst.exists():
            skipped += 1
            continue

        if not src.exists():
            warnings.warn(f"Fichier audio source manquant: {src}")
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"  Fichiers copies: {copied}, deja en cache: {skipped}")
    print(f"  Total dans {output_dir}: {len(list(output_dir.glob('*.wav')))}")


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
    test_size = prep.get("test_size", 0.15)
    random_seed = prep.get("random_seed", 42)

    dataset_csv = output_dir / "dataset.csv"

    # Train outputs
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True, parents=True)
    train_segments_dir = train_dir / "audio_segments"
    train_embeddings_file = train_dir / "all_embeddings.npz"
    train_csv = train_dir / "dataset_train.csv"

    # Test outputs
    test_dir = output_dir / "test"
    test_dir.mkdir(exist_ok=True, parents=True)
    test_csv = test_dir / "dataset_test.csv"
    test_audio_dir = test_dir / "audio"

    print("=" * 70)
    print("PREPARE DATA: TextGrid -> Split -> Segments -> Embeddings")
    print("=" * 70)

    # Step 1: Parse all annotations
    df = parse_annotations(
        textgrid_dir, audio_dir, dataset_csv,
        tier_name=tier_name, skip_if_cached=skip_cached,
    )

    # Step 2: Split participants into train/test
    df_train, df_test, train_pids, test_pids = split_participants(
        df, test_size=test_size, random_seed=random_seed,
    )

    # Save train/test CSVs
    train_csv.parent.mkdir(exist_ok=True, parents=True)
    df_train.to_csv(train_csv, index=False, encoding='utf-8')
    test_csv.parent.mkdir(exist_ok=True, parents=True)
    df_test.to_csv(test_csv, index=False, encoding='utf-8')
    print(f"  Saved: {train_csv}")
    print(f"  Saved: {test_csv}")

    # Step 3: Segment audio (train only)
    segment_audio(df_train, audio_dir, train_segments_dir, skip_if_cached=skip_cached)

    # Step 4: Extract embeddings (train only)
    extract_embeddings(df_train, train_segments_dir, train_embeddings_file,
                       skip_if_cached=skip_cached)

    # Step 5: Copy raw audio for test participants
    copy_test_audio(df_test, audio_dir, test_audio_dir, skip_if_cached=skip_cached)

    print("\n" + "=" * 70)
    print("PREPARATION TERMINEE")
    print("=" * 70)
    print(f"Sortie: {output_dir}")
    print(f"  - dataset.csv              ({len(df)} segments, all participants)")
    print(f"  - train/dataset_train.csv  ({len(df_train)} segments, {len(train_pids)} participants)")
    print(f"  - train/audio_segments/    (WAV 16 kHz)")
    print(f"  - train/all_embeddings.npz (wav2vec2 embeddings)")
    print(f"  - test/dataset_test.csv    ({len(df_test)} segments, {len(test_pids)} participants)")
    print(f"  - test/audio/              (raw WAV files)")
