"""
Parser simplifié pour annotations directes de commandes
Utilise un seul fichier TextGrid où chaque segment est annoté avec sa commande

Format du TextGrid attendu:
    Tier "Commands" avec intervals:
        xmin = 5.0
        xmax = 6.5
        text = "forward"  ← Commande directe !

Ce parser est plus simple et direct que le parser avec synchronisation joystick
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings


class SimpleCommandParser:
    """
    Parser pour TextGrid avec annotations directes de commandes
    """
    
    # Classes de commandes valides
    VALID_COMMANDS = [
        'forward', 'back', 'left', 'right', 
        'up', 'down', 'yawleft', 'yawright', 'none'
    ]
    
    def __init__(self):
        pass
    
    def _parse_textgrid_manual(self, tg_path: Path) -> Dict:
        """Parse manuellement un fichier TextGrid"""
        with open(tg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tiers = []
        current_tier = None
        current_interval = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('class = "IntervalTier"'):
                if current_tier:
                    tiers.append(current_tier)
                current_tier = {'name': None, 'intervals': []}
            
            elif line.startswith('name = '):
                name = line.split('=')[1].strip().strip('"')
                if current_tier is not None:
                    current_tier['name'] = name
            
            elif line.startswith('xmin =') and 'intervals:' not in lines[max(0, i-5):i]:
                if current_interval:
                    current_tier['intervals'].append(current_interval)
                
                xmin = float(line.split('=')[1].strip())
                current_interval = {'xmin': xmin, 'xmax': None, 'text': ''}
            
            elif line.startswith('xmax =') and current_interval is not None:
                xmax = float(line.split('=')[1].strip())
                current_interval['xmax'] = xmax
            
            elif line.startswith('text = ') and current_interval is not None:
                text = line.split('=', 1)[1].strip().strip('"')
                current_interval['text'] = text
            
            i += 1
        
        if current_interval:
            current_tier['intervals'].append(current_interval)
        if current_tier:
            tiers.append(current_tier)
        
        return {'tiers': tiers}
    
    def parse_annotated_textgrid(self, tg_path: Path, tier_name: str = "Commands") -> List[Dict]:
        """
        Parse un TextGrid avec annotations directes de commandes
        
        Args:
            tg_path: Chemin vers le fichier TextGrid
            tier_name: Nom du tier contenant les commandes (défaut: "Commands")
        
        Returns:
            Liste de dictionnaires avec start, end, command
        """
        try:
            tg = self._parse_textgrid_manual(tg_path)
        except Exception as e:
            warnings.warn(f"Erreur lors du parsing de {tg_path}: {e}")
            return []
        
        # Trouver le tier de commandes
        command_tier = None
        for tier in tg['tiers']:
            if tier['name'] == tier_name:
                command_tier = tier
                break
        
        if not command_tier:
            warnings.warn(f"Tier '{tier_name}' non trouvé dans {tg_path}")
            return []
        
        segments = []
        for interval in command_tier['intervals']:
            command = interval['text'].strip().lower()
            
            # Ignorer les intervalles vides
            if not command:
                continue
            
            # Vérifier que la commande est valide
            if command not in self.VALID_COMMANDS:
                warnings.warn(f"Commande invalide '{command}' à {interval['xmin']:.2f}s - ignorée")
                continue
            
            segments.append({
                'start': interval['xmin'],
                'end': interval['xmax'],
                'duration': interval['xmax'] - interval['xmin'],
                'command': command
            })
        
        return segments
    
    def create_dataset_from_annotations(
        self,
        textgrid_dir: Path,
        audio_dir: Path,
        output_csv: Path,
        tier_name: str = "commands",
        audio_extension: str = '.wav'
    ) -> pd.DataFrame:
        """
        Crée un dataset complet à partir d'annotations directes
        
        Args:
            textgrid_dir: Répertoire contenant les fichiers TextGrid annotés
            audio_dir: Répertoire contenant les fichiers audio correspondants
            output_csv: Chemin de sortie pour le CSV
            tier_name: Nom du tier contenant les commandes
            audio_extension: Extension des fichiers audio
        
        Returns:
            DataFrame pandas avec toutes les informations
        """
        all_segments = []
        
        # Trouver tous les TextGrid
        textgrids = list(textgrid_dir.glob('*.TextGrid'))
        
        print(f"Traitement de {len(textgrids)} fichiers TextGrid...")
        
        for tg_file in textgrids:
            # Nom du fichier audio correspondant
            audio_file = tg_file.stem + audio_extension
            audio_path = audio_dir / audio_file
            
            # Vérifier que l'audio existe
            if not audio_path.exists():
                warnings.warn(f"Fichier audio manquant: {audio_path}")
                continue
            
            # Parser le TextGrid
            segments = self.parse_annotated_textgrid(tg_file, tier_name)
            
            # Ajouter le nom du fichier audio à chaque segment
            for seg in segments:
                seg['audio_file'] = audio_file
                all_segments.append(seg)
        
        # Convertir en DataFrame
        df = pd.DataFrame(all_segments)
        
        # Réorganiser les colonnes
        df = df[['audio_file', 'start', 'end', 'duration', 'command']]
        
        # Sauvegarder
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"\n✓ Dataset créé avec {len(df)} segments")
        print(f"  Sauvegardé dans: {output_csv}")
        print(f"  Fichiers audio traités: {df['audio_file'].nunique()}")
        
        print("\n📊 Distribution des classes:")
        for cmd, count in df['command'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"  {cmd:12s}: {count:3d} segments ({pct:5.1f}%)")
        
        print(f"\n⏱️  Durée moyenne des segments: {df['duration'].mean():.2f}s")
        
        return df


# =============================================================================
# PIPELINE COMPLET: ANNOTATION → ENTRAÎNEMENT
# =============================================================================

def pipeline_annotation_vers_modele(
    textgrid_dir: str,
    audio_dir: str,
    output_dir: str,
    model_type: str = 'svm'  # 'svm' ou 'mlp'
):
    """
    Pipeline complet: TextGrid annotés → Modèle entraîné
    
    Étapes:
    1. Parser les TextGrid annotés
    2. Créer le dataset CSV
    3. Segmenter les fichiers audio
    4. Extraire les embeddings wav2vec2
    5. Entraîner le modèle
    6. Évaluer les performances
    """
    from pathlib import Path
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("=" * 80)
    print("PIPELINE COMPLET: ANNOTATION → MODÈLE")
    print("=" * 80)
    
    textgrid_path = Path(textgrid_dir)
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # ÉTAPE 1: Parser les annotations
    print("\n[1/6] Parsing des annotations TextGrid...")
    parser = SimpleCommandParser()
    csv_path = output_path / "dataset.csv"
    df = parser.create_dataset_from_annotations(
        textgrid_dir=textgrid_path,
        audio_dir=audio_path,
        output_csv=csv_path,
        tier_name="commands"
    )
    
    # ÉTAPE 2: Split train/test
    print("\n[2/6] Création du split train/test...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df['command'],
        random_state=42
    )
    
    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"  Train: {len(train_df)} segments")
    print(f"  Test:  {len(test_df)} segments")
    
    # ÉTAPE 3: Segmenter les audios
    print("\n[3/6] Segmentation des fichiers audio...")
    segments_dir = output_path / "audio_segments"
    segmenter_audio_depuis_csv(
        csv_path=str(output_path / "train.csv"),
        audio_dir=str(audio_path),
        output_dir=str(segments_dir / "train")
    )
    segmenter_audio_depuis_csv(
        csv_path=str(output_path / "test.csv"),
        audio_dir=str(audio_path),
        output_dir=str(segments_dir / "test")
    )
    
    # ÉTAPE 4: Extraire les embeddings
    print("\n[4/6] Extraction des embeddings wav2vec2...")
    embeddings_train = extraire_embeddings_batch(
        audio_dir=str(segments_dir / "train"),
        output_file=str(output_path / "embeddings_train.npy")
    )
    embeddings_test = extraire_embeddings_batch(
        audio_dir=str(segments_dir / "test"),
        output_file=str(output_path / "embeddings_test.npy")
    )
    
    # Préparer les labels
    labels_train = train_df['command'].values
    labels_test = test_df['command'].values
    np.save(output_path / "labels_train.npy", labels_train)
    np.save(output_path / "labels_test.npy", labels_test)
    
    # ÉTAPE 5: Entraîner le modèle
    print(f"\n[5/6] Entraînement du modèle {model_type.upper()}...")
    
    if model_type == 'svm':
        model = entrainer_svm_simple(
            X_train=embeddings_train,
            y_train=labels_train,
            output_path=output_path
        )
    else:  # mlp
        model = entrainer_mlp_simple(
            X_train=embeddings_train,
            y_train=labels_train,
            output_path=output_path
        )
    
    # ÉTAPE 6: Évaluer
    print("\n[6/6] Évaluation sur le test set...")
    evaluer_modele(
        model=model,
        X_test=embeddings_test,
        y_test=labels_test,
        output_path=output_path
    )
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE TERMINÉ")
    print("=" * 80)
    print(f"\nTous les fichiers sont dans: {output_path}")


# =============================================================================
# FONCTIONS AUXILIAIRES SIMPLIFIÉES
# =============================================================================

def segmenter_audio_depuis_csv(csv_path: str, audio_dir: str, output_dir: str):
    """Découpe les audios selon le CSV"""
    import pandas as pd
    import soundfile as sf
    import librosa
    from pathlib import Path
    from tqdm import tqdm
    
    df = pd.read_csv(csv_path)
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmentation"):
        audio_file = audio_path / row['audio_file']
        
        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            start_sample = int(row['start'] * sr)
            end_sample = int(row['end'] * sr)
            segment = audio[start_sample:end_sample]
            
            base_name = Path(row['audio_file']).stem
            output_file = output_path / f"{base_name}_seg_{idx:04d}.wav"
            sf.write(output_file, segment, sr)
            
        except Exception as e:
            print(f"Erreur segment {idx}: {e}")


def extraire_embeddings_batch(audio_dir: str, output_file: str):
    """Extrait les embeddings wav2vec2"""
    import numpy as np
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    import soundfile as sf
    from pathlib import Path
    from tqdm import tqdm
    
    model_name = "LeBenchmark/wav2vec2-FR-7K-large"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    embeddings = []
    
    for audio_file in tqdm(audio_files, desc="Extraction embeddings"):
        audio, sr = sf.read(audio_file)
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    
    all_embeddings = np.vstack(embeddings)
    np.save(output_file, all_embeddings)
    
    return all_embeddings


def entrainer_svm_simple(X_train, y_train, output_path):
    """Entraîne un SVM simple"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(X_train, y_encoded)
    
    joblib.dump(svm, output_path / "model_svm.pkl")
    joblib.dump(le, output_path / "label_encoder.pkl")
    
    return svm


def entrainer_mlp_simple(X_train, y_train, output_path):
    """Entraîne un MLP simple"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train, y_encoded)
    
    joblib.dump(mlp, output_path / "model_mlp.pkl")
    joblib.dump(le, output_path / "label_encoder.pkl")
    
    return mlp


def evaluer_modele(model, X_test, y_test, output_path):
    """Évalue le modèle"""
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import joblib
    
    le = joblib.load(output_path / "label_encoder.pkl")
    y_test_encoded = le.transform(y_test)
    
    y_pred = model.predict(X_test)
    
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
    print(f"\n🎯 F1-score macro: {f1_macro:.3f}")
    
    print("\n📊 Rapport de classification:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    # Sauvegarder les résultats
    with open(output_path / "evaluation.txt", 'w') as f:
        f.write(f"F1-score macro: {f1_macro:.3f}\n\n")
        f.write(classification_report(y_test_encoded, y_pred, target_names=le.classes_))


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    # OPTION 1: Juste parser les annotations
    parser = SimpleCommandParser()
    df = parser.create_dataset_from_annotations(
        textgrid_dir=Path("/path/to/textgrids"),
        audio_dir=Path("/path/to/audios"),
        output_csv=Path("/path/to/output/dataset.csv"),
        tier_name="commands"  # Nom du tier dans vos TextGrid
    )
    
    # OPTION 2: Pipeline complet automatique
    # pipeline_annotation_vers_modele(
    #     textgrid_dir="/path/to/textgrids",
    #     audio_dir="/path/to/audios",
    #     output_dir="/path/to/output",
    #     model_type='svm'  # ou 'mlp'
    # )
