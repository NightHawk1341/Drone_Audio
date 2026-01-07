# =============================================================================
# PIPELINE COMPLET: ANNOTATION → ENTRAÎNEMENT
# ==========================================================================
from pathlib import Path
from parser_simple_final import SimpleCommandParser

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
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import soundfile as sf
    from pathlib import Path
    from tqdm import tqdm
    
    model_name = "LeBenchmark/wav2vec2-FR-7K-large"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    embeddings = []
    
    for audio_file in tqdm(audio_files, desc="Extraction embeddings"):
        audio, sr = sf.read(audio_file)
        
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
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
    """
    parser = SimpleCommandParser()
    df = parser.create_dataset_from_annotations(
        textgrid_dir=Path(r"B:\drones\nikita_07-01-26\complete"),
        audio_dir=Path(r"B:\drones\nikita_07-01-26\complete"),
        output_csv=Path(r"B:\drones\nikita_07-01-26\dataset_commands.csv"),
        tier_name="commands"  # Nom du tier dans vos TextGrid
    )
    """
    # OPTION 2: Pipeline complet automatique
    pipeline_annotation_vers_modele(
        textgrid_dir=Path(r"B:\drones\nikita_07-01-26\complete"),
        audio_dir=Path(r"B:\drones\nikita_07-01-26\complete"),
        output_dir=Path(r"B:\drones\nikita_07-01-26\output_pipeline"),
        model_type='svm'  # ou 'mlp'
    )

