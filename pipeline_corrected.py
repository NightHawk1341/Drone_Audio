# =============================================================================
# PIPELINE CORRIGÉ: ANNOTATION → ENTRAÎNEMENT
# Corrections: alignement embeddings/labels, cache modèle, équilibrage classes
# =============================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm


class PipelineConfig:
    """Configuration centralisée du pipeline"""
    def __init__(self, output_dir: str):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # Chemins des fichiers intermédiaires
        self.dataset_csv = self.output_path / "dataset.csv"
        self.train_csv = self.output_path / "train.csv"
        self.test_csv = self.output_path / "test.csv"
        self.train_balanced_csv = self.output_path / "train_balanced.csv"
        
        # Segments audio
        self.segments_dir = self.output_path / "audio_segments"
        self.train_segments = self.segments_dir / "train"
        self.test_segments = self.segments_dir / "test"
        
        # Embeddings et labels (avec segment_id pour alignement)
        self.embeddings_train_npz = self.output_path / "embeddings_train.npz"
        self.embeddings_test_npz = self.output_path / "embeddings_test.npz"
        
        # Modèles
        self.model_svm = self.output_path / "model_svm.pkl"
        self.model_mlp = self.output_path / "model_mlp.pkl"
        self.label_encoder = self.output_path / "label_encoder.pkl"
        self.scaler = self.output_path / "scaler.pkl"


def pipeline_annotation_vers_modele(
    textgrid_dir: str,
    audio_dir: str,
    output_dir: str,
    model_type: str = 'svm',
    skip_if_cached: bool = True,
    balance_classes: bool = True,
    none_ratio: float = 1.5  # Ratio de "none" par rapport à la classe majoritaire non-none
):
    """
    Pipeline complet: TextGrid annotés → Modèle entraîné
    
    Args:
        textgrid_dir: Dossier contenant les TextGrid
        audio_dir: Dossier contenant les WAV
        output_dir: Dossier de sortie
        model_type: 'svm' ou 'mlp'
        skip_if_cached: Si True, réutilise les fichiers existants
        balance_classes: Si True, sous-échantillonne la classe "none"
        none_ratio: Ratio max de "none" vs classe majoritaire non-none
    """
    from parser_simple_final import SimpleCommandParser
    from sklearn.model_selection import train_test_split
    
    print("=" * 80)
    print("PIPELINE CORRIGÉ: ANNOTATION → MODÈLE")
    print("=" * 80)
    
    config = PipelineConfig(output_dir)
    textgrid_path = Path(textgrid_dir)
    audio_path = Path(audio_dir)
    
    # =========================================================================
    # ÉTAPE 1: Parser les annotations
    # =========================================================================
    print("\n[1/6] Parsing des annotations TextGrid...")
    
    if skip_if_cached and config.dataset_csv.exists():
        print(f"  → Cache trouvé: {config.dataset_csv}")
        df = pd.read_csv(config.dataset_csv)
    else:
        parser = SimpleCommandParser()
        df = parser.create_dataset_from_annotations(
            textgrid_dir=textgrid_path,
            audio_dir=audio_path,
            output_csv=config.dataset_csv,
            tier_name="commands"
        )
    
    print(f"  Total segments: {len(df)}")
    print(f"  Distribution des classes:")
    print(df['command'].value_counts().to_string(header=False))
    
    # =========================================================================
    # ÉTAPE 2: Split train/test AVANT équilibrage
    # =========================================================================
    print("\n[2/6] Création du split train/test...")
    
    if skip_if_cached and config.train_csv.exists() and config.test_csv.exists():
        print(f"  → Cache trouvé")
        train_df = pd.read_csv(config.train_csv)
        test_df = pd.read_csv(config.test_csv)
    else:
        # Ajouter un ID unique à chaque segment AVANT le split
        df['segment_id'] = [f"seg_{i:06d}" for i in range(len(df))]
        
        train_df, test_df = train_test_split(
            df,
            test_size=0.15,
            stratify=df['command'],
            random_state=42
        )
        
        # Reset index pour avoir des indices propres
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_df.to_csv(config.train_csv, index=False)
        test_df.to_csv(config.test_csv, index=False)
    
    print(f"  Train: {len(train_df)} segments")
    print(f"  Test:  {len(test_df)} segments")
    
    # =========================================================================
    # ÉTAPE 2b: Équilibrage des classes (sous-échantillonnage de "none")
    # =========================================================================
    if balance_classes:
        print("\n[2b/6] Équilibrage des classes...")
        train_df_balanced = equilibrer_classes(
            train_df, 
            none_ratio=none_ratio,
            output_csv=config.train_balanced_csv
        )
        print(f"  Train après équilibrage: {len(train_df_balanced)} segments")
        print(f"  Nouvelle distribution:")
        print(train_df_balanced['command'].value_counts().to_string(header=False))
    else:
        train_df_balanced = train_df
    
    # =========================================================================
    # ÉTAPE 3: Segmenter les audios (avec segment_id dans le nom)
    # =========================================================================
    print("\n[3/6] Segmentation des fichiers audio...")
    
    train_segments_exist = config.train_segments.exists() and len(list(config.train_segments.glob("*.wav"))) > 0
    test_segments_exist = config.test_segments.exists() and len(list(config.test_segments.glob("*.wav"))) > 0
    
    if skip_if_cached and train_segments_exist and test_segments_exist:
        print(f"  → Cache trouvé")
    else:
        segmenter_audio_avec_id(
            df=train_df_balanced,
            audio_dir=audio_path,
            output_dir=config.train_segments
        )
        segmenter_audio_avec_id(
            df=test_df,
            audio_dir=audio_path,
            output_dir=config.test_segments
        )
    
    # =========================================================================
    # ÉTAPE 4: Extraire les embeddings (avec alignement garanti)
    # =========================================================================
    print("\n[4/6] Extraction des embeddings wav2vec2...")
    
    if skip_if_cached and config.embeddings_train_npz.exists() and config.embeddings_test_npz.exists():
        print(f"  → Cache trouvé")
        train_data = np.load(config.embeddings_train_npz, allow_pickle=True)
        X_train = train_data['embeddings']
        y_train = train_data['labels']
        
        test_data = np.load(config.embeddings_test_npz, allow_pickle=True)
        X_test = test_data['embeddings']
        y_test = test_data['labels']
    else:
        X_train, y_train = extraire_embeddings_alignes(
            df=train_df_balanced,
            segments_dir=config.train_segments,
            output_file=config.embeddings_train_npz
        )
        X_test, y_test = extraire_embeddings_alignes(
            df=test_df,
            segments_dir=config.test_segments,
            output_file=config.embeddings_test_npz
        )
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_test shape:  {y_test.shape}")
    
    # Vérification de cohérence
    assert X_train.shape[0] == y_train.shape[0], "ERREUR: Désalignement train!"
    assert X_test.shape[0] == y_test.shape[0], "ERREUR: Désalignement test!"
    print("  ✓ Alignement vérifié")
    
    # =========================================================================
    # ÉTAPE 5: Entraîner le modèle (avec cache)
    # =========================================================================
    print(f"\n[5/6] Entraînement du modèle {model_type.upper()}...")
    
    model_path = config.model_svm if model_type == 'svm' else config.model_mlp
    
    if skip_if_cached and model_path.exists() and config.label_encoder.exists():
        print(f"  → Modèle trouvé en cache: {model_path}")
        model = joblib.load(model_path)
        le = joblib.load(config.label_encoder)
    else:
        if model_type == 'svm':
            model, le = entrainer_svm(X_train, y_train, config)
        else:
            model, le = entrainer_mlp(X_train, y_train, config)
    
    # =========================================================================
    # ÉTAPE 6: Évaluer
    # =========================================================================
    print("\n[6/6] Évaluation sur le test set...")
    evaluer_modele(model, le, X_test, y_test, config)
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE TERMINÉ")
    print("=" * 80)
    print(f"\nTous les fichiers sont dans: {config.output_path}")
    
    return model, le, config


# =============================================================================
# FONCTIONS AUXILIAIRES CORRIGÉES
# =============================================================================

def equilibrer_classes(df: pd.DataFrame, none_ratio: float = 1.5, output_csv: Path = None) -> pd.DataFrame:
    """
    Sous-échantillonne la classe "none" pour équilibrer le dataset.
    
    Args:
        df: DataFrame avec colonne 'command'
        none_ratio: Ratio de "none" par rapport à la classe majoritaire non-none
                   Ex: 1.5 signifie que "none" aura 1.5x le nombre de la classe 
                   majoritaire parmi les autres classes
        output_csv: Chemin pour sauvegarder le CSV équilibré
    
    Returns:
        DataFrame équilibré
    """
    # Séparer "none" des autres classes
    df_none = df[df['command'] == 'none']
    df_other = df[df['command'] != 'none']
    
    # Trouver la taille de la classe majoritaire non-none
    max_other_class_size = df_other['command'].value_counts().max()
    
    # Calculer le nombre max de "none" à garder
    max_none = int(max_other_class_size * none_ratio)
    
    print(f"  Classe 'none' originale: {len(df_none)}")
    print(f"  Classe majoritaire non-none: {max_other_class_size}")
    print(f"  Cible pour 'none' (ratio {none_ratio}): {max_none}")
    
    # Sous-échantillonner "none" si nécessaire
    if len(df_none) > max_none:
        df_none_sampled = df_none.sample(n=max_none, random_state=42)
        print(f"  → Sous-échantillonnage de 'none': {len(df_none)} → {len(df_none_sampled)}")
    else:
        df_none_sampled = df_none
        print(f"  → 'none' déjà sous le seuil, pas de changement")
    
    # Recombiner
    df_balanced = pd.concat([df_other, df_none_sampled], ignore_index=True)
    
    # Mélanger pour éviter les biais d'ordre
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if output_csv:
        df_balanced.to_csv(output_csv, index=False)
    
    return df_balanced


def segmenter_audio_avec_id(df: pd.DataFrame, audio_dir: Path, output_dir: Path):
    """
    Découpe les audios en utilisant segment_id pour garantir l'alignement.
    
    Le nom du fichier de sortie contient le segment_id, ce qui permet
    de retrouver facilement la correspondance segment ↔ label.
    """
    import soundfile as sf
    import librosa
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    audio_dir = Path(audio_dir)
    
    erreurs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmentation"):
        audio_file = audio_dir / row['audio_file']
        segment_id = row['segment_id']
        
        try:
            # Charger et rééchantillonner à 16kHz
            audio, sr = librosa.load(audio_file, sr=16000)
            
            start_sample = int(row['start'] * sr)
            end_sample = int(row['end'] * sr)
            segment = audio[start_sample:end_sample]
            
            # Vérifier que le segment n'est pas vide
            if len(segment) < 160:  # Moins de 10ms à 16kHz
                erreurs.append(f"{segment_id}: segment trop court ({len(segment)} samples)")
                continue
            
            # Nom avec segment_id pour alignement garanti
            output_file = output_dir / f"{segment_id}.wav"
            sf.write(output_file, segment, sr)
            
        except Exception as e:
            erreurs.append(f"{segment_id}: {e}")
    
    if erreurs:
        print(f"  ⚠ {len(erreurs)} erreurs de segmentation:")
        for err in erreurs[:5]:
            print(f"    - {err}")
        if len(erreurs) > 5:
            print(f"    ... et {len(erreurs) - 5} autres")


def extraire_embeddings_alignes(df: pd.DataFrame, segments_dir: Path, output_file: Path):
    """
    Extrait les embeddings wav2vec2 avec alignement garanti via segment_id.
    
    Returns:
        Tuple (embeddings, labels) avec correspondance garantie
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import soundfile as sf
    
    segments_dir = Path(segments_dir)
    
    # Charger le modèle wav2vec2
    print("  Chargement du modèle wav2vec2-FR-7K-large...")
    model_name = "LeBenchmark/wav2vec2-FR-7K-large"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    
    # Geler les paramètres
    for param in model.parameters():
        param.requires_grad = False
    
    # GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    
    embeddings = []
    labels = []
    segment_ids_traites = []
    erreurs = []
    
    # Parcourir le DataFrame dans l'ordre pour garantir l'alignement
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extraction embeddings"):
        segment_id = row['segment_id']
        audio_file = segments_dir / f"{segment_id}.wav"
        
        if not audio_file.exists():
            erreurs.append(f"{segment_id}: fichier non trouvé")
            continue
        
        try:
            audio, sr = sf.read(audio_file)
            
            # Vérifier le sample rate
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Extraire les features
            inputs = feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling sur la dimension temporelle
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            embeddings.append(embedding)
            labels.append(row['command'])
            segment_ids_traites.append(segment_id)
            
        except Exception as e:
            erreurs.append(f"{segment_id}: {e}")
    
    if erreurs:
        print(f"  ⚠ {len(erreurs)} erreurs d'extraction:")
        for err in erreurs[:5]:
            print(f"    - {err}")
    
    # Convertir en arrays numpy
    X = np.vstack(embeddings)
    y = np.array(labels)
    
    # Sauvegarder avec les segment_ids pour traçabilité
    np.savez(
        output_file,
        embeddings=X,
        labels=y,
        segment_ids=np.array(segment_ids_traites)
    )
    
    print(f"  ✓ {len(embeddings)} embeddings extraits et alignés")
    
    return X, y


def entrainer_svm(X_train, y_train, config: PipelineConfig):
    """Entraîne un SVM avec pondération des classes"""
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    
    print("  Préparation des données...")
    
    # Encoder les labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    # Normaliser les features (important pour SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    print(f"  Classes: {le.classes_}")
    print(f"  Distribution: {np.bincount(y_encoded)}")
    
    # SVM avec pondération des classes
    print("  Entraînement SVM (kernel RBF, class_weight='balanced')...")
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        verbose=False
    )
    
    # Cross-validation rapide pour estimer la performance
    print("  Validation croisée (3-fold)...")
    cv_scores = cross_val_score(svm, X_scaled, y_encoded, cv=3, scoring='f1_macro')
    print(f"  CV F1-macro: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Entraînement final sur tout le train set
    print("  Entraînement final...")
    svm.fit(X_scaled, y_encoded)
    
    # Sauvegarder
    joblib.dump(svm, config.model_svm)
    joblib.dump(le, config.label_encoder)
    joblib.dump(scaler, config.scaler)
    
    print(f"  ✓ Modèle sauvegardé: {config.model_svm}")
    
    return svm, le


def entrainer_mlp(X_train, y_train, config: PipelineConfig):
    """Entraîne un MLP avec early stopping"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    
    print("  Préparation des données...")
    
    # Encoder les labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    # Normaliser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    print(f"  Classes: {le.classes_}")
    
    # MLP avec early stopping
    print("  Entraînement MLP (512-256, early_stopping)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        alpha=0.001,  # Régularisation L2
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    
    mlp.fit(X_scaled, y_encoded)
    
    print(f"  Convergence en {mlp.n_iter_} itérations")
    
    # Sauvegarder
    joblib.dump(mlp, config.model_mlp)
    joblib.dump(le, config.label_encoder)
    joblib.dump(scaler, config.scaler)
    
    print(f"  ✓ Modèle sauvegardé: {config.model_mlp}")
    
    return mlp, le


def evaluer_modele(model, le, X_test, y_test, config: PipelineConfig):
    """Évalue le modèle avec métriques détaillées"""
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Charger le scaler
    scaler = joblib.load(config.scaler)
    X_test_scaled = scaler.transform(X_test)
    
    # Encoder les labels de test
    y_test_encoded = le.transform(y_test)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Métriques
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
    f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS D'ÉVALUATION")
    print(f"{'='*60}")
    print(f"\n🎯 F1-score macro:    {f1_macro:.3f}")
    print(f"🎯 F1-score weighted: {f1_weighted:.3f}")
    
    print(f"\n📊 Rapport de classification:")
    report = classification_report(
        y_test_encoded, 
        y_pred, 
        target_names=le.classes_,
        zero_division=0
    )
    print(report)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Sauvegarder les résultats
    with open(config.output_path / "evaluation.txt", 'w', encoding='utf-8') as f:
        f.write(f"F1-score macro: {f1_macro:.3f}\n")
        f.write(f"F1-score weighted: {f1_weighted:.3f}\n\n")
        f.write("Rapport de classification:\n")
        f.write(report)
        f.write("\n\nMatrice de confusion:\n")
        f.write(str(cm))
    
    # Visualisation de la matrice de confusion
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_
        )
        plt.title(f'Matrice de confusion (F1-macro: {f1_macro:.3f})')
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.tight_layout()
        plt.savefig(config.output_path / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"\n📈 Matrice de confusion sauvegardée: confusion_matrix.png")
    except Exception as e:
        print(f"  ⚠ Impossible de créer la visualisation: {e}")
    
    return f1_macro


def charger_modele(config: PipelineConfig, model_type: str = 'svm'):
    """Charge un modèle pré-entraîné depuis le cache"""
    model_path = config.model_svm if model_type == 'svm' else config.model_mlp
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    if not config.label_encoder.exists():
        raise FileNotFoundError(f"Label encoder non trouvé: {config.label_encoder}")
    if not config.scaler.exists():
        raise FileNotFoundError(f"Scaler non trouvé: {config.scaler}")
    
    model = joblib.load(model_path)
    le = joblib.load(config.label_encoder)
    scaler = joblib.load(config.scaler)
    
    print(f"✓ Modèle chargé: {model_path}")
    
    return model, le, scaler


def predire_commande(audio_path: str, model, le, scaler, wav2vec_model=None, feature_extractor=None):
    """
    Prédit la commande pour un fichier audio.
    
    Args:
        audio_path: Chemin vers le fichier WAV
        model: Modèle entraîné (SVM ou MLP)
        le: LabelEncoder
        scaler: StandardScaler
        wav2vec_model: Modèle wav2vec2 (optionnel, chargé si None)
        feature_extractor: Feature extractor wav2vec2 (optionnel)
    
    Returns:
        Tuple (commande_predite, probabilites)
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    import soundfile as sf
    import librosa
    
    # Charger wav2vec2 si nécessaire
    if wav2vec_model is None:
        model_name = "LeBenchmark/wav2vec2-FR-7K-large"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.eval()
    
    # Charger l'audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Extraire l'embedding
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Normaliser et prédire
    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    prediction = model.predict(embedding_scaled)
    
    commande = le.inverse_transform(prediction)[0]
    
    return commande


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    TEXTGRID_DIR = r"B:\drones\test_1\data"
    AUDIO_DIR = r"B:\drones\test_1\data"
    OUTPUT_DIR = r"B:\drones\test_1\output"
    
    # Option 1: Entraînement complet (utilise le cache si disponible)
    model, le, config = pipeline_annotation_vers_modele(
        textgrid_dir=TEXTGRID_DIR,
        audio_dir=AUDIO_DIR,
        output_dir=OUTPUT_DIR,
        model_type='svm', # ou 'mlp'
        skip_if_cached=True,      # Réutilise les fichiers existants
        balance_classes=True,      # Sous-échantillonne "none"
        none_ratio=1.5            # "none" aura max 1.5x la classe majoritaire
    )
    
    # Option 2: Charger un modèle existant pour debug
    """
    config = PipelineConfig(OUTPUT_DIR)
    model, le, scaler = charger_modele(config, model_type='svm')
    
    # Charger les données de test
    test_data = np.load(config.embeddings_test_npz)
    X_test = test_data['embeddings']
    y_test = test_data['labels']
    
    # Évaluer
    evaluer_modele(model, le, X_test, y_test, config)
    """
    
    # Option 3: Prédire une commande pour un fichier audio
    """
    config = PipelineConfig(OUTPUT_DIR)
    model, le, scaler = charger_modele(config, model_type='svm')
    
    commande = predire_commande(
        audio_path="chemin/vers/audio.wav",
        model=model,
        le=le,
        scaler=scaler
    )
    print(f"Commande prédite: {commande}")
    """
