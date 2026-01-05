"""
Guide pratique d'utilisation du parser VoiceStick
Pour le projet M2 TAL - Classification de commandes vocales pour drone

Ce script montre comment utiliser le parser dans ton workflow complet
"""

from pathlib import Path
from voicestick_parser import VoiceStickParser
import pandas as pd

# =============================================================================
# ÉTAPE 1: PARSING ET CRÉATION DU DATASET
# =============================================================================

def etape1_creer_dataset(textgrid_dir: str, output_dir: str):
    """
    Parse tous les TextGrid et crée le dataset CSV
    
    Args:
        textgrid_dir: Répertoire contenant les fichiers .TextGrid et _commands.TextGrid
        output_dir: Où sauvegarder le CSV
    """
    print("=" * 80)
    print("ÉTAPE 1: PARSING DES TEXTGRID ET CRÉATION DU DATASET")
    print("=" * 80)
    
    textgrid_path = Path(textgrid_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Créer le parser avec tolérance de 1.5s
    parser = VoiceStickParser(tolerance=1.5)
    
    # Créer le dataset complet
    csv_path = output_path / "voicestick_dataset.csv"
    df = parser.create_dataset(
        textgrid_dir=textgrid_path,
        output_csv=csv_path,
        audio_extension='.wav'
    )
    
    print(f"\n✓ Dataset créé: {csv_path}")
    print(f"  Total segments: {len(df)}")
    print(f"  Fichiers audio uniques: {df['audio_file'].nunique()}")
    
    return df


# =============================================================================
# ÉTAPE 2: ANALYSE ET STATISTIQUES
# =============================================================================

def etape2_analyser_dataset(df: pd.DataFrame):
    """
    Analyse le dataset créé et affiche les statistiques importantes
    """
    print("\n" + "=" * 80)
    print("ÉTAPE 2: ANALYSE DU DATASET")
    print("=" * 80)
    
    # Distribution des classes
    print("\n📊 DISTRIBUTION DES CLASSES:")
    print("-" * 40)
    class_dist = df['command'].value_counts()
    for cmd, count in class_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {cmd:12s}: {count:4d} segments ({pct:5.1f}%)")
    
    # Déséquilibre
    print(f"\n⚠️  DÉSÉQUILIBRE:")
    max_class = class_dist.max()
    min_class = class_dist.min()
    ratio = max_class / min_class
    print(f"  Ratio max/min: {ratio:.1f}:1")
    print(f"  → Nécessite pondération des classes et/ou data augmentation")
    
    # Qualité du matching joystick
    print(f"\n🎯 QUALITÉ DU MATCHING JOYSTICK:")
    print("-" * 40)
    conf_mean = df['joystick_confidence'].mean()
    has_joy = len(df[df['joystick_confidence'] > 0])
    no_joy = len(df[df['joystick_confidence'] == 0])
    
    print(f"  Confidence moyenne: {conf_mean:.2f}")
    print(f"  Avec joystick: {has_joy}/{len(df)} ({has_joy/len(df)*100:.1f}%)")
    print(f"  Sans joystick: {no_joy}/{len(df)} ({no_joy/len(df)*100:.1f}%)")
    
    # Segments problématiques
    low_conf = df[(df['joystick_confidence'] > 0) & (df['joystick_confidence'] < 0.5)]
    if len(low_conf) > 0:
        print(f"\n  ⚠️  Segments avec faible confidence (<0.5): {len(low_conf)}")
        print(f"     → À vérifier manuellement si nécessaire")
    
    # Durée des segments
    print(f"\n⏱️  DURÉE DES SEGMENTS:")
    print("-" * 40)
    print(f"  Moyenne: {df['duration'].mean():.2f}s")
    print(f"  Médiane: {df['duration'].median():.2f}s")
    print(f"  Min: {df['duration'].min():.2f}s")
    print(f"  Max: {df['duration'].max():.2f}s")
    
    # Segments très courts (potentiellement problématiques pour wav2vec2)
    very_short = df[df['duration'] < 0.3]
    if len(very_short) > 0:
        print(f"\n  ⚠️  Segments très courts (<0.3s): {len(very_short)}")
        print(f"     → Wav2vec2 pourrait avoir du mal avec ces segments")


# =============================================================================
# ÉTAPE 3: SPLIT TRAIN/TEST STRATIFIÉ
# =============================================================================

def etape3_split_dataset(df: pd.DataFrame, test_size: float = 0.13):
    """
    Crée le split train/test stratifié
    
    Args:
        df: Dataset complet
        test_size: Proportion du test set (0.13 = 195 segments pour vous)
    
    Returns:
        train_df, test_df
    """
    print("\n" + "=" * 80)
    print("ÉTAPE 3: SPLIT TRAIN/TEST STRATIFIÉ")
    print("=" * 80)
    
    from sklearn.model_selection import train_test_split
    
    # Split stratifié pour maintenir la distribution des classes
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['command'],
        random_state=42  # Reproductibilité
    )
    
    print(f"\n✓ Split effectué:")
    print(f"  Train: {len(train_df)} segments ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} segments ({len(test_df)/len(df)*100:.1f}%)")
    
    print(f"\n📊 Vérification de la stratification:")
    print("-" * 40)
    print(f"{'Classe':<12} {'Train %':>10} {'Test %':>10} {'Diff':>10}")
    print("-" * 40)
    
    for cmd in df['command'].unique():
        train_pct = (train_df['command'] == cmd).sum() / len(train_df) * 100
        test_pct = (test_df['command'] == cmd).sum() / len(test_df) * 100
        diff = abs(train_pct - test_pct)
        print(f"{cmd:<12} {train_pct:>9.1f}% {test_pct:>9.1f}% {diff:>9.1f}%")
    
    return train_df, test_df


# =============================================================================
# ÉTAPE 4: SAUVEGARDER LES SPLITS
# =============================================================================

def etape4_sauvegarder_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """
    Sauvegarde les splits train/test séparément
    """
    print("\n" + "=" * 80)
    print("ÉTAPE 4: SAUVEGARDE DES SPLITS")
    print("=" * 80)
    
    output_path = Path(output_dir)
    
    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Fichiers sauvegardés:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    
    return train_path, test_path


# =============================================================================
# EXEMPLE D'UTILISATION COMPLÈTE
# =============================================================================

if __name__ == "__main__":
    # Configuration des chemins
    TEXTGRID_DIR = "/path/to/voicestick/textgrids"
    OUTPUT_DIR = "/path/to/output"
    
    print("\n" + "=" * 80)
    print("PARSER VOICESTICK - WORKFLOW COMPLET")
    print("Projet M2 TAL - Classification de commandes vocales pour drone")
    print("=" * 80)
    
    try:
        # Étape 1: Parser et créer le dataset
        df = etape1_creer_dataset(TEXTGRID_DIR, OUTPUT_DIR)
        
        # Étape 2: Analyser
        etape2_analyser_dataset(df)
        
        # Étape 3: Split train/test
        train_df, test_df = etape3_split_dataset(df, test_size=0.13)
        
        # Étape 4: Sauvegarder
        train_path, test_path = etape4_sauvegarder_splits(train_df, test_df, OUTPUT_DIR)
        
        print("\n" + "=" * 80)
        print("✓ WORKFLOW TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        
        print("\n📋 PROCHAINES ÉTAPES:")
        print("-" * 40)
        print("1. Segmenter les fichiers audio selon start/end du CSV")
        print("2. Rééchantillonner les segments à 16kHz")
        print("3. Extraire les embeddings wav2vec2 (frozen, couche 12)")
        print("4. Entraîner SVM et MLP avec 5-fold CV sur train")
        print("5. Évaluer sur test set")
        print("6. Analyser les erreurs et optimiser")
        
        print("\n💡 CONSEILS:")
        print("-" * 40)
        print("• Utiliser class_weight='balanced' dans les classifieurs")
        print("• Essayer data augmentation sur train (pitch, speed, noise)")
        print("• Valider avec LOSO (Leave-One-Speaker-Out) si possible")
        print("• Surveiller particulièrement les performances sur 'stop'")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# =============================================================================

def filtrer_segments_courts(df: pd.DataFrame, min_duration: float = 0.3):
    """
    Optionnel: Filtrer les segments trop courts pour wav2vec2
    """
    avant = len(df)
    df_filtered = df[df['duration'] >= min_duration].copy()
    apres = len(df_filtered)
    
    print(f"Segments filtrés: {avant} → {apres} (supprimés: {avant - apres})")
    return df_filtered


def afficher_exemples_par_classe(df: pd.DataFrame, n: int = 3):
    """
    Affiche quelques exemples de chaque classe
    Utile pour vérifier la qualité du labelling
    """
    print("\n" + "=" * 80)
    print("EXEMPLES PAR CLASSE")
    print("=" * 80)
    
    for cmd in sorted(df['command'].unique()):
        print(f"\n{cmd.upper()}:")
        print("-" * 40)
        
        cmd_df = df[df['command'] == cmd].sample(min(n, len(df[df['command'] == cmd])))
        for _, row in cmd_df.iterrows():
            conf_str = f"(conf: {row['joystick_confidence']:.2f})" if row['joystick_confidence'] > 0 else "(sans joy)"
            print(f"  • \"{row['transcription']}\" {conf_str}")
