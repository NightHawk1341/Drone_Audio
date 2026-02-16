# =============================================================================
# TRAIN_MLP : Entraînement MLP avec validation croisée GroupKFold (par participant)
# =============================================================================
"""
Entraîne un classifieur MLP (Multi-Layer Perceptron) sur des embeddings
pré-extraits (wav2vec2 ou similaires), en reproduisant la logique de train_svm.py.

IMPORTANT (comme train_svm.py) :
- Les features X, les labels y, et les groupes (participant_ids) sont lus depuis
  all_embeddings.npz (et non depuis dataset.csv).
- Validation croisée speaker-independent via GroupKFold (group=participant_id).
- Standardisation via StandardScaler.
- Sauvegarde des résultats et artefacts dans output_dir/MLP_model/.

Usage :
    python train_mlp.py
    python train_mlp.py path/to/config.yaml
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Utilitaire : duplication console + fichier (même idée que dans train_svm.py)
# =============================================================================

class Tee:
    """
    Duplique la sortie standard :
      - écrit dans le terminal (stdout original)
      - écrit dans un fichier log
    """

    def __init__(self, filepath: Path):
        self.file = open(filepath, "w", encoding="utf-8")
        self.console = sys.__stdout__  # stdout "réel" (non remplacé)

    def write(self, msg: str):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        # nécessaire pour compatibilité avec certains environnements (debuggers, notebooks, etc.)
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.console



# =============================================================================
# Chargement config (structure identique à ton projet)
# =============================================================================

def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Charge un fichier YAML de configuration.

    - Si config_path est fourni : on charge ce fichier.
    - Sinon : on charge config.yaml à côté du script.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# (Optionnel) Sous-échantillonnage de la classe "none"
# =============================================================================

def downsample_none_class(
    X: np.ndarray,
    y: np.ndarray,
    none_label: str = "none",
    none_ratio: float = 1.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sous-échantillonne la classe "none" pour limiter son déséquilibre.

    none_ratio = 1.5 signifie :
        nb_none <= 1.5 * nb_max_autre_classe
    """
    rng = np.random.default_rng(random_state)

    y = np.asarray(y)
    X = np.asarray(X)

    # indices none vs non-none
    none_idx = np.where(y == none_label)[0]
    other_idx = np.where(y != none_label)[0]

    if len(none_idx) == 0:
        return X, y

    # calcule le plafond
    _, other_counts = np.unique(y[other_idx], return_counts=True)
    if len(other_counts) == 0:
        # cas pathologique : tout est none
        return X, y

    max_other = int(other_counts.max())
    max_none = int(none_ratio * max_other)

    if len(none_idx) <= max_none:
        return X, y

    keep_none = rng.choice(none_idx, size=max_none, replace=False)
    keep_idx = np.concatenate([other_idx, keep_none])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


# =============================================================================
# Entraînement d'un fold (MLP)
# =============================================================================

def train_one_fold_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    balance_classes: bool = True,
    none_ratio: float = 1.5,
) -> Tuple[MLPClassifier, LabelEncoder, StandardScaler]:
    """
    Entraîne un MLP sur un fold.

    Retourne :
      - le modèle MLP
      - le LabelEncoder (classes -> entiers)
      - le StandardScaler
    """
    # éventuel downsampling de 'none'
    if balance_classes:
        n_before = len(y_train)
        X_train, y_train = downsample_none_class(
            X_train, y_train, none_label="none", none_ratio=none_ratio
        )
        print(f"  Balancing: {n_before} -> {len(y_train)} (none subsampled)")

    # encodage labels (string -> int)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_scaled, y_enc)

    return mlp, le, scaler


# =============================================================================
# Cross-validation GroupKFold
# =============================================================================

def run_cross_validation_mlp(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    n_folds: int = 5,
    balance_classes: bool = True,
    none_ratio: float = 1.5,
) -> Tuple[List[Dict[str, Any]], LabelEncoder]:
    """
    Exécute un GroupKFold speaker-independent (group = participant_id).
    Retourne la liste des résultats fold par fold, + un LabelEncoder global.
    """
    gkf = GroupKFold(n_splits=n_folds)

    fold_results: List[Dict[str, Any]] = []

    # LabelEncoder global (pour reporting cohérent)
    le_global = LabelEncoder()
    le_global.fit(y)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, participant_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_participants = np.unique(participant_ids[test_idx])
        print(f"\n  --- Fold {fold_idx + 1}/{n_folds} ---")
        print(f"  Test participants ({len(test_participants)}): {list(test_participants)}")
        print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")

        # entraînement fold (balancing + scaling + fit inside)
        mlp, le_fold, scaler = train_one_fold_mlp(
            X_train, y_train,
            balance_classes=balance_classes,
            none_ratio=none_ratio,
        )

        # prédiction — le_fold peut ne pas contenir toutes les classes
        X_test_scaled = scaler.transform(X_test)
        y_pred_enc = mlp.predict(X_test_scaled)
        y_pred = le_fold.inverse_transform(y_pred_enc)

        # métriques
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"  F1-macro: {f1_macro:.3f}  |  F1-weighted: {f1_weighted:.3f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "train_size": len(y_train),
            "test_size": len(y_test),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "test_participants": list(test_participants),
        })

    return fold_results, le_global


def print_cv_summary(fold_results: List[Dict[str, Any]], le: LabelEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    Affiche un résumé et retourne toutes les prédictions concaténées
    pour produire une matrice de confusion globale.
    """
    f1m = [r["f1_macro"] for r in fold_results]
    f1w = [r["f1_weighted"] for r in fold_results]

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\n  Mean F1-macro:    {np.mean(f1m):.3f} +/- {np.std(f1m):.3f}")
    print(f"  Mean F1-weighted: {np.mean(f1w):.3f} +/- {np.std(f1w):.3f}")

    all_y_true = np.concatenate([np.array(r["y_true"], dtype=object) for r in fold_results])
    all_y_pred = np.concatenate([np.array(r["y_pred"], dtype=object) for r in fold_results])

    print(f"\n  Per-class report (aggregated across folds):")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=le.classes_, zero_division=0,
    ))

    return all_y_true, all_y_pred


# =============================================================================
# Sauvegarde matrice de confusion
# =============================================================================

def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save a confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — MLP (aggregated CV)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix: {output_path}")


# =============================================================================
# Entraînement final sur tout le dataset (comme train_svm.py)
# =============================================================================

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    balance_classes: bool = True,
    none_ratio: float = 1.5,
) -> Tuple[MLPClassifier, LabelEncoder, StandardScaler]:
    """
    Entraîne un modèle final MLP sur toutes les données.
    Retourne le modèle + encoder + scaler pour déploiement.
    """
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL (all data)")
    print("=" * 60)

    if balance_classes:
        n_before = len(y)
        X, y = downsample_none_class(X, y, none_label="none", none_ratio=none_ratio)
        print(f"  Balancing: {n_before} -> {len(y)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Training MLP on {len(y)} samples...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_scaled, y_enc)
    print(f"  Done.")

    return mlp, le, scaler


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)

    output_dir = Path(cfg["paths"]["output_dir"])
    train_cfg = cfg.get("training", {})

    balance = train_cfg.get("balance_classes", True)
    none_ratio = train_cfg.get("none_ratio", 1.5)
    n_folds = train_cfg.get("n_folds", 5)

    # Paths to prepared data (read from train/ subdirectory)
    train_dir = output_dir / "train"
    dataset_csv = train_dir / "dataset_train.csv"
    embeddings_file = train_dir / "all_embeddings.npz"

    for f in [embeddings_file]:
        if not f.exists():
            print(f"ERREUR: {f} introuvable. Lancez d'abord: python prepare_data.py")
            sys.exit(1)

    mlp_dir = output_dir / "MLP_model"
    mlp_dir.mkdir(parents=True, exist_ok=True)

    tee = Tee(mlp_dir / "mlp_output.txt")
    sys.stdout = tee

    print("=" * 60)
    print("TRAIN MLP — GroupKFold cross-validation (speaker-independent)")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Chargement données (comme train_svm.py)
    # -------------------------------------------------------------------------
    print("\nChargement des donnees...")
    data = np.load(embeddings_file, allow_pickle=True)
    X = data["embeddings"]
    y = data["labels"]
    segment_ids = data.get("segment_ids", None)
    participant_ids = data["participant_ids"]

    print(f"  Embeddings:   {X.shape}")
    print(f"  Labels:       {len(y)}")
    print(f"  Participants: {len(np.unique(participant_ids))}")

    # -------------------------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------------------------
    print(f"\nLancement du {n_folds}-fold GroupKFold...")
    fold_results, le_cv = run_cross_validation_mlp(
        X, y, participant_ids,
        n_folds=n_folds,
        balance_classes=balance,
        none_ratio=none_ratio,
    )
    all_y_true, all_y_pred = print_cv_summary(fold_results, le_cv)

    # Matrice de confusion globale
    save_confusion_matrix(
        all_y_true, all_y_pred, le_cv.classes_,
        mlp_dir / "confusion_matrix_mlp.png",
    )

    # -------------------------------------------------------------------------
    # Modèle final (sur tout)
    # -------------------------------------------------------------------------
    mlp, le, scaler = train_final_model(
        X.copy(), y.copy(),
        balance_classes=balance,
        none_ratio=none_ratio,
    )

    # Sauvegarde artefacts
    joblib.dump(mlp, mlp_dir / "model_mlp.pkl")
    joblib.dump(le, mlp_dir / "label_encoder.pkl")
    joblib.dump(scaler, mlp_dir / "scaler.pkl")

    print(f"\n  Modele sauvegarde: {mlp_dir / 'model_mlp.pkl'}")
    print(f"  Scaler:           {mlp_dir / 'scaler.pkl'}")
    print(f"  Label encoder:    {mlp_dir / 'label_encoder.pkl'}")

    # Save CV results summary
    summary = {
        'n_folds': n_folds,
        'f1_macro_mean': float(np.mean([r['f1_macro'] for r in fold_results])),
        'f1_macro_std': float(np.std([r['f1_macro'] for r in fold_results])),
        'f1_weighted_mean': float(np.mean([r['f1_weighted'] for r in fold_results])),
        'f1_weighted_std': float(np.std([r['f1_weighted'] for r in fold_results])),
        'folds': [
            {
                'fold': r['fold'],
                'f1_macro': float(r['f1_macro']),
                'f1_weighted': float(r['f1_weighted']),
                'test_participants': r['test_participants'],
            }
            for r in fold_results
        ],
    }
    with open(mlp_dir / "cv_results_mlp.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  CV results:       {mlp_dir / 'cv_results_mlp.json'}")

    print(f"  Output log:       {mlp_dir / 'mlp_output.txt'}")

    print("\n" + "=" * 60)
    print("TRAINING TERMINE")
    print("=" * 60)

    tee.close()
