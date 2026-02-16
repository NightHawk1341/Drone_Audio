# =============================================================================
# TRAIN_SVM: SVM training with 5-fold speaker-independent cross-validation
# =============================================================================
"""
Trains an SVM classifier on pre-extracted wav2vec2 embeddings using
GroupKFold cross-validation grouped by participant.

Requires: output from prepare_data.py (train/dataset_train.csv + train/all_embeddings.npz)

Usage:
    python train_svm.py                    # uses config.yaml in same directory
    python train_svm.py path/to/config.yaml
"""

from pathlib import Path
import sys
import numpy as np
import yaml
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix


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
# CLASS BALANCING
# =============================================================================

def balance_none_class(X, y, groups, none_ratio: float = 1.5):
    """Subsample the 'none' class so it has at most none_ratio * largest-other-class samples."""
    mask_none = (y == 'none')
    mask_other = ~mask_none

    if not mask_none.any() or not mask_other.any():
        return X, y, groups

    # Find size of the largest non-none class
    _, other_counts = np.unique(y[mask_other], return_counts=True)
    max_other = int(other_counts.max())
    max_none = int(max_other * none_ratio)

    none_indices = np.where(mask_none)[0]
    if len(none_indices) <= max_none:
        return X, y, groups

    rng = np.random.RandomState(42)
    keep_none = rng.choice(none_indices, size=max_none, replace=False)
    other_indices = np.where(mask_other)[0]
    keep = np.sort(np.concatenate([other_indices, keep_none]))

    return X[keep], y[keep], groups[keep]


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(X, y, groups, n_folds: int = 5,
                         balance_classes: bool = True, none_ratio: float = 1.5):
    """Run n-fold GroupKFold CV and return per-fold metrics."""
    gkf = GroupKFold(n_splits=n_folds)
    le = LabelEncoder()
    le.fit(y)

    fold_results = []
    unique_groups = np.unique(groups)
    print(f"\n  Participants totaux: {len(unique_groups)}")
    print(f"  Classes: {list(le.classes_)}")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        test_participants = np.unique(groups[test_idx])
        print(f"\n  --- Fold {fold_idx + 1}/{n_folds} ---")
        print(f"  Test participants ({len(test_participants)}): {list(test_participants)}")
        print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")

        # Balance classes in train set
        if balance_classes:
            n_before = len(y_train)
            X_train, y_train, g_train = balance_none_class(
                X_train, y_train, g_train, none_ratio
            )
            print(f"  Balancing: {n_before} -> {len(y_train)} (none subsampled)")

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Encode labels
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        # Train SVM
        svm = SVC(
            kernel='rbf', C=10, gamma='scale',
            class_weight='balanced', random_state=42,
        )
        svm.fit(X_train_s, y_train_enc)

        # Evaluate
        y_pred = svm.predict(X_test_s)
        f1_mac = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)
        f1_wt = f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)

        print(f"  F1-macro: {f1_mac:.3f}  |  F1-weighted: {f1_wt:.3f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'f1_macro': f1_mac,
            'f1_weighted': f1_wt,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'test_participants': list(test_participants),
            'y_test_enc': y_test_enc,
            'y_pred': y_pred,
        })

    return fold_results, le


def print_cv_summary(fold_results, le):
    """Print aggregated cross-validation metrics."""
    f1_macros = [r['f1_macro'] for r in fold_results]
    f1_weights = [r['f1_weighted'] for r in fold_results]

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\n  Mean F1-macro:    {np.mean(f1_macros):.3f} +/- {np.std(f1_macros):.3f}")
    print(f"  Mean F1-weighted: {np.mean(f1_weights):.3f} +/- {np.std(f1_weights):.3f}")

    # Aggregate predictions for per-class report
    all_y_true = np.concatenate([r['y_test_enc'] for r in fold_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])

    print(f"\n  Per-class report (aggregated across folds):")
    print(classification_report(
        all_y_true, all_y_pred,
        target_names=le.classes_, zero_division=0,
    ))

    return all_y_true, all_y_pred


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save a confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — SVM (aggregated CV)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix: {output_path}")


# =============================================================================
# OUTPUT TEE
# =============================================================================

class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# =============================================================================
# FINAL MODEL
# =============================================================================

def train_final_model(X, y, balance_classes: bool = True, none_ratio: float = 1.5):
    """Train the final SVM on the entire dataset."""
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL (all data)")
    print("=" * 60)

    groups_dummy = np.zeros(len(y))  # not used, just for balance function signature
    if balance_classes:
        n_before = len(y)
        X, y, _ = balance_none_class(X, y, groups_dummy, none_ratio)
        print(f"  Balancing: {n_before} -> {len(y)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Training SVM on {len(y)} samples...")
    svm = SVC(
        kernel='rbf', C=10, gamma='scale',
        class_weight='balanced', random_state=42,
    )
    svm.fit(X_scaled, y_enc)
    print(f"  Done. Support vectors: {svm.n_support_.sum()}")

    return svm, le, scaler


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

    for f in [dataset_csv, embeddings_file]:
        if not f.exists():
            print(f"ERREUR: {f} introuvable. Lancez d'abord: python prepare_data.py")
            sys.exit(1)

    svm_dir = output_dir / "SVM_model"
    svm_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(svm_dir / "svm_output.txt")
    sys.stdout = tee

    print("=" * 60)
    print("TRAIN SVM — 5-fold GroupKFold cross-validation")
    print("=" * 60)

    # Load data
    print("\nChargement des donnees...")
    data = np.load(embeddings_file, allow_pickle=True)
    X = data['embeddings']
    y = data['labels']
    segment_ids = data['segment_ids']
    participant_ids = data['participant_ids']

    print(f"  Embeddings:   {X.shape}")
    print(f"  Labels:       {len(y)}")
    print(f"  Participants: {len(np.unique(participant_ids))}")

    # Cross-validation
    print(f"\nLancement du {n_folds}-fold GroupKFold...")
    fold_results, le_cv = run_cross_validation(
        X, y, participant_ids,
        n_folds=n_folds,
        balance_classes=balance,
        none_ratio=none_ratio,
    )
    all_y_true, all_y_pred = print_cv_summary(fold_results, le_cv)

    # Save confusion matrix
    save_confusion_matrix(
        all_y_true, all_y_pred, le_cv.classes_,
        svm_dir / "confusion_matrix_svm.png",
    )

    # Train final model
    svm, le, scaler = train_final_model(
        X.copy(), y.copy(),
        balance_classes=balance,
        none_ratio=none_ratio,
    )

    # Save artifacts
    joblib.dump(svm, svm_dir / "model_svm.pkl")
    joblib.dump(le, svm_dir / "label_encoder.pkl")
    joblib.dump(scaler, svm_dir / "scaler.pkl")

    print(f"\n  Modele sauvegarde: {svm_dir / 'model_svm.pkl'}")
    print(f"  Scaler:           {svm_dir / 'scaler.pkl'}")
    print(f"  Label encoder:    {svm_dir / 'label_encoder.pkl'}")

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
    import json
    with open(svm_dir / "cv_results_svm.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  CV results:       {svm_dir / 'cv_results_svm.json'}")

    print(f"  Output log:       {svm_dir / 'svm_output.txt'}")

    print("\n" + "=" * 60)
    print("TRAINING TERMINE")
    print("=" * 60)

    tee.close()
