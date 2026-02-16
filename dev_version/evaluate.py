# =============================================================================
# EVALUATE: Comprehensive evaluation of SVM vs MLP on the held-out test set
# =============================================================================
"""
Evaluates predictions from predict.py against ground truth (dataset_test.csv).

Per model: accuracy, F1-macro, F1-weighted, per-class F1, confusion matrix.
Side-by-side comparison table when both models are available.

Usage:
    python evaluate.py                          # evaluate all available predictions
    python evaluate.py path/to/config.yaml
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

COMMAND_CLASSES = [
    'backward', 'down', 'forward', 'left',
    'none', 'right', 'up', 'yawleft', 'yawright',
]


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
# PER-MODEL EVALUATION
# =============================================================================

def evaluate_model(y_true, y_pred, model_name, eval_dir):
    """Compute all metrics, save confusion matrix and JSON for one model."""
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n  {model_name.upper()} Results:")
    print(f"    Accuracy:    {acc:.3f}")
    print(f"    F1-macro:    {f1_mac:.3f}")
    print(f"    F1-weighted: {f1_wt:.3f}")

    # Per-class report
    report_str = classification_report(
        y_true, y_pred, labels=COMMAND_CLASSES, zero_division=0,
    )
    print(f"\n  Per-class report:")
    print(report_str)

    report_dict = classification_report(
        y_true, y_pred, labels=COMMAND_CLASSES, zero_division=0, output_dict=True,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=COMMAND_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=COMMAND_CLASSES, yticklabels=COMMAND_CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {model_name.upper()} (test set)')
    fig.tight_layout()
    cm_path = eval_dir / f"confusion_matrix_eval_{model_name}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix: {cm_path}")

    # JSON results
    results = {
        'model': model_name,
        'accuracy': float(acc),
        'f1_macro': float(f1_mac),
        'f1_weighted': float(f1_wt),
        'n_samples': int(len(y_true)),
        'per_class': {
            cls: {
                'precision': float(report_dict[cls]['precision']),
                'recall': float(report_dict[cls]['recall']),
                'f1-score': float(report_dict[cls]['f1-score']),
                'support': int(report_dict[cls]['support']),
            }
            for cls in COMMAND_CLASSES if cls in report_dict
        },
    }
    json_path = eval_dir / f"eval_{model_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON:     {json_path}")

    return results


# =============================================================================
# COMPARISON
# =============================================================================

def print_comparison(results_list):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (test set)")
    print("=" * 60)

    # Header
    header = f"  {'Metric':<20}"
    for r in results_list:
        header += f" {r['model'].upper():>10}"
    print(header)
    print("  " + "-" * (20 + 11 * len(results_list)))

    # Global metrics
    for metric_label, key in [
        ('Accuracy', 'accuracy'),
        ('F1-macro', 'f1_macro'),
        ('F1-weighted', 'f1_weighted'),
    ]:
        row = f"  {metric_label:<20}"
        for r in results_list:
            row += f" {r[key]:>10.3f}"
        print(row)

    # Per-class F1
    print()
    header2 = f"  {'Per-class F1':<20}"
    for r in results_list:
        header2 += f" {r['model'].upper():>10}"
    print(header2)
    print("  " + "-" * (20 + 11 * len(results_list)))

    for cls in COMMAND_CLASSES:
        row = f"  {cls:<20}"
        for r in results_list:
            f1 = r['per_class'].get(cls, {}).get('f1-score', 0.0)
            row += f" {f1:>10.3f}"
        print(row)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)

    output_dir = Path(cfg["paths"]["output_dir"])
    test_csv = output_dir / "test" / "dataset_test.csv"
    pred_dir = output_dir / "predictions"
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Set up Tee
    tee = Tee(eval_dir / "evaluation_output.txt")
    sys.stdout = tee

    # Validate
    if not test_csv.exists():
        print(f"ERREUR: {test_csv} introuvable. Lancez d'abord: python prepare_data.py")
        sys.exit(1)

    print("=" * 60)
    print("EVALUATE — Model evaluation on test set")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth...")
    df_truth = pd.read_csv(test_csv)
    print(f"  Segments:     {len(df_truth)}")
    print(f"  Participants: {df_truth['participant_id'].nunique()}")
    print(f"  Distribution:")
    for cmd, count in df_truth['command'].value_counts().items():
        pct = (count / len(df_truth)) * 100
        print(f"    {cmd:12s}: {count:4d} ({pct:5.1f}%)")

    # Discover available prediction files
    model_names = []
    for name in ['svm', 'mlp']:
        pred_path = pred_dir / f"predictions_{name}.csv"
        if pred_path.exists():
            model_names.append(name)

    if not model_names:
        print(f"\nERREUR: No prediction files in {pred_dir}. Run predict.py first.")
        tee.close()
        sys.exit(1)

    print(f"\nModels to evaluate: {', '.join(m.upper() for m in model_names)}")

    # Evaluate each model
    all_results = []
    for model_name in model_names:
        pred_path = pred_dir / f"predictions_{model_name}.csv"
        df_pred = pd.read_csv(pred_path)

        # Merge on segment_id
        merged = df_truth.merge(
            df_pred[['segment_id', 'predicted_command']],
            on='segment_id', how='inner',
        )

        if len(merged) < len(df_truth):
            n_missing = len(df_truth) - len(merged)
            print(f"\n  Warning: {n_missing} segments without predictions for {model_name.upper()}")

        y_true = merged['command'].values
        y_pred = merged['predicted_command'].values

        results = evaluate_model(y_true, y_pred, model_name, eval_dir)
        all_results.append(results)

    # Comparison table
    if len(all_results) > 1:
        print_comparison(all_results)

        # Save comparison JSON
        comparison = {}
        for r in all_results:
            comparison[r['model']] = {
                'accuracy': r['accuracy'],
                'f1_macro': r['f1_macro'],
                'f1_weighted': r['f1_weighted'],
                'n_samples': r['n_samples'],
            }
        comp_path = eval_dir / "comparison.json"
        with open(comp_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n  Comparison JSON: {comp_path}")

    print(f"\n  Output log:      {eval_dir / 'evaluation_output.txt'}")

    print("\n" + "=" * 60)
    print("EVALUATION TERMINEE")
    print("=" * 60)
    print(f"Results: {eval_dir}")

    tee.close()
