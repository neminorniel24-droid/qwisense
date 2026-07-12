"""
QwiSense — Classical Baseline Models
SVM and Random Forest classifiers for comparison against the quantum VQC.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from preprocessing.csi_loader import load_dataset, LABELS
from preprocessing.pipeline import CSIPipeline


def train_evaluate(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    """Train and evaluate a classical model. Returns metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    f1_fall = f1_score(y_test, y_pred, average=None, labels=[3])[0]
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy       : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1 (weighted)  : {f1:.4f}")
    print(f"  F1 (Fall only) : {f1_fall:.4f}  ← key metric")
    print(f"\n{classification_report(y_test, y_pred, target_names=list(LABELS.values()))}")

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "f1_weighted": f1,
        "f1_fall": f1_fall,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str = None):
    """Plot and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(LABELS.values()),
        yticklabels=list(LABELS.values()),
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()


def plot_comparison(results: list, save_path: str = None):
    """Bar chart comparing classical models (to be extended with VQC later)."""
    names    = [r["name"] for r in results]
    accs     = [r["accuracy"] for r in results]
    f1_falls = [r["f1_fall"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accs,     width, label="Accuracy",      color="#378ADD", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_falls, width, label="F1 (Fall det.)", color="#E24B4A", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("QwiSense — Model comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=10)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=10)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, label="80% threshold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()
    return fig


def run_baselines(n_samples: int = 600, random_seed: int = 42):
    """Full baseline training pipeline."""
    # Load data
    X_raw, y, meta = load_dataset(n_synthetic=n_samples, random_seed=random_seed)
    print(f"\n[Dataset] Source: {meta['source']} | Samples: {meta['n_samples']}")

    # Preprocess
    pipe   = CSIPipeline()
    X_feat = pipe.fit_transform(X_raw)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")

    # Models
    models = [
        (SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=random_seed), "SVM (RBF)"),
        (RandomForestClassifier(n_estimators=100, random_state=random_seed), "Random Forest"),
    ]

    results = []
    for model, name in models:
        r = train_evaluate(model, X_train, X_test, y_train, y_test, name)
        results.append(r)

        os.makedirs("results", exist_ok=True)
        plot_confusion_matrix(
            r["confusion_matrix"],
            title=f"Confusion matrix — {name}",
            save_path=f"results/cm_{name.lower().replace(' ', '_')}.png"
        )

    # Comparison plot
    plot_comparison(results, save_path="results/classical_comparison.png")

    return results, X_train, X_test, y_train, y_test, pipe


if __name__ == "__main__":
    results, *_ = run_baselines()
    print("\n[Done] Classical baselines complete. Results saved to results/")
