"""Generate README charts and result files from the final experiment metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

MODEL_RESULTS = pd.DataFrame(
    [
        ("XGBoost", 0.820050, 0.902338, 0.858696, 0.841393),
        ("LightGBM", 0.812970, 0.905060, 0.855823, 0.831501),
        ("Class-weighted Random Forest", 0.714536, 0.944074, 0.812983, 0.826584),
        ("Class-weighted Logistic", 0.904449, 0.053387, 0.100743, 0.738550),
        ("SMOTE Logistic", 0.855075, 0.376081, 0.521491, 0.735547),
        ("Undersampled Logistic", 0.862155, 0.285858, 0.428446, 0.700363),
    ],
    columns=["model", "recall", "precision", "f1", "auprc"],
)

THRESHOLD_RESULTS = pd.DataFrame(
    [
        (0.05, 0.532051, 0.882979, 0.664000),
        (0.10, 0.734513, 0.882979, 0.801932),
        (0.15, 0.828283, 0.872340, 0.849741),
        (0.20, 0.843750, 0.861702, 0.852632),
        (0.25, 0.870968, 0.861702, 0.866310),
        (0.30, 0.869565, 0.851064, 0.860215),
        (0.35, 0.888889, 0.851064, 0.869565),
        (0.40, 0.898876, 0.851064, 0.874317),
        (0.45, 0.909091, 0.851064, 0.879121),
        (0.50, 0.908046, 0.840426, 0.872928),
        (0.918786, 0.951220, 0.829787, 0.886364),
    ],
    columns=["threshold", "precision", "recall", "f1"],
)

FINAL_TEST_METRICS = {
    "model": "XGBoost",
    "threshold": 0.9187861084938049,
    "precision": 0.9855072463768116,
    "recall": 0.7157894736842105,
    "f1": 0.8292682926829271,
    "auprc": 0.8214193893831631,
    "true_negatives": 56650,
    "false_positives": 1,
    "false_negatives": 27,
    "true_positives": 68,
}


def save_result_files() -> None:
    MODEL_RESULTS.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    THRESHOLD_RESULTS.to_csv(RESULTS_DIR / "threshold_summary.csv", index=False)
    (RESULTS_DIR / "final_test_metrics.json").write_text(
        json.dumps(FINAL_TEST_METRICS, indent=2) + "\n", encoding="utf-8"
    )


def plot_model_comparison() -> None:
    ordered = MODEL_RESULTS.sort_values("auprc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2563eb" if model == "XGBoost" else "#94a3b8" for model in ordered["model"]]
    ax.barh(ordered["model"], ordered["auprc"], color=colors)
    ax.set_xlim(0.65, 0.88)
    ax.set_xlabel("Mean cross-validation AUPRC")
    ax.set_title("Model comparison on five-fold stratified cross-validation")
    for y, value in enumerate(ordered["auprc"]):
        ax.text(value + 0.002, y, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def plot_threshold_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for column, color in [("precision", "#2563eb"), ("recall", "#16a34a"), ("f1", "#9333ea")]:
        ax.plot(THRESHOLD_RESULTS["threshold"], THRESHOLD_RESULTS[column], marker="o", label=column.title(), color=color)
    selected = FINAL_TEST_METRICS["threshold"]
    ax.axvline(selected, color="#ef4444", linestyle="--", label="Selected threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Validation metric")
    ax.set_title("Validation threshold trade-off for XGBoost")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_final_test_performance() -> None:
    values = [FINAL_TEST_METRICS[key] for key in ("precision", "recall", "f1", "auprc")]
    labels = ["Precision", "Recall", "F1", "AUPRC"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#2563eb", "#16a34a", "#9333ea", "#ea580c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Final XGBoost test performance")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_performance.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix() -> None:
    matrix = [
        [FINAL_TEST_METRICS["true_negatives"], FINAL_TEST_METRICS["false_positives"]],
        [FINAL_TEST_METRICS["false_negatives"], FINAL_TEST_METRICS["true_positives"]],
    ]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax)
    ax.set_xticklabels(["Predicted normal", "Predicted fraud"])
    ax.set_yticklabels(["Actual normal", "Actual fraud"], rotation=0)
    ax.set_title("Final test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    save_result_files()
    plot_model_comparison()
    plot_threshold_tradeoff()
    plot_final_test_performance()
    plot_confusion_matrix()


if __name__ == "__main__":
    main()
