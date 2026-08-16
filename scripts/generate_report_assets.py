"""Generate README figures from result files produced by run_experiment.py."""

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


def main() -> None:
    model_results = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    threshold_results = pd.read_csv(RESULTS_DIR / "threshold_analysis.csv")
    final_metrics = json.loads((RESULTS_DIR / "final_test_metrics.json").read_text())

    FIGURES_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = model_results.sort_values("auprc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2563eb" if model == "xgboost" else "#94a3b8" for model in plot_df["model"]]
    ax.barh(plot_df["model"], plot_df["auprc"], color=colors)
    ax.set_xlim(0.65, 0.88)
    ax.set_xlabel("Mean five-fold CV AUPRC")
    ax.set_title("Leakage-safe model comparison")
    for y, value in enumerate(plot_df["auprc"]):
        ax.text(value + 0.002, y, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for column, color in [("precision", "#2563eb"), ("recall", "#16a34a"), ("f1", "#9333ea")]:
        ax.plot(threshold_results["threshold"], threshold_results[column], label=column.title(), color=color)
    ax.axvline(final_metrics["threshold"], color="#ef4444", linestyle="--", label="Selected threshold")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Decision threshold", ylabel="Validation metric")
    ax.set_title("Validation threshold trade-off for XGBoost")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=180)
    plt.close(fig)

    labels = ["Precision", "Recall", "F1", "AUPRC"]
    values = [final_metrics[key] for key in ("precision", "recall", "f1", "auprc")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#2563eb", "#16a34a", "#9333ea", "#ea580c"])
    ax.set(ylim=(0, 1), ylabel="Score", title="Final XGBoost holdout performance")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_performance.png", dpi=180)
    plt.close(fig)

    matrix = [
        [final_metrics["true_negatives"], final_metrics["false_positives"]],
        [final_metrics["false_negatives"], final_metrics["true_positives"]],
    ]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax)
    ax.set_xticklabels(["Predicted normal", "Predicted fraud"])
    ax.set_yticklabels(["Actual normal", "Actual fraud"], rotation=0)
    ax.set_title("Final holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
