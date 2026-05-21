"""Create experiment comparison chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("reports/experiments/experiment_results.csv")
OUTPUT_PATH = Path("reports/experiments/experiment_micro_f1_comparison.png")


def plot_results() -> None:
    """Plot Micro F1 score for each experiment."""
    df = pd.read_csv(RESULTS_PATH)

    plt.figure(figsize=(10, 6))
    plt.bar(df["experiment_name"], df["micro_f1"])
    plt.xlabel("Experiment")
    plt.ylabel("Micro F1")
    plt.title("MLflow Experiment Comparison by Micro F1")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)

    print(f"Chart saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_results()
