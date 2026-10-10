from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

FIGURES_DIR = Path.cwd() / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, pos_label=1, average="binary"):
    kwargs = {"pos_label": pos_label} if average == "binary" else {}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0, **kwargs
    )
    return {"Precision": precision, "Recall": recall, "F1": f1}


def compare_models(y_test, predictions, pos_label=1, average="binary",
                   filename="model_comparison_metrics.png"):
    table = pd.DataFrame(
        {name: compute_metrics(y_test, pred, pos_label, average)
         for name, pred in predictions.items()}
    ).T.round(3)

    print("\n===== Model comparison =====")
    print(table.to_string())
    table.to_csv(FIGURES_DIR / "model_comparison_metrics.csv")

    metrics = table.columns.tolist()
    n_models = len(table)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (model_name, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + i * width - 0.4 + width / 2, row.values,
                      width, label=model_name)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision, recall and F1 by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=300)
    print(f"Saved chart to: {output_path}")
    plt.show()

    return table