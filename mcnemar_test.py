
from scipy.stats import binomtest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

alpha = 0.01

# Folder to save all results
os.makedirs("mcnemar_results", exist_ok=True)

# Store summary results for all comparisons
mcnemar_summary = []


def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):

    y_true = pd.Series(y_true).reset_index(drop=True)
    pred_a = pd.Series(pred_a).reset_index(drop=True)
    pred_b = pd.Series(pred_b).reset_index(drop=True)

    correct_a = (y_true == pred_a)
    correct_b = (y_true == pred_b)

    both_correct = int((correct_a & correct_b).sum())
    a_correct_b_wrong = int((correct_a & ~correct_b).sum())
    a_wrong_b_correct = int((~correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())

    table = pd.DataFrame(
        [
            [both_correct, a_correct_b_wrong],
            [a_wrong_b_correct, both_wrong]
        ],
        index=[
            f"{name_a} correct",
            f"{name_a} incorrect"
        ],
        columns=[
            f"{name_b} correct",
            f"{name_b} incorrect"
        ]
    )

    # Exact McNemar test
    b = a_correct_b_wrong
    c = a_wrong_b_correct

    if b + c == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=min(b, c),
            n=b + c,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    significant = p_value < alpha

    conclusion = (
        "Significant difference"
        if significant
        else "No significant difference"
    )

    # Print the contingency table and result
    print(f"\n----- {name_a} vs {name_b} -----")
    print(table)
    print(f"Discordant pairs: {b} and {c}")
    print(f"p-value: {p_value:.6g}")
    print(f"Conclusion at alpha={alpha}: {conclusion}")

    # Plot heatmap
    plt.figure(figsize=(8, 5))

    ax = sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar=False
    )

    ax.set_title(
        f"McNemar Test: {name_a} vs {name_b}\n"
        f"p-value = {p_value:.6g} | "
        f"alpha = {alpha} | {conclusion}"
    )

    ax.set_xlabel(name_b)
    ax.set_ylabel(name_a)

    plt.tight_layout()

    BASE_DIR = Path(__file__).resolve().parent

    # Exact output directory
    RESULTS_DIR = BASE_DIR / "mcnemar_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = (
        f"mcnemar_results/{name_a}_vs_{name_b}"
        .replace(" ", "_")
        .replace("/", "_")
        + ".png"
    )

    plt.savefig(RESULTS_DIR/filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Save the individual 2 by 2 table
    table.to_csv(RESULTS_DIR/filename.replace(".png", "_table.csv"))

    # Save summary information
    mcnemar_summary.append({
        "Model A": name_a,
        "Model B": name_b,
        "Both correct": both_correct,
        "A correct, B incorrect": b,
        "A incorrect, B correct": c,
        "Both incorrect": both_wrong,
        "p-value": p_value,
        "alpha": alpha,
        "Significant": significant,
        "Conclusion": conclusion
    })

    return table, p_value