from scipy.stats import binomtest
import pandas as pd

# Mcnemar test

alpha = 0.01

def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):

    y_true = pd.Series(y_true).reset_index(drop=True)
    pred_a = pd.Series(pred_a).reset_index(drop=True)
    pred_b = pd.Series(pred_b).reset_index(drop=True)

    correct_a = (y_true == pred_a)
    correct_b = (y_true == pred_b)

    both_correct = int((correct_a & correct_b).sum())

    a_correct_b_wrong = int(
        (correct_a & ~correct_b).sum()
    )

    a_wrong_b_correct = int(
        (~correct_a & correct_b).sum()
    )

    both_wrong = int(
        (~correct_a & ~correct_b).sum()
    )

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
    # McNemar test, compare discordant pairs
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

    print(f"\n===== {name_a} vs {name_b} =====")
    print("\n2-by-2 contingency table:")
    print(table)
    print(f"\nDiscordant pairs: {b} and {c}")
    print(f"Exact McNemar p-value: {p_value:.6f}")

    if p_value < alpha:
        print(
            "Result: Significant difference at alpha = 0.01."
        )
    else:
        print(
            "Result: No statistically significant difference "
            "at alpha = 0.01."
        )

    return table, p_value