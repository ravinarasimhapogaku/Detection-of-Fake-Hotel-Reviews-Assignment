import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"

# Use Gradient Boosting predictions
y_true = pd.Series(y_test).reset_index(drop=True)
predictions = pd.Series(gb_pred).reset_index(drop=True)

positive_mask = (y_true == POSITIVE_LABEL)
negative_mask = (y_true == NEGATIVE_LABEL)

positive_correct = int(
    ((y_true == predictions) & positive_mask).sum()
)
positive_incorrect = int(
    ((y_true != predictions) & positive_mask).sum()
)

negative_correct = int(
    ((y_true == predictions) & negative_mask).sum()
)
negative_incorrect = int(
    ((y_true != predictions) & negative_mask).sum()
)

sentiment_table = [
    [positive_correct, positive_incorrect],
    [negative_correct, negative_incorrect]
]

print("Positive vs negative review accuracy:")
print(pd.DataFrame(
    sentiment_table,
    index=["Positive reviews", "Negative reviews"],
    columns=["Correct", "Incorrect"]
))

# Chi-square test without continuity correction
chi2, p_value, dof, expected = chi2_contingency(
    sentiment_table,
    correction=False
)

# Use Fisher's exact test if expected counts are small
if (expected < 5).any():
    odds_ratio, p_value = fisher_exact(
        sentiment_table,
        alternative="two-sided"
    )
    print("Test used: Fisher's exact test")
else:
    print("Test used: Chi-square test of independence")

positive_accuracy = (
    positive_correct /
    (positive_correct + positive_incorrect)
    if positive_correct + positive_incorrect else float("nan")
)

negative_accuracy = (
    negative_correct /
    (negative_correct + negative_incorrect)
    if negative_correct + negative_incorrect else float("nan")
)

print(f"Positive review accuracy: {positive_accuracy:.4f}")
print(f"Negative review accuracy: {negative_accuracy:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < ALPHA:
    print("The accuracy difference is statistically significant.")
else:
    print("No statistically significant accuracy difference was found.")