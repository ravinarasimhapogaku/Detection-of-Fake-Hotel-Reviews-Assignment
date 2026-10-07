from preprocessing import load_and_preprocess
from multinomial_naive_bayes import train_multinomial_nb, evaluate_multinomial_nb
from gradient_boosting_model import train_gradient_boosting, evaluate_gradient_boosting
from decision_tree_baseline import train_decision_tree_baseline, evaluate_decision_tree_baseline
import pandas as pd
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test, vec = load_and_preprocess()

learning_rate = [0.01, 0.1, 0.2, 0.3]
max_depth = [2, 3, 4, 5]
n_estimators = [50, 100, 150, 200]

results = []

nb_model = train_multinomial_nb(
    X_train,
    y_train
)

naive_accuracy, naive_report = evaluate_multinomial_nb(
    nb_model,
    X_test,
    y_test
)

for lr in learning_rate:
    for md in max_depth:
        for ne in n_estimators:
            print(f"Training Gradient Boosting with learning_rate={lr}, max_depth={md}, n_estimators={ne}")
            gb_model = train_gradient_boosting(
                X_train,
                y_train,
                n_estimators=ne,
                learning_rate=lr,
                max_depth=md,
                random_state=40
            )

            gradient_boost_accuracy, gradient_boost_report = evaluate_gradient_boosting(
                gb_model,
                X_test,
                y_test
            )

            results.append({
                "learning_rate": lr,
                "max_depth": md,
                "n_estimators": ne,
                "accuracy": gradient_boost_accuracy
            })

results_df = pd.DataFrame(results)

print(results_df)

best_model = results_df.loc[results_df["accuracy"].idxmax()]

print("===== BEST GRADIENT BOOSTING MODEL =====")
print(best_model)

# Top ten gradient boosting models based on accuracy 
top_10 = results_df.sort_values(
    "accuracy",
    ascending=False
).head(10)

print('top_10', top_10)

top_10 = (
    results_df
    .sort_values("accuracy", ascending=False)
    .head(10)
    .copy()
)

# def plotGradientBoostingResults(top_10):
#     """
#     Plot the top 10 Gradient Boosting configurations based on accuracy.

#     Parameters:
#     top_10 (DataFrame): DataFrame containing the top 10 configurations.

#     Returns:
#     None
#     """
#     top_10["parameters"] = (
#         "lr=" + top_10["learning_rate"].astype(str)
#         + ", depth=" + top_10["max_depth"].astype(str)
#         + ", trees=" + top_10["n_estimators"].astype(str)
#     )

#     plt.figure(figsize=(10, 6))

#     plt.bar(
#         top_10["parameters"],
#         top_10["accuracy"]
#     )

#     plt.xlabel("Accuracy")
#     plt.ylabel("Parameters")
#     plt.title("Top 10 Gradient Boosting Configurations")

#     plt.gca().invert_yaxis()

#     plt.tight_layout()

#     plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
#     plt.show()

# plotGradientBoostingResults(top_10)


decision_tree = train_decision_tree_baseline(
    X_train,
    y_train,
    random_state=40
)

decision_tree_accuracy, decision_tree_report = evaluate_decision_tree_baseline(
    decision_tree,
    X_test,
    y_test
)

print("-----decision tree baseline-----")
print("Accuracy:", decision_tree_accuracy)
print(decision_tree_report)

naive_accuracy, naive_report = evaluate_multinomial_nb(
    nb_model,
    X_test,
    y_test
)

print("===== MODEL COMPARISON =====")
print('Multinomial Naive Bayes:', naive_accuracy)
print('---------------------------------')
print('Multimodal Naive bayes report:')
print(naive_report)
