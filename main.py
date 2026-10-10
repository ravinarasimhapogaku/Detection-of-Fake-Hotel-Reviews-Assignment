from mcnemar_test import mcnemar_test
from model_comparison import compare_models
from preprocessing import load_and_preprocess
from multinomial_naive_bayes import train_multinomial_nb, evaluate_multinomial_nb
from gradient_boosting_model import train_gradient_boosting, evaluate_gradient_boosting
from decision_tree_baseline import train_decision_tree_baseline, evaluate_decision_tree_baseline
import pandas as pd
import matplotlib.pyplot as plt

from feature_importance_analysis import (
    top_linear_features,
    top_tree_features
)

X_train, X_test, y_train, y_test, vec = load_and_preprocess()

learning_rate = 0.01
max_depth = 3
n_estimators = 100

nb_model = train_multinomial_nb(
    X_train,
    y_train
)

naive_accuracy, naive_report = evaluate_multinomial_nb(
    nb_model,
    X_test,
    y_test
)

gb_model = train_gradient_boosting(
    X_train,
    y_train,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=40
)

gradient_boost_accuracy, gradient_boost_report = evaluate_gradient_boosting(
    gb_model,
    X_test,
    y_test
)

print("Gradient bosting model")
print(gradient_boost_accuracy)

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

print("Decision tree baseline")
print("Accuracy:", decision_tree_accuracy)
print(decision_tree_report)

naive_accuracy, naive_report = evaluate_multinomial_nb(
    nb_model,
    X_test,
    y_test
)

print("------Model comparison -------")
print('Multinomial Naive Bayes:', naive_accuracy)
print('---------------------------------')
print('Multimodal Naive bayes report:')
print(naive_report)

# Get predictions for exactly the same test set
naive_base_pred = nb_model.predict(X_test)
gradient_boost_pred = gb_model.predict(X_test)
decision_tree_pred = decision_tree.predict(X_test)

# Compare Gradient Boosting with each baseline
gb_nb_table, gb_nb_p = mcnemar_test(
    y_test, gradient_boost_pred, naive_base_pred,
    "Gradient Boosting", "Naive Bayes"
)

gb_dt_table, gb_dt_p = mcnemar_test(
    y_test, gradient_boost_pred, decision_tree_pred,
    "Gradient Boosting", "Decision Tree"
)

# Feature importance
AI_LABEL = 1     
HUMAN_LABEL = 0

# Linear feature analysis
nb_features = top_linear_features(
    nb_model, "Multinomial Naive Bayes", vec,
    AI_LABEL, HUMAN_LABEL
)
gb_features = top_tree_features(
    gb_model, "Gradient Boosting", vec
)

# Models comparison
metrics_table = compare_models(
    y_test,
    {
        "Naive Bayes": naive_base_pred,
        "Gradient Boosting": gradient_boost_pred,
        "Decision Tree": decision_tree_pred,
    },
    pos_label=1,  
)