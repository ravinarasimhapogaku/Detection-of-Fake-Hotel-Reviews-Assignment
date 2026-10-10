
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path.cwd() / "model_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

"Extract features associated with AI-generated and human-written reviews for MultinomialNB or LogisticRegression"

def top_linear_features(
    model,
    model_name,
    vectorizer,
    ai_label,
    human_label,
    top_n=5
):
   
    features = vectorizer.get_feature_names_out()
    
    classes = list(model.classes_)

    if ai_label not in classes or human_label not in classes:
        raise ValueError(
            f"Labels {ai_label!r} and {human_label!r} "
            f"not found in model classes {classes}"
        )

    ai_index = classes.index(ai_label)
    human_index = classes.index(human_label)

    if hasattr(model, "feature_log_prob_"):
        # Multinomial Naive Bayes
        scores = (
            model.feature_log_prob_[ai_index]
            - model.feature_log_prob_[human_index]
        )
       # For logistic regression
    elif hasattr(model, "coef_"):
        scores = model.coef_[0].copy()

        if model.classes_[1] != ai_label:
            scores = -scores

    else:
        raise ValueError(
            "Model must be MultinomialNB or LogisticRegression."
        )

    ai_indices = np.argsort(scores)[-top_n:][::-1]
    human_indices = np.argsort(scores)[:top_n]

    results = pd.DataFrame({
        "Model": [model_name] * (2 * top_n),
        "Class": (
            ["AI generated"] * top_n
            + ["Human written"] * top_n
        ),
        "Feature": np.concatenate([
            features[ai_indices],
            features[human_indices]
        ]),
        "Score": np.concatenate([
            scores[ai_indices],
            scores[human_indices]
        ])
    })

    print(f"\n----{model_name} ----")
    print(results.to_string(index=False))

    output_path = RESULTS_DIR / (
        model_name.lower().replace(" ", "_")
        + "_top_features.csv"
    )
    results.to_csv(output_path, index=False)

    print(f"Saved results to: {output_path}")

    return results

"Extract the highest-importance features from a tree model."

def top_tree_features(
    model,
    model_name,
    vectorizer,
    top_n=5
):

    features = vectorizer.get_feature_names_out()

    results = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    results = results.sort_values(
        "Importance",
        ascending=False
    ).head(top_n).copy()

    results.insert(0, "Model", model_name)

    print(f"\n----{model_name}: Top {top_n} features ----")
    print(results.to_string(index=False))

    output_path = RESULTS_DIR / (
        model_name.lower().replace(" ", "_")
        + "_top_features.csv"
    )
    results.to_csv(output_path, index=False)

    print(f"Saved results to: {output_path}")

    return results