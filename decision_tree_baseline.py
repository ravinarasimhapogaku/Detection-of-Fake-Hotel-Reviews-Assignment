from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_decision_tree_baseline(
    X_train,
    y_train,
    max_depth=None,
    random_state=40
):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model


def evaluate_decision_tree_baseline(
    model,
    X_test,
    y_test
):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions
    )

    return accuracy, report