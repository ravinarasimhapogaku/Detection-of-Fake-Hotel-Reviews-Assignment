from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_multinomial_nb(X_train, y_train):
    """
    Train a Multinomial Naive Bayes classifier.

    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.

    Returns:
    model: Trained Multinomial Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def  evaluate_multinomial_nb(model, X_test, y_test):
    """
    Evaluate the Multinomial Naive Bayes model.

    Parameters:
    model: Trained Multinomial Naive Bayes model.
    X_test (array-like): Test feature data.
    y_test (array-like): True labels for test data.

    Returns:
    dict: Dictionary containing accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    naive_accuracy = accuracy_score(y_test, y_pred)
    naive_report = classification_report(y_test, y_pred)

    print("Multinomial Naive Bayes Classification accuracy in multimodal_naive_bayes:", naive_accuracy)
    return {
        naive_accuracy,
        naive_report
    }