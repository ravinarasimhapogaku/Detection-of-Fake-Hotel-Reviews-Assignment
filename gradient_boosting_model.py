from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# check this HistGradientBoostingClassifier and remember to check the learned weights
def train_gradient_boosting( X_train,
    y_train,
    n_estimators,
    learning_rate,
    max_depth, random_state=40):
    """
    Train a Gradient Boosting classifier.

    Parameters:
    X_train 
    y_train 
    n_estimators
    learning_rate
    max_depth
    Returns:
    model: Trained Gradient Boosting model.
    """
    # Check hyperparameters and adjust them as needed , learning rate 0.1 and max depth of 3 looks best as of now, random_state is set for reproducibility
    model = GradientBoostingClassifier(n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=40)
    model.fit(X_train, y_train)
    return model

def evaluate_gradient_boosting(model, X_test, y_test):
    """
    Evaluate the Gradient Boosting model.

    Parameters:
    model: Trained Gradient Boosting model.
    X_test 
    y_test 

    Returns:
    dict: Dictionary containing accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    gradient_boost_accuracy = accuracy_score(y_test, y_pred)
    gradient_boost_report = classification_report(y_test, y_pred)
    
    return {
       gradient_boost_accuracy,
       gradient_boost_report
    }