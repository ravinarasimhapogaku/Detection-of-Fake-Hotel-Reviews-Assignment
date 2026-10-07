from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# check this HistGradientBoostingClassifier and remember to check the learned weights
def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting classifier.

    Parameters:
    X_train 
    y_train 

    Returns:
    model: Trained Gradient Boosting model.
    """
    # check hyperparameters and adjust them as needed
    model = GradientBoostingClassifier(  n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42)
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