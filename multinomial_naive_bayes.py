from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_multinomial_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def  evaluate_multinomial_nb(model, X_test, y_test):
    y_pred = model.predict(X_test)
    naive_accuracy = accuracy_score(y_test, y_pred)
    naive_report = classification_report(y_test, y_pred)

    print("Multinomial Naive Bayes Classification accuracy in multimodal_naive_bayes:", naive_accuracy)
    return {
        naive_accuracy,
        naive_report
    }