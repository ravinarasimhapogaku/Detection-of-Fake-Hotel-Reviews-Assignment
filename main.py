from preprocessing import load_and_preprocess
from multinomial_naive_bayes import train_multinomial_nb, evaluate_multinomial_nb
from gradient_boosting_model import train_gradient_boosting, evaluate_gradient_boosting

X_train, X_test, y_train, y_test, vec = load_and_preprocess()

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
    y_train
)

gradient_boost_accuracy, gradient_boost_report = evaluate_gradient_boosting(
    gb_model,
    X_test,
    y_test
)

print("===== MODEL COMPARISON =====")
print('Multinomial Naive Bayes:', naive_accuracy)
print('---------------------------------')
print('Multimodal Naive bayes report:')
print(naive_report)

print('---------------------------------')
print('Gradient Boosting:', gradient_boost_accuracy)
print('---------------------------------')
print('Gradient Boosting report:')
print(gradient_boost_report)
