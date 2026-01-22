from sklearn.metrics import classification_report

# Lambda
accuracy_fn = lambda y_true, y_pred: (y_true == y_pred).mean()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_fn(y_test.values, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report