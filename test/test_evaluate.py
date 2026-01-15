import sys
import os

sys.path.append(os.path.abspath(".."))

from src.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_evaluate_model_runs():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[1, 2]])
    y_test = np.array([0])

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy, report = evaluate_model(model, X_test, y_test)

    assert accuracy >= 0
    assert isinstance(report, str)