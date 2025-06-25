# model.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_predict():
    """
    Trenuje model na zbiorze Iris i zwraca predykcje oraz prawdziwe etykiety.
    """
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test

def get_accuracy(preds, y_true):
    """
    Oblicza dokładność predykcji.
    """
    return accuracy_score(y_true, preds)
