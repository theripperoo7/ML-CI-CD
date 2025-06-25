# test_model.py

import pytest
from app import train_and_predict, get_accuracy

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość predykcji > 0 i pasuje do liczby próbek testowych.
    """
    preds, y_true = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_true), "Length of predictions should match true labels."

def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy predykcje są w zakresie klas [0, 1, 2].
    """
    preds, _ = train_and_predict()
    assert all(p in [0, 1, 2] for p in preds), "All predictions should be 0, 1 or 2."

def test_model_accuracy():
    """
    Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    preds, y_true = train_and_predict()
    acc = get_accuracy(preds, y_true)
    assert acc >= 0.7, f"Model accuracy should be >= 0.7, got {acc:.2f}"
