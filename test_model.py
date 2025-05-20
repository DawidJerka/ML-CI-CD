import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość predykcji jest większa od 0 i odpowiada liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions should not be empty."
    assert len(preds) == len(y_test), "Length of predictions should match length of y_test."

def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy wartości predykcji mieszczą się w zakresie 0–2 (dla zbioru Iris).
    """
    preds, _ = train_and_predict()
    unique_classes = set(preds)
    for c in unique_classes:
        assert c in [0, 1, 2], f"Prediction {c} is not a valid class."

def test_model_accuracy():
    """
    Test 4: Sprawdza, czy dokładność modelu jest co najmniej 70%.
    """
    preds, y_test = train_and_predict()
    acc = get_accuracy(preds, y_test)
    assert acc >= 0.7, f"Accuracy too low: {acc:.2f}"
