import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_predict():
    """
    Trenuje model RandomForest na zbiorze Breast Cancer i zwraca predykcje oraz etykiety testowe.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred, y_test

def get_accuracy(y_pred, y_true):
    """
    Oblicza dokładność (accuracy) modelu.
    """
    return accuracy_score(y_true, y_pred)
