"""Entraînement et évaluation du modèle."""
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data


def build_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def train_and_evaluate(seed=42):
    df = load_data()
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    return model, metrics


if __name__ == "__main__":
    model, metrics = train_and_evaluate()
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print(json.dumps(metrics))
