import json
from pathlib import Path

import pytest

from src.train import train_and_evaluate

MIN_ACCURACY = 0.90      # seuil absolu, fixé avec le métier
TOLERANCE = 0.02         # baisse tolérée par rapport au modèle en production


@pytest.fixture(scope="module")
def trained():
    return train_and_evaluate(seed=42)   # graine fixe : résultat reproductible


def test_minimum_performance(trained):
    _, metrics = trained
    assert metrics["accuracy"] >= MIN_ACCURACY


def test_no_regression(trained):
    # On compare au score du dernier modèle validé, versionné dans le dépôt
    _, metrics = trained
    baseline = json.loads(Path("metrics/baseline.json").read_text())
    assert metrics["accuracy"] >= baseline["accuracy"] - TOLERANCE


def test_predictions_are_valid(trained):
    # Le modèle répond bien 0 ou 1, et une réponse par ligne
    model, _ = trained
    from src.data import load_data
    X = load_data().drop(columns="target").head(20)
    pred = model.predict(X)
    assert len(pred) == 20
    assert set(pred) <= {0, 1}
