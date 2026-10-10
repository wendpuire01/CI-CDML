import pytest

from src.data import load_data

N_FEATURES = 30          # nombre de variables attendues par le modèle
MIN_ROWS = 500           # en dessous, le jeu est probablement tronqué


@pytest.fixture(scope="module")
def df():
    return load_data()   # chargé une seule fois pour tout le fichier


def test_schema(df):
    # Le modèle attend 30 variables + la cible, ni plus ni moins
    assert df.shape[1] == N_FEATURES + 1
    assert "target" in df.columns


def test_enough_rows(df):
    assert len(df) >= MIN_ROWS


def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_target_is_binary(df):
    assert set(df["target"].unique()) == {0, 1}


def test_classes_not_too_imbalanced(df):
    # Si une classe passe sous 20 %, l'accuracy ne veut plus rien dire
    assert df["target"].value_counts(normalize=True).min() >= 0.20


def test_features_are_positive(df):
    # Ce sont des mesures physiques : une valeur négative = bug d'ingestion
    assert (df.drop(columns="target") >= 0).all().all()
