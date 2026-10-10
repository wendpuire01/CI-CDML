
from sklearn.datasets import load_breast_cancer


def load_data():
    """Retourne un DataFrame : 30 variables numériques + la colonne 'target'."""
    return load_breast_cancer(as_frame=True).frame


#Bonjour