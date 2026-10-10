# Démo Thème 9 - CI/CD pour le ML

    pip install -r requirements.txt
    pytest -v                      # 9 tests verts

Pour montrer le blocage : dans src/train.py, remplacer le contenu de build_model() par
`return DecisionTreeClassifier(max_depth=1, random_state=0)`
(import : `from sklearn.tree import DecisionTreeClassifier`), puis commit + push.
test_no_regression échoue (0.9231 < 0.966) et le job docker est ignoré.

À adapter : src/data.py et src/train.py sont à remplacer par le code des thèmes 3 et 4.
Le Dockerfile lance src.api (l'API du thème 4), absente de ce dossier.
