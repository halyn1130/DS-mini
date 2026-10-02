from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def model_specs():
    impute = SimpleImputer(strategy="median", keep_empty_features=True)

    logistic = Pipeline(
        [
            ("imputer", clone(impute)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", clone(impute)),
            (
                "model",
                RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
            ),
        ]
    )
    return {
        "Dummy_majority": (
            Pipeline(
                [
                    ("imputer", clone(impute)),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            {},
        ),
        "Dummy_all_short": (
            Pipeline(
                [
                    ("imputer", clone(impute)),
                    ("model", DummyClassifier(strategy="constant", constant=1)),
                ]
            ),
            {},
        ),
        "LogisticRegression": (
            logistic,
            {"model__C": [0.01, 0.1, 1.0], "model__class_weight": [None, "balanced"]},
        ),
        "RandomForest": (
            forest,
            {
                "model__max_depth": [2, 4],
                "model__min_samples_leaf": [2, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }
