import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .features import FEATURES
from .models import SEED, model_specs


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_short, has_long = np.any(np.asarray(y) == 1), np.any(np.asarray(y) == 0)
    return {
        "accuracy": float(np.mean(np.asarray(y) == np.asarray(pred))),
        "recall_short": recall_score(y, pred, zero_division=0) if has_short else np.nan,
        "precision_short": precision_score(y, pred, zero_division=0),
        "f1_short": f1_score(y, pred, zero_division=0) if has_short else np.nan,
        "f2_short": fbeta_score(y, pred, beta=2, zero_division=0)
        if has_short
        else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred)
        if has_short and has_long
        else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_model(pipeline, grid, X, y, fold):

    if not grid:
        return clone(pipeline).fit(X, y), {}
    inner_n = min(3, int(y.value_counts().min()))
    if inner_n < 2:
        raise ValueError(
            "At least two training records per class are needed for tuning."
        )
    inner = StratifiedKFold(n_splits=inner_n, shuffle=True, random_state=SEED + fold)
    scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    search = GridSearchCV(
        clone(pipeline),
        grid,
        scoring=scorer,
        cv=inner,
        n_jobs=1,
        error_score="raise",
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def evaluate_models(data, n_splits=5):

    X, y = data[FEATURES], data.target
    if not data.record_id.is_unique or set(y.unique()) != {0, 1}:
        raise ValueError("Expected unique records and both target classes.")
    if y.value_counts().min() < n_splits:
        raise ValueError("Too few records per class for the requested outer folds.")
    splits = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(X, y)
    )

    fold_rows, prediction_rows, importance_rows = [], [], []
    for name, (pipeline, grid) in model_specs().items():
        for fold, (train, test) in enumerate(splits, 1):
            assert set(data.iloc[train].record_id).isdisjoint(data.iloc[test].record_id)
            fitted, params = fit_model(
                pipeline, grid, X.iloc[train], y.iloc[train], fold
            )
            pred = fitted.predict(X.iloc[test])
            train_scores = metrics(y.iloc[train], fitted.predict(X.iloc[train]))
            scores = metrics(y.iloc[test], pred)
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_n": len(train),
                    "test_n": len(test),
                    "best_params": str(params),
                    **scores,
                    "train_f2": train_scores["f2_short"],
                    "train_recall": train_scores["recall_short"],
                    "f2_train_minus_validation": train_scores["f2_short"]
                    - scores["f2_short"],
                }
            )
            probabilities = fitted.predict_proba(X.iloc[test])[
                :, list(fitted.classes_).index(1)
            ]
            for pos, label, probability in zip(test, pred, probabilities):
                prediction_rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        "record_id": data.iloc[pos].record_id,
                        "batch": data.iloc[pos].batch,
                        "y_true": int(y.iloc[pos]),
                        "y_pred": int(label),
                        "p_short": float(probability),
                    }
                )
            model = fitted.named_steps["model"]
            weights = (
                model.coef_[0]
                if name == "LogisticRegression"
                else model.feature_importances_
                if name == "RandomForest"
                else None
            )
            if weights is not None:
                for feature, weight in zip(FEATURES, weights):
                    importance_rows.append(
                        {
                            "model": name,
                            "fold": fold,
                            "feature": feature,
                            "weight": float(weight),
                        }
                    )
        print(f"{name}: {n_splits} outer folds complete", flush=True)
    folds, predictions = pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)
    assert predictions.groupby(["model", "record_id"]).size().eq(1).all()
    overall, by_batch = [], []
    for name, group in predictions.groupby("model", sort=False):
        overall.append(
            {"model": name, "n": len(group), **metrics(group.y_true, group.y_pred)}
        )
        for batch, part in group.groupby("batch"):
            by_batch.append(
                {
                    "model": name,
                    "batch": batch,
                    "n": len(part),
                    "short_n": int(part.y_true.sum()),
                    **metrics(part.y_true, part.y_pred),
                }
            )
    return {
        "fold_metrics": folds,
        "oof_predictions": predictions,
        "overall_metrics": pd.DataFrame(overall),
        "batch_metrics": pd.DataFrame(by_batch),
        "feature_weights": pd.DataFrame(importance_rows),
    }
