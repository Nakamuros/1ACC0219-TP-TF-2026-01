"""Tune LightGBM on validation data and evaluate the winner once on test."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


SEED = 42
LABELS = {"Normal": 0, "Depression": 1, "Suicidal": 2}
PARAMETER_GRID = [
    {"depression_weight": 1.00, "suicidal_weight": 2.25, "num_leaves": 31},
    {"depression_weight": 1.25, "suicidal_weight": 2.50, "num_leaves": 31},
    {"depression_weight": 1.50, "suicidal_weight": 3.00, "num_leaves": 31},
]


def rebuild_text(value: object) -> str:
    try:
        tokens = ast.literal_eval(value) if isinstance(value, str) else value
        return " ".join(map(str, tokens)) if isinstance(tokens, list) else str(value)
    except (SyntaxError, ValueError):
        return str(value)


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Prioritize Suicidal above a validated probability threshold."""
    predictions = np.argmax(probabilities[:, :2], axis=1)
    predictions[probabilities[:, 2] >= threshold] = 2
    return predictions


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    suicidal = labels == 2
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "macro_f2": float(fbeta_score(labels, predictions, beta=2, average="macro")),
        "suicidal_precision": float(precision_score(suicidal, predictions == 2)),
        "suicidal_recall": float(recall_score(suicidal, predictions == 2)),
        "suicidal_f2": float(fbeta_score(suicidal, predictions == 2, beta=2)),
    }


def main() -> None:
    frame = pd.read_csv("datos/mental_health_cleaned.csv", usecols=["tokens", "status"])
    frame = frame[frame.status.isin(LABELS)].copy()
    frame["text"] = frame.tokens.map(rebuild_text)

    # An identical input must never occur on both sides of the split. Ambiguous
    # texts carrying more than one label cannot provide a learnable target.
    label_counts = frame.groupby("text").status.transform("nunique")
    ambiguous_rows = int((label_counts > 1).sum())
    frame = frame[label_counts.eq(1)].drop_duplicates("text").reset_index(drop=True)

    texts = frame.text.to_numpy()
    labels = frame.status.map(LABELS).to_numpy()
    train_x, temporary_x, train_y, temporary_y = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        temporary_x, temporary_y, test_size=0.50,
        stratify=temporary_y, random_state=SEED,
    )

    vectorizer = TfidfVectorizer(
        max_features=20_000, ngram_range=(1, 2), min_df=3,
        max_df=0.95, sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_x)
    validation_features = vectorizer.transform(validation_x)
    test_features = vectorizer.transform(test_x)

    experiments: list[dict[str, object]] = []
    candidates: list[tuple[float, lgb.LGBMClassifier, float, dict[str, object]]] = []
    for parameters in PARAMETER_GRID:
        model = lgb.LGBMClassifier(
            n_estimators=2_000, learning_rate=0.03,
            num_leaves=parameters["num_leaves"], min_child_samples=30,
            class_weight={
                0: 1.0, 1: parameters["depression_weight"],
                2: parameters["suicidal_weight"],
            },
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.3,
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
        model.fit(
            train_features, train_y,
            eval_set=[(validation_features, validation_y)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        probabilities = model.predict_proba(validation_features)
        for threshold in np.arange(0.25, 0.551, 0.01):
            predictions = threshold_predictions(probabilities, float(threshold))
            result = metrics(validation_y, predictions)
            # Optimize the safety-oriented metric while preventing a collapse
            # in precision caused by predicting almost everything as Suicidal.
            if result["suicidal_precision"] >= 0.60:
                candidates.append((result["suicidal_f2"], model, float(threshold), parameters))
        experiments.append({
            "parameters": parameters, "best_iteration": model.best_iteration_,
            "argmax_validation": metrics(validation_y, np.argmax(probabilities, axis=1)),
        })

    if not candidates:
        raise RuntimeError("No candidate met the minimum Suicidal precision of 0.60")
    _, best_model, best_threshold, best_parameters = max(candidates, key=lambda item: item[0])
    validation_predictions = threshold_predictions(
        best_model.predict_proba(validation_features), best_threshold
    )
    test_predictions = threshold_predictions(best_model.predict_proba(test_features), best_threshold)

    report = {
        "selection_metric": "suicidal_f2",
        "minimum_suicidal_precision": 0.60,
        "decision_threshold": best_threshold,
        "best_parameters": best_parameters,
        "best_iteration": best_model.best_iteration_,
        "data": {
            "usable_rows": len(frame), "ambiguous_rows_removed": ambiguous_rows,
            "train_rows": len(train_y), "validation_rows": len(validation_y),
            "test_rows": len(test_y),
        },
        "validation": metrics(validation_y, validation_predictions),
        "test": metrics(test_y, test_predictions),
        "experiments": experiments,
    }
    # Keep the legacy 12k-feature vectorizer used by LR/SVM untouched.
    joblib.dump(vectorizer, "modelos/vectorizador_lightgbm_tfidf.pkl")
    joblib.dump(best_model, "modelos/modelo_lightgbm_salud_mental.pkl")
    Path("modelos/modelo_lightgbm_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
