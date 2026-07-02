"""Optimize class weights and decision thresholds for Suicidal F2.

The search only observes validation data. Test is evaluated once after every
model and threshold have been selected.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 42
LABELS = {"Normal": 0, "Depression": 1, "Suicidal": 2}
MIN_PRECISION = 0.60


def parse_tokens(value: object) -> list[str]:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        return list(map(str, parsed)) if isinstance(parsed, list) else str(value).split()
    except (SyntaxError, ValueError):
        return str(value).split()


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(scores)
    return values / values.sum(axis=1, keepdims=True)


def probabilities(model, features) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))
    return softmax(np.asarray(model.decision_function(features)))


def predict_with_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    result = np.argmax(probs[:, :2], axis=1)
    result[probs[:, 2] >= threshold] = 2
    return result


def metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    suicidal = y == 2
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "macro_f2": float(fbeta_score(y, predictions, beta=2, average="macro")),
        "suicidal_precision": float(precision_score(suicidal, predictions == 2)),
        "suicidal_recall": float(recall_score(suicidal, predictions == 2)),
        "suicidal_f2": float(fbeta_score(suicidal, predictions == 2, beta=2)),
    }


def best_threshold(y: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    # Coarse scan followed by a millesimal refinement around the winner.
    candidates = []
    for threshold in np.arange(0.05, 0.801, 0.01):
        score = metrics(y, predict_with_threshold(probs, float(threshold)))
        if score["suicidal_precision"] >= MIN_PRECISION:
            candidates.append((score["suicidal_f2"], score["macro_f1"], float(threshold), score))
    if not candidates:
        raise RuntimeError("No threshold met the validation precision constraint")
    coarse = max(candidates)
    refined = []
    for threshold in np.arange(max(0.01, coarse[2] - 0.015), min(0.99, coarse[2] + 0.0151), 0.001):
        score = metrics(y, predict_with_threshold(probs, float(threshold)))
        if score["suicidal_precision"] >= MIN_PRECISION:
            refined.append((score["suicidal_f2"], score["macro_f1"], float(threshold), score))
    winner = max(refined)
    return winner[2], winner[3]


def embed(documents: list[list[str]], word2vec: Word2Vec) -> np.ndarray:
    output = np.zeros((len(documents), word2vec.vector_size), dtype=np.float32)
    for index, document in enumerate(documents):
        vectors = [word2vec.wv[token] for token in document if token in word2vec.wv]
        if vectors:
            output[index] = np.mean(vectors, axis=0)
    return output


def main() -> None:
    frame = pd.read_csv("mental_health_cleaned.csv", usecols=["tokens", "status"])
    frame = frame[frame.status.isin(LABELS)].copy()
    frame["token_list"] = frame.tokens.map(parse_tokens)
    frame["text"] = frame.token_list.map(" ".join)
    label_counts = frame.groupby("text").status.transform("nunique")
    frame = frame[label_counts.eq(1)].drop_duplicates("text").reset_index(drop=True)
    indices = np.arange(len(frame))
    y = frame.status.map(LABELS).to_numpy()
    train_i, temporary_i, train_y, temporary_y = train_test_split(
        indices, y, test_size=0.30, stratify=y, random_state=SEED
    )
    validation_i, test_i, validation_y, test_y = train_test_split(
        temporary_i, temporary_y, test_size=0.50, stratify=temporary_y, random_state=SEED
    )

    vectorizer = joblib.load("vectorizador_tfidf.pkl")
    train_tfidf = vectorizer.transform(frame.text.to_numpy()[train_i])
    validation_tfidf = vectorizer.transform(frame.text.to_numpy()[validation_i])
    test_tfidf = vectorizer.transform(frame.text.to_numpy()[test_i])
    word2vec = Word2Vec.load("vectorizador_word2vec.model")
    token_lists = frame.token_list.tolist()
    train_w2v = embed([token_lists[i] for i in train_i], word2vec)
    validation_w2v = embed([token_lists[i] for i in validation_i], word2vec)
    test_w2v = embed([token_lists[i] for i in test_i], word2vec)

    datasets = {
        "logistic": (train_tfidf, validation_tfidf, test_tfidf),
        "svm": (train_tfidf, validation_tfidf, test_tfidf),
        "word2vec": (train_w2v, validation_w2v, test_w2v),
    }
    constructors = {
        "logistic": lambda c, weights: LogisticRegression(
            C=c, max_iter=2_000, solver="lbfgs", class_weight=weights, random_state=SEED
        ),
        "svm": lambda c, weights: LinearSVC(
            C=c, max_iter=5_000, class_weight=weights, random_state=SEED
        ),
        "word2vec": lambda c, weights: LogisticRegression(
            C=c, max_iter=2_000, solver="lbfgs", class_weight=weights, random_state=SEED
        ),
    }
    artifact_names = {
        "logistic": "modelo_regresion_logistica.pkl",
        "svm": "modelo_svm_lineal.pkl",
        "word2vec": "modelo_word2vec_regresion_logistica.pkl",
    }

    report = {"selection_metric": "suicidal_f2", "minimum_validation_precision": MIN_PRECISION,
              "models": {}}
    thresholds = {"lightgbm": json.loads(Path("modelo_lightgbm_metrics.json").read_text())["decision_threshold"]}

    for name in ("logistic", "svm", "word2vec"):
        train_x, validation_x, test_x = datasets[name]
        coarse_results = []
        # Broad search establishes the useful region rather than guessing one weight.
        for c in (0.25, 0.75):
            for depression_weight in (1.0, 1.5):
                for suicidal_weight in (1.75, 3.0):
                    weights = {0: 1.0, 1: depression_weight, 2: suicidal_weight}
                    model = constructors[name](c, weights).fit(train_x, train_y)
                    threshold, score = best_threshold(validation_y, probabilities(model, validation_x))
                    coarse_results.append((score["suicidal_f2"], score["macro_f1"], c,
                                           depression_weight, suicidal_weight, threshold, score, model))
        coarse_winner = max(coarse_results, key=lambda item: (item[0], item[1]))
        print(name, "coarse", coarse_winner[:7], flush=True)

        # Refine C and weights locally around the best coarse configuration.
        _, _, best_c, best_dw, best_sw, _, _, _ = coarse_winner
        refined_results = []
        c_values = sorted(set(max(0.02, best_c * factor) for factor in (0.85, 1.15)))
        dw_values = sorted(set(max(0.75, best_dw + delta) for delta in (-0.15, 0.15)))
        sw_values = sorted(set(max(1.0, best_sw + delta) for delta in (-0.3, 0.3)))
        for c in c_values:
            for depression_weight in dw_values:
                for suicidal_weight in sw_values:
                    weights = {0: 1.0, 1: depression_weight, 2: suicidal_weight}
                    model = constructors[name](c, weights).fit(train_x, train_y)
                    threshold, score = best_threshold(validation_y, probabilities(model, validation_x))
                    refined_results.append((score["suicidal_f2"], score["macro_f1"], c,
                                            depression_weight, suicidal_weight, threshold, score, model))
        winner = max(refined_results, key=lambda item: (item[0], item[1]))
        _, _, c, dw, sw, threshold, validation_score, model = winner
        test_score = metrics(test_y, predict_with_threshold(probabilities(model, test_x), threshold))
        joblib.dump(model, artifact_names[name])
        thresholds[name] = threshold
        report["models"][name] = {
            "C": c, "class_weight": {"Normal": 1.0, "Depression": dw, "Suicidal": sw},
            "decision_threshold": threshold, "validation": validation_score, "test": test_score,
            "coarse_candidates": len(coarse_results), "refined_candidates": len(refined_results),
        }
        print(name, report["models"][name], flush=True)

    Path("modelos_f2_optimized_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("model_decision_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
