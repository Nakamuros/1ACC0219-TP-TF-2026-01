"""Retrain LR, SVM and Word2Vec on the exact deduplicated LightGBM split."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 42
LABELS = {"Normal": 0, "Depression": 1, "Suicidal": 2}


def parse_tokens(value: object) -> list[str]:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        return list(map(str, parsed)) if isinstance(parsed, list) else str(value).split()
    except (SyntaxError, ValueError):
        return str(value).split()


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
    ambiguous_rows = int((label_counts > 1).sum())
    frame = frame[label_counts.eq(1)].drop_duplicates("text").reset_index(drop=True)

    indices = np.arange(len(frame))
    labels = frame.status.map(LABELS).to_numpy()
    train_i, temporary_i, train_y, temporary_y = train_test_split(
        indices, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    validation_i, test_i, validation_y, test_y = train_test_split(
        temporary_i, temporary_y, test_size=0.50,
        stratify=temporary_y, random_state=SEED,
    )
    texts = frame.text.to_numpy()
    token_lists = frame.token_list.tolist()

    vectorizer = TfidfVectorizer(
        max_features=20_000, ngram_range=(1, 2), min_df=3,
        max_df=0.95, sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(texts[train_i])
    validation_tfidf = vectorizer.transform(texts[validation_i])
    test_tfidf = vectorizer.transform(texts[test_i])

    model_specs = {
        "logistic_regression": [
            LogisticRegression(C=c, max_iter=2_000, class_weight="balanced", solver="lbfgs", random_state=SEED)
            for c in (0.5, 1.0, 2.0)
        ],
        "linear_svm": [
            LinearSVC(C=c, class_weight="balanced", max_iter=5_000, random_state=SEED)
            for c in (0.5, 1.0, 2.0)
        ],
    }
    report: dict[str, object] = {
        "selection_metric": "macro_f1",
        "data": {"usable_rows": len(frame), "ambiguous_rows_removed": ambiguous_rows,
                 "train_rows": len(train_i), "validation_rows": len(validation_i), "test_rows": len(test_i)},
        "models": {},
    }
    winners = {}
    for name, candidates in model_specs.items():
        experiments = []
        for model in candidates:
            model.fit(train_tfidf, train_y)
            validation = metrics(validation_y, model.predict(validation_tfidf))
            experiments.append({"C": model.C, "validation": validation})
        best_index = int(np.argmax([item["validation"]["macro_f1"] for item in experiments]))
        winner = candidates[best_index]
        winners[name] = winner
        report["models"][name] = {
            "best_C": winner.C,
            "validation": experiments[best_index]["validation"],
            "test": metrics(test_y, winner.predict(test_tfidf)),
            "experiments": experiments,
        }

    train_tokens = [token_lists[i] for i in train_i]
    validation_tokens = [token_lists[i] for i in validation_i]
    test_tokens = [token_lists[i] for i in test_i]
    word2vec = Word2Vec(
        sentences=train_tokens, vector_size=200, window=5, min_count=5,
        workers=8, sg=1, epochs=10, seed=SEED,
    )
    train_embeddings = embed(train_tokens, word2vec)
    validation_embeddings = embed(validation_tokens, word2vec)
    test_embeddings = embed(test_tokens, word2vec)
    w2v_experiments = []
    w2v_candidates = []
    for c in (0.5, 1.0, 2.0):
        model = LogisticRegression(
            C=c, max_iter=2_000, class_weight="balanced", solver="lbfgs", random_state=SEED
        ).fit(train_embeddings, train_y)
        validation = metrics(validation_y, model.predict(validation_embeddings))
        w2v_candidates.append(model)
        w2v_experiments.append({"C": c, "validation": validation})
    best_index = int(np.argmax([item["validation"]["macro_f1"] for item in w2v_experiments]))
    w2v_winner = w2v_candidates[best_index]
    report["models"]["word2vec_logistic_regression"] = {
        "best_C": w2v_winner.C,
        "validation": w2v_experiments[best_index]["validation"],
        "test": metrics(test_y, w2v_winner.predict(test_embeddings)),
        "experiments": w2v_experiments,
    }

    joblib.dump(vectorizer, "vectorizador_tfidf.pkl")
    joblib.dump(winners["logistic_regression"], "modelo_regresion_logistica.pkl")
    joblib.dump(winners["linear_svm"], "modelo_svm_lineal.pkl")
    word2vec.save("vectorizador_word2vec.model")
    joblib.dump(w2v_winner, "modelo_word2vec_regresion_logistica.pkl")
    Path("modelos_baseline_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
