"""Train Word2Vec embeddings and a logistic-regression classifier."""

import ast

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


LABELS = {"Normal": 0, "Depression": 1, "Suicidal": 2}


def parse_tokens(value):
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else []
        return [str(token) for token in parsed] if isinstance(parsed, list) else str(value).split()
    except (ValueError, SyntaxError):
        return str(value).split()


frame = pd.read_csv("datos/mental_health_cleaned.csv")
frame = frame[frame.status.isin(LABELS)].copy()
texts = frame.tokens.map(parse_tokens).tolist()
labels = frame.status.map(LABELS).to_numpy()
train_x, temp_x, train_y, temp_y = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=42
)
_, test_x, _, test_y = train_test_split(
    temp_x, temp_y, test_size=0.50, stratify=temp_y, random_state=42
)

word2vec = Word2Vec(
    sentences=train_x, vector_size=200, window=5, min_count=5,
    workers=8, sg=1, epochs=10, seed=42
)


def embed(documents):
    output = np.zeros((len(documents), word2vec.vector_size), dtype=np.float32)
    for index, document in enumerate(documents):
        vectors = [word2vec.wv[token] for token in document if token in word2vec.wv]
        if vectors:
            output[index] = np.mean(vectors, axis=0)
    return output


classifier = LogisticRegression(
    max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42
)
classifier.fit(embed(train_x), train_y)
predictions = classifier.predict(embed(test_x))
word2vec.save("modelos/vectorizador_word2vec.model")
joblib.dump(classifier, "modelos/modelo_word2vec_regresion_logistica.pkl")
print(f"Accuracy: {accuracy_score(test_y, predictions):.4f}")
print(f"Macro F1: {f1_score(test_y, predictions, average='macro'):.4f}")
