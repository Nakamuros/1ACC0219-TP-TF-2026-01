"""
Inferencia multi-modelo para el clasificador de salud mental.

Uso:
    python predict.py "I feel so empty and hopeless every single day"
    python predict.py  # modo interactivo
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import joblib
import nltk
import numpy as np

warnings.filterwarnings("ignore")

# ── Preprocesamiento (mismo pipeline que el notebook) ────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOP = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [_LEMMA.lemmatize(w) for w in text.split() if w not in _STOP]
    return " ".join(tokens)


# ── Carga de modelos ──────────────────────────────────────────────────────────
def load_models() -> dict:
    models: dict = {}
    vectorizer = None

    # 1. TF-IDF vectorizer (compartido por LightGBM y LR)
    try:
        vectorizer = joblib.load("vectorizador_tfidf.pkl")
    except Exception as e:
        print(f"[warning] TF-IDF vectorizer no cargado: {e}")

    # 2. LightGBM + TF-IDF (mejor modelo clásico)
    if vectorizer is not None:
        try:
            lgb_model = joblib.load("modelo_lightgbm_salud_mental.pkl")
            models["LightGBM"] = ("tfidf", vectorizer, lgb_model)
        except Exception as e:
            print(f"[warning] LightGBM no cargado: {e}")

        # 3. Regresión Logística + TF-IDF
        try:
            lr_model = joblib.load("modelo_regresion_logistica.pkl")
            models["Logistic Regression"] = ("tfidf", vectorizer, lr_model)
        except Exception as e:
            print(f"[warning] Logistic Regression no cargado: {e}")

    # 4. Word2Vec + Regresión Logística
    try:
        from gensim.models import Word2Vec as GensimW2V
        wv  = GensimW2V.load("vectorizador_word2vec.model")
        wlr = joblib.load("modelo_word2vec_regresion_logistica.pkl")
        models["Word2Vec + LR"] = ("w2v", wv, wlr)
    except Exception as e:
        print(f"[warning] Word2Vec+LR no cargado: {e}")

    # 5. RoBERTa fine-tuned
    roberta_path = Path("modelo_mental_roberta_v2/best")
    if roberta_path.exists():
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(roberta_path))
            roberta   = AutoModelForSequenceClassification.from_pretrained(str(roberta_path))
            roberta.eval()
            models["RoBERTa"] = ("roberta", tokenizer, roberta)
        except Exception as e:
            print(f"[warning] RoBERTa no cargado: {e}")

    if not models:
        raise RuntimeError("No se pudo cargar ningún modelo. Verificá las dependencias.")

    return models


# ── Predicción ────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Normal", 1: "Depression", 2: "Suicidal"}
RISK_EMOJI = {"Normal": "✅", "Depression": "🟡", "Suicidal": "🔴"}


def predict_one(text: str, model_name: str, model_type: str, vec, clf) -> dict:
    clean = preprocess(text)

    if model_type == "tfidf":
        X = vec.transform([clean])
        label_idx = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0].tolist() if hasattr(clf, "predict_proba") else None

    elif model_type == "w2v":
        words = clean.split()
        vecs  = [vec.wv[w] for w in words if w in vec.wv]
        x_vec = np.mean(vecs, axis=0) if vecs else np.zeros(200)
        label_idx = int(clf.predict([x_vec])[0])
        proba = clf.predict_proba([x_vec])[0].tolist() if hasattr(clf, "predict_proba") else None

    elif model_type == "roberta":
        import torch
        tokenizer, model = vec, clf
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0].tolist()
        label_idx = int(logits.argmax())
        proba = probs

    return {
        "model": model_name,
        "label": LABEL_MAP[label_idx],
        "proba": proba,
    }


def display_results(text: str, results: list[dict]) -> None:
    print("\n" + "═" * 65)
    print(f"  TEXTO: {text[:80]}{'...' if len(text) > 80 else ''}")
    print("═" * 65)

    votes: dict[str, int] = {}
    for r in results:
        label = r["label"]
        votes[label] = votes.get(label, 0) + 1
        emoji = RISK_EMOJI[label]
        line  = f"  {r['model']:<28} → {emoji} {label:<12}"
        if r["proba"]:
            probs = "  " + "  ".join(
                f"{LABEL_MAP[i]}:{p*100:5.1f}%" for i, p in enumerate(r["proba"])
            )
            line += probs
        print(line)

    winner = max(votes, key=lambda k: votes[k])
    print("─" * 65)
    print(f"  CONSENSO ({len(results)} modelos):  {RISK_EMOJI[winner]} {winner}")
    print("═" * 65 + "\n")


def run(text: str, models: dict) -> None:
    results = []
    for name, (mtype, vec, clf) in models.items():
        try:
            results.append(predict_one(text, name, mtype, vec, clf))
        except Exception as e:
            print(f"[{name}] error: {e}")
    display_results(text, results)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Cargando modelos...", end=" ", flush=True)
    models = load_models()
    print(f"OK ({len(models)} modelos: {', '.join(models)})\n")

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]), models)
    else:
        print("Modo interactivo. Escribe 'salir' para terminar.\n")
        while True:
            try:
                text = input("Texto > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("salir", "exit", "quit"):
                break
            run(text, models)


if __name__ == "__main__":
    main()
