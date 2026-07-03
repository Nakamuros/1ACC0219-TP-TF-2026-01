"""
Setup del proyecto Mental Health Text Classifier.
Ejecutar UNA VEZ despues de clonar el repositorio:

    python setup.py

Hace todo automaticamente:
  1. Instala las dependencias de requirements.txt
  2. Descarga los recursos de NLTK
  3. Descarga los pesos del modelo RoBERTa (~480MB) desde HuggingFace
"""

import subprocess
import sys
from pathlib import Path

def paso(n, texto):
    print(f"\n[{n}/3] {texto}")
    print("─" * 50)

# ── 1. Instalar dependencias ───────────────────────────────────────────────────
paso(1, "Instalando dependencias...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
print("OK — dependencias instaladas.")

# ── 2. Recursos NLTK ──────────────────────────────────────────────────────────
paso(2, "Descargando recursos NLTK...")
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
print("OK — stopwords y wordnet listos.")

# ── 3. Pesos de RoBERTa ───────────────────────────────────────────────────────
paso(3, "Verificando modelo RoBERTa...")
DEST = Path("modelo_mental_roberta_v2/best")
REPO = "treborDev/mhtc-roberta"

if (DEST / "model.safetensors").exists():
    print(f"Modelo ya descargado en '{DEST}'. Nada que hacer.")
else:
    print(f"Descargando desde HuggingFace ({REPO}) ~480MB...")
    from huggingface_hub import snapshot_download
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO, local_dir=str(DEST))
    print(f"OK — modelo guardado en '{DEST}'.")

# ── Listo ─────────────────────────────────────────────────────────────────────
print("\n" + "═" * 50)
print("Setup completo. El proyecto esta listo para usar.")
print("  Prueba rapida:  python predict.py \"I feel hopeless\"")
print("═" * 50)
