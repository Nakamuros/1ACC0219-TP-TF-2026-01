"""
Setup y arranque del proyecto Mental Health Text Classifier.
Ejecutar UNA VEZ despues de clonar el repositorio:

    python setup.py

Hace todo automaticamente:
  1. Instala dependencias Python (requirements.txt + backend)
  2. Descarga recursos NLTK
  3. Descarga pesos del modelo RoBERTa (~480MB) desde HuggingFace
  4. Instala dependencias del frontend (npm install)
  5. Levanta el backend (FastAPI) y el frontend (React/Vite) en paralelo

Al terminar abre: http://localhost:5173
"""

import subprocess
import sys
import shutil
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def paso(n, total, texto):
    print(f"\n[{n}/{total}] {texto}")
    print("─" * 50)

def run(cmd, **kwargs):
    subprocess.check_call(cmd, **kwargs)

TOTAL = 5

# ── 1. Dependencias Python ────────────────────────────────────────────────────
paso(1, TOTAL, "Instalando dependencias Python...")
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt", "-q"])
print("OK — dependencias Python instaladas.")

# ── 2. Recursos NLTK ──────────────────────────────────────────────────────────
paso(2, TOTAL, "Descargando recursos NLTK...")
import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
print("OK — stopwords y wordnet listos.")

# ── 3. Pesos de RoBERTa ───────────────────────────────────────────────────────
paso(3, TOTAL, "Verificando modelo RoBERTa...")
DEST = ROOT / "modelo_mental_roberta_v2" / "best"
REPO = "treborDev/mhtc-roberta"

if (DEST / "model.safetensors").exists():
    print(f"Modelo ya descargado. Nada que hacer.")
else:
    print(f"Descargando desde HuggingFace ({REPO}) ~480MB...")
    from huggingface_hub import snapshot_download
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO, local_dir=str(DEST))
    print(f"OK — modelo guardado en '{DEST}'.")

# ── 4. Frontend (npm install) ─────────────────────────────────────────────────
paso(4, TOTAL, "Instalando dependencias del frontend...")
npm = shutil.which("npm")
if not npm:
    print("ERROR: npm no encontrado. Instala Node.js desde https://nodejs.org y vuelve a correr setup.py")
    sys.exit(1)
run([npm, "install"], cwd=ROOT / "frontend")
print("OK — dependencias npm instaladas.")

# ── 5. Levantar backend + frontend ────────────────────────────────────────────
paso(5, TOTAL, "Levantando el proyecto...")

backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload", "--port", "8001"],
    cwd=ROOT,
)
frontend = subprocess.Popen(
    [npm, "run", "dev"],
    cwd=ROOT / "frontend",
)

print("\n" + "═" * 50)
print("Proyecto levantado:")
print("  Frontend → http://localhost:5173")
print("  Backend  → http://localhost:8001")
print("\nPresiona Ctrl+C para detener todo.")
print("═" * 50)

def apagar(sig, frame):
    print("\nDeteniendo servicios...")
    frontend.terminate()
    backend.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, apagar)
signal.signal(signal.SIGTERM, apagar)

# Esperar indefinidamente
backend.wait()
