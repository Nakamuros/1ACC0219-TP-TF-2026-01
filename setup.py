#!/usr/bin/env python3
"""Instalador y lanzador del proyecto MHTC en un solo comando.

Uso:
    python setup.py            Instala dependencias, descarga los modelos y
                               levanta backend (:8001) y frontend (:5173).
    python setup.py --no-run   Solo prepara el entorno, sin levantar la app.

Requisitos previos: Python 3.10+, pip y Node.js/npm instalados.
No se necesita Git LFS: los modelos grandes se descargan desde HuggingFace.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "modelos"

# (repo de HuggingFace, carpeta destino local). El traductor es el modelo
# público estándar es->en; la RoBERTa es el modelo del equipo (entrenado por
# Robert) publicado en HuggingFace.
MODELOS_REMOTOS = [
    ("Helsinki-NLP/opus-mt-es-en", MODELS_DIR / "modelo_traduccion_es_en"),
    ("treborDev/mhtc-roberta", MODELS_DIR / "modelo_mental_roberta"),
]

IS_WINDOWS = os.name == "nt"


def paso(titulo):
    print(f"\n\033[1m==> {titulo}\033[0m")


def correr(cmd, **kwargs):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, **kwargs)


def instalar_dependencias_backend():
    paso("Instalando dependencias de Python (backend)")
    correr([sys.executable, "-m", "pip", "install", "-r",
            str(ROOT / "backend" / "requirements.txt")])
    correr([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.24"])


def descargar_modelos():
    paso("Descargando modelos desde HuggingFace")
    from huggingface_hub import snapshot_download

    for repo, destino in MODELOS_REMOTOS:
        pesos = destino / "model.safetensors"
        if pesos.exists() and pesos.stat().st_size > 1_000_000:
            print(f"    ✓ {destino.name} ya está descargado, se omite.")
            continue
        destino.mkdir(parents=True, exist_ok=True)
        print(f"    Descargando {repo} -> modelos/{destino.name} ...")
        snapshot_download(repo_id=repo, local_dir=str(destino))


def instalar_dependencias_frontend():
    paso("Instalando dependencias del frontend (npm install)")
    correr(["npm", "install"], cwd=str(ROOT / "frontend"), shell=IS_WINDOWS)


def levantar_app():
    paso("Levantando la aplicación")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--reload", "--port", "8001"],
        cwd=str(ROOT),
    )
    # Pequeña espera para que el backend arranque antes que Vite.
    time.sleep(3)
    frontend = subprocess.Popen(
        ["npm", "run", "dev"], cwd=str(ROOT / "frontend"), shell=IS_WINDOWS,
    )
    print("\n\033[1mAPI en http://localhost:8001  |  Web en http://localhost:5173\033[0m")
    print("Presiona Ctrl+C para detener ambos servidores.\n")
    try:
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (frontend, backend):
            proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="Instala y levanta MHTC.")
    parser.add_argument("--no-run", action="store_true",
                        help="Solo prepara el entorno, no levanta la app.")
    args = parser.parse_args()

    instalar_dependencias_backend()
    descargar_modelos()
    instalar_dependencias_frontend()

    if args.no_run:
        paso("Listo")
        print("Entorno preparado. Para levantar la app:")
        print("    uvicorn backend.main:app --reload --port 8001")
        print("    (en otra terminal)  cd frontend && npm run dev")
    else:
        levantar_app()


if __name__ == "__main__":
    main()
