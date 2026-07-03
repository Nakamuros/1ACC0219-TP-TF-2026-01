"""
Descarga los pesos del modelo RoBERTa desde HuggingFace Hub.
Ejecutar una vez despues de clonar el repositorio:

    python setup.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download

DEST = Path("modelo_mental_roberta_v2/best")
REPO = "treborDev/mhtc-roberta"

if (DEST / "model.safetensors").exists():
    print(f"Modelo ya descargado en '{DEST}'. Nada que hacer.")
else:
    print(f"Descargando modelo RoBERTa desde {REPO} (~480MB)...")
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO, local_dir=str(DEST))
    print(f"Listo. Modelo guardado en '{DEST}'.")
