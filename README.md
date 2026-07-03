# Mental Health Text Classifier (MHTC)

**Curso:** 1ACC0219 - Aplicaciones de Data Science  
**Universidad:** Universidad Peruana de Ciencias Aplicadas  
**Ciclo:** 2026-01

---

## Interfaz web

![Vista principal de la aplicación MHTC](docs/ui-preview.png)

La app utiliza un diseño **Claymorphism + Neumorphism** con paleta cálida (Old Lace / Dry Sage / Rosy Copper). Incluye chat conversacional, panel de análisis con barra de distribución, gráfico de tendencia y recursos de emergencia.

---

## Inicio rápido

```bash
git clone https://github.com/Nakamuros/1ACC0219-TP-TF-2026-01.git
cd 1ACC0219-TP-TF-2026-01
python setup.py
```

Eso es todo — instala dependencias, descarga recursos NLTK y el modelo RoBERTa (~480MB). Después podés usar:

```bash
python predict.py "I feel hopeless every day"
```

---

## Objetivo

Desarrollar un modelo de Machine Learning basado en NLP para clasificar textos no estructurados extraídos de redes sociales (Twitter y Reddit) en tres estados de salud mental: **Normal**, **Depresión** y **Riesgo Suicida**, permitiendo una detección preventiva eficiente.

---

## Integrantes

| Código | Apellidos | Nombres |
|---|---|---|
| U20221c424 | Díaz Chávez | Ángel Gabriel |
| U202312907 | Claros Simon | Williams Giusseppi |
| U202111912 | Alvarado Valle | Robert Leonardo |

---

## Dataset

- **Nombre:** Sentiment Analysis for Mental Health
- **Fuente:** Kaggle — Sarkar, S. (2024)
- **Origen:** Compilación de publicaciones de Twitter y Reddit
- **Tamaño original:** 53,044 registros — 7 categorías
- **Tamaño utilizado:** 41,138 registros — 3 categorías (Normal, Depression, Suicidal)
- **Variables:** `statement` (texto libre), `status` (etiqueta diagnóstica)

---

## Modelo RoBERTa — Descarga

El modelo fine-tuned está disponible en HuggingFace Hub (los pesos ~480MB no se incluyen en el repo):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model     = AutoModelForSequenceClassification.from_pretrained("treborDev/mhtc-roberta")
tokenizer = AutoTokenizer.from_pretrained("treborDev/mhtc-roberta")
```

El script `setup.py` lo descarga automáticamente junto con las dependencias.

---

## Modelos implementados — Resultados en Test Set

| # | Modelo | Vectorización | Accuracy | Macro F1 | Macro F2 | F2 Suicidal |
|---|--------|--------------|----------|----------|----------|-------------|
| 1 | SVM Lineal | TF-IDF (12k, bigrams) | 0.7795 | 0.7654 | 0.7669 | 0.6482 |
| 2 | Word2Vec + Regresión Logística | Word2Vec (200d, skip-gram) | 0.7279 | 0.7227 | 0.7237 | 0.6721 |
| 3 | Regresión Logística | TF-IDF (12k, bigrams) | 0.7946 | 0.7835 | 0.7859 | 0.7054 |
| 4 | LightGBM + Custom Weights | TF-IDF (12k, bigrams) | 0.7765 | 0.7700 | 0.7714 | 0.7387 |
| 5 | **RoBERTa (fine-tuning completo)** | Transformer (roberta-base, 256 tokens) | **0.8528** | **0.8439** | **0.8475** | **0.7802** |

> El F2-Score prioriza el recall sobre la precisión (β=2), penalizando más los falsos negativos en la clase Suicidal — métrica clínicamente relevante para detección preventiva.

Resultados de prueba de los modelos adicionales:

- **Regresión Logística + TF-IDF:** accuracy 0.7963, Macro F1 0.7859 y F2
  Suicidal 0.6981.
- **SVM lineal + TF-IDF:** accuracy 0.7875, Macro F1 0.7745 y F2 Suicidal
  0.6530.
- **Word2Vec + Regresión Logística:** accuracy 0.7386, Macro F1 0.7341 y F2
  Suicidal 0.6820.
- **MentalRoBERTa:** accuracy 0.7879, Macro F1 0.7706. El encoder preentrenado se
  mantuvo congelado para hacer viable el entrenamiento sin GPU.
- **LightGBM optimizado:** F2 de Suicidal 0.7666 y recall de Suicidal 0.8260 en
  test sin textos duplicados entre particiones. La configuración y el umbral se
  eligieron exclusivamente con validación; el test se reservó para la medición
  final. Los detalles reproducibles están en `modelo_lightgbm_metrics.json`.

La comparación de LightGBM, Regresión Logística, SVM y Word2Vec emplea las
mismas 39,398 entradas, semilla y partición 70/15/15 sin duplicados entre
conjuntos. Los experimentos de los baselines están en
`modelos_baseline_metrics.json`.

Al optimizar todos los modelos bajo el mismo objetivo de seguridad —F2 de
Suicidal con precisión mínima de 0.60 en validación— los resultados de test son:

| Modelo | Accuracy | Macro F1 | F2 Suicidal | Recall Suicidal |
|---|---:|---:|---:|---:|
| Regresión Logística | 0.7765 | 0.7694 | **0.7704** | **0.8298** |
| LightGBM | **0.7836** | **0.7774** | 0.7666 | 0.8260 |
| SVM lineal | 0.7810 | 0.7731 | 0.7665 | 0.8222 |
| Word2Vec + Regresión Logística | 0.7308 | 0.7294 | 0.7048 | 0.7393 |

Esto muestra que la ventaja inicial de LightGBM en F2 provenía en parte de que
era el único modelo con pesos y umbral especializados. La búsqueda completa y
sus umbrales están en `modelos_f2_optimized_metrics.json` y
`model_decision_thresholds.json`.

Para repetir el ajuste de LightGBM:

```bash
python train_tuned_lightgbm.py
python train_fair_baselines.py
python optimize_suicidal_baselines.py
```

---

## Conclusiones

El análisis exploratorio evidenció patrones léxicos estadísticamente diferenciadores entre las tres clases: Normal presenta vocabulario diverso de cotidianidad, mientras Depression y Suicidal comparten un núcleo semántico de desesperanza con vocabulario repetitivo y carga emocional negativa. La progresión de modelos de TF-IDF hacia representaciones Transformer permitió cuantificar el aporte real de cada enfoque de representación textual para esta tarea de clasificación en dominio de salud mental.

---

## Aplicación web

El proyecto incluye una interfaz conversacional en React y una API en FastAPI para analizar textos con los modelos entrenados. Consulta [WEBAPP.md](WEBAPP.md) para ejecutarla localmente.
Opcionalmente, una clave configurada en `.env` habilita Gemini como generador de
preguntas de seguimiento. Gemini no clasifica estados ni decide alertas; esas
funciones permanecen en los modelos y reglas locales.

---

## Licencia

MIT License — libre uso con atribución.
