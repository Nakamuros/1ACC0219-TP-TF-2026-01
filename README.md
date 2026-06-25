# Mental Health Text Classifier (MHTC)

**Curso:** 1ACC0219 - Aplicaciones de Data Science  
**Universidad:** Universidad Peruana de Ciencias Aplicadas  
**Ciclo:** 2026-01

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

## Modelos implementados

| # | Modelo | Vectorización |
|---|---|---|
| 1 | Regresión Logística | TF-IDF |
| 2 | SVM (kernel lineal) | TF-IDF |
| 3 | Word2Vec + Regresión Logística | Embeddings propios |
| 4 | mental-roberta-base | Transformer |

---

## Conclusiones

El análisis exploratorio evidenció patrones léxicos estadísticamente diferenciadores entre las tres clases: Normal presenta vocabulario diverso de cotidianidad, mientras Depression y Suicidal comparten un núcleo semántico de desesperanza con vocabulario repetitivo y carga emocional negativa. La progresión de modelos de TF-IDF hacia representaciones Transformer permitió cuantificar el aporte real de cada enfoque de representación textual para esta tarea de clasificación en dominio de salud mental.

---

## Licencia

MIT License — libre uso con atribución.
