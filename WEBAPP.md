# Aplicación web MHTC

La interfaz simula una conversación y envía cada texto al modelo seleccionado. Está diseñada como una herramienta académica de detección de patrones, no como diagnóstico ni terapia.

## Ejecutar la API

Crear `.env` desde el ejemplo y colocar una clave de Gemini:

```bash
cp .env.example .env
# Editar GEMINI_API_KEY en .env
```

La clave solo se lee en el backend y `.env` está excluido de Git. Gemini se usa
exclusivamente para redactar una respuesta breve y una pregunta de seguimiento;
los modelos locales conservan la clasificación y las alertas. Sin clave o ante
un error de red, la aplicación usa preguntas locales predefinidas.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8001
```

## Ejecutar React

```bash
cd frontend
npm install
npm run dev
```

Abrir `http://localhost:5173`. LightGBM ya cuenta con un artefacto. Regresión Logística y SVM aparecerán disponibles después de ejecutar las nuevas celdas de `Entrenamiento.ipynb`, que generan sus archivos `.pkl`.

## Decisiones de producto

- El modelo analiza textos en inglés porque el dataset de entrenamiento está en inglés.
- La aplicación muestra distribución y confianza, evitando presentar el resultado como certeza clínica.
- Ante la clase `Suicidal`, muestra orientación inmediata y la Línea 113 opción 5 para Perú.
- La API no persiste los textos recibidos. Para generar la conversación envía a
  Gemini únicamente los últimos cuatro turnos, con correos y teléfonos comunes
  redactados; el proveedor externo procesa ese contenido según sus condiciones.
