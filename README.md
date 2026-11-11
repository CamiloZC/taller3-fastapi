# Taller 3 MLOps – API de predicción (Titanic)

API de FastAPI que carga un `Pipeline` de scikit-learn (preprocesamiento + regresión logística, del Taller 1)
y predice la probabilidad de supervivencia de pasajeros del Titanic.

## Archivos
- `train.py`: entrena el pipeline y lo guarda en `pipeline.joblib`.
- `pipeline.joblib`: pipeline ya entrenado (imputación, escalado, one-hot encoding y modelo).
- `main.py`: API con los endpoints `GET /` y `POST /predict`.
- `requirements.txt`: dependencias con versiones fijas.

## Ejecutar
Requiere Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Documentación interactiva: `http://<host>:8000/docs`

Para volver a entrenar el pipeline (opcional): `python train.py`

## Ejemplo de predicción
```bash
curl -X POST http://<host>:8000/predict -H "Content-Type: application/json" \
  -d '{"pasajeros": [{"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}]}'
```
Respuesta:
```json
{"predicciones": [{"sobrevive": 0, "probabilidad_supervivencia": 0.122}]}
```