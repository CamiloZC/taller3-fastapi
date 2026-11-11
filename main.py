"""API de FastAPI que usa el pipeline del Taller 1 para predecir la supervivencia en el Titanic."""
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

pipeline = joblib.load("pipeline.joblib")

app = FastAPI(title="Taller 3 MLOps - API Titanic")


class Pasajero(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: Optional[float] = None          # puede faltar: el pipeline la imputa
    sibsp: int = 0
    parch: int = 0
    fare: float
    embarked: Optional[Literal["S", "C", "Q"]] = None


class Solicitud(BaseModel):
    pasajeros: list[Pasajero]

    model_config = {"json_schema_extra": {"examples": [{"pasajeros": [
        {"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"},
        {"pclass": 1, "sex": "female", "age": None, "sibsp": 0, "parch": 0, "fare": 80.0, "embarked": None},
    ]}]}}


@app.get("/")
def home():
    return {"Universidad EIA": "MLOps - Taller 3", "documentacion": "/docs"}


@app.post("/predict")
def predict(solicitud: Solicitud):
    try:
        df = pd.DataFrame([p.model_dump() for p in solicitud.pasajeros]).replace({None: np.nan})
        probabilidades = pipeline.predict_proba(df)[:, 1]
        predicciones = pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predicciones": [
        {"sobrevive": int(pred), "probabilidad_supervivencia": round(float(prob), 3)}
        for pred, prob in zip(predicciones, probabilidades)
    ]}