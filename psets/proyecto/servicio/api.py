"""API para predecir emergencias por canton con tres dias de anticipacion."""

import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import (
    CATEGORICAS,
    NUMERICAS,
    YA_LISTAS,
    preparar_prediccion,
)


RUN_ID = os.environ.get("MLFLOW_RUN_ID", "e60512928d4f4d93afec8350fe9f1fc3")
RUTA_MODELO = Path(os.environ.get(
    "RUTA_MODELO",
    Path(__file__).with_name("modelo.joblib"),
))

modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None

app = FastAPI(title="Prediccion de emergencias ECU 911")


class EntradaEmergencias(BaseModel):
    """Datos conocidos al cerrar el dia para un canton."""

    cod_canton: str
    fecha: date
    n_emergencias: float = Field(ge=0)
    lag_1d: float = Field(ge=0)
    n_emergencias_hace_4_dias: float = Field(ge=0)
    lag_7d: float = Field(ge=0)
    lag_14d: float = Field(ge=0)
    lag_28d: float = Field(ge=0)
    media_prev_7d: float = Field(ge=0)
    media_prev_28d: float = Field(ge=0)
    es_feriado_t3: int = Field(ge=0, le=1)


@app.get("/salud")
def salud():
    return {
        "estado": "ok",
        "modelo_cargado": modelo is not None,
        "run_id": RUN_ID,
    }


@app.post("/predecir_emergencias")
def predecir_emergencias(entrada: EntradaEmergencias):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado")

    datos = preparar_prediccion(pd.DataFrame([entrada.model_dump()]))
    columnas_modelo = NUMERICAS + CATEGORICAS + YA_LISTAS
    estimacion = float(modelo.predict(datos[columnas_modelo])[0])

    return {
        "cod_canton": entrada.cod_canton,
        "fecha_objetivo": str(entrada.fecha + timedelta(days=3)),
        "emergencias_estimadas": round(estimacion, 2),
        "run_id": RUN_ID,
    }
