"""API para predecir emergencias por canton con tres dias de anticipacion."""

import json
import os
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import (
    CATEGORICAS,
    NUMERICAS,
    YA_LISTAS,
    preparar_prediccion,
)


RUTA_MODELO = Path(os.environ.get(
    "RUTA_MODELO",
    Path(__file__).with_name("modelo.joblib"),
))
RUTA_VERSION = Path(os.environ.get(
    "RUTA_VERSION_MODELO",
    Path(__file__).with_name("modelo_version.txt"),
))
RUTA_RUN_ID = Path(os.environ.get(
    "RUTA_RUN_ID_MODELO",
    Path(__file__).with_name("modelo_run_id.txt"),
))
RUTA_REGISTRO = Path(os.environ.get(
    "RUTA_REGISTRO_PREDICCIONES",
    Path(__file__).resolve().parents[1] / "monitoreo" / "predicciones.jsonl",
))

modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None
VERSION_MODELO = (
    RUTA_VERSION.read_text(encoding="utf-8").strip()
    if RUTA_VERSION.exists()
    else os.environ.get("VERSION_MODELO", "sin-exportar")
)
RUN_ID = (
    RUTA_RUN_ID.read_text(encoding="utf-8").strip()
    if RUTA_RUN_ID.exists()
    else os.environ.get("MLFLOW_RUN_ID", "sin-exportar")
)


class Metricas:
    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(float(latencia_ms))
        self.codigos.append(int(codigo))

    def resumen(self):
        if self.latencias_ms:
            p50 = round(float(np.percentile(self.latencias_ms, 50)), 2)
            p95 = round(float(np.percentile(self.latencias_ms, 95)), 2)
        else:
            p50 = None
            p95 = None
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(codigo == 422 for codigo in self.codigos),
            "errores_servidor": sum(codigo >= 500 for codigo in self.codigos),
            "latencia_p50_ms": p50,
            "latencia_p95_ms": p95,
        }


metricas = Metricas()

app = FastAPI(title="Prediccion de emergencias ECU 911")


@app.middleware("http")
async def medir_prediccion(request: Request, call_next):
    if request.url.path != "/predecir_emergencias":
        return await call_next(request)

    inicio = time.perf_counter()
    try:
        respuesta = await call_next(request)
    except Exception:
        metricas.registrar((time.perf_counter() - inicio) * 1000, 500)
        raise
    metricas.registrar(
        (time.perf_counter() - inicio) * 1000,
        respuesta.status_code,
    )
    return respuesta


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
        "version_modelo": VERSION_MODELO,
    }


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.post("/predecir_emergencias")
def predecir_emergencias(entrada: EntradaEmergencias):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no esta cargado")

    datos = preparar_prediccion(pd.DataFrame([entrada.model_dump()]))
    columnas_modelo = NUMERICAS + CATEGORICAS + YA_LISTAS
    estimacion = float(modelo.predict(datos[columnas_modelo])[0])

    id_prediccion = uuid.uuid4().hex
    fecha_objetivo = entrada.fecha + timedelta(days=3)
    registro = {
        "id_prediccion": id_prediccion,
        "momento": datetime.now().isoformat(),
        "version_modelo": VERSION_MODELO,
        **entrada.model_dump(mode="json"),
        "fecha_objetivo": str(fecha_objetivo),
        "emergencias_estimadas": round(estimacion, 2),
    }
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with RUTA_REGISTRO.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return {
        "id_prediccion": id_prediccion,
        "cod_canton": entrada.cod_canton,
        "fecha_objetivo": str(fecha_objetivo),
        "emergencias_estimadas": round(estimacion, 2),
        "run_id": RUN_ID,
        "version_modelo": VERSION_MODELO,
    }
