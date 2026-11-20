"""Calcula el MAE semanal del servicio por version del modelo."""

import os
from pathlib import Path

import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_REGISTRO = Path(os.environ.get(
    "RUTA_REGISTRO_PREDICCIONES",
    Path(__file__).with_name("predicciones.jsonl"),
))
RUTA_VERDAD = Path(os.environ.get(
    "OBT_CSV",
    RAIZ_PROYECTO.parents[1] / "laboratorios" / "data" / "obt_emergencias_muestra_6_meses.csv",
))


def preparar_verdad(datos):
    """Deja una fila por canton y fecha con el conteo real observado."""
    verdad = datos[["cod_canton", "fecha", "n_emergencias"]].copy()
    verdad["cod_canton"] = verdad["cod_canton"].astype(str).str.zfill(4)
    verdad["fecha_objetivo"] = pd.to_datetime(verdad.pop("fecha"))
    return verdad.rename(columns={"n_emergencias": "emergencias_reales"})


def calcular_desempeno(predicciones, verdad):
    predicciones = predicciones.copy()
    predicciones["cod_canton"] = predicciones["cod_canton"].astype(str).str.zfill(4)
    predicciones["fecha_objetivo"] = pd.to_datetime(predicciones["fecha_objetivo"])
    datos = predicciones.merge(
        preparar_verdad(verdad),
        on=["cod_canton", "fecha_objetivo"],
        how="inner",
        validate="many_to_one",
    )
    datos["error_absoluto"] = (
        datos["emergencias_estimadas"] - datos["emergencias_reales"]
    ).abs()
    datos["semana"] = datos["fecha_objetivo"].dt.to_period("W").dt.start_time.dt.date
    resumen = datos.groupby(["semana", "version_modelo"]).agg(
        predicciones=("id_prediccion", "count"),
        mae=("error_absoluto", "mean"),
    )
    resumen["mae"] = resumen["mae"].round(4)
    return resumen


if __name__ == "__main__":
    if not RUTA_REGISTRO.exists():
        raise SystemExit(f"No existe el registro de predicciones: {RUTA_REGISTRO}")
    predicciones = pd.read_json(RUTA_REGISTRO, lines=True)
    verdad = pd.read_csv(
        RUTA_VERDAD,
        usecols=["cod_canton", "fecha", "n_emergencias"],
        dtype={"cod_canton": str},
    )
    print(calcular_desempeno(predicciones, verdad).to_string())
