"""Crea una muestra de seis meses completos de la OBT local del ECU 911."""

import calendar
import os
from pathlib import Path

import pandas as pd


LABORATORIOS = Path(__file__).resolve().parent
RUTA_ORIGEN_PREDETERMINADA = (
    LABORATORIOS
    / "semana-09-sesion-01"
    / "data"
    / "obt_emergencias_canton_dia.csv"
)
RUTA_SALIDA_PREDETERMINADA = (
    LABORATORIOS / "data" / "obt_emergencias_muestra_6_meses.csv"
)

RUTA_ORIGEN = Path(os.environ.get("OBT_CSV", RUTA_ORIGEN_PREDETERMINADA))
RUTA_SALIDA = Path(os.environ.get("MUESTRA_CSV", RUTA_SALIDA_PREDETERMINADA))

if not RUTA_ORIGEN.exists():
    raise FileNotFoundError(f"No se encontro la OBT completa: {RUTA_ORIGEN}")

datos = pd.read_csv(
    RUTA_ORIGEN,
    dtype={"cod_canton": str, "cod_provincia": str},
    parse_dates=["fecha"],
)

periodos = sorted(datos["periodo"].dropna().astype(str).unique())
if len(periodos) < 6:
    raise ValueError("La OBT no contiene al menos seis meses.")

ultimos_seis = periodos[-6:]
muestra = datos[datos["periodo"].astype(str).isin(ultimos_seis)].copy()
muestra = muestra.sort_values(["cod_canton", "fecha"]).reset_index(drop=True)

# Comprobaciones simples para no guardar una muestra incompleta o duplicada.
if muestra.duplicated(["cod_canton", "fecha"]).any():
    raise ValueError("La muestra contiene filas duplicadas por canton y fecha.")

fechas_esperadas = 0
for periodo in ultimos_seis:
    anio, mes = map(int, periodo.split("-"))
    fechas_esperadas += calendar.monthrange(anio, mes)[1]

fechas_en_muestra = muestra["fecha"].nunique()
if fechas_en_muestra != fechas_esperadas:
    raise ValueError(
        f"Se esperaban {fechas_esperadas} fechas y se encontraron {fechas_en_muestra}."
    )

filas_esperadas = fechas_esperadas * muestra["cod_canton"].nunique()
if len(muestra) != filas_esperadas:
    raise ValueError(
        f"Se esperaban {filas_esperadas} filas y se encontraron {len(muestra)}."
    )

RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
ruta_temporal = RUTA_SALIDA.with_suffix(".csv.partial")
muestra.to_csv(ruta_temporal, index=False, date_format="%Y-%m-%d")
ruta_temporal.replace(RUTA_SALIDA)

print("Muestra creada:", RUTA_SALIDA)
print("Periodos:", ", ".join(ultimos_seis))
print("Fechas:", muestra["fecha"].min().date(), "a", muestra["fecha"].max().date())
print("Cantones:", muestra["cod_canton"].nunique())
print("Filas y columnas:", muestra.shape)
