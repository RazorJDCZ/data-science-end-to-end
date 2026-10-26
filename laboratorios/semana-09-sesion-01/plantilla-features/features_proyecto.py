import os
from pathlib import Path

import pandas as pd


RUTA_PREDETERMINADA = (
    Path(__file__).parents[1] / "data" / "obt_emergencias_canton_dia.csv"
)
RUTA_DATOS = Path(os.environ.get("OBT_CSV", RUTA_PREDETERMINADA))
OBJETIVO = "obj_n_emergencias_t3"

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        "No se encontro la OBT local. Ejecuta primero el servicio "
        f"datos-proyecto; se esperaba '{RUTA_PREDETERMINADA}'."
    )

datos = pd.read_csv(
    RUTA_DATOS,
    dtype={"cod_canton": str, "cod_provincia": str},
    parse_dates=["fecha"],
)

columnas_necesarias = {
    "cod_canton",
    "fecha",
    "dia_semana",
    "n_emergencias",
    OBJETIVO,
}
faltantes = columnas_necesarias.difference(datos.columns)
if faltantes:
    raise ValueError(f"Faltan columnas necesarias en la OBT: {sorted(faltantes)}")

datos = datos.sort_values(["cod_canton", "fecha"]).copy()

# Feature 1: t-4 y t+3 caen en el mismo dia de la semana.
datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)

# Feature 2: indica si el dia que se quiere predecir sera fin de semana.
# dia_semana usa 1=lunes y 7=domingo.
datos["dia_semana_t3"] = (datos["dia_semana"] + 2) % 7 + 1
datos["es_fin_semana_t3"] = datos["dia_semana_t3"].isin([6, 7]).astype(int)

print("\n=== FEATURES CREADAS ===")
print(
    datos[
        [
            "cod_canton",
            "fecha",
            "lag_4d",
            "dia_semana_t3",
            "es_fin_semana_t3",
            OBJETIVO,
        ]
    ].head(10).to_string(index=False)
)

print("\n=== EVIDENCIA DE LAG_4D ===")
correlacion_global = datos[["lag_4d", OBJETIVO]].corr().iloc[0, 1]
print(f"Correlacion global con el objetivo: {correlacion_global:.3f}")

if "canton" in datos.columns:
    for canton in ["QUITO", "GUAYAQUIL", "CUENCA"]:
        muestra = datos.loc[datos["canton"] == canton, ["lag_4d", OBJETIVO]]
        if len(muestra.dropna()) >= 2:
            correlacion = muestra.corr().iloc[0, 1]
            print(f"Correlacion en {canton}: {correlacion:.3f}")

print("\n=== EVIDENCIA DE FIN DE SEMANA EN T+3 ===")
comparacion = datos.groupby("es_fin_semana_t3")[OBJETIVO].agg(
    observaciones="count",
    promedio="mean",
    mediana="median",
)
print(comparacion.round(2).to_string())
