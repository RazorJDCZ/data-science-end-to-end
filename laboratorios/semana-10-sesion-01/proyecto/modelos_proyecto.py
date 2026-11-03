"""Entrena tres modelos del proyecto ECU 911 con la muestra de seis meses."""

import os
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DIRECTORIO = Path(__file__).resolve().parent
RUTA_PREDETERMINADA = (
    DIRECTORIO.parents[1] / "data" / "obt_emergencias_muestra_6_meses.csv"
)
RUTA_DATOS = Path(os.environ.get("OBT_CSV", RUTA_PREDETERMINADA))
RUTA_RESULTADOS = DIRECTORIO / "resultados_proyecto.csv"

OBJETIVO = "obj_n_emergencias_t3"
NUMERICAS = [
    "n_emergencias",
    "lag_1d",
    "lag_4d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "media_prev_7d",
    "media_prev_28d",
]
CATEGORICAS = ["cod_canton"]
YA_LISTAS = ["es_fin_semana_t3", "es_feriado_t3"]

columnas_archivo = [
    "cod_canton",
    "fecha",
    "dia_semana",
    "n_emergencias",
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "media_prev_7d",
    "media_prev_28d",
    "es_feriado_t3",
    OBJETIVO,
]

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontro {RUTA_DATOS}. Ejecuta crear_muestra_ecu911_6_meses.py."
    )

datos = pd.read_csv(
    RUTA_DATOS,
    usecols=columnas_archivo,
    dtype={"cod_canton": str},
    parse_dates=["fecha"],
)
datos = datos.sort_values(["cod_canton", "fecha"]).copy()

# Las dos features propuestas en la Semana 9, Sesion 1.
datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)
dia_semana_t3 = (datos["dia_semana"] + 2) % 7 + 1
datos["es_fin_semana_t3"] = dia_semana_t3.isin([6, 7]).astype(int)
datos = datos.dropna(subset=[OBJETIVO]).copy()

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

fechas = datos["fecha"].drop_duplicates().sort_values().reset_index(drop=True)
fecha_corte = fechas.iloc[int(len(fechas) * 0.8)]
es_entrenamiento = datos["fecha"] < fecha_corte
X_ent = X.loc[es_entrenamiento]
X_prueba = X.loc[~es_entrenamiento]
y_ent = y.loc[es_entrenamiento]
y_prueba = y.loc[~es_entrenamiento]


def crear_preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(
                handle_unknown="ignore", sparse_output=True
            )),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


resultados = []


def evaluar(nombre, modelo):
    pipeline = Pipeline([
        ("preparar", crear_preparacion()),
        ("modelo", modelo),
    ])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    resultados.append({
        "modelo": nombre,
        "mae_entrenamiento": round(mae_ent, 4),
        "mae_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "segundos": round(segundos, 3),
    })
    return pipeline


# La linea base predice siempre la mediana observada en entrenamiento.
evaluar("linea base mediana", DummyRegressor(strategy="median"))

# Ridge limita el tamano de los coeficientes del modelo lineal.
evaluar("ridge", Ridge(alpha=10.0, solver="lsqr"))

# Un bosque pequeno y limitado captura relaciones no lineales sin usar
# todos los nucleos de la computadora.
evaluar("bosque aleatorio", RandomForestRegressor(
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1,
))

tabla = pd.DataFrame(resultados)
tabla.to_csv(RUTA_RESULTADOS, index=False)

print("Archivo:", RUTA_DATOS)
print("Fecha de corte:", fecha_corte.date())
print("Entrenamiento:", X_ent.shape, "Prueba:", X_prueba.shape)
print()
print(tabla.to_string(index=False))
print("\nMejor modelo:", tabla.loc[tabla["mae_prueba"].idxmin(), "modelo"])
print("Resultados:", RUTA_RESULTADOS)
