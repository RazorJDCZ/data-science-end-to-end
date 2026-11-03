import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Se usa la muestra compartida de seis meses. Tambien se puede indicar
# otra ruta con la variable de entorno OBT_CSV.
RUTA_PREDETERMINADA = (
    Path(__file__).parents[2]
    / "data"
    / "obt_emergencias_muestra_6_meses.csv"
)
RUTA_DATOS = Path(os.environ.get("OBT_CSV", RUTA_PREDETERMINADA))

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        "No se encontro la muestra local. Ejecuta primero "
        f"crear_muestra_ecu911_6_meses.py. Se esperaba: {RUTA_PREDETERMINADA}"
    )

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

datos = pd.read_csv(
    RUTA_DATOS,
    usecols=columnas_archivo,
    dtype={"cod_canton": str},
    parse_dates=["fecha"],
)
datos = datos.sort_values(["cod_canton", "fecha"]).copy()

# Las dos features de la sesion pasada.
# t-4 y t+3 caen en el mismo dia de la semana.
datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)

# El calendario de t+3 ya se conoce al terminar el dia t.
dia_semana_t3 = (datos["dia_semana"] + 2) % 7 + 1
datos["es_fin_semana_t3"] = dia_semana_t3.isin([6, 7]).astype(int)

# Las ultimas tres fechas no tienen objetivo t+3 y no sirven para entrenar.
datos = datos.dropna(subset=[OBJETIVO]).copy()

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# Como el modelo predice el futuro, el 80 % de fechas mas antiguas se usa
# para entrenar y el 20 % mas reciente para probar.
fechas = datos["fecha"].drop_duplicates().sort_values().reset_index(drop=True)
fecha_corte = fechas.iloc[int(len(fechas) * 0.8)]
es_entrenamiento = datos["fecha"] < fecha_corte

X_ent = X.loc[es_entrenamiento]
X_prueba = X.loc[~es_entrenamiento]
y_ent = y.loc[es_entrenamiento]
y_prueba = y.loc[~es_entrenamiento]

preparacion = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]), NUMERICAS),
    ("categoricas", Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        # La salida dispersa ocupa menos memoria porque cada fila solo tiene
        # un canton activo.
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]), CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
])

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
columnas_salida = list(preparacion.get_feature_names_out())

print("Archivo:", RUTA_DATOS)
print("Fecha de corte:", fecha_corte.date())
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print("Columnas producidas:", len(columnas_salida))
print(columnas_salida)
