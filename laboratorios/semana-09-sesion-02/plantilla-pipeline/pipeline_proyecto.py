"""Preparacion de los datos de mi proyecto, lista para produccion.

Las features elegidas en la sesion pasada viven dentro de funciones. Los
notebooks de modelos pueden importar este archivo sin ejecutar el ejemplo:

    from pipeline_proyecto import cargar_y_dividir, construir_preparacion

Para comprobar que todo corre de principio a fin:

    python3 pipeline_proyecto.py
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RUTA_PREDETERMINADA = (
    Path(__file__).parents[2]
    / "data"
    / "obt_emergencias_muestra_6_meses.csv"
)
RUTA_DATOS = Path(os.environ.get("OBT_CSV", RUTA_PREDETERMINADA))

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

COLUMNAS_ARCHIVO = [
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


def agregar_features(datos):
    datos = datos.sort_values(["cod_canton", "fecha"]).copy()

    # t-4 y t+3 caen en el mismo dia de la semana.
    datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)

    # El calendario de t+3 ya se conoce al terminar el dia t.
    dia_semana_t3 = (datos["dia_semana"] + 2) % 7 + 1
    datos["es_fin_semana_t3"] = dia_semana_t3.isin([6, 7]).astype(int)
    return datos


def cargar_y_dividir(ruta=RUTA_DATOS):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            "No se encontro la muestra local. Ejecuta primero "
            f"crear_muestra_ecu911_6_meses.py. Se esperaba: {ruta}"
        )

    datos = pd.read_csv(
        ruta,
        usecols=COLUMNAS_ARCHIVO,
        dtype={"cod_canton": str},
        parse_dates=["fecha"],
    )
    datos = agregar_features(datos)
    datos = datos.dropna(subset=[OBJETIVO]).copy()

    X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
    y = datos[OBJETIVO]

    # El modelo predice el futuro: primero se entrena con fechas antiguas.
    fechas = datos["fecha"].drop_duplicates().sort_values().reset_index(drop=True)
    fecha_corte = fechas.iloc[int(len(fechas) * 0.8)]
    es_entrenamiento = datos["fecha"] < fecha_corte

    X_ent = X.loc[es_entrenamiento]
    X_prueba = X.loc[~es_entrenamiento]
    y_ent = y.loc[es_entrenamiento]
    y_prueba = y.loc[~es_entrenamiento]
    return X_ent, X_prueba, y_ent, y_prueba


def construir_preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            )),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


if __name__ == "__main__":
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)

    print("Archivo:", RUTA_DATOS)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print("Columnas producidas:", len(preparacion.get_feature_names_out()))
    print(list(preparacion.get_feature_names_out()))
