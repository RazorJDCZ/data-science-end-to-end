"""Preparacion de la muestra de emergencias para el modelo del proyecto.

El objetivo es predecir cuantas emergencias tendra cada canton dentro de
tres dias. Este modulo deja la preparacion en funciones para poder usarla
desde los experimentos, las pruebas y, mas adelante, el servicio.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RUTA_PREDETERMINADA = (
    Path(__file__).parents[3]
    / "laboratorios"
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
    """Agrega las dos features que no vienen listas en el archivo."""
    datos = datos.sort_values(["cod_canton", "fecha"]).copy()

    # t-4 y t+3 caen en el mismo dia de la semana.
    datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)

    # El calendario de t+3 ya se conoce al terminar el dia t.
    dia_semana_t3 = (datos["dia_semana"] + 2) % 7 + 1
    datos["es_fin_semana_t3"] = dia_semana_t3.isin([6, 7]).astype(int)
    return datos


def cargar_y_dividir(ruta=RUTA_DATOS):
    """Carga la muestra y hace una division temporal 80/20."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            "No se encontro la muestra local de seis meses. Se esperaba: "
            f"{ruta}"
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

    # No se mezclan fechas futuras en el entrenamiento.
    fechas = datos["fecha"].drop_duplicates().sort_values().reset_index(drop=True)
    fecha_corte = fechas.iloc[int(len(fechas) * 0.8)]
    es_entrenamiento = datos["fecha"] < fecha_corte

    return (
        X.loc[es_entrenamiento],
        X.loc[~es_entrenamiento],
        y.loc[es_entrenamiento],
        y.loc[~es_entrenamiento],
    )


def construir_preparacion():
    """Transformacion que se ajusta solo con el conjunto de entrenamiento."""
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
    print("Objetivos:", len(y_ent), "de entrenamiento y", len(y_prueba), "de prueba")

