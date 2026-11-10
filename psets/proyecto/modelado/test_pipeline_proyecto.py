"""Pruebas pequenas del pipeline, sin depender del CSV del proyecto."""

import pandas as pd

from modelado.pipeline_proyecto import (
    CATEGORICAS,
    NUMERICAS,
    OBJETIVO,
    YA_LISTAS,
    agregar_features,
    construir_preparacion,
)


def datos_de_prueba():
    return pd.DataFrame({
        "cod_canton": ["0101"] * 6,
        "fecha": pd.date_range("2026-01-01", periods=6),
        "dia_semana": [4, 5, 6, 7, 1, 2],
        "n_emergencias": [10, 20, 30, 40, 50, 60],
    })


def fila_para_preparacion(canton="0101"):
    return pd.DataFrame({
        "n_emergencias": [20.0],
        "lag_1d": [18.0],
        "lag_4d": [16.0],
        "lag_7d": [15.0],
        "lag_14d": [14.0],
        "lag_28d": [13.0],
        "media_prev_7d": [17.0],
        "media_prev_28d": [15.0],
        "cod_canton": [canton],
        "es_fin_semana_t3": [0],
        "es_feriado_t3": [0],
    })


def test_objetivo_no_entra_al_modelo():
    columnas_modelo = NUMERICAS + CATEGORICAS + YA_LISTAS
    assert OBJETIVO not in columnas_modelo


def test_lag_4d_tiene_el_valor_esperado():
    preparados = agregar_features(datos_de_prueba())
    # En el quinto dia, el valor de hace cuatro dias era 10.
    assert preparados.iloc[4]["lag_4d"] == 10


def test_una_fila_futura_no_cambia_el_pasado():
    originales = datos_de_prueba()
    modificados = originales.copy()
    modificados.loc[5, "n_emergencias"] = 9999

    features_originales = agregar_features(originales)
    features_modificados = agregar_features(modificados)
    assert features_originales.iloc[:5]["lag_4d"].equals(
        features_modificados.iloc[:5]["lag_4d"]
    )


def test_preparacion_acepta_un_canton_nuevo():
    preparacion = construir_preparacion()
    entrenamiento = pd.concat([
        fila_para_preparacion("0101"),
        fila_para_preparacion("0102"),
    ], ignore_index=True)
    preparacion.fit(entrenamiento)

    transformado = preparacion.transform(fila_para_preparacion("9999"))
    assert transformado.shape[0] == 1

