"""Pruebas del contrato, versionado y monitoreo de la API."""

import json

import pytest
from fastapi.testclient import TestClient

from modelado.pipeline_proyecto import CATEGORICAS, NUMERICAS, YA_LISTAS
from servicio import api


ENTRADA = {
    "cod_canton": "0101",
    "fecha": "2026-09-03",
    "n_emergencias": 20,
    "lag_1d": 18,
    "n_emergencias_hace_4_dias": 16,
    "lag_7d": 15,
    "lag_14d": 14,
    "lag_28d": 13,
    "media_prev_7d": 17,
    "media_prev_28d": 15,
    "es_feriado_t3": 0,
}


class ModeloDePrueba:
    def __init__(self):
        self.ultima_entrada = None

    def predict(self, datos):
        self.ultima_entrada = datos.copy()
        return [25.5]


class ModeloConError:
    def predict(self, datos):
        raise RuntimeError("error de prueba")


@pytest.fixture
def servicio_preparado(tmp_path, monkeypatch):
    modelo = ModeloDePrueba()
    monkeypatch.setattr(api, "modelo", modelo)
    monkeypatch.setattr(api, "VERSION_MODELO", "7")
    monkeypatch.setattr(api, "RUTA_REGISTRO", tmp_path / "predicciones.jsonl")
    monkeypatch.setattr(api, "metricas", api.Metricas())
    return TestClient(api.app), modelo, tmp_path / "predicciones.jsonl"


def test_salud_confirma_modelo_y_version(servicio_preparado):
    cliente, _, _ = servicio_preparado
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["modelo_cargado"] is True
    assert respuesta.json()["version_modelo"] == "7"


def test_prediccion_valida_calcula_features_y_responde(servicio_preparado):
    cliente, modelo, _ = servicio_preparado
    respuesta = cliente.post("/predecir_emergencias", json=ENTRADA)

    assert respuesta.status_code == 200
    assert respuesta.json()["emergencias_estimadas"] == 25.5
    assert respuesta.json()["fecha_objetivo"] == "2026-09-06"
    assert respuesta.json()["version_modelo"] == "7"
    assert modelo.ultima_entrada.iloc[0]["lag_4d"] == 16
    assert modelo.ultima_entrada.iloc[0]["es_fin_semana_t3"] == 1
    assert list(modelo.ultima_entrada.columns) == NUMERICAS + CATEGORICAS + YA_LISTAS


def test_entrada_sin_canton_se_rechaza(servicio_preparado):
    cliente, _, _ = servicio_preparado
    entrada = ENTRADA.copy()
    entrada.pop("cod_canton")
    respuesta = cliente.post("/predecir_emergencias", json=entrada)
    assert respuesta.status_code == 422


def test_cada_prediccion_queda_registrada(servicio_preparado):
    cliente, _, ruta_registro = servicio_preparado
    respuesta = cliente.post("/predecir_emergencias", json=ENTRADA).json()

    lineas = [json.loads(linea) for linea in ruta_registro.read_text(encoding="utf-8").splitlines()]
    assert len(lineas) == 1
    assert lineas[0]["id_prediccion"] == respuesta["id_prediccion"]
    assert lineas[0]["version_modelo"] == "7"
    assert lineas[0]["cod_canton"] == "0101"
    assert lineas[0]["fecha_objetivo"] == "2026-09-06"


def test_metricas_separan_errores_de_validacion_y_servidor(servicio_preparado, monkeypatch):
    cliente, _, _ = servicio_preparado
    cliente.post("/predecir_emergencias", json=ENTRADA)

    incompleta = ENTRADA.copy()
    incompleta.pop("cod_canton")
    cliente.post("/predecir_emergencias", json=incompleta)

    monkeypatch.setattr(api, "modelo", ModeloConError())
    cliente_que_no_propaga = TestClient(api.app, raise_server_exceptions=False)
    assert cliente_que_no_propaga.post("/predecir_emergencias", json=ENTRADA).status_code == 500

    resumen = cliente.get("/metricas").json()
    assert resumen["peticiones"] == 3
    assert resumen["errores_validacion"] == 1
    assert resumen["errores_servidor"] == 1
    assert resumen["latencia_p50_ms"] > 0
    assert resumen["latencia_p95_ms"] > 0
