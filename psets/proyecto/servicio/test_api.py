"""Pruebas del contrato de la API, sin usar el modelo guardado."""

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


modelo_de_prueba = ModeloDePrueba()
api.modelo = modelo_de_prueba
cliente = TestClient(api.app)


def test_salud_confirma_que_el_modelo_esta_cargado():
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["modelo_cargado"] is True


def test_prediccion_valida_calcula_features_y_responde():
    respuesta = cliente.post("/predecir_emergencias", json=ENTRADA)

    assert respuesta.status_code == 200
    assert respuesta.json()["emergencias_estimadas"] == 25.5
    assert respuesta.json()["fecha_objetivo"] == "2026-09-06"
    assert modelo_de_prueba.ultima_entrada.iloc[0]["lag_4d"] == 16
    assert modelo_de_prueba.ultima_entrada.iloc[0]["es_fin_semana_t3"] == 1
    assert list(modelo_de_prueba.ultima_entrada.columns) == NUMERICAS + CATEGORICAS + YA_LISTAS


def test_entrada_sin_canton_se_rechaza():
    entrada = ENTRADA.copy()
    entrada.pop("cod_canton")
    respuesta = cliente.post("/predecir_emergencias", json=entrada)

    assert respuesta.status_code == 422
