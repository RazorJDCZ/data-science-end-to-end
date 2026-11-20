"""Exporta desde MLflow el modelo elegido a un archivo para el servicio.

Se ejecuta desde psets/proyecto, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py
"""

import os
from pathlib import Path

import joblib
import mlflow.sklearn
from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
NOMBRE_MODELO = "emergencias-ecu911"
RUTA_SALIDA = Path(__file__).with_name("modelo.joblib")
RUTA_VERSION = Path(__file__).with_name("modelo_version.txt")
RUTA_RUN_ID = Path(__file__).with_name("modelo_run_id.txt")
ALIAS = "produccion"


def exportar_modelo(
    ruta_salida=RUTA_SALIDA,
    ruta_version=RUTA_VERSION,
    ruta_run_id=RUTA_RUN_ID,
):
    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS)
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}/{version.version}")
    joblib.dump(modelo, ruta_salida)
    Path(ruta_version).write_text(str(version.version), encoding="utf-8")
    Path(ruta_run_id).write_text(version.run_id, encoding="utf-8")
    return Path(ruta_salida), str(version.version), version.run_id


if __name__ == "__main__":
    ruta, version, corrida = exportar_modelo()
    print(f"Version {version} (corrida {corrida}) exportada a {ruta}")
