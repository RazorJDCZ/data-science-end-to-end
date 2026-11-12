"""Exporta desde MLflow el modelo elegido a un archivo para el servicio.

Se ejecuta desde psets/proyecto, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


# Es el bosque que resulto mas estable en la validacion temporal de la Semana 11.
RUN_ID = os.environ.get("MLFLOW_RUN_ID", "e60512928d4f4d93afec8350fe9f1fc3")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
RUTA_SALIDA = Path(__file__).with_name("modelo.joblib")


def exportar_modelo(run_id=RUN_ID, ruta_salida=RUTA_SALIDA):
    mlflow.set_tracking_uri(TRACKING_URI)
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    joblib.dump(modelo, ruta_salida)
    return Path(ruta_salida)


if __name__ == "__main__":
    ruta = exportar_modelo()
    print(f"Modelo {RUN_ID} exportado a {ruta}")
