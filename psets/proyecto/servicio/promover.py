"""Mueve el alias de produccion a una version registrada."""

import argparse
import os

import mlflow
from mlflow import MlflowClient

from servicio.registrar_modelo import NOMBRE_MODELO


ALIAS = "produccion"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")


def promover(version):
    mlflow.set_tracking_uri(TRACKING_URI)
    cliente = MlflowClient()
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS, str(version))
    actual = cliente.get_model_version_by_alias(NOMBRE_MODELO, ALIAS)
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=int)
    argumentos = parser.parse_args()
    version = promover(argumentos.version)
    print(
        f"{NOMBRE_MODELO}@{ALIAS}: version {version.version}, "
        f"corrida {version.run_id}"
    )
