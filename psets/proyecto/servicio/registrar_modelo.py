"""Entrena y registra una version del modelo de emergencias.

Se ejecuta desde ``psets/proyecto``. El primer argumento permite probar
otra cantidad de arboles sin cambiar el codigo:

    python -m servicio.registrar_modelo 60
"""

import argparse
import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from modelado.pipeline_proyecto import cargar_y_dividir, construir_preparacion


NOMBRE_MODELO = "emergencias-ecu911"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")


def costo_subestimacion(y_real, y_estimado):
    """Error medio; subestimar una emergencia cuesta el doble."""
    diferencias = y_real.to_numpy() - y_estimado
    return float(((diferencias.clip(min=0) * 2) + (-diferencias).clip(min=0)).mean())


def registrar_modelo(n_estimadores=60):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("proyecto-emergencias-registro")
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()

    modelo = Pipeline([
        ("preparar", construir_preparacion()),
        ("modelo", RandomForestRegressor(
            n_estimators=n_estimadores,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )),
    ])
    modelo.fit(X_ent, y_ent)
    estimaciones = modelo.predict(X_prueba)
    mae = float(mean_absolute_error(y_prueba, estimaciones))
    costo = costo_subestimacion(y_prueba, estimaciones)

    with mlflow.start_run(run_name=f"bosque-{n_estimadores}-arboles"):
        mlflow.log_params({
            "n_estimators": n_estimadores,
            "max_depth": 18,
            "min_samples_leaf": 2,
        })
        mlflow.log_metrics({"mae": mae, "costo_subestimacion": costo})
        informacion = mlflow.sklearn.log_model(
            modelo,
            "modelo",
            registered_model_name=NOMBRE_MODELO,
        )

    return int(informacion.registered_model_version), mae, costo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimadores", nargs="?", type=int, default=60)
    argumentos = parser.parse_args()
    version, mae, costo = registrar_modelo(argumentos.n_estimadores)
    print(
        f"Version {version} registrada: MAE {mae:.4f}, "
        f"costo de subestimacion {costo:.4f}"
    )
