"""Compara bosque, red neuronal y LLM en el proyecto ECU 911."""

import os
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DIRECTORIO = Path(__file__).resolve().parent
RUTA_PREDETERMINADA = (
    DIRECTORIO.parents[1] / "data" / "obt_emergencias_muestra_6_meses.csv"
)
RUTA_DATOS = Path(os.environ.get("OBT_CSV", RUTA_PREDETERMINADA))
RUTA_PROMPT = DIRECTORIO / "prompt_llm_proyecto.txt"
RUTA_RESPUESTAS = DIRECTORIO / "respuestas_llm_proyecto.txt"
RUTA_RESULTADOS = DIRECTORIO / "comparacion_proyecto.csv"

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
LEGIBLES = [
    "cod_canton",
    "n_emergencias",
    "lag_4d",
    "lag_7d",
    "media_prev_7d",
    "es_fin_semana_t3",
    "es_feriado_t3",
]

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

if not RUTA_DATOS.exists():
    raise FileNotFoundError(
        f"No se encontro {RUTA_DATOS}. Ejecuta crear_muestra_ecu911_6_meses.py."
    )

datos = pd.read_csv(
    RUTA_DATOS,
    usecols=columnas_archivo,
    dtype={"cod_canton": str},
    parse_dates=["fecha"],
)
datos = datos.sort_values(["cod_canton", "fecha"]).copy()
datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)
dia_semana_t3 = (datos["dia_semana"] + 2) % 7 + 1
datos["es_fin_semana_t3"] = dia_semana_t3.isin([6, 7]).astype(int)
datos = datos.dropna(subset=[OBJETIVO]).copy()

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]
fechas = datos["fecha"].drop_duplicates().sort_values().reset_index(drop=True)
fecha_corte = fechas.iloc[int(len(fechas) * 0.8)]
es_entrenamiento = datos["fecha"] < fecha_corte
X_ent = X.loc[es_entrenamiento]
X_prueba = X.loc[~es_entrenamiento]
y_ent = y.loc[es_entrenamiento]
y_prueba = y.loc[~es_entrenamiento]

muestra = X_prueba.sample(10, random_state=7)
y_muestra = y_prueba.loc[muestra.index]
ejemplos = X_ent.sample(30, random_state=7)
y_ejemplos = y_ent.loc[ejemplos.index]


def crear_preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(
                handle_unknown="ignore", sparse_output=True
            )),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


def entrenar(modelo):
    pipeline = Pipeline([
        ("preparar", crear_preparacion()),
        ("modelo", modelo),
    ])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    return pipeline, time.perf_counter() - inicio


bosque, segundos_bosque = entrenar(RandomForestRegressor(
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1,
))

red, segundos_red = entrenar(MLPRegressor(
    hidden_layer_sizes=(64,),
    early_stopping=True,
    max_iter=300,
    random_state=42,
))

modelo_red = red.named_steps["modelo"]
parametros_red = sum(matriz.size for matriz in modelo_red.coefs_)
parametros_red += sum(vector.size for vector in modelo_red.intercepts_)

filas_ejemplo = ejemplos[LEGIBLES].copy()
filas_ejemplo[OBJETIVO] = y_ejemplos.values
texto_prompt = (
    "Estima la cantidad de emergencias de cada canton dentro de tres dias. "
    "Abajo hay 30 ejemplos y 10 filas sin respuesta. Responde solamente con "
    "10 numeros no negativos, uno por linea y en el mismo orden.\n\n"
    "Ejemplos con respuesta:\n"
    + filas_ejemplo.to_csv(index=False)
    + "\nFilas sin respuesta:\n"
    + muestra[LEGIBLES].to_csv(index=False)
)
RUTA_PROMPT.write_text(texto_prompt, encoding="utf-8")

predicciones_llm = None
mae_llm = None
if RUTA_RESPUESTAS.exists():
    predicciones_llm = [
        float(linea.strip())
        for linea in RUTA_RESPUESTAS.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    if len(predicciones_llm) != len(muestra):
        raise ValueError("respuestas_llm_proyecto.txt debe tener 10 numeros.")
    mae_llm = mean_absolute_error(y_muestra, predicciones_llm)

mae_bosque_prueba = mean_absolute_error(y_prueba, bosque.predict(X_prueba))
mae_red_prueba = mean_absolute_error(y_prueba, red.predict(X_prueba))
mae_bosque_muestra = mean_absolute_error(y_muestra, bosque.predict(muestra))
mae_red_muestra = mean_absolute_error(y_muestra, red.predict(muestra))

# Precio oficial de entrada de GPT-5.4 mini consultado el 2026-11-03.
precio_millon_entrada = 0.75
tokens_por_prompt = len(texto_prompt) / 4
costo_llm_10000 = tokens_por_prompt * 1000 / 1_000_000 * precio_millon_entrada

filas = [
    {
        "opcion": "bosque aleatorio",
        "mae_prueba_completa": round(mae_bosque_prueba, 4),
        "mae_muestra_10": round(mae_bosque_muestra, 4),
        "segundos_entrenamiento": round(segundos_bosque, 3),
        "parametros_red": "",
        "costo_10000_usd": 0.0,
        "explicabilidad": "media: permite revisar importancias",
    },
    {
        "opcion": "red neuronal",
        "mae_prueba_completa": round(mae_red_prueba, 4),
        "mae_muestra_10": round(mae_red_muestra, 4),
        "segundos_entrenamiento": round(segundos_red, 3),
        "parametros_red": parametros_red,
        "costo_10000_usd": 0.0,
        "explicabilidad": "baja",
    },
    {
        "opcion": "LLM",
        "mae_prueba_completa": "",
        "mae_muestra_10": "" if mae_llm is None else round(mae_llm, 4),
        "segundos_entrenamiento": 0.0,
        "parametros_red": "",
        "costo_10000_usd": round(costo_llm_10000, 4),
        "explicabilidad": "puede dar texto, pero no garantiza fidelidad",
    },
]
tabla = pd.DataFrame(filas)
tabla.to_csv(RUTA_RESULTADOS, index=False)

print("Archivo:", RUTA_DATOS)
print("Fecha de corte:", fecha_corte.date())
print("Entrenamiento:", X_ent.shape, "Prueba:", X_prueba.shape)
print("Parametros de la red:", parametros_red)
print()
print(tabla.to_string(index=False))
print("\nPrompt:", RUTA_PROMPT)
if mae_llm is None:
    print("Falta completar:", RUTA_RESPUESTAS)
else:
    print("Respuestas LLM:", RUTA_RESPUESTAS)
print("Resultados:", RUTA_RESULTADOS)
