from pathlib import Path

import numpy as np
import pandas as pd


RUTA_DATOS = Path(__file__).with_name("viajes_taxi.csv")

# Parte 1: cargar y revisar los datos originales.
viajes = pd.read_csv(RUTA_DATOS)
print("\n=== 1. DATOS ORIGINALES ===")
print(f"Filas: {len(viajes)} | Columnas: {len(viajes.columns)}")
print(viajes.head().to_string(index=False))

# Parte 2: quitar solamente los errores confirmados durante el EDA.
viajes_clean = viajes.query("tarifa != 999.99 and duracion_min >= 0").copy()
print("\n=== 2. DATOS LIMPIOS ===")
print(f"Filas conservadas: {len(viajes_clean)} de {len(viajes)}")
print(f"Filas eliminadas: {len(viajes) - len(viajes_clean)}")

# Parte 3: extraer informacion de la fecha que si existe al pedir el taxi.
fecha_inicio = pd.to_datetime(viajes_clean["fecha_hora_inicio"])
viajes_clean["hora"] = fecha_inicio.dt.hour
viajes_clean["dia_semana"] = fecha_inicio.dt.dayofweek
viajes_clean["es_fin_semana"] = viajes_clean["dia_semana"].isin([5, 6]).astype(int)
viajes_clean["es_hora_pico"] = (
    viajes_clean["hora"].between(7, 9)
    | viajes_clean["hora"].between(16, 19)
).astype(int)

print("\n=== 3. FEATURES DE FECHA Y HORA ===")
print(
    viajes_clean[
        ["fecha_hora_inicio", "hora", "dia_semana", "es_fin_semana", "es_hora_pico"]
    ].head(10).to_string(index=False)
)

# Parte 4: representar la hora como un ciclo. Asi, las horas 23 y 0 quedan cerca.
viajes_clean["hora_sin"] = np.sin(2 * np.pi * viajes_clean["hora"] / 24)
viajes_clean["hora_cos"] = np.cos(2 * np.pi * viajes_clean["hora"] / 24)

print("\n=== 4. CODIFICACION CICLICA DE LA HORA ===")
print(
    viajes_clean[
        ["fecha_hora_inicio", "hora", "hora_sin", "hora_cos"]
    ].head(10).round(3).to_string(index=False)
)

# Parte 5: comparar tarifa y duracion dentro y fuera de hora pico.
comparacion_hora_pico = viajes_clean.groupby("es_hora_pico").agg(
    viajes=("id_viaje", "count"),
    tarifa_promedio=("tarifa", "mean"),
    duracion_promedio=("duracion_min", "mean"),
)

print("\n=== 5. COMPARACION POR HORA PICO ===")
print(comparacion_hora_pico.round(2).to_string())
