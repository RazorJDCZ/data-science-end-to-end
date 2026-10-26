from pathlib import Path

import pandas as pd


RUTA_DATOS = Path(__file__).parents[1] / "starter" / "viajes_taxi.csv"
ZONAS_MANHATTAN = [
    "Midtown",
    "Upper East Side",
    "Upper West Side",
    "Lower Manhattan",
    "Harlem",
]

viajes = pd.read_csv(RUTA_DATOS)
columnas_originales = viajes.columns.tolist()
viajes = viajes.query("tarifa != 999.99 and duracion_min >= 0").copy()

# Parte 1: marcar los viajes entre JFK y una zona de Manhattan.
desde_jfk = (viajes["zona_origen"] == "JFK Airport") & viajes[
    "zona_destino"
].isin(ZONAS_MANHATTAN)
hacia_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (
    viajes["zona_destino"] == "JFK Airport"
)
viajes["es_tarifa_fija"] = (desde_jfk | hacia_jfk).astype(int)

tarifas_fijas = viajes.loc[viajes["es_tarifa_fija"] == 1, "tarifa"]
print("\n=== 1. FEATURE DE TARIFA FIJA ===")
print(f"Viajes marcados: {len(tarifas_fijas)}")
print(f"Tarifa minima: {tarifas_fijas.min():.2f}")
print(f"Tarifa maxima: {tarifas_fijas.max():.2f}")

# Parte 2: crear una columna binaria por cada zona de origen.
origen_one_hot = pd.get_dummies(
    viajes["zona_origen"], prefix="origen", dtype=int
)
viajes_one_hot = pd.concat([viajes, origen_one_hot], axis=1)

print("\n=== 2. ONE-HOT ENCODING ===")
print(f"Columnas creadas: {len(origen_one_hot.columns)}")
print(origen_one_hot.columns.tolist())
print(origen_one_hot.head().to_string(index=False))

# Parte 3: asignar enteros segun el orden alfabetico de las zonas.
zonas_ordenadas = sorted(viajes["zona_origen"].dropna().unique())
mapa_ordinal = {zona: numero for numero, zona in enumerate(zonas_ordenadas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(mapa_ordinal)

print("\n=== 3. CODIFICACION ORDINAL ===")
print(f"Columnas creadas: 1")
print(f"Mapa usado: {mapa_ordinal}")
print(
    viajes[["zona_origen", "origen_ordinal"]]
    .drop_duplicates()
    .sort_values("origen_ordinal")
    .to_string(index=False)
)

# Parte 4: revisar la disponibilidad de cada columna al pedir el taxi.
# La distancia se considera una estimacion de la ruta, no la medida final.
disponibilidad = {
    "id_viaje": (True, "el sistema puede asignarlo al crear la solicitud"),
    "fecha_hora_inicio": (True, "se conoce la fecha y hora de la solicitud"),
    "zona_origen": (True, "el pasajero indica donde inicia"),
    "zona_destino": (True, "el pasajero indica a donde va"),
    "distancia_km": (True, "se puede estimar con la ruta antes del viaje"),
    "duracion_min": (False, "la duracion real se conoce al terminar"),
    "pasajeros": (True, "se puede solicitar antes de asignar el vehiculo"),
    "metodo_pago": (True, "se puede seleccionar al pedir el viaje"),
    "tarifa": (False, "es el resultado que se quiere estimar"),
    "propina": (False, "se decide y registra al final"),
}

print("\n=== 4. DISPONIBILIDAD AL MOMENTO DE PREDECIR ===")
for columna in columnas_originales:
    existe, motivo = disponibilidad[columna]
    respuesta = "SI" if existe else "NO"
    print(f"{columna:20} {respuesta:2} - {motivo}")

# Se usa la variable para dejar claro que el DataFrame codificado fue creado.
print(f"\nForma del DataFrame con one-hot: {viajes_one_hot.shape}")
