"""Ejercicio 2: zona, codificacion y disponibilidad en viajes_taxi.csv.

Corre el script con:

    python3 ejercicio2_features_zona.py

Al final se imprime una comprobacion por cada TODO. No la modifiques:
cuando las cuatro digan OK, pasa a las preguntas del README.
"""

import pandas as pd

viajes = pd.read_csv("viajes_taxi.csv")
columnas_originales = viajes.columns.tolist()
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# Parte 1: tarifa fija entre JFK y Manhattan.
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
desde_jfk = (viajes["zona_origen"] == "JFK Airport") & viajes["zona_destino"].isin(
    ZONAS_MANHATTAN
)
hacia_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (
    viajes["zona_destino"] == "JFK Airport"
)
viajes["es_tarifa_fija"] = (desde_jfk | hacia_jfk).astype(int)
tarifas_fijas = viajes.loc[viajes["es_tarifa_fija"] == 1, "tarifa"]
print("Viajes con tarifa fija:", len(tarifas_fijas))
print("Tarifa minima:", tarifas_fijas.min())
print("Tarifa maxima:", tarifas_fijas.max())

# Parte 2: codificacion one-hot de la zona de origen.
# (pd.get_dummies), con el prefijo "origen" y dtype=int. Imprime los
# nombres de las columnas nuevas.
codificado = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
print("Columnas one-hot:", codificado.columns.tolist())

# Parte 3: codificacion ordinal en orden alfabetico.
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas_ordenadas = sorted(viajes["zona_origen"].dropna().unique())
mapa_ordinal = {zona: numero for numero, zona in enumerate(zonas_ordenadas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(mapa_ordinal)
print("Mapa ordinal:", mapa_ordinal)

# Parte 4: disponibilidad de cada variable al solicitar el viaje.
# del dataset y True o False segun exista en el momento en que alguien
# pide el taxi (antes del viaje). Imprime la decision de cada una.
disponible_al_pedir = {
    "id_viaje": True,
    "fecha_hora_inicio": True,
    "zona_origen": True,
    "zona_destino": True,
    "distancia_km": True,
    "duracion_min": False,
    "pasajeros": True,
    "metodo_pago": True,
    "tarifa": False,
    "propina": False,
}
for columna in columnas_originales:
    print(f"{columna}: {disponible_al_pedir[columna]}")


def comprobar(numero, condicion):
    if condicion:
        print(f"TODO {numero}: OK")
    else:
        print(f"TODO {numero}: revisa")


print()
print("Comprobacion")
comprobar(1, "es_tarifa_fija" in viajes.columns and viajes["es_tarifa_fija"].sum() == 300)
comprobar(2, codificado is not None
          and len([columna for columna in codificado.columns if columna.startswith("origen_")]) == 8)
comprobar(3, "origen_ordinal" in viajes.columns
          and viajes.loc[viajes["zona_origen"] == "Brooklyn", "origen_ordinal"].iloc[0] == 0
          and viajes["origen_ordinal"].max() == 7)
comprobar(4, disponible_al_pedir is not None
          and disponible_al_pedir.get("duracion_min") is False
          and disponible_al_pedir.get("propina") is False
          and disponible_al_pedir.get("distancia_km") is True)
