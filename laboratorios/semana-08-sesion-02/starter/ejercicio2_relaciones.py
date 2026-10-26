"""Ejercicio 2: relaciones entre variables de viajes_taxi.csv.

Corre el script con:

    python3 ejercicio2_relaciones.py

Al final se imprime una comprobacion por cada TODO. No la modifiques:
cuando las cinco digan OK, pasa a los hallazgos del README.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("viajes_taxi.csv")

# Parte 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
limpios = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()
print("Filas despues de quitar errores:", len(limpios))

# Parte 2: grafico de dispersion de distancia_km contra tarifa.
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general.
plt.figure()
plt.scatter(limpios["distancia_km"], limpios["tarifa"], s=5)
plt.title("Distancia vs tarifa")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.tight_layout()
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# Parte 3: resumen de propina por metodo de pago.
# y la cantidad de viajes con propina 0, por metodo de pago.
resumen_propina = limpios.groupby("metodo_pago").agg(
    propina_promedio=("propina", "mean"),
    viajes=("propina", "size"),
    viajes_propina_0=("propina", lambda valores: (valores == 0).sum()),
)
print("Resumen de propinas:")
print(resumen_propina.round(2))

# Parte 4: correlaciones numericas con la tarifa.
# pasajeros y propina con la tarifa.
columnas_correlacion = ["distancia_km", "duracion_min", "pasajeros", "propina", "tarifa"]
correlaciones = limpios[columnas_correlacion].corr()["tarifa"]
print("Correlaciones con la tarifa:")
print(correlaciones.drop("tarifa").round(3))

# Parte 5: pasajeros vacios y pasajeros registrados como cero.
pasajeros_vacios = limpios["pasajeros"].isna().sum()
pasajeros_cero = (limpios["pasajeros"] == 0).sum()
print("Pasajeros vacios:", pasajeros_vacios)
print("Pasajeros en cero:", pasajeros_cero)


def comprobar(numero, condicion):
    if condicion:
        print(f"TODO {numero}: OK")
    else:
        print(f"TODO {numero}: revisa")


if isinstance(correlaciones, pd.DataFrame):
    correlaciones = correlaciones["tarifa"]

print()
print("Comprobacion")
comprobar(1, len(limpios) == 2975)
comprobar(2, os.path.exists("figuras/distancia_vs_tarifa.png"))
comprobar(3, resumen_propina is not None and len(resumen_propina) == 2)
comprobar(4, correlaciones is not None and round(float(correlaciones["distancia_km"]), 2) == 0.96)
comprobar(5, pasajeros_vacios == 119 and pasajeros_cero == 19)
