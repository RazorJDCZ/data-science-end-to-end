"""Ejercicio 1: resumen, distribucion y atipicos de viajes_taxi.csv.

Corre el script antes de tocar nada:

    python3 ejercicio1_explorar.py

Todos los apartados estan resueltos. Los graficos quedan en figuras/:
abrelos despues de cada corrida.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("viajes_taxi.csv")

print("Paso 1: resumen")
print("Filas y columnas:", viajes.shape)
print("Valores faltantes por columna:")
print(viajes.isna().sum())
# Parte 1: estadistica descriptiva de las columnas numericas.
print(viajes.describe().round(2))

print()
print("Paso 2: distribucion de la tarifa")
plt.figure()
plt.hist(viajes["tarifa"], bins=50)
plt.title("Tarifa de cada viaje")
plt.xlabel("tarifa")
plt.ylabel("cantidad de viajes")
plt.savefig("figuras/hist_tarifa.png")
plt.close()

plt.figure()
plt.boxplot(viajes["tarifa"])
plt.title("Tarifa, diagrama de caja")
plt.savefig("figuras/caja_tarifa.png")
plt.close()
print("Figuras guardadas en figuras/")

print()
print("Paso 3: atipicos con la regla del IQR")
# Parte 3: limite superior de la regla del IQR.
q1 = viajes["tarifa"].quantile(0.25)
q3 = viajes["tarifa"].quantile(0.75)
iqr = q3 - q1
limite_superior = q3 + 1.5 * iqr
atipicos = viajes[viajes["tarifa"] > limite_superior]
print(f"Limite superior: {limite_superior:.2f}")
print(f"Viajes sobre el limite superior: {len(atipicos)}")
print(atipicos["tarifa"].value_counts().head(3))

print()
print("Paso 4: atipico no es lo mismo que error")
# Parte 4: separar errores confirmados de tarifas validas poco comunes.
errores_de_captura = (atipicos["tarifa"] == 999.99).sum()
tarifa_exacta_70 = (atipicos["tarifa"] == 70.00).sum()
print(f"Error de captura (999.99): {errores_de_captura}")
print(f"Tarifa exacta de 70.00: {tarifa_exacta_70}")
print(f"Duraciones negativas: {(viajes['duracion_min'] < 0).sum()}")
