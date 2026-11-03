# Muestra local del proyecto ECU 911

`obt_emergencias_muestra_6_meses.csv` contiene los seis meses completos más
recientes de la OBT local: abril a septiembre de 2026.

- 40.992 filas y 37 columnas.
- 224 cantones.
- Una fila por cantón y fecha.
- Conserva las columnas y tipos de la OBT usada en las semanas 8 y 9.

Para regenerarla desde la OBT completa:

```powershell
cd laboratorios/semana-09-sesion-01/starter
docker compose run --rm --build muestra-proyecto
```

El script que realiza el recorte y valida fechas, duplicados y número de filas es
`laboratorios/crear_muestra_ecu911_6_meses.py`.
