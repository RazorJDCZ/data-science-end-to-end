# Juan Diego Cadena, 329220

## Predicción de emergencias ECU 911

Este proyecto estima cuántas emergencias tendrá cada cantón dentro de tres
días. La predicción está pensada para que el personal de operaciones pueda
anticipar la carga y planificar mejor sus recursos.

**Métrica de éxito:** se revisa el MAE temporal y un costo donde subestimar una
emergencia pesa el doble que sobreestimarla. El modelo elegido obtuvo MAE
temporal promedio de 5.6736 y costo promedio de 8.3965.

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| Portal de Datos Abiertos del ECU 911 | Emergencias mensuales por fecha, cantón, parroquia, servicio y subtipo | API pública `package_show?id=base-de-emergencias` | `laboratorios/semana-08-sesion-02/data/ecu911_emergencias_raw.csv` |
| Muestra del proyecto | Una fila por cantón y día, del 1 de abril al 30 de septiembre de 2026; 40.992 filas y 224 cantones | Scripts locales de unificación, preparación y muestreo | `laboratorios/data/obt_emergencias_muestra_6_meses.csv` |

Los datos crudos grandes no se guardan en Git. Desde la raíz del repositorio se
pueden regenerar en este orden:

```bash
python laboratorios/unificar_emergencias_raw.py
python laboratorios/preparar_obt_emergencias.py
python laboratorios/crear_muestra_ecu911_6_meses.py
```

## Cómo correr el pipeline de datos

La preparación del modelo se puede comprobar con:

```bash
python psets/proyecto/modelado/pipeline_proyecto.py
```

El script carga la muestra, agrega `lag_4d` y `es_fin_semana_t3`, y divide los
datos por fecha: las observaciones antiguas quedan en entrenamiento y las más
recientes en prueba.

## Cómo entrenar y evaluar el modelo

Los experimentos se registraron en MLflow con el notebook
`laboratorios/semana-09-sesion-02/plantilla-pipeline/experimentos_proyecto.ipynb`.
La validación adicional y las respuestas están documentadas en los README de la
Semana 11. Para revisar el código que protege la preparación:

```bash
pytest psets/proyecto -v
```

## Modelo final

Después de la validación temporal elegí el **bosque aleatorio**, porque fue más
estable que la red neuronal y tuvo menor costo al penalizar las emergencias
subestimadas. En el registro de MLflow se llama `emergencias-ecu911`. La
version 2, con 60 arboles, tiene el alias `produccion`; obtuvo MAE 5.1611 y
costo de subestimacion 7.9017. La decision de servirlo por lotes y el criterio
de versionado estan explicados en
[`docs/despliegue.md`](docs/despliegue.md).

## Cómo correr el servicio

Primero se levanta MLflow. Desde `psets/proyecto` se pueden registrar dos
versiones, promover la mejor y exportar el alias de produccion:

```bash
python -m servicio.registrar_modelo 40
python -m servicio.registrar_modelo 60
python -m servicio.promover 2
python servicio/exportar_modelo.py
```

La exportacion genera localmente `modelo.joblib`, `modelo_version.txt` y
`modelo_run_id.txt`; no se suben al repositorio.

El servicio se puede probar directamente:

```bash
uvicorn servicio.api:app --port 8000
```

O se construye y ejecuta en Docker desde `psets/proyecto`:

```bash
docker build -f servicio/Dockerfile -t emergencias-api .
docker run --rm -p 8000:8000 emergencias-api
```

La documentación interactiva queda en `http://localhost:8000/docs`. Ejemplo de
petición:

```bash
curl -X POST http://localhost:8000/predecir_emergencias \
  -H "Content-Type: application/json" \
  -d '{"cod_canton":"0101","fecha":"2026-09-03","n_emergencias":20,"lag_1d":18,"n_emergencias_hace_4_dias":16,"lag_7d":15,"lag_14d":14,"lag_28d":13,"media_prev_7d":17,"media_prev_28d":15,"es_feriado_t3":0}'
```

Respuesta observada:

```json
{
  "id_prediccion": "<identificador generado>",
  "cod_canton": "0101",
  "fecha_objetivo": "2026-09-06",
  "emergencias_estimadas": 20.28,
  "run_id": "4aa7d7dab1af451bb26c972663f6a7bf",
  "version_modelo": "2"
}
```

## Como revisar el monitoreo

La API cuenta peticiones, errores y latencias en `/metricas`. Tambien guarda
cada prediccion en `monitoreo/predicciones.jsonl`. Cuando ya existe el conteo
real para la fecha objetivo, el MAE semanal por version se calcula con:

```bash
python monitoreo/desempeno.py
```

Las metricas, sus fuentes y umbrales estan documentados en
[`docs/monitoreo.md`](docs/monitoreo.md).
