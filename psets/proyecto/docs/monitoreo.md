# Monitoreo del modelo de emergencias

El servicio se usa para una prediccion diaria por canton. Las metricas de
sistema se conocen inmediatamente. El conteo real de emergencias se conoce al
cerrar la fecha objetivo, tres dias despues de preparar la entrada.

| Metrica | Tipo | Fuente | Cuando se conoce | Umbral de alerta |
|---|---|---|---|---|
| Disponibilidad | Sistema | Codigos HTTP de `/predecir_emergencias` | En cada peticion | Mas de 1 % de respuestas 500 o superiores en un dia |
| Errores de validacion | Sistema | Respuestas 422 | En cada peticion | Mas de 2 % en un dia |
| Latencia p50 y p95 | Sistema | Middleware de la API | En cada peticion | p95 mayor a 500 ms |
| Volumen de predicciones | Sistema | Registro JSONL | El mismo dia | Falta una prediccion para algun canton esperado |
| MAE por semana y version | Modelo | Registro JSONL unido con la muestra OBT | Al cerrar la fecha objetivo, hasta tres dias despues | MAE mayor a 6.0 durante dos semanas seguidas |
| Promedio de la prediccion | Modelo indirecta | Registro JSONL | El mismo dia | Cambio mayor a 30 % frente a las cuatro semanas anteriores |

La media de las predicciones sirve como senal temprana, pero no demuestra que
el modelo se haya equivocado. El MAE necesita el conteo real y es la medida que
confirma el deterioro. El script `monitoreo/desempeno.py` calcula el MAE por
semana y por version para no mezclar el comportamiento de modelos distintos.
