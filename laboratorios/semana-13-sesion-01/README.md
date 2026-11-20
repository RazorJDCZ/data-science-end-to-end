# Juan Diego Cadena, 329220

## Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

La gobernanza de modelos es el conjunto de reglas y controles para saber que
modelos existen, quien los aprobo, con que datos y metricas se construyeron y
como se usan. Incluye el inventario, el versionado, la documentacion, la
trazabilidad de datos y experimentos, la validacion antes de produccion, el
monitoreo y las reglas para retirar o reemplazar una version.

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Guardaria el nombre y proposito del modelo, responsable, versiones, corrida y
fecha de entrenamiento, datos usados, hiperparametros, metricas, estado de
aprobacion, alias de produccion y artefactos. Una persona que no lo entreno
necesita esto para entender si sirve para su caso, reproducirlo, auditar una
prediccion y volver a una version anterior sin adivinar.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Guardaria con cada prediccion el numero de version o el `run_id`. Asi podria
buscar exactamente el modelo, sus metricas y su corrida. Para volver atras
moveria el alias `produccion` a la version anterior y reiniciaria el servicio,
sin borrar ni sobrescribir ningun artefacto.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

El error indica que no existe el modelo registrado `tarifa-taxis` o que no se
puede encontrar su alias. Eso confirma que entrenar y guardar metricas en una
corrida no crea por si solo una version en el registro.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Esperaba ver dos versiones. Al registrar la segunda, la primera siguio intacta
con su propia corrida y MAE; la version 2 se agrego, no reemplazo a la 1.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

Al arrancar con el alias en la version 2, `/salud` respondio version 2. Luego
movi el alias a la version 1 sin reiniciar y el servicio siguio respondiendo
version 2, porque el modelo se carga una sola vez al iniciar. Despues de
reiniciarlo cargo y respondio version 1.

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Fallaba `test_cada_prediccion_dice_que_version_la_calculo`, porque la respuesta
de `/predecir` no incluia `version_modelo`. `/salud` solo mostraba que version
estaba cargada al consultar el servicio; esta prueba comprueba que cada tarifa
individual queda asociada con la version que realmente la produjo.

**Lo que respondió el servicio:**

Con la version 2, de 8 neuronas, la tarifa fue **69.98** y la respuesta incluyo
`"version_modelo": 2`. Con la version 1, de 64 neuronas, fue **70.07** e
incluyo `"version_modelo": 1`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No aplica todavia: el repositorio no incluye el enunciado ni las opciones del
ConcepTest.

**Tu razonamiento:**

No escogi una letra sin conocer la pregunta. Mi criterio seria revisar si el
cambio propuesto conserva la trazabilidad, no sobrescribe versiones y permite
volver al modelo anterior.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

Queda pendiente la discusion presencial. Cambiaria mi voto si el argumento de
mi companero muestra un riesgo de versionado o despliegue que no habia tenido
en cuenta.

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

Quedo en produccion la **version 2**, el gradient boosting, con recall 0.661 y
precision 0.429. La version 3 fue la ultima registrada, pero su recall fue solo
0.102, por eso no cumplio la regla de superar a la version de produccion.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

La logistica muy regularizada tuvo precision 0.667, pero detecto muy pocos
fraudes y por eso la regla basada en recall la rechazo. Si las falsas alarmas
costaran mas, usaria una regla que exija una precision minima o una metrica de
costo que asigne valores distintos a falsos positivos y falsos negativos.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

La respuesta queda en las etiquetas de la version: `validacion=rechazada` y
`comparada_con=2`. Tambien se pueden revisar la corrida y sus metricas para ver
que el recall fue menor al de produccion.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Despues de ejecutar el ciclo dos veces quedaron **6 versiones**. La version 2
siguio en produccion porque las versiones 4 y 5 repitieron los mismos recalls,
y la regla exige una mejora estricta; la version 6 tambien fue peor.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Una version nueva pasa a produccion si mejora el MAE temporal y no empeora el
costo que penaliza el doble las subestimaciones. En este proyecto la decision
la hago yo despues de revisar las dos metricas; queda registrada en las
corridas de MLflow y en el alias `produccion`. La version 2, con 60 arboles,
quedo seleccionada con MAE 5.1611 y costo 7.9017.

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

El contenedor lleva una copia de `modelo.joblib` y el archivo
`modelo_version.txt`, generados al exportar el alias `produccion`. La API lee
ese numero al arrancar y lo devuelve en `/salud` y en cada prediccion, por lo
que no necesita consultar MLflow durante la operacion normal.

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Puede ocurrir que el servicio este disponible y sea rapido, pero que el error
del modelo aumente porque cambio la relacion entre los datos y la tarifa real.
Me enteraria guardando cada prediccion, uniendola despues con la verdad y
vigilando el MAE por periodo y por version.
