# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

Hay que versionar tres cosas: el código con Git, los datos con DVC y el modelo con MLflow. Entendí que no basta con poder volver al mismo código, porque el resultado también depende de la versión de los datos y del modelo que salió de cada entrenamiento.

**¿Por qué no alcanza con Git para versionar los datos?**

Git está pensado principalmente para archivos de código y texto. Un dataset puede ser grande, binario o cambiar muchas veces, y guardarlo completo en cada versión hace crecer demasiado el repositorio. DVC guarda en Git la referencia y la huella de la versión, mientras el archivo pesado puede vivir en otro almacenamiento.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Necesitaría conocer los hiperparámetros, la semilla y el tipo de división, qué filas quedaron en entrenamiento y prueba, la huella del archivo, la versión del código y las librerías. También compararía la métrica y, si es posible, cargaría el artefacto del modelo para comprobar la corrida sin entrenarlo otra vez.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Los dos resultados son correctos para la división que usó cada uno. La salida permite ver que uno dividió con semilla 42 y obtuvo MAE 1.2754, mientras el otro usó semilla 7 y obtuvo 1.1867. Esa impresión por sí sola no demuestra la huella de los datos, todos los parámetros ni conserva el modelo que produjo el resultado.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La diferencia más clara es la semilla de la división, porque cambia qué viajes llegan al conjunto de prueba. En la tabla esperaba ver `semilla_division=42` en la primera corrida y `semilla_division=7` en la del compañero; la huella del CSV debía ser la misma.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para poder recuperar exactamente el pipeline ya ajustado y volver a predecir sin repetir el entrenamiento. También sirve para comprobar que la métrica registrada salió de ese artefacto y no de otro modelo parecido entrenado después.

**Lo que mostró la tabla del paso 5:**

La tabla mostró:

- Bosque aleatorio: MAE de prueba 1.3016, semilla 42 y huella `cc636c7813cf`.
- Gradient boosting propio: MAE 1.2754, semilla 42 y la misma huella.
- Gradient boosting del compañero: MAE 1.1867, semilla 7 y la misma huella.

Esto confirmó que el archivo no cambió; lo que cambió fue la división.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No pude marcar una letra porque el README no incluye el enunciado ni las opciones del ConcepTest. No quise inventar una alternativa.

**Tu razonamiento:**

Mi criterio habría sido elegir la opción que exige comparar las corridas con la misma versión de datos, la misma división y los parámetros registrados, no solo mirar cuál número es menor.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No hay una discusión de pareja registrada en el material. Después de ejecutar el ejercicio sí reforcé la idea de que una mejora en una sola división no prueba por sí sola que un modelo sea mejor.

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

Quedó primero la red neuronal con MAE 1.1695. El segundo fue gradient boosting con 1.2754, así que la diferencia fue aproximadamente 0.1058 de MAE.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Coincide porque el paso carga el mismo pipeline entrenado que quedó guardado como artefacto y lo evalúa sobre el mismo `X_prueba`. No garantiza que el modelo sea estable con otra división, que generalice a datos futuros ni que siga siendo el mejor si cambia la distribución de los viajes.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

La huella `huella_datos` sería distinta aunque se modificara un solo byte. Al comparar la nueva corrida con las anteriores vería que ya no se entrenó con la misma versión del CSV.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Además del error, MLflow muestra su capa de 64 neuronas, `max_iter=1000`, la semilla 42, el tiempo de entrenamiento, la huella de los datos, el número de filas y el artefacto del pipeline. Una tabla manual podría olvidar alguno de esos datos o no permitir cargar el modelo.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

En esta primera selección elegí la **red neuronal**. Su `run_id` es `cfe8821f136546399f2c313a53586424`: obtuvo MAE 5.4637 en entrenamiento y 4.9613 en prueba, usando 32.256 filas para entrenar y 8.064 para prueba. El bosque quedó segundo con 5.1231, una diferencia de 0.1617. El artefacto cargado reprodujo exactamente el MAE 4.9613453737.

Los entregables quedaron junto al pipeline:

- [`experimentos_proyecto.ipynb`](../semana-09-sesion-02/plantilla-pipeline/experimentos_proyecto.ipynb)
- [`corridas_proyecto.csv`](../semana-09-sesion-02/plantilla-pipeline/corridas_proyecto.csv)

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita el mismo CSV, el mismo código de preparación, la fecha de corte, los hiperparámetros, las semillas y versiones compatibles de las librerías. MLflow registró la huella `c0c505ad3ef8`, el corte `2026-08-25`, los parámetros, las métricas y el modelo entrenado. El repositorio sí contiene el CSV, el pipeline, el notebook y los requisitos, pero la carpeta local `.mlflow` no se sube; por eso, otra persona puede repetir el entrenamiento y verificar la métrica, aunque no obtiene el artefacto local de mi corrida solo con clonar Git.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Sí cambió al probar más divisiones. Para la red neuronal, la validación al azar dio MAE promedio 5.6465 con desviación 0.1641, mientras la temporal dio 6.4700 con desviación 2.3505 y pliegues entre 4.9402 y 9.1765. Esa variación no aparecía en la corrida original: MLflow había registrado el resultado de una sola fecha de corte.
