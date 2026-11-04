# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada, y tu `pipeline_proyecto.py`.

Como la sesión pasada, los laboratorios son notebooks que importan la preparación de un script (`preparacion_taxis.py` en los Ejercicios 1 y 2, tu `pipeline_proyecto.py` en el 3).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

El aprendizaje automático clásico conviene para datos tabulares y etiquetados, cuando la tarea está bien definida, como clasificación, regresión o pronóstico. Un LLM conviene más cuando la entrada o la salida principal es lenguaje: resumir, conversar, extraer información de texto o generar contenido.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

Para mi proyecto pesa más el tipo de datos. Tengo una tabla con conteos diarios, cantones, calendario y rezagos, y quiero predecir un número. Un modelo clásico puede aprender directamente esa relación, cuesta poco al reutilizarlo y da una salida más estable que pedir el conteo a un LLM.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Mediría el tiempo de entrenamiento, la latencia de cada predicción, el costo por consulta, el uso de memoria, la estabilidad al repetir una entrada y la posibilidad de explicar por qué se estimó una tarifa.

## Ejercicio 1: una red neuronal frente al ensamblaje

Notebook: [`starter/ejercicio1_red_neuronal.ipynb`](./starter/ejercicio1_red_neuronal.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). Ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división, importadas de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo completo una vez antes de tocar nada (**Run All**).

La solución que ya se realizó se conserva en [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py).

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

La advertencia significa que la red llegó al límite de iteraciones antes de terminar de optimizar sus pesos. Al permitir hasta 1000 iteraciones esperaba que convergiera y redujera su error. La advertencia desapareció, terminó en 692 iteraciones y obtuvo MAE 1,1695 en prueba, mejor que el 1,2754 del ensamblaje en esta corrida.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

Aprende 1.729 números: `25 × 64 = 1600` pesos de entrada, 64 sesgos de la capa oculta, 64 pesos hacia la salida y 1 sesgo de salida.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

Esperaba que la distancia fuera la más importante, y obtuvo importancia 0,9824. Para explicar una predicción usaría el ensamblaje porque permite revisar importancias y el recorrido de sus árboles. La red neuronal mezcla 1.729 parámetros y es menos directa de explicar.

**Lo que confirmó la salida, TODO por TODO:**

Gradient boosting obtuvo MAE 1,0705 en entrenamiento y 1,2754 en prueba en 0,195 segundos. La red obtuvo 1,0882 y 1,1695 en 7,388 segundos. La red aprendió 1.729 parámetros y terminó en 692 iteraciones. Las tres columnas más importantes del ensamblaje fueron `distancia_km` (0,9824), `es_tarifa_fija` (0,0120) y `es_hora_pico` (0,0028).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No puedo elegir A, B, C o D porque el repositorio no incluye el enunciado ni las opciones del ConcepTest.

**Tu razonamiento:**

Necesitaría conocer la pregunta y el significado de cada alternativa. Elegir una letra sin esa información sería inventarla.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica todavía. Lo completaría durante la clase después de conocer el enunciado y escuchar el razonamiento de mi compañero.

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Notebook: [`starter/ejercicio2_tres_soluciones.ipynb`](./starter/ejercicio2_tres_soluciones.ipynb), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

La solución anterior permanece en [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), junto con los prompts y respuestas generados.

1. Completa los `TODO` 1 a 3 y corre las celdas hasta la que escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y corre sus celdas y la de la comprobación hasta que todas digan `OK`. Los modelos ya están entrenados en el kernel: no hace falta volver a correr las celdas de arriba. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé el LLM de esta conversación para responder el prompt. Para estimar un posible costo de API tomé como referencia GPT-5.4 mini, cuyo precio oficial de entrada consultado el 3 de noviembre de 2026 es 0,75 USD por millón de tokens.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0,8957 | 0 USD (0,27 s local) | 0 USD de API | Sí, mediante importancias |
| Red neuronal | 0,9637 | 0 USD (8,75 s local) | 0 USD de API | Difícil |
| LLM | 1,4045 | 0 USD para nosotros | 0,2721 USD | Puede generar una explicación, pero no garantiza que sea fiel |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No dio exactamente lo mismo: una segunda respuesta manual cambió 17 de los 20 números. Esa variación es peligrosa en una app de cobro porque dos pasajeros con el mismo viaje podrían recibir tarifas distintas. Para precios se necesita una salida determinista, versionada y fácil de auditar.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor fue un viaje de JFK Airport a JFK Airport de 1,01 km. El valor real fue 11,46 USD y el LLM respondió 6,80, con error 4,66. Es particular porque sale y llega al aeropuerto, pero no es la ruta fija entre JFK y Manhattan. El ensamblaje pudo ver más combinaciones en sus 2.380 filas y aprender esa excepción mejor que el prompt con solo 30 ejemplos.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Evaluar las 595 filas exigiría prompts más largos o muchas consultas, con más costo y trabajo para conservar el orden. Una muestra de 20 permite hacer la práctica, pero su MAE tiene mucha variación y puede depender de qué viajes salieron. La decisión principal debería usar el conjunto completo de prueba.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí tu preparación y tu división, y entrene el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

| Opción | MAE prueba completa | MAE en las mismas 10 filas | Entrenamiento | Costo estimado para 10.000 | Explicabilidad |
|---|---:|---:|---:|---:|---|
| Bosque aleatorio | 5,1231 | 31,4083 | 36,395 s | 0 USD de API | Media; permite revisar importancias |
| Red neuronal | 4,9613 | 18,1299 | 37,607 s; 15.105 parámetros | 0 USD de API | Baja |
| LLM | No medido | 26,1000 | No se entrenó localmente | 0,4174 USD | Puede dar texto, pero no garantiza fidelidad |

Usaría el bosque aleatorio. Su MAE completo es solo 0,1618 mayor que el de la red, pero es más sencillo de explicar, siempre produce lo mismo con el mismo modelo y no cobra por cada consulta. No escogería según las 10 filas porque esa muestra salió difícil y cambió mucho el orden de los modelos. El LLM quedó sin evaluación completa y agrega costo y variación innecesarios para predecir un conteo tabular.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Revisaría la versión exacta del CSV, qué filas se limpiaron, las columnas usadas, la división y su semilla, el pipeline, los valores imputados, los hiperparámetros, la versión de scikit-learn y la métrica. También comprobaría que ambos ajustaron el preprocesamiento solo con entrenamiento.

## Datos, precio y ejecución

El proyecto usa `laboratorios/data/obt_emergencias_muestra_6_meses.csv`, con datos de abril a septiembre de 2026. El precio usado para el cálculo del LLM está documentado en la [página oficial de GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

Desde `semana-10-sesion-02/starter`:

```powershell
docker compose build
docker compose run --rm ejercicio1
docker compose run --rm ejercicio2
docker compose run --rm proyecto
```

El último comando crea `proyecto/comparacion_proyecto.csv` y los archivos de la prueba del LLM.
