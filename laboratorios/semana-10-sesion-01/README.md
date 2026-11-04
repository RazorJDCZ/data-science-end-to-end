# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina las predicciones de varios modelos para obtener una respuesta más estable. En bagging los modelos se entrenan de manera independiente sobre muestras distintas y al final se promedian. En boosting se entrenan uno después de otro, y cada modelo nuevo intenta corregir los errores que dejaron los anteriores.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque los árboles no cometen exactamente los mismos errores. Al promediar árboles construidos con muestras y variables distintas, parte de los errores se cancela y baja la variación de la predicción. Esto funciona mejor cuando los árboles son diversos y cada uno aporta alguna señal útil.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Está sobreajustando. Aprendió detalles y ruido del entrenamiento que no se repiten igual en datos nuevos, por eso la brecha entre entrenamiento y prueba aumenta.

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

La solución anterior se conserva en [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py) y sus métricas en [`starter/resultados_ejercicio1.csv`](./starter/resultados_ejercicio1.csv).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Esperaba un error cercano a 12 o 13 dólares; la salida dio 12,5308 en prueba. Sirve como punto de comparación mínimo: un modelo que usa las columnas solo vale la pena si mejora claramente esa referencia.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

Esperaba un error casi cero en entrenamiento y bastante mayor en prueba. Ocurrió eso: el MAE fue 0,0029 en entrenamiento y 1,7017 en prueba, con una brecha de 1,6988.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad esperaba una brecha menor, porque el árbol ya no puede memorizar cada viaje. Bajó de 1,6988 a 0,2630. El bosque redujo el error de prueba frente al árbol sin límite, aunque su brecha fue 0,8276; en esta corrida el árbol de profundidad 8 quedó un poco mejor en prueba que el bosque.

**Lo que confirmó la tabla, TODO por TODO:**

| Modelo | MAE entrenamiento | MAE prueba | Brecha |
|---|---:|---:|---:|
| Línea base | 13,1998 | 12,5308 | -0,6690 |
| Árbol sin límite | 0,0029 | 1,7017 | 1,6988 |
| Árbol `max_depth=8` | 1,0098 | 1,2728 | 0,2630 |
| Bosque aleatorio | 0,4739 | 1,3016 | 0,8276 |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No puedo elegir A, B, C o D porque el repositorio no incluye el enunciado ni las opciones del ConcepTest.

**Tu razonamiento:**

Necesitaría conocer la pregunta y qué representa cada alternativa. Elegir una letra sin esa información sería inventar la respuesta.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica todavía. Lo completaría durante la clase después de leer el enunciado y escuchar la explicación de mi compañero.

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

La solución anterior también permanece en [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), con su tabla en [`starter/resultados_ejercicio2.csv`](./starter/resultados_ejercicio2.csv).

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

La regresión recibe 3.275 columnas y solo hay 2.380 filas de entrenamiento. Hay más variables que observaciones, por lo que el modelo puede ajustar combinaciones muy específicas y volverse inestable. Eso explica que el MAE de entrenamiento sea 0,8953, pero el de prueba suba a 3.483.977.296,5699.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

Con Ridge el MAE de entrenamiento subió un poco, de 0,8953 a 1,0272, pero el de prueba bajó de miles de millones a 1,3209. Ridge empeora un poco el ajuste de entrenamiento porque penaliza coeficientes grandes, pero esa restricción evita que el modelo dependa de combinaciones exageradas y mejora la generalización.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

El mejor fue gradient boosting con MAE 1,2754. Mejora a la línea base en 11,2554 dólares de MAE, aproximadamente un 89,8%.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

La línea base sí aprende la mediana de entrenamiento, pero no aprende relaciones entre columnas. Por azar, las tarifas del conjunto de prueba quedaron un poco más cerca de esa mediana que las de entrenamiento. Una brecha negativa pequeña puede aparecer por la división de los datos y no significa que el modelo sea mejor en datos nuevos.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.

### Entregable 1: `modelos_proyecto.ipynb`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Usé `DummyRegressor(strategy="median")`. Es una referencia sencilla y resistente a los cantones con cantidades muy altas: siempre predice la mediana aprendida en las 32.256 filas de entrenamiento.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

El bosque muestra algo de sobreajuste: obtiene MAE 3,5108 en entrenamiento y 5,1231 en prueba, una brecha de 1,6122. No es tan extremo como el árbol sin límite del ejercicio de taxis, pero sí es una diferencia que conviene vigilar.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

En el modelo lineal usé Ridge con `alpha=10.0`, que reduce coeficientes demasiado grandes. Ridge obtuvo MAE 6,3885 en entrenamiento y 5,9268 en prueba. En el bosque limité la profundidad a 14 y exigí al menos 2 observaciones por hoja para que los árboles no memoricen filas individuales.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El bosque aleatorio fue el mejor con MAE 5,1231 frente a 37,2822 de la línea base. La mejora absoluta es 32,1591 emergencias y la reducción del error es aproximadamente 86,3%.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Mediría el tiempo de entrenamiento, la latencia por predicción, el costo por uso, la memoria, la estabilidad al repetir la misma entrada, la facilidad de actualizar el modelo y qué tan bien se puede explicar una tarifa al conductor o al pasajero.

## Datos y ejecución

El proyecto usa `laboratorios/data/obt_emergencias_muestra_6_meses.csv`, con datos de abril a septiembre de 2026. Desde `semana-10-sesion-01/starter`:

```powershell
docker compose build
docker compose run --rm ejercicio1
docker compose run --rm ejercicio2
docker compose run --rm proyecto
```

El último comando crea `proyecto/resultados_proyecto.csv`.
