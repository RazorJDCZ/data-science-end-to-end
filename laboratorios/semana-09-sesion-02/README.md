# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano tu notebook de features de la sesión pasada. Los tres ejercicios de hoy son scripts, no notebooks: un pipeline de preparación es código de producción, que tiene que correr igual cada vez y que otros archivos importan. En el Ejercicio 3, la preparación de tu proyecto pasa del notebook donde la probaste a ese script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

La fuga de información ocurre cuando el entrenamiento usa datos que no deberían estar disponibles en el momento de hacer la predicción, por ejemplo información del futuro o estadísticas calculadas también con prueba. El modelo puede verse excelente porque ya recibió pistas sobre los casos con los que se lo evalúa, pero en producción esas pistas no existen y el error real aumenta.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

Fue fuga del objetivo o del futuro. La duración real del viaje ayuda a explicar la tarifa, pero todavía no se conoce cuando el pasajero pide el taxi. Si se usa para predecir la tarifa en ese momento, el modelo aprende con una columna que no tendrá en producción.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

La media y la desviación se calculan solo con los viajes de entrenamiento. Después se usan esos mismos valores para transformar prueba y datos nuevos. Sí cambia: calcularlos con todos deja que prueba influya en el entrenamiento y la evaluación deja de ser completamente independiente.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

Entra en la línea 26, cuando `fit_transform` calcula la imputación y el escalado usando `X` completo antes de dividir. Lo delatan las 2975 filas que vio el escalador, porque entrenamiento solo debería tener 2380.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Está midiendo el error sobre las mismas filas usadas para ajustar el modelo, es decir, un error de entrenamiento. Ese número es optimista y no muestra qué tan bien generaliza a viajes que no ha visto.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Esperaba un cambio pequeño porque las distribuciones de entrenamiento y prueba son parecidas. En efecto, el MAE cambió de 2,5375 a 2,5373. Que casi no cambie no elimina la fuga: la fuga está en el procedimiento y podría causar una diferencia mayor con otros datos.

**Lo que confirmó la salida, TODO por TODO:**

El escalador del compañero vio las 2975 filas, usó una media de distancia de 8,5034 y obtuvo MAE 2,5375. El pipeline correcto vio solamente las 2380 filas de entrenamiento, usó una media de 8,5576 y obtuvo MAE 2,5373 sobre 595 filas de prueba.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No puedo elegir A, B, C o D porque el repositorio no incluye el enunciado ni las opciones del ConcepTest.

**Tu razonamiento:**

Necesitaría leer la pregunta y lo que significa cada opción. Marcar una letra sin esa información sería inventar la respuesta.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica todavía. Lo completaría durante la clase después de conocer el enunciado y escuchar el argumento de mi compañero.

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan 25 columnas: 2 numéricas escaladas, 5 que ya estaban listas y 18 creadas por one-hot. Estas 18 salen de 8 zonas de origen, 8 zonas de destino y 2 métodos de pago.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con entrenamiento: 6,94 km y 1 pasajero. En todo el dataset son 6,93 km y 1 pasajero. Esto demuestra que el imputador se ajustó solo con entrenamiento y que prueba no participó en el aprendizaje.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

Como Bronx no apareció en entrenamiento y se usó `handle_unknown="ignore"`, sus 8 columnas de zona de origen quedan en cero y el viaje se transforma sin error. Con `pd.get_dummies` aplicado por separado aparecería una columna nueva para Bronx y faltarían las columnas aprendidas, por lo que la forma no coincidiría con la que espera el modelo.

## Ejercicio 3: el pipeline de tu proyecto, como script

Copia la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py) a la carpeta de tu proyecto, al lado de tu notebook de features. Desde la próxima sesión, tus notebooks de modelos importan este archivo en vez de copiar su código:

Para el proyecto se usa la muestra compartida de seis meses, no la OBT completa.

El script expone las funciones que usarán los notebooks de las siguientes semanas:

```python
from pipeline_proyecto import cargar_y_dividir, construir_preparacion
```

### Paso 1: pasa tus features a `agregar_features` (5 minutos)

Copia de tu notebook de la sesión pasada solo las líneas que construyen tus 2 features, dentro de la función `agregar_features`. Sin gráficos, sin `print` y sin las celdas que usaste para probar ideas: esas se quedan en el notebook, como evidencia de por qué elegiste cada feature.

### Paso 2: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

El objetivo es `obj_n_emergencias_t3`, la cantidad de emergencias del cantón dentro de tres días.

- Numéricas: `n_emergencias`, `lag_1d`, `lag_4d`, `lag_7d`, `lag_14d`, `lag_28d`, `media_prev_7d` y `media_prev_28d`.
- Categórica: `cod_canton`.
- Ya listas: `es_fin_semana_t3` y `es_feriado_t3`, porque son indicadores binarios.

Dejé fuera la fecha, las columnas descriptivas y los demás objetivos futuros. `lag_4d` y `es_fin_semana_t3` son las dos features creadas en la sesión anterior.

### Paso 3: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha dentro de `cargar_y_dividir`; si no, al azar. Justifica.

Dividí por fecha porque se quiere predecir el futuro. El 80 % de fechas más antiguas se usa para entrenamiento y el 20 % más reciente para prueba; en la muestra la fecha de corte es 2026-08-23. Una división al azar mezclaría días futuros en entrenamiento y daría una evaluación poco realista.

### Paso 4: corre el script (10 minutos)

Desde una terminal, en la carpeta de tu proyecto:

```
python3 pipeline_proyecto.py
```

El pipeline imprimió entrenamiento con forma `(32256, 234)` y prueba con forma `(8064, 234)`. Las 234 columnas producidas son:

- 8 numéricas: `numericas__n_emergencias`, `numericas__lag_1d`, `numericas__lag_4d`, `numericas__lag_7d`, `numericas__lag_14d`, `numericas__lag_28d`, `numericas__media_prev_7d` y `numericas__media_prev_28d`.
- 224 columnas one-hot con el formato `categoricas__cod_canton_<código>`, una por cada cantón aprendido.
- 2 listas: `ya_listas__es_fin_semana_t3` y `ya_listas__es_feriado_t3`.

El script también imprime la lista exacta de los 224 códigos de cantón. Se dejó la codificación one-hot como matriz dispersa para no gastar memoria guardando tantos ceros. Como corre de principio a fin sin un notebook abierto, los notebooks de modelos ya pueden importarlo.

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Está sobreajustando: aprendió demasiado bien los detalles y el ruido del entrenamiento, pero no generaliza igual a datos nuevos. Compararía la diferencia entre ambos errores y probaría regularización, un modelo más simple o mejores datos y features.

## Cómo ejecutar las soluciones

Desde `laboratorios/semana-09-sesion-02/starter`:

```powershell
docker compose build
docker compose run --rm ejercicio1
docker compose run --rm ejercicio2
docker compose run --rm proyecto
```

El primer comando construye las imágenes con las versiones necesarias de pandas y scikit-learn. El servicio `proyecto` lee la muestra compartida de seis meses sin modificar el CSV.
