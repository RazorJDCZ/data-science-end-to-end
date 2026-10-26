# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

### Datos locales del proyecto ECU 911

El CSV crudo está en `data/ecu911_emergencias_raw.csv`. Para descargar nuevamente los
63 meses y después construir la OBT local, ejecuta desde `starter`:

```powershell
docker compose run --rm --build datos-emergencias
docker compose run --rm --build datos-proyecto
```

La OBT queda en `data/obt_emergencias_canton_dia.csv` y comparte el mismo archivo
físico con la Semana 8. Los CSV grandes están excluidos de Git.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio representa algo real del problema, se puede explicar y está disponible en el momento de predecir. Una columna que solo sube una métrica puede estar aprovechando azar, información futura o un patrón que no se mantendrá en producción.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

Me quedo con transformar una fecha cruda en información que el modelo pueda relacionar con el problema, como hora, día semanal, fin de semana o antigüedad. En el ejercicio separé la fecha del viaje y codifiqué la hora como un ciclo.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Crearía `es_tarifa_fija`, que vale 1 cuando el viaje conecta JFK con Manhattan, y `es_hora_pico`, que vale 1 en las franjas de tráfico de días laborables. También usaría `hora_sin` y `hora_cos` para que las 23:00 y las 00:00 queden cerca.

## Ejercicio 1: features de fecha y hora

Archivo: [`starter/ejercicio1_features_tiempo.py`](./starter/ejercicio1_features_tiempo.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. El script quita los errores confirmados y resuelve los 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo.

```
cd laboratorios/semana-09-sesion-01/starter
python3 ejercicio1_features_tiempo.py
```

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como números, 23 y 0 están a una distancia de 23; en el reloj están separados por una hora. Con seno y coseno su distancia es 0,26, la misma que entre 11 y 12.

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Espero una tarifa por kilómetro más alta en hora pico, porque recorrer la misma distancia puede tomar más tiempo por la congestión. La salida lo confirma: la mediana sube de 3,22 USD/km fuera de hora pico a 3,52 USD/km en hora pico.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

`es_hora_pico` aporta una señal indirecta de congestión. No reemplaza la duración real, pero representa parte de su efecto usando información que ya existe antes de iniciar el viaje y evita fuga de información.

**Lo que confirmó la salida, TODO por TODO:**

La tarifa mediana por kilómetro es 3,22 fuera de hora pico y 3,52 en hora pico. Como números, 23 y 0 están a distancia 23, mientras 11 y 12 están a distancia 1; con la codificación cíclica ambos pares quedan a distancia 0,26.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No puedo escoger A, B, C o D porque el repositorio no incluye el enunciado del ConcepTest ni sus opciones.

**Tu razonamiento:**

Necesitaría conocer la afirmación y qué representa cada opción. Elegir una letra sin esa información sería inventar una respuesta.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica todavía. Lo completaría durante la clase y anotaría qué argumento hizo que mantuviera o cambiara mi respuesta.

## Ejercicio 2: zona, codificación y disponibilidad

Archivo: [`starter/ejercicio2_features_zona.py`](./starter/ejercicio2_features_zona.py), sobre el mismo `viajes_taxi.csv`. Los 4 `TODO` están resueltos y las cuatro comprobaciones finales muestran `OK`.

```
python3 ejercicio2_features_zona.py
```

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

Hay 300 viajes con `es_tarifa_fija = 1`; la tarifa mínima y máxima son 70 USD. Esta columna permite que un modelo lineal aprenda un ajuste específico para esas rutas en vez de explicar el precio solo mediante la distancia.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas y la codificación ordinal crea una. Para una regresión lineal usaría one-hot. La ordinal introduce un orden y distancias numéricas falsas entre zonas que en realidad son categorías sin jerarquía.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

La duración real, la tarifa final y la propina no existen al pedir el taxi. La distancia recorrida tampoco, aunque sí puede usarse una estimación de ruta. Entrenar con información posterior produciría fuga: el modelo parecería preciso en pruebas, pero no tendría esas variables cuando deba predecir.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

El modelo predice si un cantón tendrá demanda alta de emergencias dentro de tres días (`y_3`) al cierre del día t. En ese momento se conocen el cantón, la fecha, el calendario y el historial hasta t. Todavía no existen `obj_n_emergencias_t3` ni `y_3`; esas columnas se usan solo como objetivo durante entrenamiento y evaluación.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

```python
datos = datos.sort_values(["cod_canton", "fecha"]).copy()

# t-4 cae en el mismo día de la semana que t+3.
datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)

# El calendario de t+3 se conoce al realizar la predicción.
datos["dia_semana_t3"] = (datos["dia_semana"] + 2) % 7 + 1
datos["es_fin_semana_t3"] = datos["dia_semana_t3"].isin([6, 7]).astype(int)
```

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`lag_4d` sale del ciclo semanal encontrado en el EDA. Su correlación con el objetivo es 0,731 en Quito, 0,845 en Guayaquil y 0,665 en Cuenca. El dato ya existe al cierre de t y se conserva como variable numérica; cualquier escalado se ajustaría solo con entrenamiento.

**Feature 2:**

`es_fin_semana_t3` resume una parte conocida del calendario. El objetivo promedia 38,30 emergencias cuando t+3 cae entre semana y 50,47 cuando cae en fin de semana; las medianas son 7 y 10. Como ya es binaria, no necesita one-hot.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Calcularía la media y la desviación solamente con los viajes de entrenamiento. Después aplicaría esos mismos valores a validación, prueba y producción. Usar todos los viajes filtraría información de prueba al entrenamiento y haría menos honesta la evaluación.

## Cómo ejecutar las soluciones

Desde `semana-09-sesion-01/starter`:

```powershell
docker compose run --rm --build ejercicio1-oficial
docker compose run --rm ejercicio2-oficial
docker compose run --rm proyecto-semana09
```

El archivo anterior `starter/mi_primer_feature_eng.py` y la solución de
`plantilla-ejercicio2/features_de_zona.py` se conservaron. Los nuevos
`starter/ejercicio1_features_tiempo.py` y `starter/ejercicio2_features_zona.py` del
profesor también quedaron completos.
