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

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio representa algo real del problema, se puede explicar y está disponible en el momento de hacer la predicción. No basta con que suba una métrica. Una columna puede mejorar el resultado por casualidad, porque copia información del futuro o porque aprovecha un patrón que después cambia. Por eso primero trataría de justificar la feature con lo que conozco del proceso y luego comprobaría si ayuda al modelo.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

Me quedo con la idea de transformar una fecha cruda en una medida que el modelo pueda relacionar con el problema, por ejemplo tiempo transcurrido o antigüedad. El conocimiento de negocio es que normalmente importa cuánto tiempo ha pasado y no el número de la fecha. En este ejercicio apliqué la misma idea al separar la fecha del viaje en hora, día de la semana, fin de semana y hora pico.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Crearía `es_tarifa_fija`, que vale 1 cuando el viaje conecta JFK con una zona de Manhattan, y `es_hora_pico`, que vale 1 en las franjas con más tráfico. También conservaría la hora con una codificación cíclica (`hora_sin` y `hora_cos`) para que las 23:00 y las 00:00 queden cerca. De esta forma el modelo recibe los patrones como números en lugar de tener que descubrirlos solamente a partir de las columnas originales.

## Ejercicio 1: features de fecha y hora

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada, sin los errores confirmados. A partir de `fecha_hora_inicio`, construir hora, día de la semana, fin de semana, hora pico y la hora como ciclo, y comprobar si la hora pico aporta algo a la tarifa.

**Respuesta 1, antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como números quedan a una distancia de 23, pero en el reloj solo están separadas por una hora. Por eso una variable numérica normal representa mal el ciclo diario. Con seno y coseno, ambas horas quedan cerca en el círculo.

**Respuesta 2:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

`es_hora_pico` puede traer una señal indirecta de congestión. En ciertas horas es razonable esperar más tráfico y, por lo tanto, viajes más lentos o costosos. No reemplaza la duración real, pero aporta información disponible antes de que el viaje comience y evita usar una variable del futuro.

**La formulación completa, tal como la resuelvo yo:**

La solución está en [`starter/mi_primer_feature_eng.py`](./starter/mi_primer_feature_eng.py). Primero quité solamente los errores confirmados: 15 tarifas de 999,99 y 10 duraciones negativas. Quedaron 2.975 de los 3.000 viajes. Después creé `hora`, `dia_semana`, `es_fin_semana`, `es_hora_pico`, `hora_sin` y `hora_cos`.

Definí hora pico como 07:00–09:59 y 16:00–19:59. El resultado fue:

| Hora pico | Viajes | Tarifa promedio | Duración promedio |
|---:|---:|---:|---:|
| 0 | 1.760 | 25,92 | 22,71 min |
| 1 | 1.215 | 27,39 | 27,94 min |

La diferencia muestra que la hora pico sí aporta una señal: los viajes duran aproximadamente 5,23 minutos más y la tarifa promedio aumenta 1,47. Esto no demuestra por sí solo que la hora sea la causa, porque la ruta y la distancia también influyen, pero sí justifica probar la feature en el modelo.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No puedo escoger A, B, C o D porque en este README no aparece el enunciado del ConcepTest ni sus opciones.

**Tu razonamiento:**

Para votar necesitaría conocer la afirmación y qué significa cada opción. Elegir una letra sin esa información sería inventar una respuesta y no demostraría que entendí el tema.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica todavía. Lo completaría durante la clase cuando se presente el ConcepTest y anotaría qué argumento hizo que mantuviera o cambiara mi respuesta.

## Ejercicio 2: zona, codificación y disponibilidad

La solución en [`plantilla-ejercicio2/features_de_zona.py`](./plantilla-ejercicio2/features_de_zona.py) completa la tarifa fija, las dos codificaciones de zona y la revisión de disponibilidad de las variables.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

Hay 300 viajes con `es_tarifa_fija = 1`. La tarifa mínima y la máxima son 70 USD. Esta columna permite que un modelo lineal aprenda un ajuste específico para esas rutas, en vez de intentar explicar los 70 USD solamente con la distancia. Como todos los casos marcados tienen la misma tarifa, es una señal bastante directa.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea 8 columnas, una por cada zona de origen, mientras que la codificación ordinal crea una sola columna. Para una regresión lineal usaría one-hot. La versión ordinal asigna números de 0 a 7 y hace parecer que las zonas tienen un orden y una distancia entre ellas; por ejemplo, que una zona con código 6 es el doble de otra con código 3. Esa relación no existe en la realidad.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

No existen la duración real, la tarifa final ni la propina. La distancia exacta recorrida tampoco se conoce, aunque sí se puede usar una distancia estimada de la ruta. Si uso información que solo aparece al final, el modelo tendría fuga de información: durante el entrenamiento parecería muy preciso, pero al pedir un taxi no tendría esas columnas y fallaría en producción. El identificador del viaje puede existir, pero no lo usaría como predictor porque no aporta una relación de negocio.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

El modelo predice si un cantón tendrá demanda alta de emergencias dentro de tres días (`y_3`). La predicción se hace al terminar el día `t`. En ese momento ya existen el cantón, la fecha, el día de la semana, los feriados programados y el historial de emergencias hasta `t`. Todavía no existen el conteo real de `t+3`, `obj_n_emergencias_t3` ni la etiqueta `y_3`; esas columnas solo se usan como objetivo durante el entrenamiento y la evaluación.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

El código completo está en [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py). Las dos features principales son:

```python
datos = datos.sort_values(["cod_canton", "fecha"]).copy()

# El día t-4 cae en el mismo día de la semana que t+3.
datos["lag_4d"] = datos.groupby("cod_canton")["n_emergencias"].shift(4)

# Saber si t+3 será fin de semana no usa información futura desconocida.
datos["dia_semana_t3"] = (datos["dia_semana"] + 2) % 7 + 1
datos["es_fin_semana_t3"] = datos["dia_semana_t3"].isin([6, 7]).astype(int)
```

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`lag_4d` sale del ciclo semanal encontrado en la exploración. Para predecir `t+3`, el dato de `t-4` corresponde al mismo día de la semana y ya se conoce al predecir. Con la OBT local su correlación con el objetivo es 0,731 en Quito, 0,845 en Guayaquil y 0,665 en Cuenca. La correlación global es 0,991, pero está inflada por las diferencias de escala entre cantones, por eso prefiero la evidencia dentro de cada ciudad. Es numérica y la dejaría como número; si se escala, los parámetros se calculan solo con entrenamiento.

**Feature 2:**

`es_fin_semana_t3` resume parte del patrón semanal en una variable binaria conocida por calendario. En el CSV, el objetivo promedia 38,30 emergencias cuando t+3 cae entre semana y 50,47 cuando cae en fin de semana; las medianas son 7 y 10. Como ya vale 0 o 1, no necesita one-hot ni codificación ordinal.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Calcularía la media y la desviación solamente con los viajes de entrenamiento. Luego aplicaría esos mismos valores a validación, prueba y producción. Si uso todos los viajes, la distribución del conjunto de prueba se filtra al entrenamiento y la evaluación deja de ser completamente honesta. Puede cambiar tanto los valores escalados como la métrica final, especialmente si el periodo de prueba tiene distancias diferentes.

## Cómo ejecutar las soluciones

### Crear el CSV crudo unificado del ECU 911

Desde la carpeta `starter`, este comando consulta el catálogo oficial, descarga cada
mes y crea `../data/ecu911_emergencias_raw.csv`:

```powershell
docker compose run --rm --build datos-emergencias
```

El archivo conserva los siete campos originales con separador `;` y agrega
`_periodo`, `_archivo_origen` y `_fila_archivo`. La misma información aparece en la
carpeta `data/` de la Semana 8 mediante un enlace al mismo archivo físico.

### Preparar la OBT local

Este comando transforma el crudo en 429.632 filas de cantón y día, ejecuta los
controles de integridad y crea la misma OBT en las carpetas `data/` de ambas semanas:

```powershell
docker compose run --rm --build datos-proyecto
```

### Ejecutar los ejercicios

Desde la carpeta `starter`, la primera ejecución construye la imagen y corre el Ejercicio 1:

```powershell
docker compose run --rm --build python-app
```

Para el Ejercicio 2:

```powershell
docker compose run --rm python-app python plantilla-ejercicio2/features_de_zona.py
```

Para el Ejercicio 3 del proyecto:

```powershell
docker compose run --rm proyecto-semana09
```

Para repetir el EDA del proyecto de la Semana 8 desde el mismo entorno:

```powershell
docker compose run --rm proyecto-semana08
```
