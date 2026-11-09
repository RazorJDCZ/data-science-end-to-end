# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los laboratorios de hoy son notebooks de Jupyter, como el diagnóstico de la Semana 4: cada gráfico aparece debajo de la celda que lo dibuja. Una exploración se lee para decidir qué hacer con los datos; por eso va en un notebook y no en un script.

### Datos locales del proyecto ECU 911

El archivo crudo unificado se conserva en `data/ecu911_emergencias_raw.csv`. Tiene
17.953.877 incidentes entre julio de 2021 y septiembre de 2026. Para regenerarlo,
desde `semana-09-sesion-01/starter` se ejecuta:

```powershell
docker compose run --rm --build datos-emergencias
```

Después, este comando construye la OBT local completa:

```powershell
docker compose run --rm --build datos-proyecto
```

La OBT completa queda en `data/obt_emergencias_canton_dia.csv`. Para los talleres no
se usan todos esos años. Este comando crea una muestra con los seis meses completos
más recientes:

```powershell
docker compose run --rm --build muestra-proyecto
```

La muestra compartida queda en
`laboratorios/data/obt_emergencias_muestra_6_meses.csv`: contiene 40.992 filas, 224
cantones y fechas desde el 1 de abril hasta el 30 de septiembre de 2026.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

La exploración responde qué contiene el dataset, cuántas observaciones y variables hay, qué tipo tiene cada columna, dónde faltan datos, si existen duplicados, cómo se distribuyen los valores y qué anomalías, desequilibrios o relaciones podrían afectar al modelo. Con eso se decide qué limpiar, transformar o excluir antes de entrenar.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para una variable numérica se usan estadísticas descriptivas y un histograma; para una categórica, frecuencias y un gráfico de barras. Para estudiar relaciones entre variables se usan gráficos de interacción, como un diagrama de dispersión, y medidas de correlación cuando corresponden al tipo de datos.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

Las 15 tarifas extremas aumentan mucho el error y pueden empujar los coeficientes o las predicciones hacia valores irreales. Mirando solamente la métrica final quizá notaría un peor resultado, pero no sabría la causa. Para detectarla revisaría el máximo, la distribución, un boxplot y los casos con mayor error.

## Ejercicio 1: resumen, distribución y atípicos

Notebook nuevo del profesor: [`starter/ejercicio1_explorar.ipynb`](./starter/ejercicio1_explorar.ipynb). También se conserva la solución en [`starter/ejercicio1_explorar.py`](./starter/ejercicio1_explorar.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York. El script resuelve los 4 `TODO`, arma los gráficos y los guarda en `figuras/`:

```
cd laboratorios/semana-08-sesion-02/starter
python3 ejercicio1_explorar.py
```

El notebook se puede ejecutar con **Run All** para comparar sus celdas con el script resuelto.

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El máximo de `tarifa` (999,99) y el mínimo de `duracion_min` (−74) son sospechosos. La tarifa está muy lejos del percentil 75 (32,55) y una duración negativa es imposible.

**Respuesta 2 (TODO 2):** mira el histograma después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

Las tarifas de 999,99 estiran el eje horizontal hasta casi 1.000 USD. Como la mayoría de los viajes cuesta menos de 70 USD, sus barras quedan comprimidas en una fracción pequeña del gráfico.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

No espero que todos sean errores. El IQR identifica valores poco comunes, no datos inválidos. Revisaría sus tarifas, rutas, distancias y duraciones. En este caso, la mayoría corresponde a la tarifa fija de JFK y otros viajes largos reales; solo los valores 999,99 son errores confirmados.

**Lo que confirmó la salida, TODO por TODO:**

El límite superior es 60,40 USD y aparecen 334 tarifas atípicas: 15 errores de 999,99, 302 viajes con tarifa exacta de 70 USD y 17 viajes reales con otras tarifas superiores al límite. También hay 10 duraciones negativas.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No puedo escoger A, B, C o D porque el repositorio no incluye el enunciado del ConcepTest ni sus opciones.

**Tu razonamiento:**

Necesitaría conocer la afirmación y qué representa cada opción. Elegir una letra sin esa información sería inventar una respuesta.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No aplica todavía. Lo completaría durante la clase y anotaría qué argumento hizo que mantuviera o cambiara mi respuesta.

## Ejercicio 2: relaciones entre variables

Notebook nuevo del profesor: [`starter/ejercicio2_relaciones.ipynb`](./starter/ejercicio2_relaciones.ipynb). Se conserva además [`starter/ejercicio2_relaciones.py`](./starter/ejercicio2_relaciones.py), sobre el mismo `viajes_taxi.csv`. La solución quita solo los errores confirmados y completa los 5 `TODO`; las comprobaciones muestran `OK`.

```
python3 ejercicio2_relaciones.py
```

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

Además de la diagonal general, hay una línea horizontal en 70 USD: 300 viajes entre JFK y Manhattan o Harlem pagan la tarifa fija sin importar la distancia. También hay 25 viajes con distancia 0 que duran entre 3,9 y 81,1 minutos, lo que apunta a kilómetros no registrados.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Los 888 viajes pagados en efectivo tienen propina 0, mientras que los 2.027 viajes con tarjeta tienen una propina promedio de 4,73 USD y ninguno tiene propina 0. Esto describe principalmente el sistema de registro: las propinas entregadas en efectivo no quedan guardadas.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. Aunque la correlación de la duración con la tarifa es 0,956, la duración real solo existe después del viaje. Usarla produciría fuga de información. La distancia estimada de la ruta sí puede calcularse al pedir el taxi.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar. Copia la plantilla [`plantilla-eda/explorar_proyecto.ipynb`](./plantilla-eda/explorar_proyecto.ipynb) a `psets/proyecto/modelado/`: el notebook es tu entregable, con el código, lo que mostró y lo que concluyes en el mismo lugar.

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

La muestra tiene 40.992 filas, 37 columnas, 224 cantones y fechas desde abril hasta septiembre de 2026. Como se recortó después de construir la OBT, los rezagos iniciales ya están calculados. Los faltantes esperados son 672 en `obj_n_emergencias_t3`, 1.568 en `obj_n_emergencias_t7` y 39.872 en `nombre_feriado`, porque la mayoría de días no son feriado. El IQR marca 5.001 valores en `obj_n_emergencias_t3`.

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

Comparé `obj_n_emergencias_t3` con las emergencias de t, t−7 y t−4. En Quito las correlaciones son −0,543, −0,508 y 0,664, respectivamente. `lag_4d` es mejor porque t−4 y t+3 corresponden al mismo día semanal. Su correlación también es 0,784 en Guayaquil y 0,797 en Cuenca.

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

En el paso 5 del notebook, para cada hallazgo: una celda de código con el número o el gráfico que lo confirma, y debajo, en la celda de texto, qué encontraste y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar). Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de exploración en tu fork:**

Los 5.001 valores marcados por IQR pertenecen a 54 cantones, principalmente los de mayor volumen; no son errores que deban eliminarse. Quito y Guayaquil reúnen el 39,7% de todas las emergencias, mientras 36 cantones tienen una mediana de 2 o menos por día. **Cambio:** comparar cada cantón con su propia historia y reportar métricas por grupos de tamaño.

**Hallazgo 2:**

Existe un ciclo semanal fuerte: el domingo promedia 50,4 emergencias y el martes 33,8. El rezago de t−4 coincide con el día semanal de t+3 y resulta más informativo que el conteo de t. **Cambio:** agregar `lag_4d` para el horizonte de tres días.

**Hallazgo 3:**

El promedio nacional cambia dentro de los seis meses: pasa de 9.384 emergencias diarias en mayo a 8.395 en julio y 9.119 en septiembre. Con un P90 aprendido antes del corte del 23 de agosto, la tasa de alta demanda sube de 5,8% en julio a 13,1% en septiembre; en feriado t+3 es 9,7% frente a 8,8% en días normales. **Cambio:** usar división temporal, vigilar la deriva mes a mes y conservar `es_feriado_t3`.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Crearía `es_tarifa_fija`, con valor 1 cuando el viaje conecta JFK con una zona de Manhattan, y variables de tiempo como `hora`, `es_hora_pico`, `hora_sin` y `hora_cos`. Así el modelo recibe de forma numérica los patrones encontrados en el EDA.

## Cómo ejecutar las soluciones

Desde `semana-08-sesion-02/starter`:

```powershell
docker compose run --rm --build ejercicio1-oficial
docker compose run --rm ejercicio2-oficial
docker compose run --rm proyecto
```

Los archivos resueltos anteriores (`EDA.py` y
`plantilla-ejercicio2/explorar_relaciones.py`) se conservaron para no perder el trabajo
ya realizado. Los nuevos `starter/ejercicio1_explorar.py` y
`starter/ejercicio2_relaciones.py` también están completos.
