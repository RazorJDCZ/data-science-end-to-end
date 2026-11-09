# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

La paradoja ocurre cuando un modelo parece muy bueno por accuracy aunque no detecta la clase importante, especialmente si esa clase es poco frecuente. Por ejemplo, predecir siempre “no fraude” da cerca de 95% en estos datos y recall 0. En ese caso conviene revisar precision, recall, F1 y el área bajo la curva precision-recall, según el costo de los errores.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

El ejemplo que entendí fue el de una empresa de noticias que puso como meta aumentar 20% la duración de las sesiones. El equipo alargó introducciones y agregó relleno: la métrica subió, pero la calidad bajó. Con un modelo puede pasar algo parecido si se optimiza una sola métrica hasta ignorar la utilidad real o los costos del negocio.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Con una sola división no lo sabía. Después de validar, la red neuronal tuvo MAE 5.6465 ± 0.1641 al azar y 6.4700 ± 2.3505 en validación temporal. La corrida de la sesión pasada solo mostraba 4.9613 para una fecha de corte, por lo que resultó bastante optimista frente al promedio temporal.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Esperaba valores cercanos, con unas pocas centésimas de diferencia. La ejecución dio promedio 1.2288 y desviación 0.0361; los pliegues fueron 1.28, 1.24, 1.16, 1.24 y 1.23. Entre el menor y el mayor hubo cerca de 0.12.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Esperaba un error un poco mayor porque el esquema temporal no mezcla viajes futuros con pasados y puede encontrar cambios en el tiempo. Se confirmó: gradient boosting pasó de 1.2288 al azar a 1.3099 temporal.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

Pensaba que el peor sería el que recibiera una zona con un patrón de tarifas poco parecido a las zonas usadas para entrenar. Resultó ser el pliegue 5, con MAE 6.43, muy por encima de los otros cuatro.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes subestimados en más de 20% responde directamente cuántos reclamos se esperan. El MAE resume el tamaño medio del error y el RMSE castiga errores grandes, pero ninguno distingue exactamente esa condición de reclamo. En la división usada fue 4.37%.

**Lo que confirmó la tabla, esquema por esquema:**

- Aleatorio, gradient boosting: promedio 1.2288, desviación 0.0361; peor pliegue 1 con aproximadamente 1.28.
- Aleatorio, bosque: promedio 1.2880, desviación 0.0291; peor pliegue 4 con 1.34.
- Temporal, gradient boosting: promedio 1.3099, desviación 0.0693; peor pliegue 1 con 1.40.
- Por zona, gradient boosting: promedio 2.2996, desviación 2.0648; peor pliegue 5 con 6.43.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

No marqué una letra porque en el repositorio no aparecen el enunciado ni las opciones del ConcepTest 1.

**Tu razonamiento:**

Mi razonamiento sería escoger el esquema que imita el uso real: temporal para predecir el futuro y por grupo cuando se quiere medir el desempeño en entidades nunca vistas.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

No hay una respuesta de pareja en el material. Después de ver los resultados mantendría ese criterio, porque el esquema por zona reveló un error que la división al azar ocultaba.

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

Tampoco marqué una letra en el ConcepTest 2 porque faltan su pregunta y alternativas.

**Tu razonamiento:**

No elegiría una métrica solo por costumbre. Usaría la que representa el evento que cuesta dinero o afecta al usuario, como recall para fraudes no detectados o el porcentaje de tarifas subestimadas para reclamos.

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

La línea base obtuvo accuracy 0.9508 (95.08%) y recall 0.0000. Parece acertar casi todo, pero no detecta ninguno de los 295 fraudes.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

Gradient boosting logró accuracy 0.9712, una mejora de 0.0204 (2.04 puntos porcentuales) frente a la línea base. Su recall fue 0.4610, una mejora de 46.10 puntos. El recall describe mejor la ganancia porque mide cuántos fraudes reales sí encuentra.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con umbral 0.5 dejó pasar 159 fraudes y produjo 14 falsas alarmas. El costo mínimo entre los umbrales probados apareció en 0.1: dejó pasar 122 fraudes, generó 222 falsas alarmas y costó 5.546 dólares según las constantes del ejercicio.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

El umbral sube. Recalculando con 20 dólares por falsa alarma, entre los valores probados conviene 0.5: su costo sería 6.640 dólares, menor que 6.780 con 0.3. Como revisar falsos positivos se vuelve caro, el sistema exige más probabilidad antes de marcar fraude.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

Corresponde validación **temporal**, porque cada fila usa información de un día para predecir las emergencias de tres días después. Mezclar fechas al azar dejaría información de periodos futuros en entrenamiento y daría una evaluación demasiado fácil.

El entregable está junto al pipeline en [`validacion_proyecto.ipynb`](../semana-09-sesion-02/plantilla-pipeline/validacion_proyecto.ipynb), y su tabla exportada es [`resultados_validacion_proyecto.csv`](../semana-09-sesion-02/plantilla-pipeline/resultados_validacion_proyecto.csv).

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

Entre los dos esquemas no cambió el ganador: el bosque tuvo MAE 5.4008 al azar y 5.6736 temporal; la red obtuvo 5.6465 y 6.4700, respectivamente. Sin embargo, el esquema temporal mostró una penalización y una variación mucho mayores, sobre todo en la red.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

La diferencia temporal de los promedios fue 0.7964 a favor del bosque. Es mayor que la desviación del bosque (0.5059), pero menor que la de la red (2.3505), así que no es una ventaja enorme frente a toda la variación. Aun así, ya no mantendría la red neuronal elegida con una sola división: escogería el **bosque aleatorio**, porque fue más estable y también obtuvo menor costo de negocio.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Elegí un costo asimétrico donde cada emergencia subestimada pesa 2 y cada una sobreestimada pesa 1. El bosque obtuvo costo promedio temporal 8.3965 y la red 10.1396. El MAE decía el tamaño medio del error, pero esta métrica además indica qué modelo se equivoca de la manera más peligrosa para la planificación de personal y vehículos.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

Las predicciones de ambos modelos dependen de datos del caso, así que no se pueden calcular una sola vez por noche para todos los casos posibles. La tarifa debe aparecer al pedir el viaje y el fraude debe decidirse al intentar pagar, idealmente en menos de uno o dos segundos. Lo que sí se puede hacer por lotes durante la noche es reentrenar los modelos o precalcular algunas características.
