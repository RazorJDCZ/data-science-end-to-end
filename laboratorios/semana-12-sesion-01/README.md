# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

En batch las predicciones se calculan antes de que llegue una solicitud, se
guardan y se actualizan cada cierto tiempo. Por ejemplo, una tienda puede
calcular cada noche las recomendaciones de todos sus clientes. En online la
predicción se calcula después de recibir la solicitud; por ejemplo, un banco
puede evaluar una transacción justo cuando el cliente intenta pagar.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

La predicción batch puede quedar atrasada hasta que se ejecute el siguiente
lote y también puede calcular resultados que nadie va a consultar. Online
resuelve la falta de frescura porque responde con el caso que acaba de llegar,
pero exige baja latencia, un servicio siempre disponible y más infraestructura
para obtener las features y atender las solicitudes.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La tarifa de los viajes reservados sí se puede calcular durante la noche,
porque esos viajes ya se conocen. La de un viaje que se pide en ese momento no
se puede dejar precalculada para todos los orígenes y destinos posibles. Si el
modelo de fraude se usa para bloquear o pedir una verificación antes de cobrar,
tiene que ser online y responder aproximadamente en menos de 100 milisegundos.
Si solo se usa para una revisión posterior, sí puede procesarse en un lote
diario, como en el Ejercicio 2.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Pasan porque las dos funciones que tenían los `TODO 4` y `TODO 5` todavía no
contenían ningún `assert`. Pytest puede ejecutar esas funciones sin encontrar
un error y las marca como aprobadas, aunque en realidad aún no estaban
comprobando nada. La única prueba completa era la de las columnas y por eso era
la que fallaba.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Espero que estime los 60 viajes. El pipeline de preparación imputa la mediana
en `pasajeros` y el valor más frecuente en las columnas categóricas, entre
ellas `metodo_pago`. Así el modelo recibe valores válidos sin eliminar viajes,
que es importante porque cada reserva necesita una tarifa.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Sirve para rastrear con qué archivo de modelo se calculó cada tarifa. Si luego
se cambia el modelo, todavía se puede auditar una fila antigua, comparar
versiones o investigar una estimación extraña sin depender de recordar qué
modelo estaba activo ese día.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

La prueba del `TODO 4` fallaría si el script eliminara un viaje, cambiara su
orden o devolviera otros identificadores; por ejemplo, si descartara las filas
incompletas antes de predecir. La del `TODO 5` fallaría si el viaje sin
pasajeros quedara con una tarifa nula o si las filas no guardaran exactamente
la versión del modelo recibida por la función.

**Lo que mostró el lote:**

El modelo quedó identificado con la versión `d552a7c9f850`. La tarifa más baja
fue **4.67** (viaje 90014) y la más alta fue **70.26** (viaje 90051). Los
viajes sin pasajeros registrados también recibieron tarifa: 90023 = 33.94,
90024 = 29.88, 90025 = 25.80 y 90048 = 17.58.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No aplica todavía: en este README no aparece el enunciado del ConcepTest ni
las opciones A, B, C y D.

**Tu razonamiento:**

No escogí una letra al azar porque no sería posible justificarla sin conocer
la pregunta y sus opciones. Esta parte se debe completar con la diapositiva o
la discusión realizada en clase.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

Queda pendiente la discusión presencial por la misma razón. Sí cambiaría mi
voto si la otra persona muestra que la latencia o la disponibilidad de los
datos no son las que asumí inicialmente.

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

Con el umbral de 0.10 quedaron **26 pagos** en la cola. En el entrenamiento
hubo 295 fraudes de 6000 pagos, una proporción aproximada de 4.92 %. Si los 300
pagos mantienen esa proporción, esperaría encontrar cerca de **15 fraudes**
(300 x 0.0492 = 14.75).

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

La empresa se entera después de que el pago ya ocurrió, por lo que pierde la
oportunidad de bloquearlo o pedir una verificación adicional antes de aceptar
el dinero. Esto importa si el equipo revisa cuando el pago ya fue liquidado y
no se puede recuperar fácilmente. Si el cobro se mantiene pendiente hasta que
termine la revisión, el retraso del lote puede ser menos costoso.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

La prueba queda pequeña, rápida y determinista. Como conoce de antemano las
probabilidades, revisa solamente la regla del umbral, que incluye el 0.10, y el
orden descendente. Si usara el modelo, un cambio en el entrenamiento podría
hacer fallar la prueba aunque la función de la cola estuviera correcta.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con un umbral de 0.50 quedan **13 pagos**. Las cuatro pruebas siguen pasando
porque prueban el formato y los límites de las probabilidades, y la prueba de
la cola entrega explícitamente su propio umbral de 0.10. No depende del valor
global usado por el lote real.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

Pendiente de pegar el enlace exacto después de subir este trabajo al fork y
comprobar que termine la ejecución.

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

Elegí predicción **batch diaria**. El número que decide es una latencia máxima
de **una hora** después de cerrar y cargar los datos del día, porque la salida
se usa para planificación de emergencias a tres días y no para responder a un
usuario en ese instante.

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

Hice la comprobación de forma local antes de subir: cambié el valor esperado
del `lag_4d` de 10 a 11 y pytest mostró `1 failed`, indicando la línea del
`assert` y que el valor real era 10. Después restauré el valor correcto y las
cuatro pruebas volvieron a pasar. No subí deliberadamente el código roto a la
rama principal para no dejar el fork en un estado incorrecto; el mismo fallo
aparecería en la sección **Actions**, dentro del paso que corre pytest.

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Tendría que existir un servicio online, por ejemplo una API con un endpoint de
predicción que mantenga cargado el modelo. La app le enviaría un JSON con las
features del viaje nuevo (origen, destino, distancia, pasajeros, hora y método
de pago) y recibiría la tarifa estimada en la respuesta.
