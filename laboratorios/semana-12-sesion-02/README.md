# Juan Diego Cadena, 329220

## Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

Primero se guarda el modelo entrenado en un archivo y se define qué datos debe
recibir. Después se crea una API que carga el modelo, convierte el JSON a las
columnas esperadas y expone una ruta para predecir. Finalmente se prueba el
contrato, se fijan las dependencias y se empaqueta el servicio para que pueda
ejecutarse fuera del notebook.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para empaquetar el código, el modelo, Python y las versiones de las librerías en
un mismo ambiente. Así otra persona no tiene que reproducir manualmente la
computadora del autor y se evitan diferencias de sistema operativo o de
dependencias.

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Tendría que existir una API disponible en línea con el modelo cargado. La app le
mandaría un JSON con la fecha y hora, origen, destino, distancia, pasajeros y
método de pago. La API calcularía las features de tiempo y devolvería la tarifa
estimada en la misma petición.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo parece correcto aunque `/predecir` devuelve siempre 0.0 y las dos
pruebas incompletas no tenían ningún `assert`. Además, el esquema solo exigía
las zonas. Un código 200 y tres pruebas marcadas como aprobadas no demostraban
que el modelo hubiera recibido los datos ni calculado una tarifa.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Sin `distancia_km` espero una respuesta 200 con una tarifa, porque el campo es
opcional y el pipeline imputa la mediana. Sin `zona_origen` espero un 422,
porque es un campo obligatorio para construir la predicción.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El servicio seguiría respondiendo, pero el significado de la feature habría
cambiado. El modelo podría tratar como normal una hora que durante el
entrenamiento era pico, o al revés, y producir tarifas equivocadas sin mostrar
un error técnico. Por eso el servicio importa la misma función de features.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce FastAPI al validar el esquema de Pydantic antes de llamar a la
función de la ruta. La petición no llega al modelo.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque uvicorn escucha por defecto en `127.0.0.1`, que dentro del contenedor
solo es accesible desde el mismo contenedor. Publicar el puerto no lo vuelve
accesible si el proceso no escucha en la interfaz externa. El `CMD` debe usar
`--host 0.0.0.0`.

**Lo que respondió el servicio:**

El viaje de Midtown a JFK Airport recibió `{"tarifa_estimada": 70.39}`. Sin
zona de origen respondió 422 con:

```json
{
  "detail": [{
    "type": "missing",
    "loc": ["body", "zona_origen"],
    "msg": "Field required",
    "input": {
      "fecha_hora_inicio": "2026-11-05 08:30",
      "zona_destino": "JFK Airport",
      "distancia_km": 26.4,
      "pasajeros": 2,
      "metodo_pago": "tarjeta"
    }
  }]
}
```

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

No aplica todavía: el repositorio no incluye el enunciado ni las opciones del
ConcepTest.

**Tu razonamiento:**

No escogí una letra al azar porque no sería posible justificarla sin ver la
pregunta. Esta parte se completa con la diapositiva usada en clase.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

Queda pendiente la discusión presencial por la misma razón. Cambiaría el voto
si el argumento de la otra persona muestra una condición distinta de latencia,
validación o disponibilidad del modelo.

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

El pago riesgoso recibió probabilidad **0.9996** y `revisar: true`. El pago de
la tarde recibió **0.0159** y `revisar: false`.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app recibe una decisión lista y no necesita copiar la regla de negocio. Si
el umbral cambia a 0.20 se modifica `UMBRAL` en el servicio y sus pruebas; la
app puede seguir consumiendo el mismo campo `revisar` sin cambiar su lógica.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí, debe dar la misma probabilidad si ambos cargan la misma versión del modelo,
calculan `hora` y `monto_por_km` con la misma función y entregan las columnas en
el mismo orden. El umbral puede cambiar la decisión, pero no la probabilidad que
produce el modelo.

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde 422 y señala `body -> monto` con el mensaje de que el texto no se puede
convertir a un número. La validación nace en la declaración `monto: float` de
la clase `Pago`; FastAPI la aplica antes de ejecutar `evaluar_pago`.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe el cantón, fecha, conteo actual, rezagos de 1, 7, 14 y 28 días, el
conteo de hace 4 días, promedios anteriores de 7 y 28 días, y si la fecha
objetivo es feriado. El servicio convierte el conteo de hace cuatro días en
`lag_4d`, obtiene el día de la semana desde la fecha y calcula
`es_fin_semana_t3`. También informa la fecha objetivo
tres días después. Sé que usa la misma lógica porque importa
`preparar_prediccion` desde `modelado/pipeline_proyecto.py`; la fórmula no está
copiada dentro de la API.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

Petición:

```json
{
  "cod_canton": "0101",
  "fecha": "2026-09-03",
  "n_emergencias": 20,
  "lag_1d": 18,
  "n_emergencias_hace_4_dias": 16,
  "lag_7d": 15,
  "lag_14d": 14,
  "lag_28d": 13,
  "media_prev_7d": 17,
  "media_prev_28d": 15,
  "es_feriado_t3": 0
}
```

Respuesta 200 observada dentro del contenedor:

```json
{
  "cod_canton": "0101",
  "fecha_objetivo": "2026-09-06",
  "emergencias_estimadas": 19.61,
  "run_id": "e60512928d4f4d93afec8350fe9f1fc3"
}
```

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Guardaría el `run_id` junto a cada predicción; por eso la API lo incluye en la
respuesta. Así se puede buscar en MLflow el modelo, sus parámetros y métricas.
Para volver atrás exportaría nuevamente el artefacto de un `run_id` anterior,
reconstruiría la imagen y desplegaría esa versión conocida.
