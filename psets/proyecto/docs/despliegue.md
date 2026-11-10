# Decisión de despliegue del proyecto

## Patrón elegido

Elegí servir el modelo **por lotes (batch)** una vez al día. El modelo estima
cuántas emergencias tendrá cada cantón dentro de tres días. Esa predicción
ayuda al personal de operaciones a planificar con anticipación la distribución
de recursos y la atención de los cantones con mayor carga esperada.

La predicción no tiene que aparecer en milisegundos. Puede ejecutarse al cerrar
los datos de cada día y dejar una tabla lista en **menos de una hora**. Luego un
tablero o reporte puede consultar esos resultados durante el día.

## Casos que se predicen

Los casos se conocen antes de ejecutar el modelo: se necesita una predicción
por cada cantón y fecha objetivo. Por eso se puede armar el lote completo sin
esperar una petición de un usuario. En el desarrollo estoy usando la muestra
local de seis meses que se preparó en las semanas anteriores.

## Datos disponibles al predecir

Las features usan el número de emergencias del día actual, los rezagos de 1,
4, 7, 14 y 28 días, los promedios de los 7 y 28 días anteriores, el cantón y
datos del calendario para el día que se quiere predecir. Los conteos y
promedios quedan disponibles cuando termina la carga diaria. Los feriados y
fines de semana se conocen de antemano, así que no requieren información del
futuro.

## Qué se pierde al no usar predicción online

Con batch no se obtiene una estimación nueva cada vez que llega una emergencia
durante el día. Si ocurre un evento inesperado, la predicción queda igual hasta
el siguiente lote. Un servicio online sería más fresco, pero también necesitaría
actualizar los agregados casi en tiempo real y mantener una API disponible. Para
una decisión de planificación a tres días, ese costo extra no parece necesario
por ahora.
