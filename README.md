# Uso de Google Trends para proyectar nuevos contagios por COVID-19

El uso de Google Trends para pronosticar distintos fenómenos epidemiológicos, financieros y sociales ha crecido desde 2009. Su primera aplicación fue el desarrollo una herramienta para detectar casos de influenza en Estados Unidos. Es tal el valor el que aporta que existen propuestas para implementar dicho índice en los modelos de vigilancia de bancos centrales.

## ¿Por qué escogí este proyecto?

Si bien el impacto económico ocasionado por la pandemia es irrefutable, los verdaderos efectos se verán hasta el 2021. Agustín Cárstens<sup>1</sup> menciona que "históricamente hay un desfase de un año entre la desaceleración del PIB y el incremento de las quiebras y desempleo." Por lo tanto, **tratar de medir los efectos en la economía es aún prematuro**.

Sin embargo, la evidencia recolectada por Baker et al<sup>2</sup> sugiere que los shocks en el mercado de capitales de Estados Unidos se debió a las medidas de contención de los contagios, principalmente el confinamiento voluntario. **Es decir, hubo un cambio en los hábitos de las personas con la pandemia**.

El peligro de una caída en la actividad económica es también un problema de salud pública. Kimura<sup>3</sup> recolecta evidencia que muestra los efectos del **desempleo** en la salud. Al perder el empleo la situación financiera de las familias se complica, lo que genera incrementos en enfermedades cardiovasculares, muertes con violencia y propagación de enfermedades infecciosas debido a que las familias sin ingresos se ven obligadas al hacinamiento.

En resúmen, el incremento de los contagios modifica los hábitos de las personas lo que impacta directamente la actividad económica incrementando el desempleo y otros problemas de salud pública.

## Si el desempleo empezará a verse hasta el siguiente año, ¿a qué viene tu proyecto?

El punto es poder anticipar políticas públicas que ataquen directamente dos cosas: la propagación del COVID-19 y empezar a recolectar evidencia de desempleo. Si detenemos la propagación, la economía puede mantenerse activa.

En este proyecto me enfoqué en la propagación del COVID-19 porque el desempleo debería aparecer el primer o segundo trimestre del siguiente año, además es una métrica que difícilmente puede encontrarse desagregada por semana. Pero en esencia, el proceso es el mismo.

## Google Trends y la propagación del COVID-19

Google Trends no solamente es una herramienta para la optimización de motores de búsqueda. Su esencia permite también saber en qué están interesadas las personas durante un periodo de tiempo. Resulta sumamente interesante lo preciso que puede llegar a ser la búsqueda de palabras clave con los eventos epidemiológicos, tal y como se puede ver en la siguiente gráfica.

![Casos confirmados y búsqueda en Google](https://raw.githubusercontent.com/aakimura/desafio_covid/img/cases_perdida_olfato.png)

## Datos

Se utilizó la [API de Data México](https://datamexico.org/api/data.jsonrecords?Nation=mex&cube=gobmx_covid_stats_nation&drilldowns=Time,Nation&measures=Accum+Cases,Daily+Cases,AVG+7+Days+Accum+Cases,AVG+7+Days+Daily+Cases,Rate+Daily+Cases,Rate+Accum+Cases,Days+from+50+Cases&parents=true&sparse=false&s=Casospositivosdiarios&q=Fecha&r=withoutProcessOption) para obtener el número de casos diarios de COVID-19. Por otro lado, se utilizan data sets de tendencias de búsqueda en [Google Trends](https://trends.google.com/trends/?geo=MX). Se probaron diferentes frases que representan los síntomas característicos de COVID-19 (e.g. "perdida olfato", "perdida gusto", "fiebre", "sintomas covid"), insumos para evitar contagios (e.g. "cubrebocas", "n95") o actividades (e.g. "hoteles abiertos", "cines abiertos", "restaurantes abiertos".

## ¿Cuáles son las frases correlacionadas con los contagios?

Se seleccionaron diversas frases que representen tanto los mismos síntomas de la enfermedad, así como actividades que podrían propiciar el contagio como por ejemplo "restaurantes" y "cines" ya que se llevan a cabo en lugares cerrados.

En lo que va del 2020 la frase que presentó una mayor correlación fue "perdida olfato" seguida de "perdida gusto". Por otro lado, los insumos médicos tienen una pobre correlación con los contagios.

Frases relacionadas con vacaciones tienen una correlación débil, sin embargo, actividades en lugares cerrados como "restaurantes" o "cines" tienen correlaciones significativas.

|Clase      | Frase               | Correlación
|-----------|---------------------|------------:
|Síntomas   |perdida olfato       |     0.897261
|           |perdida gusto        |     0.871580
|           |sintomas covid       |     0.814489
|           |cansancio            |     0.624288
|           |fiebre               |    -0.001196
|Insumos    |caretas              |     0.312766
|           |guantes latex        |    -0.164120
|           |cubrebocas           |     0.154195
|Actividades|airbnb               |    -0.033366
|           |airbnb covid         |     0.289632
|           |hoteles              |    -0.302206
|           |hoteles abiertos     |     0.026534
|           |hoteles covid        |     0.482530
|           |restaurantes         |    -0.008360
|           |restaurantes abiertos|     0.674170
|           |restaurantes covid   |     0.539001
|           |cines                |     0.302642
|           |cines abiertos       |     0.692476
|           |cines covid          |     0.603122

Algunas frases mantienen su trayectoria, aunque claramente otras evolucionan de acuerdo a los casos de COVID.

![Evolucion de casos y frases](https://raw.githubusercontent.com/aakimura/desafio_covid/img/all_corr.png)

### Síntomas

Retomando la correlación entre "perdida_olfato" y los casos confirmados resultan interesantes tres puntos:

1) Picos bruscos en las primeras semanas hasta el máximo histórico de la semana 29. Probablemente debido a la atención generada al conocerse los síntomas característicos del COVID-19.

2) El pico de la semana 29 es predecido 2 semanas antes por las búsquedas en Google y con la misma magnitud que los casos.

3) Sin embargo, el segundo pico de la semana 43 no pudo ser predecido a tiempo. De hecho los picos de ambas series se da la misma semana pero con magnitudes distintas.

Este último punto nos podría indicar el conocimiento adquirido del público respecto a este síntoma o fatiga respecto al tema. Por lo tanto, podríamos intuir que las **búsquedas pierden poder con el paso del tiempo**. He ahí la necesidad de combinar este método con otros modelos.

Al presentar altas correlaciones la intuición sería correr regresiones sobre estos indicadores para obtener el número de casos confirmados. Sin embargo, se ha documentado en Lazer et al.<sup>4</sup> que depender únicamente de éstos generaría un sesgo que no precisamente corresponde a la dinámica de la enfermedad. Los autores sugieren combinar indicadores de Google con formas tradicionales de estimación. En el caso de eventos epidemiológicos, ARIMA es uno de ellos.

### Insumos médicos

Otro caso interesante son las búsquedas relacionadas a insumos médicos (e.g. "cubrebocas" y "N95"). Como se puede ver, alrededor de la semana 15 alcanzaron sus puntos máximos para después descender consistentemente. Esto nos hablaría del pánico que generó la enfermedad en sus primeras fases y la necesidad auto-generada de a población de abastecerse de dichos insumos de protección.

### Vacaciones y actividades

Durante 2020 las búsquedas de hoteles tocaron su máximo la segunda semana del año, justo la fecha más baja en un año habitual. Esto nos habla de la caída estrepitosa que tuvo el sector este año.

Por otro lado, de las actividades en lugares cerrados, las búsquedas de "restaurantes abiertos" parecen imitar la curva de contagios hasta la semana 40 cuando pierden su eficacia.

## ¿Qué podemos hacer con esta información?

Definitivamente el índice de Google debería aportar valor a las metodolgías tradicionales de estimación.

## ARIMA y COVID

ARIMA o *Autocorreation Integrated Moving Average* es un modelo estadístico que permite generar predicciones a partir de series temporales. Su supuesto principal es que la serie de tiempo debe ser estacionaria, es decir, no debe tener tendencia ni estacionalidad.

La estimación de ARIMA es iterativa y en su mayoría puede evaluarse visualmente. Por ejemplo, para estimar las transformaciones iniciales, es necesario identificar algúna transforación no lineal que deba realizarse como preparativo. Revisando la serie que nos ocupa podemos comprobar la que es una serie no estacionaria con un incremento explosivo los primeros meses.

![Casos confirmados por semana](https://raw.githubusercontent.com/aakimura/desafio_covid/img/cases_weekly.png)

La serie de casos fue transformada mediante logaritmo natural. Al inspeccionarse las funciones ACF y PACF (Función de Autocorrelación y Función de Autocorrelación Parcial, respectivamente) se encontró que el mejor modelo para predecir los movimientos del COVID sería ARIMA(1, 1, 0). Dicha configuración corresponde a un modelo autorregresivo de primer orden. Este modelo alcanzó un RMSE de 0.09852.

![Resultados de la autocorrelación](https://raw.githubusercontent.com/aakimura/desafio_covid/img/ln_fcst_arima100_train80.png)

## ARIMA y Google Trends

La estimación de ARIMA con variable exógena, sin embargo, no corrió la misma suerte que el modelo anterior. Tanto los p-values como los t-stats del lag como del Google index resultaron no estadísticamente significativos.

## ¿Qué implica esto?

Es necesario continuar con el trabajo de prueba para encontrar el modelo idóneo paa combinar el modelo autorregresivo con la variable explicatoria Google Index.

Esto nos permitiría poder anticipar movimientos de la pandemia que impacten directamente en la economía, como los picos de pánico de los primeros meses o el confinamiento voluntario en ambos picos.

## Limitaciones

El uso de índices basado en búsquedas en Google aportan gran valor al complementar modelos tradicionales de predicción. Sin embargo, la pandemia de COVID-19 tiene elementos muy particulares que impiden recolectar datos anteriores a su aparición y por lo tanto, el *dataset* será limitado. Por ejemplo, COVID-19 es la primera pandemia a la cual se le ha encontrado una fuerte correlación con la pérdida del olfato y el gusto, lo que apunta a ser un síntoma previamente no reconocido en otras enfermedades.

Google ya ha intendado crear herramientas basadas en sus búsquedas para rastrear eventos epidemiológicos. Sin embargo, su iniciativa fracasó principalmente por los cambios realizados en su motor de búsqueda. Mantener constante el mismo modelo de predicción resulta peligroso por lo que necesita un constante mantenimiento.

## Referencias

1. Carstens, Agustín, 2020. The Great Reallocation. Project Syndicate. [Link](https://www.project-syndicate.org/commentary/covid19-crisis-recovery-structural-reform-by-agustin-carstens-2020-10?barrier=accesspaylog)

2. Baker, S.R., Bloom, N., Davis, S.J., Kost, K.J., Sammon, M.C. and Viratyosin, T., 2020. The unprecedented stock market impact of COVID-19 (No. w26945). National Bureau of Economic Research.

3. Kimura, A., 2020. Los efectos del gran confinamiento en la salud. [Link](https://medium.com/@akimura/efectos-del-gran-confinamiento-en-la-salud-9753526c458)

4. Lazer, D., Kennedy, R., King, G. and Vespignani, A., 2014. The parable of Google Flu: traps in big data analysis. Science, 343(6176), pp.1203-1205.

D’Amuri, F. and Marcucci, J., 2017. The predictive power of Google searches in forecasting US unemployment. International Journal of Forecasting, 33(4), pp.801-816.

Lazer, D., Kennedy, R., King, G. and Vespignani, A., 2014. The parable of Google Flu: traps in big data analysis. Science, 343(6176), pp.1203-1205.

Zhang X, Zhang T, Young AA, Li X (2014) Applications and Comparisons of Four Time Series Models in Epidemiological Surveillance Data. PLoS ONE 9(2): e88075.
