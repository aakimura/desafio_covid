# Uso de Google Trends para proyectar nuevos contagios por COVID-19

El uso de Google Trends para pronosticar distintos fenómenos epidemiológicos, financieros y sociales ha crecido desde 2009. Su primera aplicación fue el desarrollo una herramienta para detectar casos de influenza en Estados Unidos. Es tal el valor el que aporta que existen propuestas para implementar dicho índice en los modelos de vigilancia de bancos centrales.

## ¿Por qué escogí este proyecto?

Si bien el impacto económico ocasionado por la pandemia es irrefutable, los verdaderos efectos se verán hasta el 2021. Agustín Cárstens<sup>1</sup> menciona que "históricamente hay un desfase de un año entre la desaceleración del PIB y el incremento de las quiebras y desempleo." Por lo tanto, **tratar de medir los efectos en la economía es aún prematuro**.

Sin embargo, la evidencia recolectada por Baker et al<sup>2</sup> sugiere que los shocks en el mercado de capitales de Estados Unidos se debió a las medidas de contención de los contagios, principalmente el confinamiento voluntario. **Es decir, hubo un cambio en los hábitos de las personas con la pandemia**.

El peligro de una caída en la actividad económica es también un problema de salud pública. Kimura<sup>3</sup> recolecta evidencia que muestra los efectos del **desempleo** en la salud. Al perder el empleo la situación financiera de las familias se complica, lo que genera incrementos en enfermedades cardiovasculares, muertes con violencia y propagación de enfermedades infecciosas debido a que las familias sin ingresos se ven obligadas al hacinamiento.

En resúmen, el incremento de los contagios modifica los hábitos de las personas lo que impacta directamente la actividad económica incrementando el desempleo y otros problemas de salud pública.

## Si el desempleo empezará a verse hasta el siguiente año, ¿a qué viene tu proyecto?

El punto es poder anticipar políticas públicas que ataquen directamente dos cosas: la propagación del COVID-19 y empezar a recolectar evidencia de desempleo. Si detenemos la propagación, la economía puede mantenerse activa.

En este proyecto me enfoqué en la propagación del COVID-19 porque el desempleo es una métrica que difícilmente puede encontrarse desagregada por semana. Pero en esencia, el proceso es el mismo.

## Google Trends y la propagación del COVID-19

Google Trends no solamente es una herramienta para la optimización de motores de búsqueda. Su esencia permite también saber en qué están interesadas las personas durante un periodo de tiempo. Resulta sumamente interesante lo preciso que puede llegar a ser la búsqueda de palabras clave con los eventos epidemiológicos, tal y como se puede ver en la siguiente gráfica.

## Datos

Se utilizan data sets de tendencias de búsqueda en [Google Trends](https://trends.google.com/trends/?geo=MX). Se probaron diferentes frases que representan los síntomas característicos de COVID-19 (e.g. "perdida olfato", "perdida gusto", "fiebre", "sintomas covid"), o insumos para evitar contagios (e.g. "cubrebocas", "n95"). Por otro lado, se utilizó la [API de Data México](https://datamexico.org/api/data.jsonrecords?Nation=mex&cube=gobmx_covid_stats_nation&drilldowns=Time,Nation&measures=Accum+Cases,Daily+Cases,AVG+7+Days+Accum+Cases,AVG+7+Days+Daily+Cases,Rate+Daily+Cases,Rate+Accum+Cases,Days+from+50+Cases&parents=true&sparse=false&s=Casospositivosdiarios&q=Fecha&r=withoutProcessOption) para obtener el número de casos diarios de COVID-19.

## Metodología

Como se muestra en la siguiente tabla, la frase que presentó una mayor correlación fue "perdida olfato" seguida de "perdida gusto". Por otro lado, los insumos médicos tienen una pobre correlación con los contagios.

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

## Limitaciones

El uso de índices basado en búsquedas en Google aportan gran valor al complementar modelos tradicionales de predicción. Sin embargo, la pandemia de COVID-19 tiene elementos muy particulares que impiden recolectar datos anteriores a su aparición y por lo tanto, el *dataset* será limitado. Por ejemplo, COVID-19 es la primera pandemia a la cual se le ha encontrado una fuerte correlación con la pérdida del olfato y el gusto, lo que apunta a ser un síntoma previamente no reconocido en otras enfermedades.

## Referencias

1. Carstens, Agustín, 2020. The Great Reallocation. Project Syndicate. [Link](https://www.project-syndicate.org/commentary/covid19-crisis-recovery-structural-reform-by-agustin-carstens-2020-10?barrier=accesspaylog)

2. Baker, S.R., Bloom, N., Davis, S.J., Kost, K.J., Sammon, M.C. and Viratyosin, T., 2020. The unprecedented stock market impact of COVID-19 (No. w26945). National Bureau of Economic Research.

3. Kimura, A., 2020. Los efectos del gran confinamiento en la salud. [Link](https://medium.com/@akimura/efectos-del-gran-confinamiento-en-la-salud-9753526c458)

D’Amuri, F. and Marcucci, J., 2017. The predictive power of Google searches in forecasting US unemployment. International Journal of Forecasting, 33(4), pp.801-816.

Lazer, D., Kennedy, R., King, G. and Vespignani, A., 2014. The parable of Google Flu: traps in big data analysis. Science, 343(6176), pp.1203-1205.

Zhang X, Zhang T, Young AA, Li X (2014) Applications and Comparisons of Four Time Series Models in Epidemiological Surveillance Data. PLoS ONE 9(2): e88075.
