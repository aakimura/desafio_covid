# Uso de Google Trends para proyectar nuevos contagios por COVID-19

El uso de Google Trends para pronosticar distintos fenómenos epidemiológicos, financieros y sociales ha crecido desde 2009. Su primera aplicación fue el desarrollo una herramienta para detectar casos de influenza en Estados Unidos.

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

D’Amuri, F. and Marcucci, J., 2017. The predictive power of Google searches in forecasting US unemployment. International Journal of Forecasting, 33(4), pp.801-816.

Lazer, D., Kennedy, R., King, G. and Vespignani, A., 2014. The parable of Google Flu: traps in big data analysis. Science, 343(6176), pp.1203-1205.

Zhang X, Zhang T, Young AA, Li X (2014) Applications and Comparisons of Four Time Series Models in Epidemiological Surveillance Data. PLoS ONE 9(2): e88075. https://doi.org/10.1371/journal.pone.0088075
