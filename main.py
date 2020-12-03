import os
import re
import fnmatch
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.metrics
import warnings

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

# Filtrar alertas
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

# Definir paths
root_path = os.getcwd()
data_path = os.path.join(root_path, 'data')
gt_path = os.path.join(data_path, 'google_trends')

# Definir URLs
covid_url = 'https://datamexico.org/api/data.jsonrecords?Nation=mex&cube=gobmx\
             _covid_stats_nation&drilldowns=Time,Nation&measures=Accum+Cases,\
             Daily+Cases,AVG+7+Days+Accum+Cases,AVG+7+Days+Daily+Cases,Rate+\
             Daily+Cases,Rate+Accum+Cases,Days+from+50+Cases&parents=true&\
             sparse=false&s=Casos positivos diarios&q=Fecha&r=\
             withoutProcessOption'.replace(' ', '')


# Definir funciones
def find(pattern, path):
    """Busca el archivo más reciente con el patrón proporcionado"""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    # Trae el archivo más reciente
    result = max(result, key=os.path.getctime)
    return result


def get_covid(url):
    """Hace un request a la API de Data México y convierte las fechas a
       datetime"""
    # Obtener data frame
    res = requests.get(url).json()
    df = pd.DataFrame.from_dict(res['data'])

    # Reformatear nombres de columnas
    df = df.rename(columns={'Daily Cases': 'cases_daily'})
    # Convertir Time a datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Crear columna de número de semana del año
    df['wk'] = df['Time'].dt.isocalendar().week

    return df


def import_gt(pattern, lag=0):
    """Importa archivos CSV de resultados de búsquedas de Google Trends,
       resetea el índice y transforma los datos a sus tipos apropiados"""
    df = pd.read_csv(find(pattern, gt_path), header=1, index_col='Week')

    # Reformatear columnas
    keyword = pattern.replace('*', '')
    df.columns = [keyword]
    df.index = df.index.rename('google_date')

    # Convertir columnas a respectivos data types
    df.index = pd.to_datetime(df.index)

    # Desplazar curva x número de días determinados por la variable offset
    df.index = df.index + datetime.timedelta(lag)

    try:
        df[keyword] = df[keyword].str.replace('<', '').astype(int)
    except AttributeError:
        pass

    # Crear columna de número de semana del año
    df['wk'] = df.index.to_series().dt.isocalendar().week

    # Indexar data frame con el número de la semana
    df = df.reset_index()
    df = df.set_index('wk')

    return df


def correlate_gt(path, target_df, lag=0):
    # Obtener DataFrame de casos confirmados
    df1 = target_df.groupby('wk').sum()

    # Normalizar casos diarios
    df1['cases_norm'] = round(
        df1['cases_daily'] / max(df1['cases_daily']) * 100, 0)

    df1 = df1[['cases_daily', 'cases_norm']]

    # Procesar Google Trends
    files = [i for i in os.listdir(path) if i.endswith('.csv')]

    # Obtener keywords usados en Googlt Trends
    keywords = [re.match(r'(.+?)(_\d+\.csv)', i).group(1) for i in files]
    keywords = list(set(keywords))

    # Crear data frame

    for keyword in keywords:
        # Checar que keyword sea único
        check = [i for i in keywords if keyword in i]

        if len(check) > 1:
            keyword = keyword + '_2'

        # Obtener data frame del keyword
        try:
            df2 = import_gt(keyword + '*', lag=lag)
        except ValueError:
            print(keyword)
            break

        df1 = pd.concat([df1, df2], axis=1)

    # Limpiar data frame
    df1 = df1.drop('google_date', axis=1).dropna()

    # Correlacionar variables
    df2 = df1.corr()

    # Limpiar resultados
    df2 = df2.drop(['cases_daily', 'cases_norm']).sort_values(
        by='cases_daily', ascending=False)
    df2 = df2['cases_daily']

    return df1, df2


def plot_grid(corr_df):
    fig, axs = plt.subplots(5, 4, sharex=True, sharey=True)
    i = 0
    j = 0
    order = ['sintomas_covid', 'perdida_olfato', 'perdida_gusto', 'cansancio',
             'fiebre', 'cubrebocas', 'caretas', 'guantes_latex', 'n95',
             'hoteles_2', 'hoteles_covid', 'hoteles_abiertos', 'airbnb_2',
             'airbnb_covid', 'restaurantes_2', 'restaurantes_covid',
             'restaurantes_abiertos', 'cines_2', 'cines_covid',
             'cines_abiertos']
    for t in order:
        name = re.sub(r'_2', r'', t)
        df = corr_df[['cases_norm', t]]
        df.plot(ax=axs[i, j], legend=False, color=['gray', 'red'])
        axs[i, j].set_title(name)

        if j == 3:
            j = 0
            i += 1
        else:
            j += 1

    plt.tight_layout()
    plt.show()


def arima_tests(df, exog, p, d, q):
    """df se refiere a la serie después de transformaciones. Este método es
       útil para encontrar las transformaciones más efectivas, así como los
       parámetros p y q de ARIMA(p, d, q)."""
    # Prueba de Dickey-Fuller de series estacionarias
    result = adfuller(df)
    if result[1] > 0.05:
        print('Serie no estacionaria.')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        df = df.diff().dropna()
        result = adfuller(df)
    else:
        print('Serie estacionaria.')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Valores críticos:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    # Funciones de autocorrelación y autocorrelación parcial
    sm.graphics.tsa.plot_acf(df, title='y ACF', zero=False)
    sm.graphics.tsa.plot_pacf(df, title='y PACF', zero=False)

    # Modelar ARIMA
    model = ARIMA(endog=df, exog=exog, order=(p, d, q))
    model_fit = model.fit(disp=0)
    residuals = model_fit.resid

    # Regresar evaluadores e interpetación
    p_interp = []
    for v in model_fit.pvalues:
        if v < 0.05:
            p_interp.append('OK')
        else:
            p_interp.append('Falló')

    t_interp = []
    for v in model_fit.tvalues:
        if v > 2:
            t_interp.append('OK')
        else:
            t_interp.append('Falló')

    frames = [model_fit.pvalues, model_fit.tvalues]
    evaluation = pd.concat(frames, axis=1).rename(
        columns={0: 'p-values', 1: 't-values'})
    evaluation['Evaluacion p-value'] = p_interp
    evaluation['Evaluacion t-value'] = t_interp

    print(evaluation)

    # Graficar residuales
    sm.graphics.tsa.plot_acf(residuals, title='residuals ACF', zero=False)
    sm.graphics.tsa.plot_pacf(residuals, title='residuals PACF', zero=False)

    plt.show()


def arima_fcst(df, exog, p, d, q, train_pct=0):
    """Calcula el modelo ARIMA(p, d, q) y obtiene las funciones ACF y PACF de
    los residuales.
    train_pct indica el porcentaje de observaciones que se reservan para
    evaluar el modelo. Por lo tanto (1 - train_pct) es la proporción del
    dataset destinada a entrenar el modelo."""
    # Dividir dataset
    if train_pct <= 1:
        size = int(len(df) * train_pct)
        train, test = df[0:size], df[size:len(df)]
        history = [x for x in train]
        predictions = list()

        # Acotar variables exógenas a las semanas del train set
        if exog is not None:
            exog = exog.loc[(exog.index >= min(train.index)) &
                            (exog.index <= max(train.index))]

        # Predecir cada uno de los pasos utilizando una "historia" actualizada
        # con las predicciones anteriores
        for t in range(len(test)):
            model = ARIMA(endog=history, exog=exog, order=(p, d, q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test.iloc[t]
            history.append(obs)

        # Mediciones de error
        rmse = sklearn.metrics.mean_squared_error(test, predictions,
                                                  squared=False)
        mae = sklearn.metrics.mean_absolute_error(test, predictions)

        print('Errores \n' + '-' * 20 + '\n' + 'RMSE:\t' + str(rmse) +
              '\nMAE:\t' + str(mae))

        # Graficar resultados
        plt.plot(test.to_list())
        plt.plot(predictions, color='red')
        plt.show()

    else:
        print('El porcentaje de train set es mayor a 1.')


# Ejecutar codigo
def main():
    # Obtener DataFrame de casos confirmados
    covid_df = get_covid(covid_url)
    wk_df = covid_df.groupby('wk').sum()

    # Las siguientes transformaciones fueron iteradas manualmente utilizando
    # la función arima_tests() para encontrar la serie ideal.

    # 1) Filtrar serie de tiempo (eliminar semanas sin casos confirmados)
    arima_target = wk_df.iloc[8:]['cases_daily']

    # 2) Transformacion por ln
    arima_target = np.log(arima_target)
    # ACF y PACF muestran lags tardíos más estables en ln que diferenciación,
    # cuyo PACF muestra que los lags 11-14 son significativos.
    # Diferenciación:
    #     ADF coeff: -2.9235 (valor crítico 5% = -2.9412)
    #     p-value: 0.04267
    # Logaritmo natural:
    #     ADF coeff: -16.3568 (valor crítico 1% = -3.6155)
    #     p-value: 0.0000

    # Modelar ARIMA
    model = ARIMA(endog=arima_target,
                  exog=None,
                  order=(1, 1, 0))
    # El orden ARIMA(1, 0, 0) proviene de las gráficas ACF y PACF donde PACF
    # presenta picos en k=1 y una caída significativa y definitiva en k=2,
    # mientras que ACF muestra un pico en k=1 y un descenso gradual.
    # Por lo tanto la serie presenta un patrón AR(1) distintivo.

    return covid_df, arima_target


if __name__ == '__main__':
    main()
