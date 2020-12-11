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
from sklearn.ensemble import RandomForestRegressor


# Filtrar alertas
warnings.filterwarnings(action='ignore', module='statsmodels.tsa.arima_model',
                        category=FutureWarning)

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


# Definir MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


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


def plot_single(df, title, vline=0):
    df.plot(title=title)
    plt.xlabel('Semana')
    plt.ylabel('Indice')
    if vline > 0:
        plt.axvline(vline, color='red')
    plt.show()


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


def arima_fcst(df, exog, p, d, q, train_pct=0, trans='diff', exog_lag=0):
    """Calcula el modelo ARIMA(p, d, q) y obtiene las funciones ACF y PACF de
    los residuales.
    train_pct indica el porcentaje de observaciones que se reservan para
    evaluar el modelo. Por lo tanto (1 - train_pct) es la proporción del
    dataset destinada a entrenar el modelo."""
    # Dividir dataset
    if train_pct <= 1:
        # Modelado ARIMA y variables exógenas
        if exog is not None:
            # Incluir lag en variables exogenas
            exog.index = exog.index + exog_lag

            model_type = 'ARIMAX'
            # Semanas que se intersectan en ambas series
            wks = exog.index.intersection(df.index)
            # Semanas a proyectar con el modelo entrenado
            # wksx = exog.index.difference(df.index)
            # Variables exogenas acotadas
            exog = exog.loc[wks]
            # Variables endogenas acotadas
            df = df.loc[wks]

            size = int(len(df) * train_pct)
            train, test = df[0:size], df[size:len(df)]
            train_exog, test_exog = exog[0:size], exog[size:len(exog)]
            history = [x for x in train]
            history_exog = [x for x in train_exog]
            predictions = list()

            # Predecir cada uno de los pasos utilizando "historia" actualizada
            # con las predicciones anteriores
            for t in range(len(test)):
                # Crear modelo
                model = ARIMA(endog=history, exog=history_exog,
                              order=(p, d, q))
                model_fit = model.fit(disp=0)
                # Proyectar con autorregresión y exog
                target_exog = pd.DataFrame(train_exog).iloc[t]
                output = model_fit.forecast(exog=target_exog)
                yhat = output[0][0]
                predictions.append(yhat)
                # Agregar observaciones a variables endógenas
                obs = test.iloc[t]
                gt = test_exog.iloc[t]
                history.append(obs)
                history_exog.append(gt)

        # Modelado solo ARIMA
        else:
            model_type = 'ARIMA'
            size = int(len(df) * train_pct)
            train, test = df[0:size], df[size:len(df)]
            history = [x for x in train]
            predictions = list()

            # Predecir cada uno de los pasos utilizando"historia" actualizada
            # con las predicciones anteriores
            for t in range(len(test)):
                # Crear modelo
                model = ARIMA(endog=history, exog=None, order=(p, d, q))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0][0]
                predictions.append(yhat)
                # Agregar observaciones a variables endógenas
                obs = test.iloc[t]
                history.append(obs)

        # Reversar la transformación
        if trans == 'ln':
            test = np.exp(test)
            predictions = np.exp(predictions)
            df = np.exp(df)

        # Mediciones de error
        rmse = sklearn.metrics.mean_squared_error(test, predictions,
                                                  squared=False)
        mae = sklearn.metrics.mean_absolute_error(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)

        # Imprimir reporte
        print('\nModelo ' + model_type + '(' + str(p) + ', ' +
              str(d) + ', ' + str(q) + ')\n' + '-' * 20)
        print('\nErrores \n' + '-' * 20 +
              '\nRMSE:\t' + str(rmse) +
              '\nMAE:\t' + str(mae),
              '\nMAPE:\t' + str(mape) + '\n' + '-' * 20 +
              '\nÚltimas observaciones:\n')
        chart = pd.DataFrame({'Observaciones': df,
                              'Predicciones': pd.Series(predictions,
                                                        index=test.index)})
        chart['Error'] = (chart['Observaciones'] - chart['Predicciones']) /\
            chart['Observaciones']
        print(chart.tail(5))

        # Graficar resultados
        chart[['Observaciones', 'Predicciones']].plot(title='ARIMA(X)')
        plt.ylim(bottom=min(train))

    else:
        print('El porcentaje de train set es mayor a 1.')


def rfr_fcst(X, y, X_lag=0, train_pct=0.8):
    """Random forest regression para proyectar el número semanal de casos respecto al índice de Google Trends."""
    # Adicionar lag a la variable independiente
    X.index = X.index + X_lag

    # Encontrar intersección entre semanas
    wks = X.index.intersection(y.index)
    X = np.array(X.loc[wks])
    y = np.array(y.loc[wks])

    # Reformatear figuras
    X = X.reshape(-1, 1)

    # Partir dataset en train y test
    size = int(len(y) * train_pct)
    X_train, X_test = X[0:size], X[size:len(X)]
    y_train, y_test = y[0:size], y[size:len(y)]

    # Crear regresión lineal
    regr = RandomForestRegressor(n_estimators=500)

    # Entrenar modelo con los training sets
    regr.fit(X_train, y_train)

    # Crear predicciones con el test set
    y_pred = regr.predict(X_test)

    # Calcular errores
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Crear reportes
    print('\nModelo Random Forest Regression\n' + '-' * 20)
    print('\nCoeficiente de determinación\n', '-' * 20)
    print('R^2: %.2f' % regr.score(X_test, y_test))
    print('Errores \n' + '-' * 20 +
          '\nRMSE:\t' + str(rmse) +
          '\nMAE:\t' + str(mae),
          '\nMAPE:\t' + str(mape) + '\n' + '-' * 20 +
          '\nÚltimas observaciones:\n')

    chart = pd.DataFrame({'Observaciones': y_test,
                          'Predicciones': y_pred})
    chart['Error'] = (chart['Observaciones'] - chart['Predicciones']) /\
        chart['Observaciones']
    print(chart.tail(5))

    # Graficar resultados
    chart[['Observaciones', 'Predicciones']].plot(
        title='Random Forest Regression')


# Ejecutar codigo
def main():
    # Obtener DataFrame de casos confirmados
    df = get_covid(covid_url)
    this_week = datetime.date.today().isocalendar()[1]
    df = df[~df['wk'].isin([this_week])]
    wk_df = df.groupby('wk').sum()

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

    # Obtener variables exógenas
    keyword = 'perdida_olfato'
    target_gt = import_gt(keyword.replace(' ', '_') + '*')

    # a) Obtener intersección entre los dos conjuntos de semanas

    # b) Construir serie exógena
    exog = target_gt.loc[:, keyword]

    # Modelar ARIMA
    arima_fcst(arima_target, exog=None, p=1, d=0, q=0, train_pct=0.8,
               trans='ln', exog_lag=1)
    # El orden ARIMA(1, 0, 0) proviene de las gráficas ACF y PACF donde PACF
    # presenta picos en k=1 y una caída significativa y definitiva en k=2,
    # mientras que ACF muestra un pico en k=1 y un descenso gradual.
    # Por lo tanto la serie presenta un patrón AR(1) distintivo.

    # Modelar Random Forest Regression
    rfr_fcst(wk_df['cases_daily'], wk_df['cases_daily'], X_lag=1, train_pct=0.8)

    # Mostrar graficas
    plt.show()


if __name__ == '__main__':
    main()
