import os
import re
import fnmatch
import requests
import pandas as pd
import datetime

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
    df1 = target_df

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
    df1 = df1.corr()

    # Limpiar resultados
    df1 = df1.drop(['cases_daily', 'cases_norm']).sort_values(
        by='cases_daily', ascending=False)
    df1 = df1['cases_daily']

    return df1


# Ejecutar codigo
def main():
    # Obtener DataFrame de casos confirmados
    covid_df = get_covid(covid_url)

    corr_df = correlate_gt(gt_path, target_df=covid_df.groupby('wk').sum(),
                           lag=0)

    return covid_df, corr_df


if __name__ == '__main__':
    main()
