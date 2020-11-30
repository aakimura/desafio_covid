import os
import fnmatch
import requests
import pandas as pd


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


def import_gt(pattern):
    """Importa archivos CSV de resultados de búsquedas de Google Trends,
       resetea el índice y transforma los datos a sus tipos apropiados"""
    df = pd.read_csv(find(pattern, gt_path), header=1, index_col='Week')

    # Reformatear columnas
    keyword = pattern.replace('*', '')
    df.columns = [keyword]
    df.index = df.index.rename('google_date')

    # Convertir columnas a respectivos data types
    df.index = pd.to_datetime(df.index)

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


# Ejecutar codigo
def main():
    # Obtener DataFrame de casos confirmados
    covid_df = get_covid(covid_url).groupby('wk').sum()['cases_daily']
    gt_sym = import_gt('sintomas_covid*')
    gt_sl = import_gt('perdida_olfato*')
    gt_tl = import_gt('perdida_gusto*')
    df = pd.concat([covid_df, gt_sym, gt_sl, gt_tl], axis=1)

    # Crear columna con número normalizado de casos confirmados
    df['cases_norm'] = round(
        (df['cases_daily'] / max(df['cases_daily']) * 100), 0)

    # Eliminar clutter
    df = df.drop('google_date', axis=1)
    df = df.dropna()

    return df


if __name__ == '__main__':
    main()
