import pandas as pd
import numpy as np
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense


'''
FUNCIONES DE IMPORTACIÓN
'''

def import_his_mini_ibex(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)

  df_completo = pd.read_csv("{0}/Input/min_ibx_fut.csv".format(ruta), sep = ";", parse_dates = ['Fecha'], date_parser = parser, decimal= ',')
  df_completo = df_completo.sort_values('Contrato', ascending=True).groupby('Fecha').head(1)
  df_completo = df_completo.sort_values('Fecha', ascending=True)
  df_completo.reset_index(drop = True, inplace = True)
  df_reducido = df_completo[['Fecha', 'Cierre', 'VolatCierre', 'VolatApertura']].copy()
  df_reducido = df_reducido.rename(columns={'Cierre': 'FmI_Cierre', 'VolatCierre': 'FmI_VolatCierre', 
                              'VolatApertura': 'FmI_VolatApertura'})
  #df_reducido.info()
  return df_reducido

def import_his_ibx_plus(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)

  df_completo = pd.read_csv("{0}/Input/ibx_plus_volatilidad.csv".format(ruta), sep = ';', parse_dates = ['Fecha'], date_parser = parser)
  df_reducido = df_completo[['Fecha', 'FutIbexPlus', 'Volat10', 'Volat30', 'Volat60', 'Volat90', 'Volat120']].copy()
  df_reducido = df_reducido.rename(columns={'FutIbexPlus': 'FIp_Cierre', 'Volat10': 'FIp_Volat10', 
                              'Volat30': 'FIp_Volat30', 'Volat60': 'FIp_Volat60', 
                              'Volat90': 'FIp_Volat90', 'Volat120': 'FIp_Volat120'})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido

def import_benchmark_ibex(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  nombre = 'IBEX_Cierre'
  
  df_completo = pd.read_csv("{0}/Input/benchmark_ibex.csv".format(ruta), parse_dates = ['Fecha'], date_parser = parser)
  df_completo = df_completo.loc[df_completo['Ind'] == 'ibex_div']
  df_completo = df_completo.reset_index(drop = True)
  df_reducido = df_completo[['Fecha', 'Close']].copy()
  df_reducido = df_reducido.rename(columns={'Close': nombre})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido

def import_benchmark_estx(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  nombre = 'ESTX_Cierre'
  
  df_completo = pd.read_csv("{0}/Input/benchmark_estx.csv".format(ruta), parse_dates = ['Fecha'], date_parser = parser)
  df_completo = df_completo.loc[df_completo['Ind'] == 'ESTXNR']
  df_completo = df_completo.reset_index(drop = True)
  df_reducido = df_completo[['Fecha', 'Close']].copy()
  df_reducido = df_reducido.rename(columns={'Close': nombre})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido

def import_benchmark_sp(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  nombre = 'SP_Cierre'

  df_completo = pd.read_csv("{0}/Input/benchmark_sp.csv".format(ruta), parse_dates = ['Fecha'], date_parser = parser)
  df_reducido = df_completo[['Fecha', 'Close']].copy()
  df_reducido = df_reducido.rename(columns={'Close': nombre})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido

def import_vivex(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  nombre = 'vivex_Cierre'

  df_completo = pd.read_csv("{0}/Input/vivex.csv".format(ruta),sep=";", decimal=',',parse_dates = ['Fecha'], date_parser = parser)
  df_reducido = df_completo[['Fecha', 'Close']].copy()
  df_reducido = df_reducido.rename(columns={'Close': nombre})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido

def import_skew(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  nombre = 'skew_Cierre'
  
  df_completo = pd.read_csv("{0}/Input/skew.csv".format(ruta),sep=";", decimal=',', parse_dates = ['Fecha'], date_parser = parser)
  df_reducido = df_completo[['Fecha', 'Close']].copy()
  df_reducido = df_reducido.rename(columns={'Close': nombre})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido
  
def import_vix(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  nombre = 'vix_Cierre'
  
  df_completo = pd.read_csv("{0}/Input/vix.csv".format(ruta),sep=";", parse_dates = ['Fecha'], date_parser = parser)
  df_reducido = df_completo[['Fecha', 'Close']].copy()
  df_reducido = df_reducido.rename(columns={'Close': nombre})
  df_reducido = df_reducido.sort_values('Fecha', ascending=True)
  #df_reducido.info()
  return df_reducido

def load_ficheros(ruta, dateFormat1 = '%d/%m/%Y', dateFormat2 = '%Y-%m-%d', dateFormat3 = '%m/%d/%Y'):

  df_minFutVol = import_his_mini_ibex(ruta, dateFormat1)
  df_ibxFutVol = import_his_ibx_plus(ruta, dateFormat1)

  df_benchIbex = import_benchmark_ibex(ruta, dateFormat2)
  df_benchEstx = import_benchmark_estx(ruta, dateFormat2)
  df_benchSp = import_benchmark_sp(ruta, dateFormat2)

  df_benchVix = import_vix(ruta, dateFormat3)

  df_vivex = import_vivex(ruta, dateFormat1)
  df_skew = import_skew(ruta, dateFormat1)

  return df_minFutVol, df_ibxFutVol, df_benchIbex, df_benchEstx, df_benchSp, df_benchVix, df_vivex, df_skew


'''
COMPROBACIONES INDIVIDUALES
'''

def ffill_nulos(df):
  df_col_nan = df.isnull().any()
  df_nan = df_col_nan.any()
  # Seleccionamos aquellas columnas con NaN y aplicamos el método Forward-Fill
  if df_nan == True:
    for i in df_col_nan.index:
      if (df_col_nan[i] == True) & (i != 'Fecha'):
        df[i] = df[i].fillna(method = 'ffill')

  return df

def remove_duplicados(df, resetIndex = True):
  df_dup = df.duplicated().any()
  if df_dup == True:
    df.drop_duplicates(keep = 'first', inplace = True)
    if resetIndex == True:
      df.reset_index(drop = True, inplace = True)

  return df

def remove_duplicados_fecha(df, resetIndex = True):
  df_dup_f = df['Fecha'].duplicated().any()
  if df_dup_f == True:
    df.drop_duplicates(subset = 'Fecha', keep = 'first', inplace = True)
    if resetIndex == True:
      df.reset_index(drop = True, inplace = True)

  return df

def check_basicos(df):
  df = ffill_nulos(df)
  df = remove_duplicados(df)
  df = remove_duplicados_fecha(df)
  return df


'''
FUNCIONES PARA UNIÓN DE LOS FICHEROS
'''

def get_espacio_temporal(df_miniIbex, df_ibexPlus, df_bencIbex, df_bencEstx, 
                         df_bencSp, df_bencVix, df_vivex, df_skew):
  df_fechas = pd.DataFrame({'F_Fut_Mini_Ibex': df_miniIbex['Fecha'], 
                            'F_Fut_Ibex_Plus': df_ibexPlus['Fecha'], 
                            'F_Bench_Ibex': df_bencIbex['Fecha'], 
                            'F_Bench_Estx': df_bencEstx['Fecha'],
                            'F_Benc_Sp': df_bencSp['Fecha'],
                            'F_Benc_Vix': df_bencVix['Fecha'],
                            'F_Vivex': df_vivex['Fecha'],
                            'F_Skew': df_skew['Fecha']})
  inicio_comun = df_fechas.min().max().to_pydatetime()
  fin_comun = df_fechas.max().min().to_pydatetime()
  return inicio_comun, fin_comun

def filer_df(df, iniDate, endDate):
  df = df.loc[(df['Fecha'] >= iniDate) & (df['Fecha'] <= endDate)]
  return df

def merge_dataFrames(df_miniIbex, df_ibexPlus, df_bencIbex, df_bencEstx, 
                     df_bencSp, df_bencVix, df_vivex, df_skew):
  # Calculamos el espacio temporal común
  iniDate, endDate = get_espacio_temporal(df_miniIbex, df_ibexPlus, df_bencIbex, df_bencEstx,
                                          df_bencSp, df_bencVix, df_vivex, df_skew)
  
  # Filtramos los DF por el espacio temporal común
  df_miniIbex = filer_df(df_miniIbex, iniDate, endDate)
  df_ibexPlus = filer_df(df_ibexPlus, iniDate, endDate)
  df_bencIbex = filer_df(df_bencIbex, iniDate, endDate)
  df_bencEstx = filer_df(df_bencEstx, iniDate, endDate)
  df_bencSp = filer_df(df_bencSp, iniDate, endDate)
  df_bencVix = filer_df(df_bencVix, iniDate, endDate)
  df_vivex = filer_df(df_vivex, iniDate, endDate)
  df_skew = filer_df(df_skew, iniDate, endDate)

  # Unificamos todos los DF en uno solo
  df = pd.merge(df_miniIbex, df_ibexPlus, how='left', on=['Fecha'])
  df = pd.merge(df, df_bencIbex, how='left', on=['Fecha'])
  df = pd.merge(df, df_bencEstx, how='left', on=['Fecha'])
  df = pd.merge(df, df_bencSp, how='left', on=['Fecha'])
  df = pd.merge(df, df_bencVix, how='left', on=['Fecha'])
  df = pd.merge(df, df_vivex, how='left', on=['Fecha'])
  df = pd.merge(df, df_skew, how='left', on=['Fecha'])
    
  return df

def check_df_features(df):
  feat_null = df.isnull().any().any()
  feat_dup = df.duplicated().any()
  feat_dup_f = df['Fecha'].duplicated().any()

  if (feat_null == True) or (feat_dup == True) or (feat_dup_f == True):
    df = check_basicos(df)
  
  return df


'''
GENERACIÓN DE VOLATILIDADES
'''

def calc_volatilidad(df, prfx, ventana):
  nombre = prfx + '_Vol' + str(ventana)
  col = prfx + '_Cierre'
  # Nos traemos las columnas a un nuevo DF sobre el que trabajaremos
  df_vol = df[['Fecha', col]].copy()
  # Calculamos los rendimientos logarítmicos
  df_vol['Log_Ret'] = np.log(df_vol[col] / df_vol[col].shift(1))
  # Calculamos la STDV para la ventana temporal
  df_vol['Stdv'] = df_vol.Log_Ret.rolling(ventana).std()
  # Multiplicamos la media por la raiz cuadrada de 252 y por 100(%)
  df_vol[nombre] = df_vol.Stdv * np.sqrt(252) * 100
  # Devolvemos al DF original la columna con las volatilidades calculadas
  df[nombre] = df_vol[nombre]

  return df

def get_volatilidades(df, prfx, volatilidades):
  for i in volatilidades:
    df = calc_volatilidad(df, prfx, i)
  
  return df


'''
GENERACIÓN DE Y
'''

def calc_desv_optima(df, dist = 22):
  # Nos traemos las columnas a un nuevo DF sobre el que trabajaremos
  df_desv = df[['Fecha', 'FmI_Cierre']].copy()
  # Calculamos los rendimientos logarítmicos
  df_desv['shift'] = df_desv['FmI_Cierre'].shift(-1 * dist)
  # Redondeamos el precio del mes siguiente para obtener el Strike que maxmimiza el rendimiento
  df_desv['round'] = np.floor((df_desv['shift'])/100) * 100
  # Obtenemos la desviación del Strike, respecto del precio actual
  df_desv['desv'] = df_desv['round'] / df_desv['FmI_Cierre']
  # Devolvemos al DF original la columna con la desviación calculada
  df['Y_FmI_Dv_Opt'] = df_desv['desv']

  return df


'''
FUNCIÓN DE AJUSTE DE LA VENTANA TEMPORAL
'''

def ajuste_ventana_temporal(df, max_vol, dist, ruta):
  datasetValido = False

  feat_null_completo = df.isnull().any().any()
  tratarDatasetCompleto = feat_null_completo == True
  
  if tratarDatasetCompleto == True:
    feat_null_reducido = df.iloc[max_vol:(-1 * dist)].isnull().any().any()
    datasetValido = feat_null_reducido == False
    
    if datasetValido == True:
      df = df.iloc[max_vol:(-1 * dist)]
      df.reset_index(drop = True, inplace = True)

  else: 
    datasetValido = True

  df.to_csv(path_or_buf = '{0}/Output/Estudio_Caracteristicas/dataset.csv'.format(ruta), sep = ';', float_format = '%.4f', index = False, encoding = 'utf-8')
    
  if 'Fecha' in df.columns:
    del df['Fecha']
  
  return datasetValido, df


'''
INSPECCIÓN DE LOS DATOS
'''

def plot_caracteristicas(df, ruta, conj_volatilidades):
  
  sns_plot1 = sns.pairplot(df[["Y_FmI_Dv_Opt", "FmI_Cierre", "FIp_Cierre", "IBEX_Cierre", "ESTX_Cierre", "SP_Cierre"]], diag_kind="kde")
  sns_plot1.savefig('{0}/Output/Estudio_Caracteristicas/Corr_Y_vs_Bench1.png'.format(ruta))

  sns_plot2 = sns.pairplot(df[["Y_FmI_Dv_Opt", "vivex_Cierre", "skew_Cierre", "vix_Cierre"]], diag_kind="kde")
  sns_plot2.savefig('{0}/Output/Estudio_Caracteristicas/Correlacion_Y_vs_Bench2.png'.format(ruta))

  for i in conj_volatilidades:
    sns_plot3 = sns.pairplot(df[["Y_FmI_Dv_Opt", "FmI_Vol{}".format(i), "FIp_Volat{}".format(i), "IBEX_Vol{}".format(i), "ESTX_Vol{}".format(i), "SP_Vol{}".format(i)]], diag_kind="kde")
    sns_plot2.savefig('{0}/Output/Estudio_Caracteristicas/Corr_Y_vs_Vol{1}.png'.format(ruta, i))

def analisis_caracteristicas(df, ruta, conj_volatilidades):
  plot_caracteristicas(df, ruta, conj_volatilidades)
  
  df = df.describe()
  df = df.transpose()
  df.to_csv(path_or_buf = '{0}/Output/Estudio_Caracteristicas/caracteristicas_dataset.csv'.format(ruta), sep = ';', float_format = '%.4f', index = False, encoding = 'utf-8')
  
  return df


'''
ESTRUCTURAS PARA EL MODELO - TRAIN, TEST Y NORMALIZACIÓN
'''

def create_train_test(df, porc_entrenamiento = 0.8):
  df_x_train = df.sample(frac = porc_entrenamiento, random_state = 0)
  df_x_test = df.drop(df_x_train.index)

  s_y_train = df_x_train.pop('Y_FmI_Dv_Opt')
  s_y_test = df_x_test.pop('Y_FmI_Dv_Opt')

  return df_x_train, df_x_test, s_y_train, s_y_test

def min_max_scaler_datos(df):
  normalizer = preprocessing.MinMaxScaler()
  norm_array = normalizer.fit_transform(df)

  df_norm = pd.DataFrame(norm_array, columns = df.columns)

  return df_norm


'''
GENERACIÓN DE MODELOS
'''

def plot_save(hist, ruta, nombre_modelo):

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Y_FmI_Dv_Opt]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.legend()
  plt.savefig('{0}/Output/Estudio_Modelos/{1}_mae.png'.format(ruta, nombre_modelo))
  plt.close()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [Y_FmI_Dv_Opt^2]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.legend()
  plt.savefig('{0}/Output/Estudio_Modelos/{1}_mse.png'.format(ruta, nombre_modelo))
  plt.close()


def crear_compilar_entrenar_evaluar_modelo(x_train, y_train, x_test, y_test, topologia, opt, lr, epocas, lote, ruta, nombre_modelo):
  # Construimos el modelo con la topologia especificada
  modelo = tf.keras.Sequential()
  capa_entrada = Dense(topologia[0], activation='relu', input_shape=[len(x_train.keys())])
  modelo.add(capa_entrada)
  for i in topologia:
    if topologia.index(i)>0:
      capa_oculta = Dense(i, activation='relu')
      modelo.add(capa_oculta)
  capa_salida = Dense(1, activation='sigmoid')
  modelo.add(capa_salida)

  # Establecemos el optimizador especificado
  if opt == 'SGD':
    optimizador = tf.keras.optimizers.SGD(learning_rate = lr)
  elif opt == 'Adam':
    optimizador = tf.keras.optimizers.Adam(learning_rate = lr)

  # Compilamos el modelo
  modelo.compile(loss='mse', optimizer=optimizador, metrics=['mae', 'mse'])

  # Establecemos el Earlystop para evitar el overfitting
  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

  # Entrenamos el modelo
  historia = modelo.fit(x_train, y_train, epochs = epocas, batch_size=lote, validation_split = 0.2, verbose = 0, callbacks = [early_stop])
  df_historia = pd.DataFrame(historia.history)
  df_historia['epoch'] = historia.epoch

  # Ploteamos las métricas MAE y MSE
  plot_save(df_historia, ruta, nombre_modelo)

  # Evaluamos el modelo
  loss, mae, mse = modelo.evaluate(x_test, y_test, verbose=2)
  s_evaluation = pd.Series([nombre_modelo, loss, mae, mse], index = ['modelo','loss','mae','mse'])
  s_evaluation
  
  return modelo, s_evaluation 


'''
BÚSQUEDA EN REJILLA
'''

def busqueda_rejilla(ruta, x_train, y_train, x_test, y_test, conj_topologias, conj_optimizadores, conj_lr, conj_epocas, conj_lotes):
  df_his_evaluacion = pd.DataFrame([['',0.0,0.0,0.0]],columns=['modelo','loss','mae','mse'])

  n = 1
  for i in conj_topologias:
    for j in conj_optimizadores:
      for k in conj_lr:
        for l in conj_epocas:
          for m in conj_lotes:
            # Creamos el nombre del modelo
            str_nom = 'modelo_{0}_{1}ly_{2}_{3}lr_{4}ep_{5}bs'.format(n, len(conj_topologias), j, str(k).replace('.',''), str(l), str(m))
            
            # Creamos y evaluamos el modelo
            modelo, s_evaluacion = crear_compilar_entrenar_evaluar_modelo(x_train, y_train, x_test, y_test, i, j, k, l, m, ruta, str_nom)
            
            # Salvamos el modelo y su evaluacion
            modelo.save('{0}/Output/Estudio_Modelos/{1}.h5'.format(ruta, str_nom))
            df_his_evaluacion = df_his_evaluacion.append(s_evaluacion,ignore_index=True)
            
            n = n + 1

  df_his_evaluacion = df_his_evaluacion.loc[df_his_evaluacion['modelo'] != '']
  df_his_evaluacion.reset_index(drop = True, inplace = True)
  df_his_evaluacion.to_csv(path_or_buf = '{0}/Output/Estudio_Modelos/evaluacion_modelos.csv'.format(ruta), sep = ';', float_format = '%.4f', index = False, encoding = 'utf-8')
  
  return df_his_evaluacion

def run(ruta, conj_vol, distancia, conj_top, conj_opt, conj_lr, conj_epc, conj_bs, analisis = False):
  # Importamos los ficheros
  minFutVol, ibxFutVol, benchIbex, benchEstx, benchSP, benchVix, vivex, skew = load_ficheros(ruta)

  # Comprobamos para cada fichero si hay valores nulos y duplicados
  minFutVol = check_basicos(minFutVol)
  ibxFutVol = check_basicos(ibxFutVol)
  benchIbex = check_basicos(benchIbex)
  benchEstx = check_basicos(benchEstx)
  benchSP = check_basicos(benchSP)
  benchVix = check_basicos(benchVix)
  vivex = check_basicos(vivex)
  skew = check_basicos(skew)

  # Creamos un único fichero y comprobamos si existen valores nulos y duplicados
  df_Features = merge_dataFrames(minFutVol, ibxFutVol, benchIbex, benchEstx, benchSP, benchVix, vivex, skew)
  df_Features = check_df_features(df_Features)

  # Calculamos las volatilidades para cada índice
  df_Features = get_volatilidades(df_Features, 'FmI', conj_vol)
  df_Features = get_volatilidades(df_Features, 'IBEX', conj_vol)
  df_Features = get_volatilidades(df_Features, 'ESTX', conj_vol)
  df_Features = get_volatilidades(df_Features, 'SP', conj_vol)

  # Ordenamos las columnas del DF
  '''
  cols = ['Fecha',
          'FmI_Cierre', 'FmI_VolSMA30', 'FmI_VolSMA60','FmI_VolSMA90', 'FmI_VolSMA120',
          'FmI_VolatApertura', 'FmI_VolatCierre',
          'FIp_Cierre', 'FIp_Volat10', 'FIp_Volat30', 'FIp_Volat60', 'FIp_Volat90', 'FIp_Volat120',
          'IBEX_Cierre', 'IBEX_VolSMA30', 'IBEX_VolSMA60', 'IBEX_VolSMA90', 'IBEX_VolSMA120',
          'ESTX_Cierre', 'ESTX_VolSMA30', 'ESTX_VolSMA60', 'ESTX_VolSMA90', 'ESTX_VolSMA120',
          'SP_Cierre', 'SP_VolSMA30', 'SP_VolSMA60', 'SP_VolSMA90', 'SP_VolSMA120',
          'vivex_Cierre',
          'skew_Cierre',
          'vix_Cierre']
  df_Features = df_Features[cols]
  '''
  # Generamos la columna Y del dataset
  df_Dataset = calc_desv_optima(df_Features)

  # Ajustamos la ventana temporal del dataset
  dataset_OK, df_Dataset = ajuste_ventana_temporal(df_Dataset, max(conj_vol), distancia, ruta)

  # Si el dataset es correcto, continua la ejecución
  if dataset_OK == True:
    
    if analisis == True:
      # Generamos estadísticas sobre los datos de entrenamiento
      df_datasetStats = analisis_caracteristicas(df_Dataset, ruta, conj_vol)

    # Creamos las estructuras normalizadas necesarias para el entrenaiento de modelos
    X_train, X_test, Y_train, Y_test = create_train_test(df_Dataset)
    X_train = min_max_scaler_datos(X_train)
    X_test = min_max_scaler_datos(X_test)

    # Ejecutamos el estudio de modelos
    df_Evaluation = busqueda_rejilla(ruta, X_train, Y_train, X_test, Y_test, 
                                      conj_top, conj_opt, 
                                      conj_lr, conj_epc, 
                                      conj_bs)
  else:
    print("El dataset no es adecuado: NaN, duplicados...")
  
  return df_Features, df_Dataset, df_Evaluation

'''
HIPERPARAMETROS
'''

'''
ruta = '/content/drive/My Drive/Colab Notebooks/TFM/'
conjunto_volatilidades = (10, 30, 60, 90, 120)
distancia = 22
analisis = False
conjunto_topologias = ((32,8), (64, 32, 16, 4))
conjunot_optimizadores = ('SGD', 'Adam')
conjunto_learning_rates = (0.1, 0.001)
conjunto_epocas = (500, 1000)
conjunto_batch_size = (64, 256)

df_Caracteristicas, df_Dataset, df_Evaluaciones = run(ruta, conjunto_volatilidades, distancia, conjunto_topologias, conjunot_optimizadores, 
                                                                  conjunto_learning_rates, conjunto_epocas, conjunto_batch_size, analisis)

'''
