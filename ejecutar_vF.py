import pandas as pd
import datetime as dt

import estrategia_basica_vF as estrategia_basica
import estrategia_DL_vF as estrategia_DL
import estudio_modelos_vF as estudio_modelos


'''
HIPERPARÁMETROS
'''
_ruta = '/content/drive/My Drive/Colab Notebooks/TFM'
#_ruta = 'https://bucketmiax.s3.us-east-2.amazonaws.com'

_fecha_inicial = dt.datetime(year=2010,month=6,day=23)
_fecha_final = dt.datetime(year=2019,month=2,day=22)

_nombre_modelo1 = 'modelo_212_3ly_Adam_0001lr_500ep_256bs'
_nombre_modelo2 = 'modelo_102_3ly_SGD_0001lr_100ep_512bs'
_nombre_modelo3 = 'modelo_107_3ly_SGD_0001lr_1000ep_256bs'

_capital_inicial = 100000.0
_desv_strike = 0.98
_comision = 2.5

_conjunto_volatilidades = (10, 30, 60, 90, 120)
_distancia = 22
_analisis = False

_conjunto_topologias = ((32,8), (20, 10, 5), (64, 32, 16, 4))
_conjunot_optimizadores = ('SGD', 'Adam')
_conjunto_learning_rates = (1.0, 0.1, 0.01, 0.001)
_conjunto_epocas = (100, 500, 1000)
_conjunto_batch_size = (64, 256, 512)


# Ejecutamos la estrategia básica
Bas_Futuro, Bas_Opciones, Bas_Posiciones, Bas_Tesoreria, Bas_Liquidaciones = estrategia_basica.run(ruta = _ruta, 
                                                                                                   iniDate = _fecha_inicial, 
                                                                                                   endDate = _fecha_final, 
                                                                                                   capitalInicial = _capital_inicial, 
                                                                                                   desvStrike = _desv_strike, 
                                                                                                   comisionBroker = _comision)

# Ejecutamos el entrenamiento de los modelos
Est_Caracteristicas, Est_Dataset, Est_Evaluaciones = estudio_modelos.run(ruta = _ruta, 
                                                                         conj_vol = _conjunto_volatilidades, 
                                                                         distancia = _distancia, 
                                                                         conj_top = _conjunto_topologias, 
                                                                         conj_opt = _conjunot_optimizadores, 
                                                                         conj_lr = _conjunto_learning_rates, 
                                                                         conj_epc = _conjunto_epocas, 
                                                                         conj_bs = _conjunto_batch_size, 
                                                                         analisis = True)

# Ejecutamos la estrategia tomando el modelo 1
DL1_Futuro, DL1_Opciones, DL1_Posiciones, DL1_Tesoreria, DL1_Liquidaciones = estrategia_DL.run(ruta = _ruta, 
                                                                                               modelo = _nombre_modelo1, 
                                                                                               iniDate = _fecha_inicial, 
                                                                                               endDate = _fecha_final,
                                                                                               capitalInicial = _capital_inicial,
                                                                                               comisionBroker = _comision)

# Ejecutamos la estrategia tomando el modelo 2
DL2_Futuro, DL2_Opciones, DL2_Posiciones, DL2_Tesoreria, DL2_Liquidaciones = estrategia_DL.run(ruta = _ruta, 
                                                                                               modelo = _nombre_modelo2, 
                                                                                               iniDate = _fecha_inicial, 
                                                                                               endDate = _fecha_final,
                                                                                               capitalInicial = _capital_inicial,
                                                                                               comisionBroker = _comision)

# Ejecutamos la estrategia tomando el modelo 3
DL3_Futuro, DL3_Opciones, DL3_Posiciones, DL3_Tesoreria, DL3_Liquidaciones = estrategia_DL.run(ruta = _ruta, 
                                                                                               modelo = _nombre_modelo3, 
                                                                                               iniDate = _fecha_inicial, 
                                                                                               endDate = _fecha_final,
                                                                                               capitalInicial = _capital_inicial,
                                                                                               comisionBroker = _comision)


def plot_backtesting(df, iniDate, endDate, modelo1, modelo2, modelo3):
  periodo = str(iniDate)[:10].replace('-', '') + '-' + str(endDate)[:10].replace('-', '')

  f = plt.figure(figsize=(14,6))
  ax = f.add_subplot()
  ax.title.set_text('Backtesting')
  ax.plot_date(df.Fecha, df['basica'], linestyle = 'solid', color = 'k', marker = '', label = 'Estrategia básica (-2%)')
  ax.plot_date(df.Fecha, df[modelo1], linestyle = 'solid', color = 'g', marker = '', label = 'Modelo 212. Mejor MSE')
  ax.plot_date(df.Fecha, df[modelo2], linestyle = 'solid', color = 'r', marker = '', label = 'Modelo 102. Peor MSE')
  ax.plot_date(df.Fecha, df[modelo3], linestyle = 'solid', color = 'b', marker = '', label = 'Modelo 107.')
  ax.legend(loc = "upper left")
  ax.set_xlim([iniDate, endDate])
  plt.savefig('{0}/Output/Estrategia_DL/Backtesting_{1}.png'.format(_ruta, periodo))
  plt.close()

# Ploteamos la evolución del capital de todas las ejecuciones
df = pd.merge(Bas_Liquidaciones, DL1_Liquidaciones, how='outer', on=['Fecha'])
df = pd.merge(df, DL2_Liquidaciones, how='outer', on=['Fecha'])
df = pd.merge(df, DL3_Liquidaciones, how='outer', on=['Fecha'])
df.columns = ['Fecha','basica', _nombre_modelo1, _nombre_modelo2, _nombre_modelo3]

plot_backtesting(df, _fecha_inicial, _fecha_final, _nombre_modelo1, _nombre_modelo2, _nombre_modelo3)
