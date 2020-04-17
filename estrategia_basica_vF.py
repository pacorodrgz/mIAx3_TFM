import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
from dateutil import relativedelta

'''
FUNCIONES DE IDENTIFICACIÓN DE FECHAS DE VENCIMIENTO
'''

def is_tercer_viernes(dt):
  return dt.weekday() == 4 and 15 <= dt.day <= 21

def set_codigo_vencimiento(dt, codigoMes):
  # Creamos el código de vencimiento de la fecha pasada por parámetro
  cod = '-'
  if is_tercer_viernes(dt):
    dty = str(dt.year)[2:4]
    dtm = codigoMes[dt.month]
    cod = dtm + dty

  return cod

def get_codigo_vencimiento(dt, codigoMes, mon = 0):
  # Creamos el código de vencimiento en función de los meses que queden para ello
  dt2 = dt + relativedelta.relativedelta(months = mon)
  dt2y = str(dt2.year)[2:4]
  dt2m = codigoMes[dt2.month]
  cod = dt2m + dt2y
  return cod


'''
FUNCIONES DE CONSTRUCCIÓN DEL DF DE FUTUROS
'''

def import_his_futuro(ruta, dateFormat):
  parser = lambda f: pd.datetime.strptime(f, dateFormat)
  df = pd.read_csv("{0}/Input/min_ibx_fut.csv".format(ruta), sep = ";", parse_dates = ['Fecha'], date_parser = parser, decimal = ",")
  return df

def filter_his_futuro(df_completo, iniDate, endDate):
  # Filtramos por el rango de fechas
  df_completo = df_completo.loc[(df_completo['Fecha'] >= iniDate) & (df_completo['Fecha'] <= endDate)]
  # Nos quedamos con el contrato próximo a vencer
  df_completo = df_completo.sort_values('Contrato', ascending=True).groupby('Fecha').head(1)
  df_completo = df_completo.sort_values('Fecha', ascending=True)
  # Nos quedamos con ls columnas de Fecha y Precio de cierre
  df_reducido = df_completo[['Fecha', 'Cierre']].copy()
  df_reducido.reset_index(drop = True, inplace = True)
  return df_reducido

def set_venc_his_futuro(df, codigoMes):
  df['F_Venc'] = df.Fecha.apply(lambda d: (is_tercer_viernes(d)))
  df['CodVenc'] = df.Fecha.apply(lambda d: (set_codigo_vencimiento(d, codigoMes)))
  return df

def make_his_futuro(ruta, iniDate, endDate, dateFormat, codigoMes):
  df = import_his_futuro(ruta, dateFormat)
  df = filter_his_futuro(df, iniDate, endDate)
  df = set_venc_his_futuro(df, codigoMes)
  return df

# FUNCIONES PARA REALIZAR COMPROBACIONES
def get_info_futuro(d = 1, m = 1, y = 2000):
  jornada = dt.datetime(year=y,month=m,day=d)
  registro = FutData.loc[(FutData['Fecha'] == jornada)]
  return registro

'''
FUNCIONES DE CONSTRUCCIÓN DEL DF DE OPCIONES
'''

def import_his_opciones(ruta, dateformat):
  parser = lambda f: pd.datetime.strptime(f[:10], dateformat)

  df1 = pd.read_csv("{0}/Input/min_ibx_opc_1.csv".format(ruta), sep = ";", parse_dates = ['Fecha'], date_parser = parser)
  df2 = pd.read_csv("{0}/Input/min_ibx_opc_2.csv".format(ruta), sep = ";", parse_dates = ['Fecha'], date_parser = parser)
  df3 = pd.read_csv("{0}/Input/min_ibx_opc_3.csv".format(ruta), sep = ";", parse_dates = ['Fecha'], date_parser = parser)

  df = df1.append(df2)
  df = df.append(df3)

  df = df.sort_values('Fecha', ascending=True)
  return df

def filter_his_opciones(df, iniDate, endDate):
  df = df.loc[(df['Fecha'] >= iniDate) & (df['Fecha'] <= endDate)]
  return df

def make_his_opciones(ruta, iniDate, endDate, dateFormat):
  df = import_his_opciones(ruta, dateFormat)
  df = filter_his_opciones(df, iniDate, endDate)
  return df

# FUNCIONES PARA REALIZAR COMPROBACIONES
def get_info_opciones(cont, d = 1, m = 1, y = 2000):
  jornada = dt.datetime(year=y,month=m,day=d)
  registro = OpcData.loc[(OpcData['Fecha'] == jornada) & (OpcData['Contrato'] == cont)]
  return registro

'''
FUNCIONES DE CONSTRUCCIÓN DEL DF DE POSICIONES
'''
def make_his_posiciones():
  df = pd.DataFrame([[0,'','',0.0,0,False,0.0,False,0.0]],columns=['Strike_Cont','Cod_Venc','Cod_Cont','Prima_Cont','Num_Oper','Pos_Abierta','Cotz_Futr','Ejerc_Opc','Import_Ejerc'])
  return df


# FUNCIONES PARA REALIZAR COMPROBACIONES
def get_info_posiciones(cont):
  registro = PosData.loc[PosData['Cod_Cont'] == cont]
  return registro

'''
FUNCIONES DE CONSTRUCCIÓN DEL DF DE TESORERIA
'''
def make_his_tesoreria(df_futuros, capitalInicial):
  s = pd.Series(np.zeros(df_futuros['Fecha'].size))
  s[0] = capitalInicial
  df = pd.DataFrame({'Fecha': df_futuros['Fecha'], 'Capital_Disp': s})
  return df


# FUNCIONES PARA REALIZAR COMPROBACIONES
def get_info_tesoreria(d = 1, m = 1, y = 2000):
  jornada = dt.datetime(year=y,month=m,day=d)
  registro = TesData.loc[(TesData['Fecha'] == jornada)]
  return registro

'''
FUNCIONES DE GESTIÓN DE LA TESORERÍA
'''

def update_tesoreria(df, i):
  if i > 0:
    df.loc[i, 'Capital_Disp'] = df.loc[i-1, 'Capital_Disp']
  return df

def set_liquidacion(df, i, importe, apertura = True):
  if apertura == True:
    cap_previo = df.loc[i, 'Capital_Disp']
    df.loc[i, 'Capital_Disp'] = cap_previo + importe
  else:
    cap_previo = df.loc[i - 1, 'Capital_Disp']
    df.loc[i, 'Capital_Disp'] = cap_previo + importe
  return df


'''
FUNCIONES RELATIVAS AL CIERRE DE POSICIONES
'''

def set_cotizacion_cierre(df, j, ctzFutr):
  df.at[j, 'Cotz_Futr'] = ctzFutr
  return df

def calc_ejercicio_opcion(df_posiciones, j, df_tesoreria, i):
  diferencia = df_posiciones.at[j, 'Cotz_Futr'] - df_posiciones.at[j, 'Strike_Cont']
  if diferencia < 0:
    perdida = diferencia * df_posiciones.at[j, 'Num_Oper']
    df_posiciones.at[j, 'Ejerc_Opc'] = True
    df_posiciones.at[j, 'Import_Ejerc'] = perdida
    df_tesoreria = set_liquidacion(df_tesoreria, i, perdida, False)
  else:
    df_tesoreria = update_tesoreria(df_tesoreria, i)
  return df_posiciones, df_tesoreria

def set_posicion_cerrada(df, j):
  df.at[j, 'Pos_Abierta'] = False
  return df

def cierre_posicion(df_posiciones, codVenc, ctzFutr, df_tesoreria, i):
  j = df_posiciones.index[df_posiciones['Cod_Venc'] == codVenc]
  if j.size == 1:
    j = j[0]
    if df_posiciones.at[j, 'Pos_Abierta'] == True:
      df_posiciones = set_cotizacion_cierre(df_posiciones, j, ctzFutr)
      df_posiciones, df_tesoreria = calc_ejercicio_opcion(df_posiciones, j, df_tesoreria, i)
      df_posiciones = set_posicion_cerrada(df_posiciones, j)
    else:
      df_tesoreria = update_tesoreria(df_tesoreria, i)
  else:
    df_tesoreria = update_tesoreria(df_tesoreria, i)
  return df_posiciones, df_tesoreria


'''
FUNCIONES RELATIVAS A LA APERTURA DE POSICIONES
'''

def calc_strike(ctzPrvFutr, desvStrike):
  stk = int(np.floor((ctzPrvFutr * desvStrike)/100) * 100)
  return stk

def set_codigos_opcion(strike, fecVenc, codigoMes):
  codVenc = get_codigo_vencimiento(fecVenc, codigoMes, 1)
  codStrike = str(strike).rjust(5)
  codCont = 'PIBX' + codStrike + codVenc
  return codVenc, codCont

def get_prima_contrato(df_opciones, fecVenc, codCont):
  prima = float(df_opciones.Cierre.loc[(df_opciones['Fecha'] == fecVenc) & (df_opciones['Contrato'] == codCont)])
  return prima

def calc_num_operaciones(df_tesoreria, stk, i):
  capitalDisp = df_tesoreria.Capital_Disp[i]
  numOper = int(np.floor(capitalDisp / stk))
  return numOper

def apertura_posicion(df_opciones, df_posiciones, df_tesoreria, codigoMes, fecVenc, ctzPrvFutr, desvStrike, comisionBroker, i):

  # Cálculo de parámetros de la posición
  strike = calc_strike(ctzPrvFutr, desvStrike)
  codVenc, codCont = set_codigos_opcion(strike, fecVenc, codigoMes)
  primaCont = get_prima_contrato(df_opciones, fecVenc, codCont)
  numOper = calc_num_operaciones(df_tesoreria, strike, i)

  # Actualización del histórico de Tesorería
  primas = primaCont * numOper
  comisiones = comisionBroker * numOper
  importe = primas - comisiones
  df_tesoreria = set_liquidacion(df_tesoreria, i, importe, True)

  # Actualización del histórico de Posiciones
  s_posicion = pd.Series([strike,codVenc,codCont,primaCont,numOper,True,0.0,False,0.0],index=['Strike_Cont','Cod_Venc','Cod_Cont','Prima_Cont','Num_Oper','Pos_Abierta','Cotz_Futr','Ejerc_Opc','Import_Ejerc'])
  df_posiciones = df_posiciones.append(s_posicion,ignore_index=True)
  return df_posiciones, df_tesoreria


'''
FUNCIÓN PARA GENERAR EL DF DE RESULTADOS
'''

def make_resultados(df_futuro, df_tesoreria, ruta, iniDate, endDate, modelo, desvStrike, estilo = 'seaborn'):
  # Preparamos el DF de tesorería para que sólo contenga el capital disponible los días de vencimiento
  df_solo_dia_vencimiento = df_tesoreria.groupby('Capital_Disp').head(1)
  df_capital_vencimiento = pd.DataFrame({'Fecha': df_futuro['Fecha'], 'Capital_Disp': df_solo_dia_vencimiento.Capital_Disp})
  mascara_plot = np.isfinite(df_capital_vencimiento.Capital_Disp)

  periodo = str(iniDate)[:10].replace('-', '') + '-' + str(endDate)[:10].replace('-', '')
  desviacion = str(round((1.0 - desvStrike) * 100, 2)) + '%'

  plt.style.use(estilo)
  
  # Ploteamos la evolución del precio del futuro sobre Mini-Ibex
  f1 = plt.figure(figsize=(14,6))
  ax1 = f1.add_subplot()
  ax1.title.set_text('Evolución diaria de precios')
  ax1.plot_date(df_futuro.Fecha, df_futuro.Cierre, linestyle = 'solid', color = 'b', marker = '_', label = 'Ptos. del futuro Mini-Ibex')
  ax1.legend(loc = "upper right")
  ax1.set_xlim([iniDate, endDate])
  #ax1.set_ylim([0, 16000])
  plt.savefig('{0}/Output/Estrategia_Basica/Futuro_MiniIbex{1}.png'.format(ruta, periodo))
  plt.close()

  # Ploteamos la evolución del capital disponible
  f2 = plt.figure(figsize=(14,6))
  ax2 = f2.add_subplot()
  ax2.title.set_text('Evolución del capital')
  ax2.plot_date(df_tesoreria.Fecha[mascara_plot], df_capital_vencimiento.Capital_Disp[mascara_plot], linestyle = 'solid', color = 'r', marker = '*', label = 'Capital en miles de €')
  ax2.legend(loc = "upper right")
  ax2.set_xlim([iniDate, endDate])
  #ax2.set_ylim([50000, 150000])
  plt.savefig('{0}/Output/Estrategia_Basica/Resultado_{1}_{2}_{3}.png'.format(ruta, modelo, periodo, desviacion))
  plt.close()

  #Devolvemos el DF con las liquidaciones
  df = df_solo_dia_vencimiento.reset_index(drop = True)

  return df

def export_CSV(df, ruta, nombre, iniDate, endDate, desvStrike):
  periodo = str(iniDate)[:10].replace('-', '') + '-' + str(endDate)[:10].replace('-', '')
  desviacion = str(round((1.0 - desvStrike) * 100, 2)) + '%'
  
  df.to_csv(path_or_buf = '{0}/Output/Estrategia_Basica/{1}_{2}_{3}.csv'.format(ruta, nombre, periodo, desviacion), sep = ';', float_format = '%.4f', index = False, encoding = 'utf-8')

'''
FUNCIÓN PARA EJECUTAR LA ESTRATEGIA
'''

def run(ruta = '/content/drive/My Drive/Colab Notebooks/TFM',
                       iniDate = dt.datetime(year=2010,month=1,day=4),
                       endDate = dt.datetime(year=2010,month=12,day=30),
                       futuroDateFormat = '%d/%m/%Y',
                       opcionDateFormat = '%Y-%m-%d',
                       capitalInicial = 100000.0,
                       desvStrike = 0.98,
                       comisionBroker = 2.50,
                       modelo = 'basico',
                       codigoMes = pd.Series(['F','G','H','J','K','M','N','Q','U','V','X','Z'], index = [1,2,3,4,5,6,7,8,9,10,11,12])):
  
  # Creamos los DF que vamos a utilizar en la ejecución
  FutData = make_his_futuro(ruta, iniDate, endDate, futuroDateFormat, codigoMes)
  OpcData = make_his_opciones(ruta, iniDate, endDate, opcionDateFormat)
  PosData = make_his_posiciones()
  TesData = make_his_tesoreria(FutData, capitalInicial)

  # Iteramos sobre los precios del futuro Mini-Ibex para operar
  for i in FutData.index:
    if FutData.F_Venc[i] == True:
      fecVenc = FutData.Fecha[i]
      codVenc = FutData.CodVenc[i]
      ctzFutr = FutData.Cierre[i]
      ctzPrvFutr = FutData.Cierre[i-1]

      PosData, TesData = cierre_posicion(PosData, codVenc, ctzFutr, TesData, i)
      PosData, TesData = apertura_posicion(OpcData, PosData, TesData, codigoMes, fecVenc, ctzPrvFutr, desvStrike, comisionBroker, i)
    else:
      TesData = update_tesoreria(TesData, i)

  LiqData = make_resultados(FutData, TesData, ruta, iniDate, endDate, modelo, desvStrike)

  # Exportamos los resultados a CSV
  export_CSV(FutData, ruta, '_01_fut_mI', iniDate, endDate, desvStrike)
  export_CSV(OpcData, ruta, '_02_Opc_mI', iniDate, endDate, desvStrike)
  export_CSV(PosData, ruta, '_03_Posiciones', iniDate, endDate, desvStrike)
  export_CSV(TesData, ruta, '_04_Tesoreria', iniDate, endDate, desvStrike)
  export_CSV(LiqData, ruta, '_05_Liquidaciones', iniDate, endDate, desvStrike)

  return FutData, OpcData, PosData, TesData, LiqData

'''
HIPERPARÁMETROS
'''

'''
_ruta = '/content/drive/My Drive/Colab Notebooks/TFM'

_codigo_de_mes = pd.Series(['F','G','H','J','K','M','N','Q','U','V','X','Z'], index = [1,2,3,4,5,6,7,8,9,10,11,12])

_fecha_inicial = dt.datetime(year=2010,month=1,day=4)
_fecha_final = dt.datetime(year=2019,month=12,day=31)

_capital_inicial = 100000.0
_desv_strike = 0.98
_comision = 2.5

Futuro, Opciones, Posiciones, Tesoreria, LiqData = run(endDate = _fecha_final)

'''