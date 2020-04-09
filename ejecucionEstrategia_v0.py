import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
from dateutil import relativedelta

'''
Importamos los datos de los futuros del Mini-Ibex y de las opciones entre 2010 y 2012, y acotamos ambos conjuntos de 
datos al rango de fechas menor [4/1/2010 : 30/12/2011].
'''


dateIndex = False
iniDate = dt.datetime(year=2010,month=1,day=4)
endDate = dt.datetime(year=2011,month=12,day=30)

parser1 = lambda f: pd.datetime.strptime(f, '%d/%m/%Y')
parser2 = lambda f: pd.datetime.strptime(f[:10], '%Y-%m-%d')


if dateIndex == True:
  print("Índices de fechas.")
  # Importamos los datos de futuros y acotamos las fechas
  MiniIbex_Vol = pd.read_csv("/content/drive/My Drive/Colab Notebooks/TFM/ibx_volatilidad.csv", sep = ";", parse_dates = ['Fecha'], index_col = ['Fecha'], date_parser = parser1)
  MiniIbex_Vol = MiniIbex_Vol.loc[iniDate:endDate]
  FutData = MiniIbex_Vol[['Fecha', 'FutMiniIbex']].copy()

  # Importamos los datos de opciones
  OpcData = pd.read_csv("/content/drive/My Drive/Colab Notebooks/TFM/opc 10-12.csv", sep = ";", parse_dates = ['Fecha'], index_col = ['Fecha','Contrato'], date_parser = parser2)
else:
  print("Índices numéricos.")
  # Importamos los datos de futuros y acotamos las fechas
  MiniIbex_Vol = pd.read_csv("/content/drive/My Drive/Colab Notebooks/TFM/ibx_volatilidad.csv", sep = ";", parse_dates = ['Fecha'], date_parser = parser1)
  MiniIbex_Vol = MiniIbex_Vol.loc[(MiniIbex_Vol['Fecha'] >= iniDate) & (MiniIbex_Vol['Fecha'] <= endDate)]
  FutData = MiniIbex_Vol[['Fecha', 'FutMiniIbex']].copy()

  # Importamos los datos de opciones
  OpcData = pd.read_csv("/content/drive/My Drive/Colab Notebooks/TFM/opc 10-12.csv", sep = ";", parse_dates = ['Fecha'], date_parser = parser2)


FutData.dtypes
#FutData.info()
#FutData.index
#FutData

#OpcData.dtypes
#OpcData.info()
#OpcData.index
#OpcData


'''
Creamos la tabla con los códigos de mes empleados en la nomenclatura de los contratos de Opciones:
'''
monthTranslate = pd.Series(['F','G','H','J','K','M','N','Q','U','V','X','Z'], index = [1,2,3,4,5,6,7,8,9,10,11,12])
monthTranslate


'''
Creamos la tabla con los datos históricos de la cotización del futuro de Mini-Ibex, identicamos las fechas de 
vencimiento (tercer viernes de mes) y añadimos lo códigos de vencimiento correspondientes a cada fecha:
'''

def is_tercer_viernes(dt):
  return dt.weekday() == 4 and 15 <= dt.day <= 21

def get_codigo_vencimiento(dt, mon = 0):
  # Se puede calcular siguiente cósigo de vencimiento.
  dt2 = dt + relativedelta.relativedelta(months = mon)
  dt2y = str(dt2.year)[2:4]
  dt2m = monthTranslate[dt2.month]
  return dt2m + dt2y

FutData['F_Venc'] = FutData.Fecha.apply(lambda d: (is_tercer_viernes(d)))
FutData['CodVenc'] = FutData.Fecha.apply(lambda d: (get_codigo_vencimiento(d)))
FutData.head(15)

'''
Creamos la tabla con los datos de las operaciones con el siguiente formato:
'''

operData = pd.DataFrame([[0,'','',0.0,0,False,0.0,False,0.0]],columns=['Strike_Cont','Cod_Venc','Cod_Cont','Prima_Cont','Num_Oper','Pos_Abierta','Cotz_Futr','Ejerc_Opc','Import_Ejerc'])
operData

'''
Creamos la tabla donde se registran los flujos de caja.

*   Asumimos que el Capital Inicial es de 100.000€
*   Tomamos el mismo espacio temporal del Histórico del Futuro
'''

capital = pd.Series(np.zeros(FutData['Fecha'].size))
capital[0] = 100000.0
tesoreriaData = pd.DataFrame({'Fecha': FutData['Fecha'], 'Capital_Disp': capital})
tesoreriaData



'''
Descripción del algoritmo a alto nivel:

1.   Comprobar si es día de vencimiento (tercer viernes de cada mes).

2.   Acciones de Cierre de posiciones:
    *   Recuperar la posición abierta.
    *   Calcular el resultado de la operación.
    *   Liquidar el pago, en caso de haberlo.
    *   Registrar la posición como cerrada.

3.   Acciones de Apertura de posiciones:
    *   Calcular las operaciones que se pueden llevar a cabo.
    *   Construir código de contrato y recuperar el valor de la prima.
    *   Liquidar el cobro de las primas.
    *   Registrar la posición como abierta.

Asunciones y detalles a tener en cuenta:
    *   Asumimos que la liquidación de pagos y cobros se realizan de manera instantánea, no el siguiente día hábil.
    *   Por el momento, asumimos que el mercado MEFF no requiere de ninguna garantía por nuestra parte.
    *   Asumimos que únicamente se podrá mantener una posición abierta, compuesta de N contratos.
    *   Actualmente, las cotizaciones empleadas pertenecen a futuros sobre IBEX, no MINI-IBEX. Para la programación, nos sirve...

Comisiones de Opciones sobre Futuros de Mini-Ibex:
[Bankinter](https://www.bankinter.com/broker/tarifas-comisiones/derivados-futuros): 2,50€/Contrato
[DeGiro](https://www.degiro.es/data/pdf/es/Relacion_de_tarifas.pdf): 0,50€/Contrato
'''

'''
HIPERPARÁMETROS
'''
desv_strike = 0.98
comision_Bankinter = 2.5
comision_DeGiro = 0.5

'''
FUNCIONES DE GESTIÓN DE LA TESORERÍA
'''

def update_tesoreria(df, i):
  if i > 0:
    df.loc[i, 'Capital_Disp'] = df.loc[i-1, 'Capital_Disp']
  return df

def set_liquidacion(df, i, importe):
  df.loc[i, 'Capital_Disp'] = df.loc[i-1, 'Capital_Disp'] + importe
  return df

'''
FUNCIONES RELATIVAS AL CIERRE DE POSICIONES
'''

def set_cotizacion_cierre(df, j, ctzFutr):
  df.at[j, 'Cotz_Futr'] = ctzFutr
  print("Cotización del futuro: {0} ptos.".format(ctzFutr, ))
  return df

def calc_ejercicio_opcion(df_posiciones, j, df_tesoreria, i):
  diferencia = df_posiciones.at[j, 'Cotz_Futr'] - df_posiciones.at[j, 'Strike_Cont']
  if diferencia < 0:
    perdida = diferencia * df_posiciones.at[j, 'Num_Oper']
    print("El comprador EJERCE la opción: {0}€".format(perdida))
    df_posiciones.at[j, 'Ejerc_Opc'] = True
    df_posiciones.at[j, 'Import_Ejerc'] = perdida
    df_tesoreria = set_liquidacion(df_tesoreria, i, perdida)
  else:
    print("El comprador NO EJERCE la opción")
  return df_posiciones, df_tesoreria

def set_posicion_cerrada(df, j):
  df.at[j, 'Pos_Abierta'] = False
  print("Posición CERRADA")
  print("------------------------------------")
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
      print("Posición con vencimiento {0} ya CERRADA".format(codVenc))
      df_tesoreria = update_tesoreria(df_tesoreria, i)
  else:
    print("No hay posición con vencimiento {0}".format(codVenc))
    df_tesoreria = update_tesoreria(df_tesoreria, i)
  return df_posiciones, df_tesoreria


'''
FUNCIONES RELATIVAS A LA APERTURA DE POSICIONES
'''


def calc_strike(ctzPrvFutr):
    stk = int(np.floor((ctzPrvFutr * desv_strike) / 100) * 100)
    return stk


def set_codigos_opcion(strike, fecVenc):
    codVenc = get_codigo_vencimiento(fecVenc, 1)
    codStrike = str(strike).rjust(5)
    codCont = 'PIBX' + codStrike + codVenc
    return codVenc, codCont


def get_prima_contrato(OpcData, fecVenc, codCont):
    prima = float(OpcData.Cierre.loc[(OpcData['Fecha'] == fecVenc) & (OpcData['Contrato'] == codCont)])
    return prima


def calc_num_operaciones(df_tesoreria, strike):
    capitalDisp = df_tesoreria.Capital_Disp[i]
    numOper = int(np.floor(capitalDisp / strike))
    return numOper


def apertura_posicion(OpcData, df_posiciones, fecVenc, ctzPrvFutr, df_tesoreria, i):
    # Cálculo de parámetros de la posición
    strike = calc_strike(ctzPrvFutr)
    codVenc, codCont = set_codigos_opcion(strike, fecVenc)
    primaCont = get_prima_contrato(OpcData, fecVenc, codCont)
    numOper = calc_num_operaciones(tesoreriaData, strike)

    # Actualización del histórico de Tesorería
    importe = primaCont * numOper
    df_tesoreria = set_liquidacion(df_tesoreria, i, importe)

    # Actualización del histórico de Posiciones
    print("------------------------------------")
    print("Posición ABIERTA [{0}]".format(i))
    print("Se abren {0} contratos {1}".format(numOper, codCont))
    print("Se cobra {0}€ por primas({1}€)".format(importe, primaCont))
    s_posicion = pd.Series([strike, codVenc, codCont, primaCont, numOper, True, 0.0, False, 0.0],
                           index=['Strike_Cont', 'Cod_Venc', 'Cod_Cont', 'Prima_Cont', 'Num_Oper', 'Pos_Abierta',
                                  'Cotz_Futr', 'Ejerc_Opc', 'Import_Ejerc'])
    df_posiciones = df_posiciones.append(s_posicion, ignore_index=True)
    return df_posiciones, df_tesoreria

'''
MAIN
'''

#for i in FutData.index:
for i in range(99):
  if FutData.F_Venc[i] == True:
    fecVenc = FutData.Fecha[i]
    codVenc = FutData.CodVenc[i]
    ctzFutr = FutData.FutMiniIbex[i]
    ctzPrvFutr = FutData.FutMiniIbex[i-1]

    operData, tesoreriaData = cierre_posicion(operData, codVenc, ctzFutr, tesoreriaData, i)
    operData, tesoreriaData = apertura_posicion(OpcData, operData, fecVenc, ctzPrvFutr, tesoreriaData, i)
  else:
    tesoreriaData = update_tesoreria(tesoreriaData, i)






