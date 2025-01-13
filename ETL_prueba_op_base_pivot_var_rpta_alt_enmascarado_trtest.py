# -*- coding: utf-8 -*-
"""
@author: Sebastian Echeverri
"""
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Configuraciones para la visualización de DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Ruta de la carpeta donde están los archivos
data_path = r'C:\Users\JULI\OneDrive\Escritorio\Pureba Analitica Bancolombia cobranzas\Data'

# Archivos disponibles
files = {
    'base_pivot': 'prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv',
    'probabilidad_oblig': 'prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv',
    'master_customer': 'prueba_op_master_customer_data_enmascarado_completa.csv',
    'cuotas_pagos_hist': 'prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv',
    'prueba_op': 'prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv'
}

# Cargar los archivos en DataFrames
df_base_pivot = pd.read_csv(os.path.join(data_path, files['base_pivot']))
df_probabilidad_oblig = pd.read_csv(os.path.join(data_path, files['probabilidad_oblig']))
df_master_customer = pd.read_csv(os.path.join(data_path, files['master_customer']))
df_cuotas_pagos_hist = pd.read_csv(os.path.join(data_path, files['cuotas_pagos_hist']))
df_prueba_op= pd.read_csv(os.path.join(data_path, files['prueba_op']))

# Verificar las primeras filas de cada DataFrame
print("Base pivot:")
print(df_base_pivot.head(), "\n")
print("Probabilidad de obligación:")
print(df_probabilidad_oblig.head(), "\n")
print("Master customer:")
print(df_master_customer.head(), "\n")
print("Cuotas y pagos históricos:")
print(df_cuotas_pagos_hist.head(), "\n")
print("Datos de validación:")
print(df_prueba_op.head(), "\n")

#Validacion de dataframes

def validar_dataframe(df, nombre_df, columnas_duplicados=None):
    """
    Realiza validaciones comunes los Dataframes.

    Args:
        df (pd.DataFrame): El DataFrame a validar.
        nombre_df (str): El nombre del DataFrame para mostrar en los resultados.
        columnas_duplicados (list, optional): Lista de columnas para verificar duplicados.
    """
    print(f"\n--- Validaciones para: {nombre_df} ---")
    
    # Información general del DataFrame
    print("\nInformación del DataFrame:")
    df.info()
    
    # Conteo de valores nulos
    null_counts = df.isnull().sum()
    print("\nConteo de valores nulos por columna:")
    print(null_counts)

    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())

    # Detección de duplicados en todo el DataFrame
    duplicados_total = df.duplicated()
    hay_duplicados = duplicados_total.any()
    num_duplicados = duplicados_total.sum()

    print(f"\n¿Existen duplicados en el DataFrame? {hay_duplicados}")
    print(f"Número total de filas duplicadas: {num_duplicados}")

    if hay_duplicados:
        print("\nFilas duplicadas:")
        print(df[duplicados_total])
    
    # Detección de duplicados en columnas específicas
    if columnas_duplicados:
        duplicados_columnas = df.duplicated(subset=columnas_duplicados).sum()
        print(f"\nNúmero de filas duplicadas en las columnas {columnas_duplicados}: {duplicados_columnas}")

# Validar df_base_pivot
validar_dataframe(
    df_base_pivot,
    "df_base_pivot",
    columnas_duplicados=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado']
)

# Validar df_probabilidad_oblig
validar_dataframe(
    df_probabilidad_oblig,
    "df_probabilidad_oblig",
    columnas_duplicados=['nit_enmascarado', 'num_oblig_enmascarado']
)

# Validar df_master_customer
validar_dataframe(
    df_master_customer,
    "df_master_customer",
    columnas_duplicados=['nit_enmascarado']
)

# Validar df_cuotas_pagos_hist
validar_dataframe(
    df_cuotas_pagos_hist,
    "df_cuotas_pagos_hist",
    columnas_duplicados=['nit_enmascarado','num_oblig_enmascarado']
)

# Validar df_prueba_op
validar_dataframe(
    df_prueba_op,
    "df_prueba_op",
    columnas_duplicados=['nit_enmascarado','num_oblig_enmascarado']
)

#Inspección de datos
def inspeccionar_datos(df, nombre_df, max_columns=None):
     """
     Calcula y muestra la frecuencia y el porcentaje de valores únicos por columna en un DataFrame.
 
     Args:
         df (pd.DataFrame): El DataFrame a inspeccionar.
         nombre_df (str): El nombre del DataFrame para mostrar en los resultados.
         max_columns (int, optional): Número máximo de columnas a procesar. Defaults to None (procesar todas).
     """
     print(f"\n--- Inspección de datos para: {nombre_df} ---")
     value_counts_dict = {}

     # Si se especifica un número máximo de columnas, solo procesar esas
     if max_columns:
         columns_to_process = df.columns[:max_columns]
     else:
         columns_to_process = df.columns
 
     for column in columns_to_process:
         counts = df[column].value_counts()
         percentages = df[column].value_counts(normalize=True) * 100
         value_counts_dict[column] = pd.DataFrame({
             'Cantidad': counts,
             'Porcentaje': percentages
         })
 
     for column, counts_percentages in value_counts_dict.items():
         print(f'\nCantidad y porcentaje de valores en columna: {column}')
         print(counts_percentages)
         print('-' * 50)

# Inspeccionar
inspeccionar_datos(df_base_pivot, "df_base_pivot", max_columns=30)
inspeccionar_datos(df_probabilidad_oblig, "df_probabilidad_oblig", max_columns=30)
inspeccionar_datos(df_master_customer, "df_master_customer", max_columns=30)
inspeccionar_datos(df_cuotas_pagos_hist, "df_cuotas_pagos_hist", max_columns=30)
inspeccionar_datos(df_prueba_op, "df_prueba_op", max_columns=30)

# Preprocesamiento de variables y creacion de indicadores

df_base_pivot['fecha'] = pd.to_datetime(df_base_pivot['fecha_var_rpta_alt'], format='%Y%m')

# Crear variables de tendencia (cambio porcentual respecto al mes anterior)
df_base_pivot['tendencia_dias_mora'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['dias_mora_fin'].pct_change()
df_base_pivot['tendencia_dias_mora'] = df_base_pivot['tendencia_dias_mora'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_vlr_obligacion'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['vlr_obligacion'].pct_change()
df_base_pivot['tendencia_vlr_obligacion'] = df_base_pivot['tendencia_vlr_vencido'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_vlr_vencido'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['vlr_vencido'].pct_change()
df_base_pivot['tendencia_vlr_vencido'] = df_base_pivot['tendencia_vlr_vencido'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_vlr_endeudamiento'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['endeudamiento'].pct_change()
df_base_pivot['tendencia_vlr_endeudamiento'] = df_base_pivot['tendencia_vlr_endeudamiento'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_valor_cuota_mes'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['valor_cuota_mes'].pct_change()
df_base_pivot['tendencia_valor_cuota_mes'] = df_base_pivot['tendencia_valor_cuota_mes'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_pago_cuota'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['pago_cuota'].pct_change()
df_base_pivot['tendencia_pago_cuota'] = df_base_pivot['tendencia_pago_cuota'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_porc_pago_cuota'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['porc_pago_cuota'].pct_change()
df_base_pivot['tendencia_porc_pago_cuota'] = df_base_pivot['tendencia_porc_pago_cuota'].replace([np.inf, -np.inf,np.nan], 0)
df_base_pivot['tendencia_pago_mes'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['pago_mes'].pct_change()
df_base_pivot['tendencia_pago_mes'] = df_base_pivot['tendencia_porc_pago_cuota'].replace([np.inf, -np.inf,np.nan], 0)

# Crear tasa de cumplimiento (pagos realizados / valor cuota)
df_base_pivot['tasa_cumplimiento'] = df_base_pivot['pago_mes'] / (df_base_pivot['valor_cuota_mes'] + 1e-6)  # Evitar división por cero

# Crear tasa de endeudamiento: (enduedamiento / valor obligacion)
df_base_pivot['tasa_endeudamiento'] = (df_base_pivot['endeudamiento'] / (df_base_pivot['vlr_obligacion'] + 1e-6))  # Evitar división por cero

# Crear Tasa de cumplimiento de pago total: (pago_mes/vlr_obligacion)
df_base_pivot['tasa_pago_total'] = (df_base_pivot['pago_mes'] / (df_base_pivot['vlr_obligacion'] + 1e-6))  # Evitar división por cero

# Crear Tasa de cumplimiento de cuota: (pago_cuota/valor_cuota_mes)
df_base_pivot['tasa_cumplimiento_cuota'] = (df_base_pivot['pago_cuota'] / (df_base_pivot['valor_cuota_mes'] + 1e-6))  # Evitar división por cero

# Crear Relación entre saldo de capital y valor vencido: (saldo_capital/vlr_vencido)
df_base_pivot['tasa_saldo_capital_vencido'] = (df_base_pivot['saldo_capital'] / (df_base_pivot['vlr_vencido'] + 1e-6))  # Evitar división por cero

#Crear Tasa de mora respecto al valor de la obligación:
df_base_pivot['tasa_de_mora'] = (df_base_pivot['vlr_vencido'] / (df_base_pivot['vlr_obligacion'] + 1e-6))  # Evitar división por cero

#Crear Dias promedio de mora:
df_base_pivot['promedio_mora'] = (df_base_pivot['min_mora']+df_base_pivot['max_mora'])/ 2  # Evitar división por cero    

#Crear Tasa de contacto efectivo:
df_base_pivot['contacto_efectivo'] = (df_base_pivot['rpc']/df_base_pivot['cant_gestiones']+ 1e-6)  # Evitar división por cero
    
#Crear Tasa de acuerdos exitosos:
df_base_pivot['tasa_acuerdos'] = (df_base_pivot['cant_acuerdo']/df_base_pivot['cant_gestiones']+ 1e-6)  # Evitar división por cero    

#Crear Tasa de acuerdos exitosos:
df_base_pivot['tasa_promesa_cumplida'] = (df_base_pivot['promesas_cumplidas']/df_base_pivot['cant_alter_posibles']+ 1e-6)  # Evitar división por cero    

# Resago variable anterior
df_base_pivot['lag_aceptacion'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['var_rpta_alt'].shift(1)
df_base_pivot['lag_aceptacion'] = df_base_pivot['lag_aceptacion'].replace([np.inf, -np.inf,np.nan], 0)
# Crear una media móvil de los últimos 3 meses de aceptación
df_base_pivot['lag_aceptacion_3m'] = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha']).groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado'])['var_rpta_alt'].rolling(3).mean().reset_index(level=[0, 1, 2], drop=True)
df_base_pivot['lag_aceptacion_3m'] = df_base_pivot['lag_aceptacion_3m'].replace([np.inf, -np.inf,np.nan], 0)

## Frecuencia de aceptación por cliente
# Frecuencia de aceptación por cliente
frecuencia_aceptacion = df_base_pivot[df_base_pivot['var_rpta_alt'] == 1].groupby('nit_enmascarado').size().reset_index(name='frecuencia_aceptacion')
# Total de contactos por cliente
total_contactos = df_base_pivot.groupby('nit_enmascarado').size().reset_index(name='total_contactos')
# Unir frecuencia y total de contactos
aceptacion_con_total = pd.merge(frecuencia_aceptacion, total_contactos, on='nit_enmascarado', how='left')
# Calcular la proporción de aceptación
aceptacion_con_total['proporcion_aceptacion'] = aceptacion_con_total['frecuencia_aceptacion'] / aceptacion_con_total['total_contactos']
# Unir al df_base_pivot
df_base_pivot = pd.merge(df_base_pivot, aceptacion_con_total[['nit_enmascarado', 'frecuencia_aceptacion', 'proporcion_aceptacion']], on='nit_enmascarado', how='left')
prueba=df_base_pivot.head() 

columnas_a_llenar = [
    'tendencia_dias_mora', 'tendencia_vlr_obligacion', 'tendencia_vlr_vencido',
    'tendencia_vlr_endeudamiento', 'tendencia_pago_cuota',
    'tendencia_porc_pago_cuota', 'tendencia_pago_mes', 'tasa_cumplimiento',
    'tasa_endeudamiento', 'tasa_pago_total', 'tasa_cumplimiento_cuota',
    'tasa_saldo_capital_vencido', 'tasa_de_mora', 'promedio_mora',
    'contacto_efectivo', 'tasa_acuerdos', 'tasa_promesa_cumplida'
]

df_base_pivot.loc[:, columnas_a_llenar] = df_base_pivot.loc[:, columnas_a_llenar].fillna(0)
#prueba=df_base_pivot.head()

#creacion de agregaciones
df_base_pivot_ordenado = df_base_pivot.sort_values(by=['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado', 'fecha'])

agregadas = df_base_pivot_ordenado.groupby(['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado']).agg({
    'dias_mora_fin': ['mean', 'min', 'max'],
    'vlr_obligacion': ['mean', 'last'],
    'vlr_vencido': ['mean', 'last'],
    'saldo_capital': ['last'],
    'endeudamiento': ['mean', 'last'],
    'pago_cuota': ['mean','sum','last'],
    'pago_mes': ['mean','sum','last'],
    'porc_pago_mes': ['mean', 'min', 'max'],
    'cant_gestiones': 'sum'
})

# Guardar las columnas de agrupamiento (índice)
columnas_de_indice = ['nit_enmascarado', 'num_oblig_orig_enmascarado','num_oblig_enmascarado']

# Renombrar solo las columnas agregadas
nuevas_columnas = []
for col in agregadas.columns:
    if isinstance(col, tuple):  # Si es una columna agregada (tiene tupla)
        nuevas_columnas.append('_'.join(col).strip())
    else:
        nuevas_columnas.append(col)

agregadas.columns = nuevas_columnas

# Resetear el índice para pasar las columnas de agrupamiento a columnas regulares
agregadas = agregadas.reset_index()
agregadas = agregadas.fillna(0)

# Eliminar los guiones bajos de las columnas de índice después del reset_index()
for col in columnas_de_indice:
  agregadas.rename(columns={f'{col}': col}, inplace=True)

agregadas_prueba=agregadas.head()
# Unir las variables agregadas al DataFrame original
df_base_pivot = df_base_pivot.merge(agregadas, on=columnas_de_indice, how='left')
def limpiar_valores_numericos(df):
    """
    Reemplaza valores -inf, inf y NaN con 0 en todas las columnas numéricas de un DataFrame.
    
    Args:
        df (pd.DataFrame): El DataFrame a limpiar.
        
    Returns:
        pd.DataFrame: El DataFrame con los valores numéricos limpios.
    """
    df_copy = df.copy() # Evitar modificación inplace
    numeric_cols = df_copy.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_copy

df_base_pivot = limpiar_valores_numericos(df_base_pivot)

#Validacion de datos atipicos
variables_numericas=['vlr_obligacion', 'vlr_vencido', 'saldo_capital', 
                     'endeudamiento', 'cant_gestiones', 'rpc',
                     'cant_acuerdo', 'valor_cuota_mes','pago_cuota', 
                     'porc_pago_cuota','pago_mes', 'porc_pago_mes',
                     'tendencia_dias_mora','tendencia_vlr_obligacion','tendencia_vlr_vencido',
                     'tendencia_vlr_endeudamiento', 'lag_aceptacion', 'lag_aceptacion_3m',
                     'tendencia_valor_cuota_mes', 'tendencia_pago_cuota', 'tendencia_porc_pago_cuota',
                     'tendencia_pago_mes', 'frecuencia_aceptacion', 'proporcion_aceptacion',
                     'dias_mora_fin_mean', 'vlr_obligacion_mean', 'vlr_obligacion_last',
                     'vlr_vencido_mean', 'vlr_vencido_last', 'saldo_capital_last',
                     'endeudamiento_mean', 'endeudamiento_last', 'pago_cuota_mean',
                     'pago_cuota_sum', 'pago_cuota_last', 'pago_mes_mean',
                     'pago_mes_sum', 'pago_mes_last', 'porc_pago_mes_mean',
                     'porc_pago_mes_min', 'porc_pago_mes_max', 'cant_gestiones_sum']

def visualizar_distribucion(df, columnas):
     """
     Visualiza la distribución de columnas en un DataFrame usando histogramas y boxplots.
 
     Args:
         df (pd.DataFrame): El DataFrame a analizar.
         columnas (list): La lista de columnas a visualizar.
     """
     for columna in columnas:
         fig, axes = plt.subplots(1, 2, figsize=(14, 5))

         # Histograma
         sns.histplot(df[columna], kde=True, ax=axes[0])
         axes[0].set_title(f'Histograma de {columna}')

         # Boxplot
         sns.boxplot(x=df[columna], ax=axes[1])
         axes[1].set_title(f'Boxplot de {columna}')

         plt.tight_layout()
         plt.show()


columnas_a_visualizar = variables_numericas
visualizar_distribucion(df_base_pivot, columnas_a_visualizar)

df_base_pivot.info()

correlacion_antes = variables_numericas[variables_numericas].corr()

variables_numericas = ['vlr_obligacion', 'vlr_vencido', 'saldo_capital', 
                     'endeudamiento', 'cant_gestiones', 'rpc',
                     'cant_acuerdo', 'valor_cuota_mes','pago_cuota', 
                     'porc_pago_cuota','pago_mes', 'porc_pago_mes',
                     'tendencia_dias_mora','tendencia_vlr_obligacion','tendencia_vlr_vencido',
                     'tendencia_vlr_endeudamiento', 'lag_aceptacion', 'lag_aceptacion_3m',
                     'tendencia_valor_cuota_mes', 'tendencia_pago_cuota', 'tendencia_porc_pago_cuota',
                     'tendencia_pago_mes', 'frecuencia_aceptacion', 'proporcion_aceptacion',
                     'dias_mora_fin_mean', 'vlr_obligacion_mean', 'vlr_obligacion_last',
                     'vlr_vencido_mean', 'vlr_vencido_last', 'saldo_capital_last',
                     'endeudamiento_mean', 'endeudamiento_last', 'pago_cuota_mean',
                     'pago_cuota_sum', 'pago_cuota_last', 'pago_mes_mean',
                     'pago_mes_sum', 'pago_mes_last', 'porc_pago_mes_mean',
                     'porc_pago_mes_min', 'porc_pago_mes_max', 'cant_gestiones_sum']

# Acceder a las columnas del DataFrame utilizando la lista de nombres
correlacion_matriz = df_base_pivot[variables_numericas].corr()

# Crear el mapa de calor
plt.figure(figsize=(20, 18))  # Ajusta el tamaño de la figura según tus necesidades
sns.heatmap(correlacion_matriz, annot=False, cmap='coolwarm', vmin=-1, vmax=1) # vmin y vmax establecen los valores minimos y máximos de la escala de colores del mapa
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()


plt.figure(figsize=(20, 18))  # Ajusta el tamaño de la figura según tus necesidades
sns.heatmap(correlacion_matriz, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()

#Matriz de correlacion en dataframe
correlaciones_df = correlacion_matriz.stack().reset_index()
correlaciones_df.columns = ['Variable 1', 'Variable 2', 'Correlacion']

# Calcular el valor absoluto de la correlación
correlaciones_df['Correlacion_Abs'] = abs(correlaciones_df['Correlacion'])

# Excluir correlaciones iguales a 1
correlaciones_df = correlaciones_df[correlaciones_df['Correlacion_Abs'] != 1]

# Ordenar por valor absoluto de la correlación de mayor a menor
correlaciones_df_ordenado = correlaciones_df.sort_values(by='Correlacion_Abs', ascending=False)

# Imprimir el DataFrame ordenado
print(correlaciones_df_ordenado)

#Matriz de correlacion para la variable respuesta 0
# Calcular la matriz de correlación para los casos donde var_rpta_alt es 0
correlacion_0 = df_base_pivot[df_base_pivot['var_rpta_alt'] == 0][variables_numericas].corr()

# Calcular la matriz de correlación para los casos donde var_rpta_alt es 1
correlacion_1 = df_base_pivot[df_base_pivot['var_rpta_alt'] == 1][variables_numericas].corr()

# Crear los mapas de calor

# Mapa de calor para var_rpta_alt = 0
plt.figure(figsize=(20, 18))
sns.heatmap(correlacion_0, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Matriz de Correlación (var_rpta_alt = 0)')
plt.show()

# Mapa de calor para var_rpta_alt = 1
plt.figure(figsize=(20, 18))
sns.heatmap(correlacion_1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Matriz de Correlación (var_rpta_alt = 1)')
plt.show()

# Calcular la matriz de correlación para los casos donde var_rpta_alt es 0
correlacion_0 = df_base_pivot[df_base_pivot['var_rpta_alt'] == 0][variables_numericas].corr()

# Convertir a DataFrame
correlaciones_0_df = correlacion_0.stack().reset_index()
correlaciones_0_df.columns = ['Variable 1', 'Variable 2', 'Correlacion']
correlaciones_0_df['Correlacion_Abs'] = abs(correlaciones_0_df['Correlacion'])
correlaciones_0_df = correlaciones_0_df[correlaciones_0_df['Correlacion_Abs'] != 1]
correlaciones_0_df_ordenado = correlaciones_0_df.sort_values(by='Correlacion_Abs', ascending=False)


# Imprimir las correlaciones para los casos donde var_rpta_alt es 0
print("Correlaciones para var_rpta_alt = 0:")
print(correlaciones_0_df_ordenado)

# Calcular la matriz de correlación para los casos donde var_rpta_alt es 1
correlacion_1 = df_base_pivot[df_base_pivot['var_rpta_alt'] == 1][variables_numericas].corr()

# Convertir a DataFrame
correlaciones_1_df = correlacion_1.stack().reset_index()
correlaciones_1_df.columns = ['Variable 1', 'Variable 2', 'Correlacion']
correlaciones_1_df['Correlacion_Abs'] = abs(correlaciones_1_df['Correlacion'])
correlaciones_1_df = correlaciones_1_df[correlaciones_1_df['Correlacion_Abs'] != 1]
correlaciones_1_df_ordenado = correlaciones_1_df.sort_values(by='Correlacion_Abs', ascending=False)


# Imprimir las correlaciones para los casos donde var_rpta_alt es 1
print("\nCorrelaciones para var_rpta_alt = 1:")
print(correlaciones_1_df_ordenado)



#Winzorizacion para manipular atipicos
def winsorizar_dataframe(df, lower_quantile=0.05, upper_quantile=0.95, columns_to_winsorize = None):
    """
    Aplica la winsorización a las columnas numéricas especificadas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame a winsorizar.
        lower_quantile (float, optional): Cuantil inferior. Defaults to 0.05.
        upper_quantile (float, optional): Cuantil superior. Defaults to 0.95.
        columns_to_winsorize (list, optional): Columnas a winsorizar. Defaults to None.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las columnas winsorizadas.
    """
    df_copy = df.copy() # No modificar el dataframe original
    if columns_to_winsorize is None:
          cols_to_process = df_copy.select_dtypes(include=np.number).columns
    else:
         cols_to_process = columns_to_winsorize
    for col in cols_to_process:
        lower_limit = df_copy[col].quantile(lower_quantile)
        upper_limit = df_copy[col].quantile(upper_quantile)
        df_copy[col] = df_copy[col].clip(lower=lower_limit, upper=upper_limit)
    return df_copy

variables_numericas = ['vlr_obligacion', 'vlr_vencido', 'saldo_capital', 
                     'endeudamiento', 'cant_gestiones', 'rpc',
                     'cant_acuerdo', 'valor_cuota_mes','pago_cuota', 
                     'porc_pago_cuota','pago_mes', 'porc_pago_mes',
                     'tendencia_dias_mora','tendencia_vlr_obligacion','tendencia_vlr_vencido',
                     'tendencia_vlr_endeudamiento', 'lag_aceptacion', 'lag_aceptacion_3m',
                     'tendencia_valor_cuota_mes', 'tendencia_pago_cuota', 'tendencia_porc_pago_cuota',
                     'tendencia_pago_mes', 'frecuencia_aceptacion', 'proporcion_aceptacion',
                     'dias_mora_fin_mean', 'vlr_obligacion_mean', 'vlr_obligacion_last',
                     'vlr_vencido_mean', 'vlr_vencido_last', 'saldo_capital_last',
                     'endeudamiento_mean', 'endeudamiento_last', 'pago_cuota_mean',
                     'pago_cuota_sum', 'pago_cuota_last', 'pago_mes_mean',
                     'pago_mes_sum', 'pago_mes_last', 'porc_pago_mes_mean',
                     'porc_pago_mes_min', 'porc_pago_mes_max', 'cant_gestiones_sum']

# Winsorizar por separado para var_rpta_alt = 0
df_winsorizado_0 = winsorizar_dataframe(
    df_base_pivot[df_base_pivot['var_rpta_alt'] == 0].copy(),
    upper_quantile=0.95,
    columns_to_winsorize=variables_numericas
)
# Winsorizar por separado para var_rpta_alt = 1
df_winsorizado_1 = winsorizar_dataframe(
    df_base_pivot[df_base_pivot['var_rpta_alt'] == 1].copy(),
    upper_quantile=0.95,
    columns_to_winsorize=variables_numericas
)

# Concatenar los resultados winsorizados
df_base_pivot = pd.concat([df_winsorizado_0, df_winsorizado_1])

#preuba_df_winsorizado_1=df_winsorizado_1.head()

#hot encoding
# Lista de variables cualitativas
categorical_columns = ['banca', 'segmento', 'producto', 'producto_cons', 'aplicativo', 
                       'rango_mora', 'desc_alternativa1', 'desc_alternativa2', 'desc_alternativa3',
                       'alter_posible1_2', 'alter_posible2_2', 'alter_posible3_2',
                       'descripcion_ranking_mejor_ult', 'descripcion_ranking_post_ult',
                       'marca_alt_rank', 'marca_alt_apli', 'marca_agrupada_rgo', 
                       'marca_pago', 'marca_alternativa', 'marca_alternativa_orig']

# Función para manejar categorías con baja frecuencia
def replace_low_frequency_categories(df, col, threshold=0.1):
    value_counts = df[col].value_counts(normalize=True)
    low_freq_categories = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(low_freq_categories, 'otros')
    return df

# Aplicar la función a cada columna cualitativa del DataFrame
for col in categorical_columns:
    if col in df_base_pivot.columns:  # Verificar que la columna exista en el DataFrame
        df_base_pivot = replace_low_frequency_categories(df_base_pivot, col)

# Aplicar One-Hot Encoding a las columnas cualitativas
df_encoded = pd.get_dummies(df_base_pivot, columns=categorical_columns, drop_first=True)
#test_df_base_pivot=df_base_pivot.head()
# Mostrar el resultado
print(df_encoded.head())
print(df_encoded.info)

# 1. Definir la variable objetivo y las variables predictoras

variables_predictoras = ['vlr_obligacion', 'vlr_vencido', 'saldo_capital',
                     'endeudamiento', 'cant_gestiones', 'rpc',
                     'cant_acuerdo', 'valor_cuota_mes','pago_cuota',
                     'porc_pago_cuota','pago_mes', 'porc_pago_mes',
                     'tendencia_dias_mora','tendencia_vlr_obligacion','tendencia_vlr_vencido',
                     'tendencia_vlr_endeudamiento', 'lag_aceptacion', 'lag_aceptacion_3m',
                     'tendencia_valor_cuota_mes', 'tendencia_pago_cuota', 'tendencia_porc_pago_cuota',
                     'tendencia_pago_mes', 'frecuencia_aceptacion', 'proporcion_aceptacion',
                     'dias_mora_fin_mean', 'vlr_obligacion_mean', 'vlr_obligacion_last',
                     'vlr_vencido_mean', 'vlr_vencido_last', 'saldo_capital_last',
                     'endeudamiento_mean', 'endeudamiento_last', 'pago_cuota_mean',
                     'pago_cuota_sum', 'pago_cuota_last', 'pago_mes_mean',
                     'pago_mes_sum', 'pago_mes_last', 'porc_pago_mes_mean',
                     'porc_pago_mes_min', 'porc_pago_mes_max', 'cant_gestiones_sum',
                    'banca_otros', 'segmento_Personal plus', 'segmento_otros',
                    'producto_ROTATIVOS', 'producto_TARJETA DE CREDITO', 'producto_otros',
                    'producto_cons_Rotativos', 'producto_cons_Tarjeta de Credito', 'producto_cons_otros',
                    'aplicativo_L', 'aplicativo_M', 'aplicativo_V', 'aplicativo_otros',
                    'rango_mora_b.31-90', 'desc_alternativa1_Consolidación de pasivos',
                    'desc_alternativa1_Reestructuración novacion', 'desc_alternativa1_otros',
                    'desc_alternativa2_Sin alivio', 'desc_alternativa2_otros',
                    'desc_alternativa3_Reestructuración novacion', 'desc_alternativa3_Sin alivio',
                    'desc_alternativa3_otros', 'alter_posible1_2_CON22', 'alter_posible1_2_TDC10',
                    'alter_posible1_2_otros', 'alter_posible2_2_TDC11', 'alter_posible2_2_otros',
                    'alter_posible3_2_CON03', 'alter_posible3_2_CON22', 'alter_posible3_2_otros',
                    'descripcion_ranking_mejor_ult_NO ACEPTA ACUERDO', 'descripcion_ranking_mejor_ult_PLAN DE PAGO',
                    'descripcion_ranking_mejor_ult_otros', 'descripcion_ranking_post_ult_NO ACEPTA ACUERDO',
                    'descripcion_ranking_post_ult_NO CONTESTA', 'descripcion_ranking_post_ult_PLAN DE PAGO',
                    'descripcion_ranking_post_ult_otros', 'marca_alt_rank_Cliente compromiso de pago',
                    'marca_alt_rank_No acepta acuerdo', 'marca_alt_rank_otros', 'marca_alt_apli_SI',
                    'marca_agrupada_rgo_MANTENIMIENTO', 'marca_agrupada_rgo_otros', 'marca_pago_Sin pago',
                    'marca_pago_otros', 'marca_alternativa_N.A', 'marca_alternativa_otros',
                    'marca_alternativa_orig_N.A', 'marca_alternativa_orig_otros']
variable_objetivo = 'var_rpta_alt'

# 2. Dividir los datos en conjuntos de entrenamiento y prueba
X = df_encoded[variables_predictoras]
y = df_encoded[variable_objetivo]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar los modelos
# Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight = 'balanced')
modelo_rf.fit(X_train, y_train)

# Gradient Boosting
modelo_gb = GradientBoostingClassifier(n_estimators=100, random_state=42,  learning_rate = 0.1, max_depth = 3)
modelo_gb.fit(X_train, y_train)

# 4. Realizar predicciones
y_pred_rf = modelo_rf.predict(X_test)
y_pred_gb = modelo_gb.predict(X_test)

# 5. Evaluar los modelos
print("Resultados Random Forest:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_rf):.4f}")
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_rf))

print("\nResultados Gradient Boosting:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"   AUC: {roc_auc_score(y_test, y_pred_gb):.4f}")
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_gb))

#Ajuste del modelo por multicolinealidad y sobreajuste
backup_X_filtered=X_filtered
# Importancia de variables en Random Forest
feature_importance = pd.Series(modelo_rf.feature_importances_, index=X.columns)
important_features = feature_importance[feature_importance > 0.01].index  # Seleccionar variables con importancia > 1%
# Filtrar el dataset con las variables seleccionadas
X_filtered = X[important_features_2]
#Eliminacion de variables con alta correlacion
correlation_matrix = X_filtered.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlación")
plt.show()
high_corr_vars = correlation_matrix.columns[(correlation_matrix.abs() > 0.80).sum() > 1]
X_filtered.drop(columns=high_corr_vars, inplace=True)

# Suponiendo que X_filtered y y ya están definidos
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)


# Modelo Random Forest con regularización
rf_model = RandomForestClassifier(
    n_estimators=100,          # Número de árboles
    max_depth=10,              # Profundidad máxima de los árboles
    min_samples_split=10,      # Mínimo de muestras para dividir un nodo
    min_samples_leaf=5,        # Mínimo de muestras en una hoja
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predicciones y evaluación del modelo Random Forest
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nResultados Random Forest:")
print(f"  Accuracy: {rf_model.score(X_test, y_test):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))


feature_importance_2= pd.Series(rf_model.feature_importances_, index=X_filtered.columns)
important_features_2= feature_importance[feature_importance > 0.01].index  # Seleccionar variables con importancia > 1%

# Modelo Gradient Boosting con regularización
gb_model = GradientBoostingClassifier(
    n_estimators=100,          # Número de iteraciones
    learning_rate=0.05,        # Tasa de aprendizaje
    max_depth=3,               # Profundidad máxima de los árboles
    min_samples_split=10,      # Mínimo de muestras para dividir un nodo
    min_samples_leaf=5,        # Mínimo de muestras en una hoja
    random_state=42
)
gb_model.fit(X_train, y_train)

# Predicciones y evaluación del modelo Gradient Boosting
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

print("\nResultados Gradient Boosting:")
print(f"  Accuracy: {gb_model.score(X_test, y_test):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_prob_gb):.4f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_gb))

# Lista de variables más importantes
variables_importantes = ['nit_enmascarado','num_oblig_orig_enmascarado','num_oblig_enmascarado','fecha','var_rpta_alt',
    'frecuencia_aceptacion',
    'proporcion_aceptacion',
    'porc_pago_mes_max',
    'descripcion_ranking_mejor_ult_PLAN DE PAGO',
    'descripcion_ranking_mejor_ult_otros',
    'marca_alt_rank_otros',
    'marca_alt_apli_SI',
    'marca_agrupada_rgo_MANTENIMIENTO',
    'marca_agrupada_rgo_otros',
    'marca_pago_Sin pago'
]

# Crear un nuevo DataFrame con solo las variables importantes
df_seleccionado_trtest= df_encoded[variables_importantes].copy()
df_seleccionado_trtest_renamed = df_seleccionado_trtest.rename(columns={'fecha': 'fecha_corte'})