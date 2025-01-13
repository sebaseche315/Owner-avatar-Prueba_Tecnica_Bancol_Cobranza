# -*- coding: utf-8 -*-
"""
@author: Sebastian Echeverri Parra
"""
# Archivo disponibles
files = {
    'cuotas_pagos_hist': 'prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv'
}

# Cargar los archivos en DataFrames
df_cuotas_pagos_hist = pd.read_csv(os.path.join(data_path, files['cuotas_pagos_hist']))

df=df_cuotas_pagos_hist
# Asegurar que las fechas sean interpretadas correctamente
df['fecha_corte'] = pd.to_datetime(df['fecha_corte'], format='%Y%m%d')
df['fecha_pago_minima'] = pd.to_datetime(df['fecha_pago_minima'], format='%Y%m%d', errors='coerce')
df['fecha_pago_maxima'] = pd.to_datetime(df['fecha_pago_maxima'], format='%Y%m%d', errors='coerce')

# ======= 1. Conversión de variables categóricas =======
# One-Hot Encoding para 'producto', 'aplicativo', 'segmento' y 'marca_pago'
cat_vars = ['producto', 'aplicativo', 'segmento', 'marca_pago']

def group_low_frequency_categories(df, column, threshold=0.7):
    """
    Agrupa categorías menos frecuentes en 'Otros' si su proporción acumulada es menor al threshold.
    Args:
        df (pd.DataFrame): DataFrame original
        column (str): Nombre de la columna a procesar
        threshold (float): Umbral de proporción acumulada para agrupar en 'Otros'
    """
    # Calcular la proporción de cada categoría
    category_counts = df[column].value_counts(normalize=True)
    
    # Identificar las categorías que representan el 70% o más
    cumulative_sum = category_counts.cumsum()
    major_categories = cumulative_sum[cumulative_sum <= threshold].index.tolist()
    
    # Agrupar categorías menos frecuentes en 'Otros'
    df[column] = df[column].apply(lambda x: x if x in major_categories else 'Otros')

# Aplicar la función a cada variable categórica
for col in cat_vars:
    group_low_frequency_categories(df, col, threshold=0.7)

# Realizar la codificación one-hot después de agrupar
df = pd.get_dummies(df, columns=cat_vars, drop_first=True)

# ======= 2. Manipulacion de outliers=======
def winsorizar_variables(df, num_vars, lower_quantile=0.05, upper_quantile=0.95):
    """
    Aplica winsorización a variables numéricas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame a procesar.
        num_vars (list): Lista de nombres de las variables numéricas a procesar.
        lower_quantile (float, optional): Cuantil inferior para winsorizar. Defaults to 0.05.
        upper_quantile (float, optional): Cuantil superior para winsorizar. Defaults to 0.95.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las variables procesadas.
    """
    df_copy = df.copy() # Para no modificar el dataframe original
    for col in num_vars:
        # Winsorización
        lower_limit = df_copy[col].quantile(lower_quantile)
        upper_limit = df_copy[col].quantile(upper_quantile)
        df_copy[col] = df_copy[col].clip(lower=lower_limit, upper=upper_limit)
    
    return df_copy
# Definir las variables a procesar
num_cols = ['valor_cuota_mes', 'pago_total', 'porc_pago']

# Ejemplo de uso (asegúrate de que df sea tu DataFrame)
df = winsorizar_variables(df, num_cols )
print(df[num_cols].head())

# ======= 3. Manipulación de fechas =======
# Descomponer fecha_corte en componentes de año y mes
df['año_corte'] = df['fecha_corte'].dt.year
df['mes_corte'] = df['fecha_corte'].dt.month

# Calcular diferencia en días entre fecha_corte y fecha_pago_maxima
df['dias_diferencia_pago'] = (df['fecha_corte'] - df['fecha_pago_maxima']).dt.days

# ======= 4. Creación de variables de rezago =======
# Ordenar el dataset por nit, obligación y fecha
df.sort_values(by=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_corte'], inplace=True)

# Crear rezago de porcentaje de pago (lag de 1 mes)
df['lag_porc_pago_1m'] = df.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['porc_pago'].shift(1)

# Crear media móvil de pago total (últimos 3 meses)
df['media_movil_pago_3m'] = df.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['pago_total'].rolling(3).mean().reset_index(level=[0, 1, 2], drop=True)

# ======= 5. Variables adicionales =======
# Crear variable binaria de cumplimiento de pago
df['cumplimiento_pago'] = (df['pago_total'] >= df['valor_cuota_mes']).astype(int)

# ======= 6. Eliminar filas con valores nulos generados por los rezagos =======
df.dropna(subset=['lag_porc_pago_1m', 'media_movil_pago_3m'], inplace=True)

# ================================
# 7. Seleccionar las variables creadas
# ================================
new_vars = ['nit_enmascarado', 'num_oblig_enmascarado','fecha_corte',
    'año_corte', 'mes_corte', 'dias_diferencia_pago',
    'lag_porc_pago_1m', 'media_movil_pago_3m', 'cumplimiento_pago',
    'valor_cuota_mes', 'pago_total', 'porc_pago' # Agregar variables winsorizadas
]

new_vars.extend([col for col in df.columns if col.startswith('producto_')])
new_vars.extend([col for col in df.columns if col.startswith('aplicativo_')])
new_vars.extend([col for col in df.columns if col.startswith('segmento_')])
new_vars.extend([col for col in df.columns if col.startswith('marca_pago_')])

print("Variables seleccionadas:")
print(new_vars)

# Usar el DataFrame original df y filtrar las columnas new_vars
df_cuotas=df[new_vars]
df_cuotas=df_cuotas.fillna(0)
# Ahora que tenemos el dataframe con las variables seleccionadas, podemos aplicar el filtro
#df_cuotas_prueba = df_cuotas[df_cuotas['nit_enmascarado'] == 205391]
