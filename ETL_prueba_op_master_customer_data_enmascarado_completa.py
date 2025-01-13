# -*- coding: utf-8 -*-
"""
@author: Sebastian Echeverri Parra
"""
# Archivos disponibles
files = {
    'master_customer': 'prueba_op_master_customer_data_enmascarado_completa.csv'
}

# Cargar los archivos en DataFrames
df_master_customer = pd.read_csv(os.path.join(data_path, files['master_customer']))
# Cargar el dataset
df = df_master_customer

# ================================
# 1. Imputación de valores faltantes
# ================================
df['tipo_vivienda'].fillna('Desconocido', inplace=True)
df['nivel_academico'].fillna('Desconocido', inplace=True)
df['num_hijos'].fillna(0, inplace=True)
df['personas_dependientes'].fillna(0, inplace=True)

# ================================
# 2. Creación de nuevas variables derivadas
# ================================
df['relacion_ing_egresos'] = df['total_ing'] / (df['egresos_mes'] + 1)
df['relacion_act_pas'] = df['tot_activos'] / (df['tot_pasivos'] + 1)
df['porc_patrimonio'] = df['tot_patrimonio'] / (df['tot_activos'] + 1)
df['carga_familiar'] = df['num_hijos'] + df['personas_dependientes']

# ================================
# 3. Codificación cíclica de la variable 'month'
# ================================
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ================================
# 4. Transformación logarítmica de variables numéricas
# ================================
num_log_vars = ['total_ing', 'tot_activos', 'tot_pasivos', 'tot_patrimonio']
for col in num_log_vars:
    df[col] = np.log1p(df[col])  # log1p evita problemas con ceros

# ================================
# 5. Manejo de atipicos
# ================================
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
num_vars = ['edad_cli', 'total_ing', 'tot_activos', 'tot_pasivos', 'egresos_mes', 'tot_patrimonio']

# Ejemplo de uso (asegúrate de que df sea tu DataFrame)
df = winsorizar_variables(df, num_vars)

# ================================
# 6. Codificación de variables categóricas
# ================================
# Lista de variables categóricas

cat_vars = ['cod_tipo_doc', 'tipo_cli', 'ctrl_terc', 'genero_cli', 'estado_civil', 
            'tipo_vivienda', 'nivel_academico', 'ocup', 'act_econom', 'sector', 
            'subsector', 'declarante', 'origen_fondos', 'canal_actualizacion', 
            'cli_actualizado', 'segm', 'subsegm', 'nicho', 'region_of', 'nombre_dpto_dirp', 'ciiu']

# Función para agrupar categorías menos frecuentes
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
df = pd.get_dummies(df, columns=cat_vars)

# ================================
# 7. Seleccionar las variables creadas
# ================================
new_vars = ['nit_enmascarado',
    'relacion_ing_egresos', 'relacion_act_pas', 'porc_patrimonio', 'carga_familiar',
    'month_sin', 'month_cos'
]
new_vars.extend([col for col in df.columns if col.startswith('tipo_vivienda_')])
new_vars.extend([col for col in df.columns if col.startswith('nivel_academico_')])
new_vars.extend([col for col in df.columns if col.startswith('cod_tipo_doc_')])
new_vars.extend([col for col in df.columns if col.startswith('tipo_cli_')])
new_vars.extend([col for col in df.columns if col.startswith('ctrl_terc_')])
new_vars.extend([col for col in df.columns if col.startswith('genero_cli_')])
new_vars.extend([col for col in df.columns if col.startswith('estado_civil_')])
new_vars.extend([col for col in df.columns if col.startswith('ocup_')])
new_vars.extend([col for col in df.columns if col.startswith('act_econom_')])
new_vars.extend([col for col in df.columns if col.startswith('sector_')])
new_vars.extend([col for col in df.columns if col.startswith('subsector_')])
new_vars.extend([col for col in df.columns if col.startswith('declarante_')])
new_vars.extend([col for col in df.columns if col.startswith('origen_fondos_')])
new_vars.extend([col for col in df.columns if col.startswith('canal_actualizacion_')])
new_vars.extend([col for col in df.columns if col.startswith('cli_actualizado_')])
new_vars.extend([col for col in df.columns if col.startswith('segm_')])
new_vars.extend([col for col in df.columns if col.startswith('subsegm_')])
new_vars.extend([col for col in df.columns if col.startswith('nicho_')])
new_vars.extend([col for col in df.columns if col.startswith('region_of_')])
new_vars.extend([col for col in df.columns if col.startswith('nombre_dpto_dirp_')])
new_vars.extend([col for col in df.columns if col.startswith('ciiu_')])

print("Variables seleccionadas:")
print(new_vars)

# Usar el DataFrame original df y filtrar las columnas new_vars
df_master=df[new_vars]
# Ahora que tenemos el dataframe con las variables seleccionadas, podemos aplicar el filtro
df_prueba_master = df_master[df_master['nit_enmascarado'] == 205391]
df_prueba_master = df_master[df_master['nit_enmascarado'] == 630611]




print(df_prueba_master)