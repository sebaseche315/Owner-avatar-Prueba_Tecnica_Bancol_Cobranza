# -*- coding: utf-8 -*-
"""
@author: Sebastian Echeverri Parra
"""
# Datos de ejemplo (remplazar por la carga real del dataset)
data_path = r'C:\Users\JULI\OneDrive\Escritorio\Pureba Analitica Bancolombia cobranzas\Data'
# Archivos disponibles
files = {
    'probabilidad_oblig': 'prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv'

}
# Cargar los archivos en DataFrames
df_probabilidad_oblig = pd.read_csv(os.path.join(data_path, files['probabilidad_oblig']))

# Crear DataFrame
df = pd.DataFrame(df_probabilidad_oblig)

# Convertir 'fecha_corte' a tipo datetime para manipulación temporal
df['fecha_corte'] = pd.to_datetime(df['fecha_corte'], format='%Y%m')

# Extraer componentes temporales
df['year'] = df['fecha_corte'].dt.year
df['month'] = df['fecha_corte'].dt.month
df['quarter'] = df['fecha_corte'].dt.quarter

# Crear variables de rezago (lag) para las probabilidades
df = df.sort_values(by=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_corte'])
df['lag_prob_propension_1m'] = df.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['prob_propension'].shift(1)
df['lag_prob_propension_3m'] = df.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['prob_propension'].shift(3)

# Crear variables de tendencia (diferencia porcentual respecto al mes anterior)
df['trend_prob_prop'] = df.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['prob_propension'].pct_change()
df['trend_prob_alrt'] = df.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['prob_alrt_temprana'].pct_change()

# Manejo de nulos en las nuevas columnas
df.fillna({
    'lag_prob_propension_1m': 0,
    'lag_prob_propension_3m': 0,
    'trend_prob_prop': 0,
    'trend_prob_alrt': 0
}, inplace=True)

# Calcular la frecuencia de aparición de cada obligación por cliente
df['freq_obligation'] = df.groupby('nit_enmascarado')['num_oblig_enmascarado'].transform('count')

# Manejo de la variable categórica 'lote' con One-Hot Encoding
df = pd.get_dummies(df, columns=['lote'], prefix='lote')
df_probabilidad_models=df
# Resultado final
#df_prueba_probabilidad2=df_probabilidad_models[
#    df_probabilidad_models['nit_enmascarado'] == 205391
#]	