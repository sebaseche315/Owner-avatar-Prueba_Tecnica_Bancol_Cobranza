# -*- coding: utf-8 -*-
"""
@author: Sebastian Echeverri Parra
"""
# Datos de prueba
data_path = r'C:\Users\JULI\OneDrive\Escritorio\Pureba Analitica Bancolombia cobranzas\Data'

# Archivos disponibles
files = {
    'prueba_op': 'prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv'
}

# Cargar los archivos en DataFrames
df_prueba = pd.read_csv(os.path.join(data_path, files['prueba_op']))
df_prueba_original= pd.read_csv(os.path.join(data_path, files['prueba_op']))


important_features= ['descripcion_ranking_mejor_ult_otros', 'marca_pago_Sin pago',
       'marca_alt_apli_SI', 'prob_alrt_temprana', 'porc_pago_mes_max',
       'marca_agrupada_rgo_MANTENIMIENTO', 'frecuencia_aceptacion',
       'prob_auto_cura', 'prob_propension', 'marca_alt_rank_otros',
       'freq_obligation', 'marca_agrupada_rgo_otros', 'proporcion_aceptacion',
       'descripcion_ranking_mejor_ult_PLAN DE PAGO', 'trend_prob_prop']

from sklearn.impute import SimpleImputer
df_merged = df_prueba.merge(df_unido_final, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='left')
X_test = df_merged[important_features]

imputer = SimpleImputer(strategy='constant', fill_value=0)  # Imputar con 0
X_test = imputer.fit_transform(X_test)

random_forest_model = rf_model

# ============================
# 6. Predecir probabilidades
# ============================
probabilidades = random_forest_model.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1
predicciones = (probabilidades >= 0.5).astype(int)  # Umbral de 0.5 para clasificación binaria

# Crear DataFrame temporal con predicciones y probabilidades
df_predicciones = df_merged[['nit_enmascarado', 'num_oblig_enmascarado']].copy()
df_predicciones['probabilidad_predicha'] = probabilidades
df_predicciones['prediccion'] = predicciones

# Hacer merge con df_prueba para asegurar que todas las filas estén presentes
df_prueba = df_prueba.merge(df_predicciones, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='left')

# Rellenar con 0 las filas que no tengan predicciones
df_prueba['probabilidad_predicha'].fillna(0, inplace=True)
df_prueba['prediccion'].fillna(0, inplace=True)


# Crear la columna ID concatenando las tres columnas clave
df_prueba['ID'] = df_prueba['nit_enmascarado'].astype(str) + '#' + \
                   df_prueba['num_oblig_orig_enmascarado'].astype(str) + '#' + \
                   df_prueba['num_oblig_enmascarado'].astype(str)

# Seleccionar solo las columnas requeridas y renombrar 'prediccion' a 'var_rpta_alt'
df_final = df_prueba[['ID', 'prediccion']].rename(columns={'prediccion': 'var_rpta_alt'})

data_path = r'C:\Users\JULI\OneDrive\Escritorio\Pureba Analitica Bancolombia cobranzas\Data'
file_path = os.path.join(data_path, 'file_submission.csv')
df_final.to_csv(file_path, index=False)