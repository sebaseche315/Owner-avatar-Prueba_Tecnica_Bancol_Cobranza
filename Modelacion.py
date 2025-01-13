# -*- coding: utf-8 -*-
"""
@author: Sebastian Echeverri Parra
"""
# Paso 1: Unir df_seleccionado_trtest con df_cuotas
# Identifica las columnas comunes entre df_seleccionado_trtest y df_cuotas
# Reemplaza 'columna_comun_1' con el nombre de tu columna común
df_unido_1 = pd.merge(df_seleccionado_trtest_renamed, df_cuotas, on=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_corte'], how='left')

# Paso 2: Unir el resultado con df_prueba_master
# Identifica las columnas comunes entre df_unido_1 y df_prueba_master
# Reemplaza 'columna_comun_2' con el nombre de tu columna común
df_unido_2 = pd.merge(df_unido_1, df_prueba_master, on=['nit_enmascarado'], how='left')

# Paso 3: Unir el resultado con df_probabilidad_models
# Identifica las columnas comunes entre df_unido_2 y df_probabilidad_models
# Reemplaza 'columna_comun_3' con el nombre de tu columna común
df_unido_final = pd.merge(df_unido_2, df_probabilidad_models, on=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_corte'], how='left')
df_unido_final = df_unido_final.fillna(0)
# Modelacion
# 1. Definir la variable objetivo y las variables predictoras
variables_predictoras =  [
    'frecuencia_aceptacion',
    'proporcion_aceptacion',
    'porc_pago_mes_max',
    'descripcion_ranking_mejor_ult_PLAN DE PAGO',
    'descripcion_ranking_mejor_ult_otros',
    'marca_alt_rank_otros',
    'marca_alt_apli_SI',
    'marca_agrupada_rgo_MANTENIMIENTO',
    'marca_agrupada_rgo_otros',
    'marca_pago_Sin pago',
    'año_corte', 'mes_corte', 'dias_diferencia_pago',
    'lag_porc_pago_1m', 'media_movil_pago_3m', 'cumplimiento_pago',
    'valor_cuota_mes', 'pago_total', 'porc_pago',
    'relacion_ing_egresos', 'relacion_act_pas', 'porc_patrimonio',
    'carga_familiar', 'month_sin', 'month_cos',
    'prob_propension', 'prob_alrt_temprana', 'prob_auto_cura',
     'lag_prob_propension_1m',
    'lag_prob_propension_3m', 'trend_prob_prop', 'trend_prob_alrt',
    'freq_obligation', 'lote_1', 'lote_2', 'lote_3'
]


variable_objetivo = 'var_rpta_alt'

# Imprimir las columnas para identificar los nombres correctos
print("Columnas en df_unido_final:")
print(df_unido_final.columns)
print("----------------------------------")

# Variables categóricas a considerar
categorical_prefixes = ['producto_', 'aplicativo_', 'segmento_', 'marca_pago_']

# Generar una lista con las variables categoricas que si existen en df_unido_final
cat_vars = [col for col in df_unido_final.columns if any(col.startswith(prefix) for prefix in categorical_prefixes)]

# Juntar las variables numéricas y las variables categóricas que si existen
variables_predictoras = variables_predictoras + cat_vars

# Eliminar duplicados por si existen variables que se hayan agregado por error en ambas listas
variables_predictoras = list(set(variables_predictoras))

# 2. Dividir los datos en conjuntos de entrenamiento y prueba
X = df_unido_final[variables_predictoras]
y = df_unido_final[variable_objetivo]


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

# Importancia de variables
feature_importance = pd.Series(modelo_rf.feature_importances_, index=X.columns)
important_features = feature_importance[feature_importance > 0.01].index  # Seleccionar variables con importancia > 1%
# Filtrar el dataset con las variables seleccionadas
X_filtered_consolidado = X[important_features]
#Eliminacion de variables con alta correlacion
correlation_matrix = X_filtered_consolidado.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlación")
plt.show()
high_corr_vars = correlation_matrix.columns[(correlation_matrix.abs() > 0.80).sum() > 1]
X_filtered_consolidado.drop(columns=high_corr_vars, inplace=True)
X_filtered_consolidado.shape
X_filtered_consolidado.columns
# Suponiendo que X_filtered y y ya están definidos
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_filtered_consolidado , y, test_size=0.2, random_state=42)

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

