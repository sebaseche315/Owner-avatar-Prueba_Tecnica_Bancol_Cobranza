# Prueba Tecnica Bancolombia Cobranza
Link del reto:
https://www.kaggle.com/competitions/prueba-analitica-modelo-opciones-de-pago/overview

Este repositorio contiene la solución desarrollada para una prueba técnica de predicción utilizando un modelo de Random Forest y de Gradient Boosting. El objetivo principal es predecir una variable binaria (0 o 1) asociada a una entidad identificada mediante un ID compuesto por tres variables concatenadas. La solución se implementa en Python, utilizando bibliotecas como pandas, scikit-learn, y numpy.

Estructura del Proyecto
bash
├── src/                      # Carpeta con los scripts de procesamiento y predicción
│   ├── ETL_prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.py    # Script ETL del dataset prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest
│   ├── ETL_prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.py     # Script ETL del dataset op_maestra_cuotas_pagos_mes_hist_enmascarado_completa
│   ├── ETL_prueba_op_master_customer_data_enmascarado_completa.py     # Script ETL del dataset prueba_op_master_customer_data_enmascarado_completa
│   ├── ETL_prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.py     # Script ETL del dataset prueba_op_probabilidad_oblig_base_hist_enmascarado_completa
│   ├── Modelacion.py     # Script para entrenar el modelo Random Forest y Gradient Boosting
│   └── Datos de Prueba.py         # Script para generar las predicciones y el archivo final
├── README.md                 # Archivo de documentación del proyecto
└── requirements.txt          # Lista de dependencias necesarias para ejecutar el proyecto

Flujo de Trabajo
1. Preparación de Datos
El primer paso consiste en la preparación de los datos. Los datos de entrada contienen múltiples registros, algunos de los cuales pueden estar duplicados. Se realizaron los siguientes pasos:

Concatenación de columnas nit_enmascarado, num_oblig_orig_enmascarado y num_oblig_enmascarado para generar el campo ID.
Validación y eliminación de duplicados en la columna ID.
Imputación de valores faltantes (NaN) en el conjunto de datos utilizando ceros (0).
2. Entrenamiento del Modelo
Se utilizó un modelo de Random Forest Classifier y Gradient Boosting de scikit-learn para entrenar el modelo.
Pasos realizados:

Selección de las características más importantes para el modelo.
Entrenamiento del modelo con los datos preparados.
Validación del modelo utilizando métricas de precisión y recall.
3. Generación de Predicciones
Una vez entrenado el modelo, se realizaron predicciones sobre un conjunto de datos de prueba. Las predicciones generadas son probabilidades que se convierten en valores binarios (0 o 1) utilizando un umbral de 0.5.

4. Archivo de Salida
El archivo de salida final contiene dos columnas:

ID: Identificador único compuesto por nit_enmascarado, num_oblig_orig_enmascarado y num_oblig_enmascarado concatenados por el separador #.
var_rpta_alt: Variable binaria con el valor 0 o 1 correspondiente a la predicción.
Copiar código
ID,var_rpta_alt
250631#175418#912682,1
217161#1054045#26297,0
443187#754930#325412,1
224370#328405#754753,0

Sebastian Echeverri Parra
