

import pandas as pd
import numpy as np

def limpieza_base(df):
    # """
    # Filtra columnas irrelevantes, maneja nulos y codifica el target.
    # """
    df_clean = df.copy()
    
    # 1. Definimos las columnas que SÍ queremos (Fácil de modificar en el futuro)
    columnas_a_mantener = [
        'age', 'job', 'contact', 'month', 'campaign', 'poutcome', 
        'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed', 'y'
    ]
    
    # Nos aseguramos de mantener solo estas columnas (y verificamos si 'y' existe, 
    # útil por si luego pasamos datos nuevos sin la respuesta)
    cols_presentes = [col for col in columnas_a_mantener if col in df_clean.columns]
    df_clean = df_clean[cols_presentes]
    
    # 2. Tratamiento de Nulos: Convertir 'unknown' a NaN en categóricas
    cols_categoricas = ['job', 'contact', 'month', 'poutcome']
    for col in cols_categoricas:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace('unknown', np.nan)
            
    # 3. Eliminar filas con NaN (Como decidiste, ya que son muy pocas)
    df_clean = df_clean.dropna()
    
    # 4. Mapear la variable objetivo 'y' a números (1 y 0)
    if 'y' in df_clean.columns:
        df_clean['y'] = df_clean['y'].map({'no': 0, 'yes': 1})
        
    return df_clean





from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

def construir_pipeline_ml():
    # """
    # Construye el pipeline de transformaciones matemáticas (Escalado, Encoding y PCA).
    # """
    # Si quieres agregar columnas en el futuro, solo las añades a estas listas
    num_cols = ['age', 'campaign']
    cat_cols = ['job', 'contact', 'month', 'poutcome']
    eco_cols = ['emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed']

    # Rama 1: Variables Numéricas (Solo escalado)
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Rama 2: Variables Categóricas (One-Hot Encoding)
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Rama 3: Variables Económicas (Escalado + PCA para fusionarlas en 1 componente)
    eco_pipeline = Pipeline([
        ('scaler', StandardScaler()), # El PCA SIEMPRE requiere datos escalados previamente
        ('pca', PCA(n_components=1))  # Esto comprime las 4 variables en 1 "Índice de Crisis"
    ])

    # Unimos las 3 ramas en un solo procesador maestro
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
            ('eco', eco_pipeline, eco_cols)
        ],
        remainder='drop' # Si se nos cuela otra columna, la ignora automáticamente
    )
    
    return preprocesador