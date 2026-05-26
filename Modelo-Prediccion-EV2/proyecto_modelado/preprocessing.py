import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

def limpieza_base(df):
    # """
    # Filtra columnas irrelevantes, maneja nulos y codifica el target.
    # """
    df_clean = df.copy()
    
    # 1. Definimos las columnas que SÍ queremos
    columnas_a_mantener = [
        'age', 'job', 'contact', 'month', 'campaign', 'poutcome', 
        'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed', 'y'
    ]
    
    cols_presentes = [col for col in columnas_a_mantener if col in df_clean.columns]
    df_clean = df_clean[cols_presentes]
    
    # 2. Tratamiento de Nulos: Convertir 'unknown' a NaN en categóricas
    cols_categoricas = ['job', 'contact', 'month', 'poutcome']
    for col in cols_categoricas:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace('unknown', np.nan)
            
    # 3. Eliminar filas con NaN
    df_clean = df_clean.dropna()
    
    # 4. Mapear la variable objetivo 'y' a números (1 y 0)
    if 'y' in df_clean.columns:
        df_clean['y'] = df_clean['y'].map({'no': 0, 'yes': 1})
        
    return df_clean


def construir_pipeline_ml():
    # """
    # Construye el pipeline de transformaciones matemáticas (Escalado, Encoding y PCA).
    # """
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

    # Rama 3: Variables Económicas (Escalado + PCA)
    eco_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=1))  
    ])

    # Procesador maestro
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols),
            ('eco', eco_pipeline, eco_cols)
        ],
        remainder='drop'
    )
    
    return preprocesador


def preparar_todo_el_dataset(ruta_csv=None):
    # """
    # Función maestra: Lee, limpia, divide (train/test) y procesa los datos de una sola vez.
    # Devuelve todo lo necesario para los modelos y los perfilamientos.
    # """
    if ruta_csv is None:
        base_dir = os.path.dirname(__file__)
        ruta_csv = os.path.normpath(os.path.join(base_dir, '..', 'dataset', 'bank-additional-full.csv'))

    # 1. Cargar y Limpiar
    df_raw = pd.read_csv(ruta_csv, sep=';')
    df_limpio = limpieza_base(df_raw)

    # 2. Separar Features (X) y Target (y)
    X = df_limpio.drop('y', axis=1)
    y = df_limpio['y']

    # 3. Dividir en Entrenamiento y Prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Construir y aplicar el Pipeline matemático
    pipeline = construir_pipeline_ml()
    X_train_procesado = pipeline.fit_transform(X_train)
    X_test_procesado = pipeline.transform(X_test)
    
  
    return X_train_procesado, X_test_procesado, y_train, y_test, df_limpio, X_train, X_test, pipeline