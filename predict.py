import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Cargar modelo y preprocesador
model = tf.keras.models.load_model('so2_predictor.h5')
preprocessor = joblib.load('preprocessor.pkl')

# Obtener nombres de columnas categóricas para saber los valores posibles
# (se necesitan para la generación aleatoria en threads)
# Leemos el dataset original para extraer valores únicos
df_original = pd.read_csv('d3_aire01_49_1.csv')
categorical_cols = ['Entidad_federativa', 'Tipo_de_Fuente']
possible_values = {}
for col in categorical_cols:
    possible_values[col] = df_original[col].dropna().unique().tolist()

# También necesitamos estadísticas de las columnas numéricas para generar datos realistas
numerical_cols = ['CO', 'NOx', 'COV', 'PM_010', 'PM_2_5', 'NH_3']
stats = {}
for col in numerical_cols:
    # Convertir a numérico, reemplazar no numéricos por NaN
    series = pd.to_numeric(df_original[col], errors='coerce')
    stats[col] = {
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max()
    }

def predict(input_data):
    """
    input_data: diccionario con las claves:
        'Entidad_federativa': str,
        'Tipo_de_Fuente': str,
        'CO': float,
        'NOx': float,
        'COV': float,
        'PM_010': float,
        'PM_2_5': float,
        'NH_3': float
    Retorna: valor predicho de SO₂ (float)
    """
    # Convertir a DataFrame de una fila
    df_input = pd.DataFrame([input_data])
    # Preprocesar usando el pipeline guardado
    X_input = preprocessor.transform(df_input)
    # Predicción
    pred = model.predict(X_input, verbose=0)
    return float(pred[0, 0])

# Exportar funciones y datos para threads
__all__ = ['predict', 'possible_values', 'stats']