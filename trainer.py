import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Cargar datos
df = pd.read_csv('d3_aire01_49_1.csv')

# Columnas a utilizar
numerical_cols = ['CO', 'NOx', 'COV', 'PM_010', 'PM_2_5', 'NH_3']
categorical_cols = ['Entidad_federativa', 'Tipo_de_Fuente']
target_col = 'SO_2'

# Limpieza: reemplazar cadenas vacías por 0 y convertir a float
for col in numerical_cols + [target_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Eliminar filas con valores infinitos o nulos restantes (si los hay)
df = df.replace([np.inf, -np.inf], 0).dropna(subset=[target_col])

# Separar X, y
X = df[numerical_cols + categorical_cols]
y = df[target_col]

# Preprocesador: numérico escalado + categórico one‑hot
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Aplicar preprocesamiento
X_preprocessed = preprocessor.fit_transform(X)

# Guardar el preprocesador completo para usar en predict.py
joblib.dump(preprocessor, 'preprocessor.pkl')

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

# Construir el modelo
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # regresión
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=1)

# Guardar modelo
model.save('so2_predictor.h5')

print("Modelo y preprocesador guardados correctamente.")