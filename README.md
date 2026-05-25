# Predicción de Emisiones de SO₂ con Red Neuronal e Hilos

Este proyecto entrena una red neuronal (TensorFlow/Keras) para predecir las emisiones de **dióxido de azufre (SO₂)** a partir de otras variables contaminantes y datos categóricos (estado y tipo de fuente). Además, utiliza múltiples hilos para generar datos aleatorios realistas y simular predicciones en tiempo real.

## 📌 Descripción

- **Dataset**: Emisiones por municipio y tipo de fuente en México (archivo `d3_aire01_49_1.csv`).
- **Variables de entrada**:
  - Numéricas: `CO`, `NOx`, `COV`, `PM_010`, `PM_2_5`, `NH_3`
  - Categóricas: `Entidad_federativa`, `Tipo_de_Fuente`
- **Variable objetivo**: `SO_2`
- **Modelo**: Red neuronal densa con 3 capas ocultas (128, 64, 32 neuronas), dropout para regularización.
- **Preprocesamiento**: Escalado estándar para numéricas y codificación one‑hot para categóricas.
- **Simulación concurrente**: Se lanzan hilos que generan valores aleatorios (basados en la media y desviación de los datos reales) y obtienen la predicción del modelo.

## ⚙️ Cómo funciona

1. **Entrenamiento** (`trainer.py`):
   - Lee el CSV, limpia datos nulos, separa características y objetivo.
   - Aplica preprocesamiento y guarda el transformador (`preprocessor.pkl`).
   - Entrena la red neuronal y guarda el modelo (`so2_predictor.h5`).

2. **Predicción** (`predict.py`):
   - Carga el modelo y el preprocesador.
   - Exporta la función `predict()` y estadísticas de las variables numéricas para generar datos sintéticos.

3. **Hilos** (`threads.py`):
   - Cada hilo genera `n` muestras aleatorias (categorías al azar, valores numéricos con distribución normal truncada).
   - Llama a `predict()` y muestra el resultado.

4. **Ejecución principal** (`main.py`):
   - Verifica si el modelo existe; si no, ejecuta `trainer.py`.
   - Luego lanza los hilos de simulación.

## 🚀 Ejecución desde cero

### Requisitos
- Python 3.8 o superior
- pip

### Pasos

1. **Clonar o descargar el proyecto** y colocar el archivo `d3_aire01_49_1.csv` en el mismo directorio.

2. **Crear y activar entorno virtual** (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows