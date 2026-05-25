import os
import subprocess
import sys

def main():
    # Verificar si el modelo ya está entrenado
    if not os.path.exists('so2_predictor.h5') or not os.path.exists('preprocessor.pkl'):
        print("Modelo no encontrado. Ejecutando entrenamiento...")
        subprocess.run([sys.executable, 'trainer.py'], check=True)
    else:
        print("Modelo encontrado. Cargando...")

    # Ejecutar los hilos de predicción
    print("\nIniciando simulación con hilos...\n")
    import threads
    threads.run_threads(num_threads=3, predictions_per_thread=5)

if __name__ == "__main__":
    main()