import threading
import random
import time
import numpy as np
from predict import predict, possible_values, stats

def generate_random_input():
    """Genera un diccionario con valores aleatorios realistas."""
    # Elegir categorías al azar
    entidad = random.choice(possible_values['Entidad_federativa'])
    tipo_fuente = random.choice(possible_values['Tipo_de_Fuente'])
    
    # Generar valores numéricos con distribución normal truncada
    random_input = {'Entidad_federativa': entidad, 'Tipo_de_Fuente': tipo_fuente}
    for col, stat in stats.items():
        # Evitar valores negativos para contaminantes (mínimo 0)
        val = np.random.normal(stat['mean'], stat['std'])
        val = max(0, val)  # no negativos
        # Opcional: limitar a máximos observados
        val = min(val, stat['max'] * 1.5)  # permitir un poco más del máximo
        random_input[col] = val
    return random_input

def worker(thread_id, num_predictions=5):
    """Función ejecutada por cada hilo."""
    for i in range(num_predictions):
        data = generate_random_input()
        prediction = predict(data)
        print(f"[Hilo {thread_id}] Predicción #{i+1}: SO₂ = {prediction:.2f} (datos: {data})")
        time.sleep(1)  # simular intervalo entre predicciones

def run_threads(num_threads=3, predictions_per_thread=5):
    threads = []
    for tid in range(num_threads):
        t = threading.Thread(target=worker, args=(tid, predictions_per_thread))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    print("Todos los hilos han terminado.")

if __name__ == "__main__":
    run_threads()