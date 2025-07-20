import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys # <--- ¡Añade esta línea!
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union

# Asegúrate de que las rutas de importación coincidan con tu estructura de proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from services.ai_models import FaultDetectionModel, HealthPredictionModel, DataPreprocessor, ContinuousMonitoringEngine, BatteryMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Funciones Auxiliares (pueden ser de ai_analysis.py o utilities.py) ---
def generate_enhanced_sample_data(battery_id: int, count: int = 1000, metadata: Optional[BatteryMetadata] = None) -> List[Dict]:
    """Generar datos de ejemplo mejorados con mayor realismo para entrenamiento."""
    import random

    sample_data = []
    base_time = datetime.now(timezone.utc)

    if metadata:
        base_voltage = (metadata.voltage_limits[0] + metadata.voltage_limits[1]) / 2
        max_current = metadata.charge_current_limit
        temp_range = metadata.operating_temp_range
    else:
        base_voltage = 3.7
        max_current = 10.0
        temp_range = (20, 40)

    scenarios = ['charging', 'discharging', 'idle', 'stress']
    current_scenario = random.choice(scenarios)

    for i in range(count):
        timestamp = base_time - timedelta(minutes=i * 5)

        if current_scenario == 'charging':
            voltage = base_voltage + random.uniform(0.1, 0.5)
            current = random.uniform(2.0, max_current * 0.8)
            soc = min(100, 60 + (count - i) * 0.05 + random.uniform(-5, 5))
            fault_label = 0 # Normal
        elif current_scenario == 'discharging':
            voltage = base_voltage - random.uniform(0.0, 0.3)
            current = -random.uniform(1.0, max_current * 0.6)
            soc = max(10, 80 - (count - i) * 0.03 + random.uniform(-5, 5))
            fault_label = 0 # Normal
        elif current_scenario == 'stress': # Simular una condición que podría llevar a falla
            voltage = base_voltage - random.uniform(0.5, 1.0) # Caída de voltaje
            current = random.uniform(max_current * 0.9, max_current * 1.2) # Corriente alta
            temperature = temp_range[1] + random.uniform(5, 15) # Sobrecalentamiento
            soc = max(0, 50 - (count - i) * 0.1 + random.uniform(-10, 10))
            fault_label = random.choice([2, 4, 5]) # Short circuit, overheat, thermal runaway
        else:  # idle
            voltage = base_voltage + random.uniform(-0.1, 0.1)
            current = random.uniform(-0.5, 0.5)
            soc = 70 + random.uniform(-10, 10)
            fault_label = 0 # Normal

        base_temp = (temp_range[0] + temp_range[1]) / 2
        temp_variation = abs(current) * 0.1
        temperature = base_temp + temp_variation + random.uniform(-2, 2)
        if temperature > 60: temperature = 60 # Cap at max temp

        base_soh = 90 - (i * 0.005)
        soh = max(70, base_soh + random.uniform(-2, 2))

        cycles = 500 + i + random.randint(-10, 10)
        internal_resistance = 0.05 + (100 - soh) * 0.001 + random.uniform(-0.005, 0.005)

        data_point = {
            'battery_id': battery_id,
            'timestamp': timestamp.isoformat(),
            'voltage': round(voltage, 3),
            'current': round(current, 3),
            'temperature': round(temperature, 1),
            'soc': round(soc, 1),
            'soh': round(soh, 1),
            'cycles': cycles,
            'internal_resistance': round(internal_resistance, 4),
            'scenario': current_scenario,
            'fault_label': fault_label # Añadir etiquetas de falla para entrenamiento supervisado
        }
        sample_data.append(data_point)

        if random.random() < 0.05: # Cambiar escenario más a menudo para variedad de datos
            current_scenario = random.choice(scenarios)

    return sample_data

# --- Directorio para guardar modelos ---
MODELS_DIR = 'trained_models'
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_save_models():
    logger.info("Iniciando el proceso de entrenamiento y guardado de modelos...")

    # Generar datos de entrenamiento de ejemplo (reemplaza con tus datos reales)
    # Idealmente, cargarías esto desde una base de datos o un archivo CSV grande
    # Ejemplo: training_df = pd.read_csv('path/to/your_real_training_data.csv')
    
    # Generar un dataset de ejemplo más grande para un mejor entrenamiento
    num_data_points = 5000 
    sample_data_list = generate_enhanced_sample_data(battery_id=1, count=num_data_points)
    training_df = pd.DataFrame(sample_data_list)
    
    # Asegúrate de que las columnas de características necesarias estén en el DataFrame
    # y de que 'fault_label' exista si vas a entrenar un clasificador de fallas
    if 'fault_label' not in training_df.columns:
        logger.warning("Columna 'fault_label' no encontrada en los datos de entrenamiento. El Random Forest no será entrenado.")
        training_labels = None
    else:
        training_labels = training_df['fault_label']
        
    # --- 1. Entrenamiento de FaultDetectionModel ---
    logger.info("Entrenando FaultDetectionModel...")
    fault_model = FaultDetectionModel()
    try:
        fault_model.fit(training_data=training_df, training_labels=training_labels)
        fault_model_path = os.path.join(MODELS_DIR, 'fault_detection_model.joblib')
        joblib.dump(fault_model, fault_model_path)
        logger.info(f"FaultDetectionModel entrenado y guardado en: {fault_model_path}")
    except Exception as e:
        logger.error(f"Error al entrenar FaultDetectionModel: {e}", exc_info=True)

    # --- 2. Entrenamiento de HealthPredictionModel ---
    logger.info("Entrenando HealthPredictionModel...")
    health_model = HealthPredictionModel()
    try:
        # HealthPredictionModel también necesitaría un método 'fit' similar
        # que tome datos históricos para entrenar sus modelos (DL o ML)
        # Por ahora, solo simulo la carga si es un modelo DL que se entrena con keras
        # Esto es un placeholder; necesitas implementar un método fit real en HealthPredictionModel
        # health_model.fit(training_data=training_df, training_labels=training_soh_targets)
        
        # Si HealthPredictionModel se entrena con Keras/TF, puede que necesites guardar/cargar
        # con sus métodos save/load (ej. model.save('model.h5'))
        # Para el propósito de este ejemplo, si no tiene un 'fit' que entrene el escalador,
        # asegúrate de que su inicialización sea completa si es pre-entrenado o no usa scaler.
        
        health_model_path = os.path.join(MODELS_DIR, 'health_prediction_model.joblib')
        joblib.dump(health_model, health_model_path)
        logger.info(f"HealthPredictionModel guardado (asumiendo entrenamiento interno/previo) en: {health_model_path}")
    except Exception as e:
        logger.error(f"Error al guardar HealthPredictionModel: {e}", exc_info=True)

    # El ContinuousMonitoringEngine y XAIExplainer probablemente no necesitan 'fit'
    # en el mismo sentido que FaultDetectionModel/HealthPredictionModel,
    # ya que ContinuousMonitoringEngine usa reglas o modelos muy ligeros.
    # XAIExplainer es más una utilidad.

    logger.info("Proceso de entrenamiento y guardado de modelos completado.")

if __name__ == "__main__":
    # Asegúrate de que los paths de src estén correctamente manejados
    import sys
    import os
    # Asegurarse de que el directorio raíz del proyecto esté en PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    train_and_save_models()