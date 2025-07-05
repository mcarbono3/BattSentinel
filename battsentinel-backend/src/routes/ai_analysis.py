# src/routes/ai_analysis.py

from flask import Blueprint, request, jsonify, current_app
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
from src.models.battery import db, Battery, BatteryData, BatteryAnalysis
# Importa tus modelos de IA y explicabilidad
from src.services.ai_models import FaultDetectionModel, HealthPredictionModel, XAIExplainer

ai_bp = Blueprint('ai', __name__)

# --- Funciones auxiliares para datos de ejemplo y lógica de soporte ---

def generate_sample_analysis_data(battery_id, count=100):
    """Generar datos de ejemplo para análisis"""
    import random
    
    sample_data = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(count):
        timestamp = base_time - timedelta(hours=i)
        
        # Simular degradación gradual
        degradation_factor = 1 - (i * 0.001)
        
        voltage = (12.0 + random.uniform(-0.3, 0.3)) * degradation_factor
        current = 2.5 + random.uniform(-1.0, 1.0)
        temperature = 25.0 + random.uniform(-5.0, 10.0)
        soc = max(20, min(100, 75 + random.uniform(-15, 15)))
        soh = max(70, min(100, (85 * degradation_factor) + random.uniform(-5, 5)))
        cycles = 150 + i
        
        data_point = {
            'battery_id': battery_id,
            'timestamp': timestamp.isoformat(),
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'cycles': cycles
        }
        sample_data.append(data_point)
    
    return sample_data

def generate_sample_analyses_history(battery_id):
    """Generar historial de análisis de ejemplo"""
    import random
    
    analyses = []
    base_time = datetime.now(timezone.utc)
    
    analysis_types = ['fault_detection', 'health_prediction', 'anomaly_detection', 'performance_analysis']
    
    for i in range(10):
        timestamp = base_time - timedelta(days=i * 3)
        analysis_type = random.choice(analysis_types)
        
        analysis = {
            'id': i + 1,
            'battery_id': battery_id,
            'analysis_type': analysis_type,
            'timestamp': timestamp.isoformat(),
            'status': 'completed',
            'results_summary': f'Análisis de {analysis_type} completado exitosamente',
            'score': random.uniform(70, 95),
            'issues_found': random.randint(0, 3)
        }
        analyses.append(analysis)
    
    return analyses

def get_performance_grade(score):
    """Obtener calificación de rendimiento"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def _detect_anomalies_statistical(df, columns):
    """
    Detectar anomalías en columnas especificadas usando Z-score e IQR.
    Retorna una lista de diccionarios con las anomalías detectadas.
    """
    anomalies = []
    
    for col in columns:
        if col in df.columns and not df[col].empty:
            values = pd.to_numeric(df[col], errors='coerce').dropna() # Asegurar que los valores sean numéricos y manejar NaN
            
            if len(values) < 2: # Necesita al menos 2 puntos para calcular STD/IQR
                continue

            # Detección basada en Z-score
            mean = values.mean()
            std_dev = values.std()
            
            if std_dev == 0: # Evitar división por cero si todos los valores son iguales
                z_scores = pd.Series(0, index=values.index)
            else:
                z_scores = (values - mean) / std_dev
            
            z_anomalies = values[np.abs(z_scores) > 3].index.tolist() # Z-score > 3 como umbral
            
            # Detección basada en IQR (Interquartile Range)
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = values[(values < lower_bound) | (values > upper_bound)].index.tolist()
            
            # Combinar y eliminar duplicados
            combined_anomalies_indices = list(set(z_anomalies + iqr_anomalies))
            
            for idx in combined_anomalies_indices:
                # Asegúrate de que el índice exista en el DataFrame original y no sea NaN
                if idx in df.index and pd.notna(df.loc[idx, col]):
                    anomaly = {
                        'index': int(idx),
                        'parameter': col,
                        'value': float(df.loc[idx, col]),
                        'timestamp': df.loc[idx, 'timestamp'] if 'timestamp' in df.columns else None,
                        'z_score': float(z_scores.loc[idx]) if idx in z_scores.index else None,
                        'method': 'statistical'
                    }
                    anomalies.append(anomaly)
    
    return anomalies

def _classify_anomaly_severity(anomalies, df):
    """Clasificar severidad de anomalías detectadas."""
    classified = []
    
    for anomaly in anomalies:
        severity = 'low'
        
        param = anomaly['parameter']
        value = anomaly['value']
        
        # Obtener valores para contexto de severidad (si la columna existe y no está vacía)
        if param in df.columns and not df[param].empty:
            values = pd.to_numeric(df[param], errors='coerce').dropna()
            if not values.empty:
                mean = values.mean()
                std_dev = values.std()
                if std_dev != 0:
                    z_score = abs((value - mean) / std_dev)
                else:
                    z_score = 0 # No hay desviación si std_dev es 0

                if param == 'temperature':
                    if z_score > 5 or value > 60 or value < -20: # Umbrales específicos para temperatura
                        severity = 'critical'
                    elif z_score > 3.5 or value > 45 or value < 0:
                        severity = 'high'
                    elif z_score > 2:
                        severity = 'medium'
                elif param == 'voltage':
                    if z_score > 4 or value < 10.0 or value > 15.0: # Umbrales específicos para voltaje
                        severity = 'critical'
                    elif z_score > 3 or value < 11.0 or value > 14.0:
                        severity = 'high'
                    elif z_score > 2:
                        severity = 'medium'
                elif param == 'current':
                    if z_score > 4 or abs(value) > 100: # Umbrales para corriente
                        severity = 'critical'
                    elif z_score > 3 or abs(value) > 50:
                        severity = 'high'
                    elif z_score > 2:
                        severity = 'medium'
                elif param == 'soh':
                    if value < 60: # SOH muy bajo
                        severity = 'critical'
                    elif value < 75:
                        severity = 'high'
                    elif value < 85:
                        severity = 'medium'
                else: # Default para otros parámetros si no hay reglas específicas
                    if z_score > 3.5:
                        severity = 'high'
                    elif z_score > 2:
                        severity = 'medium'
        
        anomaly['severity'] = severity
        classified.append(anomaly)
    
    return classified


def perform_anomaly_detection(df):
    """
    Realizar detección de anomalías usando métodos estadísticos.
    Esta función utiliza las helpers _detect_anomalies_statistical y _classify_anomaly_severity.
    """
    try:
        if df.empty:
            return {'anomalies': [], 'status': 'no_data'}
        
        # Columnas a analizar para anomalías
        columns_to_analyze = ['voltage', 'current', 'temperature', 'soh', 'soc']
        
        detected_anomalies = _detect_anomalies_statistical(df, columns_to_analyze)
        classified_anomalies = _classify_anomaly_severity(detected_anomalies, df)
        
        return {
            'anomalies': classified_anomalies[:20], # Limitar a X anomalías para la respuesta
            'total_anomalies': len(classified_anomalies),
            'status': 'completed',
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        current_app.logger.error(f"Error en perform_anomaly_detection: {e}")
        return {
            'anomalies': [],
            'status': 'error',
            'error': str(e)
        }

def perform_performance_analysis(df):
    """Realizar análisis de rendimiento básico"""
    try:
        if df.empty:
            return {'performance_metrics': {}, 'status': 'no_data'}
        
        metrics = {}
        
        # Eficiencia energética (Power)
        if 'voltage' in df.columns and 'current' in df.columns and not df[['voltage', 'current']].empty:
            # Asegúrate de que las columnas sean numéricas
            voltage_numeric = pd.to_numeric(df['voltage'], errors='coerce')
            current_numeric = pd.to_numeric(df['current'], errors='coerce')
            power = (voltage_numeric * current_numeric).dropna() # Calcula la potencia y elimina NaN
            
            if not power.empty:
                metrics['average_power'] = float(power.mean())
                # Asumiendo 50W como referencia para eficiencia, escala a 100%
                metrics['power_efficiency'] = min(100.0, float(power.mean() / 50 * 100))
            else:
                metrics['average_power'] = 0.0
                metrics['power_efficiency'] = 0.0

        # Estabilidad de voltaje
        if 'voltage' in df.columns and not df['voltage'].empty:
            voltage_numeric = pd.to_numeric(df['voltage'], errors='coerce').dropna()
            if not voltage_numeric.empty and voltage_numeric.mean() != 0:
                voltage_stability = 100 - (voltage_numeric.std() / voltage_numeric.mean() * 100)
                metrics['voltage_stability'] = max(0.0, float(voltage_stability))
            else:
                metrics['voltage_stability'] = 100.0 # Estable si no hay variacion o datos

        # Consistencia de temperatura
        if 'temperature' in df.columns and not df['temperature'].empty:
            temperature_numeric = pd.to_numeric(df['temperature'], errors='coerce').dropna()
            if not temperature_numeric.empty:
                # Asumiendo que una desviación estándar de 10°C es el 0% de consistencia
                temp_consistency = 100 - (temperature_numeric.std() / 10 * 100)
                metrics['temperature_consistency'] = max(0.0, float(temp_consistency))
            else:
                metrics['temperature_consistency'] = 100.0 # Consistente si no hay variacion o datos
        
        # Score general de rendimiento
        performance_scores = [v for v in metrics.values() if isinstance(v, (int, float))]
        overall_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        
        return {
            'performance_metrics': metrics,
            'overall_performance_score': float(overall_performance),
            'performance_grade': get_performance_grade(overall_performance),
            'status': 'completed',
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        current_app.logger.error(f"Error en perform_performance_analysis: {e}")
        return {
            'performance_metrics': {},
            'status': 'error',
            'error': str(e)
        }

# --- Rutas de la API ---

@ai_bp.route('/analyze/<int:battery_id>', methods=['POST'])
def analyze_battery(battery_id):
    """Realizar análisis completo de IA en una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        data = request.get_json() or {}
        analysis_types = data.get('analysis_types', ['fault_detection', 'health_prediction', 'anomaly_detection', 'performance_analysis'])
        time_window = data.get('time_window_hours', 24) # Últimas 24 horas por defecto
        
        # Obtener datos históricos para análisis
        # Filtrar por ventana de tiempo para eficiencia
        historical_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= (datetime.now(timezone.utc) - timedelta(hours=time_window))
        ).order_by(BatteryData.timestamp.desc()).limit(500).all() # Aumentado límite para más datos
        
        df = pd.DataFrame() # Inicializa un DataFrame vacío por si acaso
        if not historical_data or len(historical_data) < 10:
            current_app.logger.warning(f"Datos insuficientes para batería {battery_id}, generando datos de ejemplo para análisis completo.")
            df = pd.DataFrame(generate_sample_analysis_data(battery_id, 100)) # Generar un mínimo de datos
        else:
            df = pd.DataFrame([point.to_dict() for point in historical_data])

        if df.empty:
            current_app.logger.error(f"DataFrame vacío incluso después de intentar cargar/generar datos para batería {battery_id}.")
            return jsonify({
                'success': False,
                'error': 'No data available for analysis, even with sample data generation.'
            }), 400
        
        # Asegurarse de que 'timestamp' sea un tipo de dato correcto para operaciones
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').reset_index(drop=True) # Ordenar por tiempo ascendente para análisis de series

        analysis_results = {}
        
        if 'fault_detection' in analysis_types:
            try:
                fault_model = FaultDetectionModel() # Instancia tu modelo de detección de fallas
                fault_result = fault_model.analyze(df) # Llama al método de análisis de tu modelo
                analysis_results['fault_detection'] = fault_result
            except Exception as e:
                current_app.logger.error(f"Error al ejecutar FaultDetectionModel para batería {battery_id}: {e}")
                analysis_results['fault_detection'] = {'status': 'error', 'error': str(e)}
            
        if 'health_prediction' in analysis_types:
            try:
                health_model = HealthPredictionModel() # Instancia tu modelo de predicción de salud
                health_prediction_result = health_model.analyze(df) # Llama al método de análisis de tu modelo
                
                # Integrar la explicabilidad
                explainer = XAIExplainer()
                explanation = explainer.explain_health_prediction(df, health_prediction_result)
                health_prediction_result['explanation'] = explanation
                
                analysis_results['health_prediction'] = health_prediction_result
            except Exception as e:
                current_app.logger.error(f"Error al ejecutar HealthPredictionModel/XAIExplainer para batería {battery_id}: {e}")
                analysis_results['health_prediction'] = {'status': 'error', 'error': str(e)}
        
        if 'anomaly_detection' in analysis_types:
            analysis_results['anomaly_detection'] = perform_anomaly_detection(df)
            
        if 'performance_analysis' in analysis_types:
            analysis_results['performance_analysis'] = perform_performance_analysis(df)
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'analysis_types': analysis_types,
                'results': analysis_results,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'data_points_analyzed': len(df)
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error general en analyze_battery para batería {battery_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/detect-faults/<int:battery_id>', methods=['POST'])
def detect_faults(battery_id):
    """Detectar fallas en la batería usando IA - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(100).all() # Mayor límite para más datos

        df = pd.DataFrame()
        if not recent_data:
            current_app.logger.warning(f"No hay datos recientes para batería {battery_id}, generando datos de ejemplo para detección de fallas.")
            df = pd.DataFrame(generate_sample_analysis_data(battery_id, 50))
        else:
            df = pd.DataFrame([point.to_dict() for point in recent_data])
            
        if df.empty:
            return jsonify({'success': False, 'error': 'No data available for fault detection.'}), 400
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').reset_index(drop=True)

        try:
            fault_model = FaultDetectionModel()
            fault_results = fault_model.analyze(df)
        except Exception as e:
            current_app.logger.error(f"Error al ejecutar FaultDetectionModel en detect_faults para batería {battery_id}: {e}")
            fault_results = {'status': 'error', 'error': str(e)}

        return jsonify({
            'success': True,
            'data': fault_results
        })
            
    except Exception as e:
        current_app.logger.error(f"Error general en detect_faults para batería {battery_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/predict-health/<int:battery_id>', methods=['POST'])
def predict_health(battery_id):
    """Predecir salud futura de la batería - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        data = request.get_json() or {}
        prediction_horizon = data.get('prediction_horizon', 30) # días
        
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(300).all() # Mayor límite

        df = pd.DataFrame()
        if not historical_data:
            current_app.logger.warning(f"No hay datos históricos para batería {battery_id}, generando datos de ejemplo para predicción de salud.")
            df = pd.DataFrame(generate_sample_analysis_data(battery_id, 200))
        else:
            df = pd.DataFrame([point.to_dict() for point in historical_data])

        if df.empty:
            return jsonify({'success': False, 'error': 'No data available for health prediction.'}), 400
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').reset_index(drop=True)

        try:
            health_model = HealthPredictionModel()
            health_prediction = health_model.analyze(df, prediction_horizon=prediction_horizon) # Pasa el horizonte si tu modelo lo usa
            
            # Integrar la explicabilidad
            explainer = XAIExplainer()
            explanation = explainer.explain_health_prediction(df, health_prediction)
            health_prediction['explanation'] = explanation

        except Exception as e:
            current_app.logger.error(f"Error al ejecutar HealthPredictionModel/XAIExplainer en predict_health para batería {battery_id}: {e}")
            health_prediction = {'status': 'error', 'error': str(e)}
        
        return jsonify({
            'success': True,
            'data': health_prediction
        })
            
    except Exception as e:
        current_app.logger.error(f"Error general en predict_health para batería {battery_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/detect-anomalies/<int:battery_id>', methods=['POST'])
def detect_anomalies(battery_id):
    """Detectar anomalías en el comportamiento de la batería - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(150).all() # Mayor límite

        df = pd.DataFrame()
        if not recent_data:
            current_app.logger.warning(f"No hay datos recientes para batería {battery_id}, generando datos de ejemplo para detección de anomalías.")
            df = pd.DataFrame(generate_sample_analysis_data(battery_id, 100))
        else:
            df = pd.DataFrame([point.to_dict() for point in recent_data])

        if df.empty:
            return jsonify({'success': False, 'error': 'No data available for anomaly detection.'}), 400
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').reset_index(drop=True)

        anomaly_results = perform_anomaly_detection(df) # Usa la función auxiliar avanzada
        
        return jsonify({
            'success': True,
            'data': anomaly_results
        })
            
    except Exception as e:
        current_app.logger.error(f"Error general en detect_anomalies para batería {battery_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/analyses-history/<int:battery_id>', methods=['GET'])
def get_analyses_history(battery_id):
    """Obtener historial de análisis de IA - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Simular historial de análisis (esto es siempre con datos de ejemplo ya que no hay una tabla real de historial de análisis)
        analyses_history = generate_sample_analyses_history(battery_id)
        
        return jsonify({
            'success': True,
            'data': analyses_history
        })
            
    except Exception as e:
        current_app.logger.error(f"Error general en get_analyses_history para batería {battery_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
