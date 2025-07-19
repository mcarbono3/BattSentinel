"""
BattSentinel AI Analysis Routes - Versión Mejorada 2.0
Sistema de Monitoreo de Baterías de Clase Industrial

Implementa endpoints para sistema de doble nivel:
- Nivel 1: Monitoreo Continuo (Ligero y Eficiente)
- Nivel 2: Análisis Avanzado (Profundo y Preciso)
"""

from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Dict, List, Optional, Any
import time
from functools import wraps

from flask_cors import cross_origin

# Importaciones locales
import sys
import os

# Importar modelos mejorados
from src.models.battery import db, Battery, BatteryData, AnalysisResult as DBAnalysisResult # Renombrar para evitar conflicto
# Asumiendo que AnalysisResult en ai_models es el dataclass de src.models.schemas
from src.services.ai_models import (
    FaultDetectionModel,
    HealthPredictionModel,
    XAIExplainer,
    ContinuousMonitoringEngine,
    BatteryMetadata,
    DataPreprocessor,
    AnalysisResult as AIAnalysisResult # Renombrar el dataclass para claridad
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ai_bp = Blueprint('ai', __name__)

# Cache global para modelos y motores
MODEL_CACHE = {
    'continuous_engine': None,
    'fault_model': None,
    'health_model': None,
    'xai_explainer': None,
    'last_updated': None
}

# Configuración del sistema
SYSTEM_CONFIG = {
    'level1_cache_timeout': 300,  # 5 minutos
    'level2_cache_timeout': 3600,  # 1 hora
    'max_processing_time_level1': 1.0,  # 1 segundo
    'max_processing_time_level2': 30.0,  # 30 segundos
    'min_data_points_level1': 5,
    'min_data_points_level2': 20,
    'level2_analysis_data_window_days': 7 # Valor por defecto para Nivel 2
}

def timing_decorator(func):
    """Decorador para medir tiempo de ejecución"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Agregar tiempo de procesamiento al resultado si es un dict
        if isinstance(result, tuple) and len(result) == 2:
            response, status_code = result
            if isinstance(response.data, bytes):
                try:
                    data = json.loads(response.data.decode('utf-8'))
                    if 'data' in data:
                        data['data']['processing_time_ms'] = (end_time - start_time) * 1000
                    response.data = json.dumps(data).encode('utf-8')
                except:
                    pass

        return result
    return wrapper

def get_or_create_models():
    """Obtener o crear modelos en cache"""
    global MODEL_CACHE

    current_time = datetime.now()

    # Verificar si necesitamos actualizar el cache
    if (MODEL_CACHE['last_updated'] is None or
        (current_time - MODEL_CACHE['last_updated']).seconds > SYSTEM_CONFIG['level1_cache_timeout']):

        logger.info("Inicializando/actualizando modelos en cache")

        try:
            MODEL_CACHE['continuous_engine'] = ContinuousMonitoringEngine()
            MODEL_CACHE['fault_model'] = FaultDetectionModel()
            MODEL_CACHE['health_model'] = HealthPredictionModel()
            MODEL_CACHE['xai_explainer'] = XAIExplainer()
            MODEL_CACHE['last_updated'] = current_time

            logger.info("Modelos inicializados correctamente")
        except Exception as e:
            logger.error(f"Error inicializando modelos: {str(e)}")
            # Mantener modelos existentes si falló la actualización

    return MODEL_CACHE

def extract_battery_metadata(battery: Battery) -> Optional[BatteryMetadata]:
    """Extraer metadatos de la batería desde el modelo de base de datos"""
    try:
        # Intentar extraer metadatos si están disponibles en el modelo Battery
        # Esto asume que el modelo Battery tiene campos adicionales para metadatos
        metadata = BatteryMetadata(
            design_capacity=getattr(battery, 'design_capacity', 100.0),
            design_cycles=getattr(battery, 'design_cycles', 2000),
            voltage_limits=(getattr(battery, 'min_voltage', 3.0), getattr(battery, 'max_voltage', 4.2)),
            charge_current_limit=getattr(battery, 'charge_current_limit', 50.0),
            discharge_current_limit=getattr(battery, 'discharge_current_limit', 50.0),
            operating_temp_range=(getattr(battery, 'min_temp', -10), getattr(battery, 'max_temp', 60)),
            chemistry=getattr(battery, 'chemistry', 'Li-ion'),
            manufacturer=getattr(battery, 'manufacturer', 'Unknown'),
            model=getattr(battery, 'model', 'Generic')
        )
        return metadata
    except Exception as e:
        logger.warning(f"No se pudieron extraer metadatos de la batería: {str(e)}")
        return None

def generate_enhanced_sample_data(battery_id: int, count: int = 20, metadata: Optional[BatteryMetadata] = None) -> List[Dict]:
    """Generar datos de ejemplo mejorados con mayor realismo"""
    import random

    sample_data = []
    base_time = datetime.now(timezone.utc)

    # Parámetros base basados en metadatos si están disponibles
    if metadata:
        base_voltage = (metadata.voltage_limits[0] + metadata.voltage_limits[1]) / 2
        max_current = metadata.charge_current_limit
        temp_range = metadata.operating_temp_range
    else:
        base_voltage = 3.7
        max_current = 10.0
        temp_range = (20, 40)

    # Simular diferentes escenarios operacionales
    scenarios = ['charging', 'discharging', 'idle']
    current_scenario = random.choice(scenarios)

    for i in range(count):
        timestamp = base_time - timedelta(minutes=i * 5)  # Datos cada 5 minutos

        # Simular datos más realistas basados en escenario
        if current_scenario == 'charging':
            voltage = base_voltage + random.uniform(0.1, 0.5)
            current = random.uniform(2.0, max_current * 0.8)
            soc = min(100, 60 + (count - i) * 2 + random.uniform(-5, 5))
        elif current_scenario == 'discharging':
            voltage = base_voltage - random.uniform(0.0, 0.3)
            current = -random.uniform(1.0, max_current * 0.6)
            soc = max(10, 80 - (count - i) * 1.5 + random.uniform(-5, 5))
        else:  # idle
            voltage = base_voltage + random.uniform(-0.1, 0.1)
            current = random.uniform(-0.5, 0.5)
            soc = 70 + random.uniform(-10, 10)

        # Temperatura con variación realista
        base_temp = (temp_range[0] + temp_range[1]) / 2
        temp_variation = abs(current) * 0.1  # Calentamiento por corriente
        temperature = base_temp + temp_variation + random.uniform(-2, 2)

        # SOH con degradación gradual
        base_soh = 90 - (i * 0.1)  # Degradación lenta
        soh = max(70, base_soh + random.uniform(-2, 2))

        # Ciclos incrementales
        cycles = 500 + i + random.randint(-10, 10)

        # Resistencia interna (aumenta con degradación)
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
            'scenario': current_scenario
        }
        sample_data.append(data_point)

        # Cambiar escenario ocasionalmente
        if random.random() < 0.1:
            current_scenario = random.choice(scenarios)

    return sample_data

def validate_analysis_request(data: Dict) -> Dict[str, Any]:
    """Validar y normalizar solicitud de análisis"""
    validated = {
        'analysis_types': data.get('analysis_types', ['fault_detection', 'health_prediction']),
        'time_window_hours': max(1, min(168, data.get('time_window_hours', 24))),  # Entre 1 hora y 1 semana
        'analysis_level': data.get('analysis_level', 1),  # 1 o 2
        'include_explanation': data.get('include_explanation', True),
        'force_refresh': data.get('force_refresh', False),
        'level2_data_window_days': data.get('level2_data_window_days', SYSTEM_CONFIG.get('level2_analysis_data_window_days', 7))
    }

    # Validar tipos de análisis
    valid_types = ['fault_detection', 'health_prediction', 'anomaly_detection', 'continuous_monitoring']
    validated['analysis_types'] = [t for t in validated['analysis_types'] if t in valid_types]

    if not validated['analysis_types']:
        validated['analysis_types'] = ['fault_detection']

    # Validar nivel de análisis
    if validated['analysis_level'] not in [1, 2]:
        validated['analysis_level'] = 1

    return validated

@ai_bp.route('/analyze/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def analyze_battery(battery_id: int):
    """
    Realiza un análisis completo de IA en una batería.
    Este endpoint está diseñado para realizar un ANÁLISIS AVANZADO (Nivel 2).
    Utiliza estrictamente datos históricos reales y no permite la generación de datos de ejemplo.
    """
    try:
        battery = Battery.query.get_or_404(battery_id)

        request_data = request.get_json() or {}
        params = validate_analysis_request(request_data)

        # FORZAR EL NIVEL DE ANÁLISIS A 2 para este endpoint, según lo solicitado.
        analysis_level = 2
        logger.info(f"Iniciando análisis de Nivel {analysis_level} para batería {battery_id}.")

        battery_metadata = extract_battery_metadata(battery)

        # --- LÓGICA DE OBTENCIÓN DE DATOS PARA ANÁLISIS DE NIVEL 2 ---
        # 1. Encontrar el timestamp del último registro para esta batería.
        latest_data_point = BatteryData.query.filter_by(battery_id=battery_id) \
                                            .order_by(BatteryData.timestamp.desc()) \
                                            .first()

        if not latest_data_point:
            error_message = f"Error: No se encontraron datos históricos para la batería {battery_id}. El análisis de Nivel 2 requiere datos históricos."
            logger.error(error_message)
            return jsonify({
                'success': False,
                'error': error_message,
                'error_type': 'no_historical_data_for_level2'
            }), 400

        # Asegurar que el timestamp sea timezone-aware para evitar comparaciones erróneas
        latest_timestamp = latest_data_point.timestamp.replace(tzinfo=timezone.utc)

        # 2. Calcular la fecha de inicio de la ventana de datos para Nivel 2.
        # Usa el valor del frontend o el predeterminado de SYSTEM_CONFIG.
        data_window_days = params.get('level2_data_window_days', SYSTEM_CONFIG['level2_analysis_data_window_days'])
        cutoff_time_for_level2 = latest_timestamp - timedelta(days=data_window_days)

        logger.info(f"Nivel {analysis_level}: Consultando datos históricos desde {cutoff_time_for_level2} hasta {latest_timestamp} (últimos {data_window_days} días).")

        # 3. Consultar los datos históricos dentro de la ventana definida.
        recent_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time_for_level2,
            BatteryData.timestamp <= latest_timestamp # Asegura que no se capturen datos 'futuros'
        ).order_by(BatteryData.timestamp.asc()).all() # Ordenar ascendentemente para series de tiempo

        # 4. Validar la cantidad mínima de datos para Nivel 2.
        min_data_required_level2 = 50 # Requisito fijo para los modelos de Nivel 2

        if len(recent_data) < min_data_required_level2:
            error_message = (f"Error: Datos históricos insuficientes para análisis de Nivel {analysis_level}. "
                             f"Se requieren al menos {min_data_required_level2} puntos en los últimos {data_window_days} días, "
                             f"pero se encontraron {len(recent_data)}. NO se generarán datos de ejemplo para Nivel 2.")
            logger.error(error_message)
            return jsonify({
                'success': False,
                'error': error_message,
                'error_type': 'insufficient_historical_data_for_level2'
            }), 400

        # Convertir la lista de objetos BatteryData a un DataFrame de Pandas
        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in recent_data
        ])

        if df.empty:
            error_message = "Error: No se pudieron procesar los datos históricos en un DataFrame válido. El DataFrame resultante está vacío."
            logger.error(error_message)
            return jsonify({
                'success': False,
                'error': error_message,
                'error_type': 'dataframe_processing_error'
            }), 500
            
        # --- AÑADIR PASOS DE PREPROCESAMIENTO DE DATOS AQUÍ ---
        # 1. Convertir la columna 'timestamp' a tipo datetime
        if 'timestamp' in df.columns:
            # Asegúrate de que el formato sea el correcto o que pandas pueda inferirlo.
            # Convertir a UTC para consistencia.
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            # Eliminar filas donde el timestamp no pudo ser convertido
            df.dropna(subset=['timestamp'], inplace=True)
            df = df.sort_values(by='timestamp').reset_index(drop=True) # Asegurar orden cronológico

        # 2. Manejar valores no numéricos, NaN o infinitos en columnas relevantes
        # Excluir 'timestamp' y otras columnas no numéricas que no deben ser tratadas así
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ['battery_id', 'cycles']] # Ajusta si tienes más columnas no relevantes para este tratamiento
        
        if numeric_cols: # Asegúrate de que haya columnas numéricas para procesar
            # Reemplazar infinitos con NaN
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # Rellenar NaN con la media de la columna (o el método que sea apropiado para tu aplicación)
            # Es importante no usar una media NaN si toda la columna es NaN.
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    if pd.isna(mean_val):
                        df[col] = df[col].fillna(0) # Si toda la columna es NaN, rellena con 0
                        logger.warning(f"Columna '{col}' contenía solo NaN después de eliminar infinitos. Rellenada con 0.")
                    else:
                        df[col] = df[col].fillna(mean_val)
                        
        # Obtener los modelos de IA inicializados
        models = get_or_create_models()

        # Diccionario para almacenar los resultados de los análisis (objetos AIAnalysisResult)
        results: Dict[str, AIAnalysisResult] = {}
        analysis_metadata = {
            'battery_id': battery_id,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'data_points_analyzed': len(df),
            'analysis_level': analysis_level,
            'has_metadata': battery_metadata is not None,
            'level2_data_window_days_used': data_window_days # Indicar la ventana usada
        }
        # Para este endpoint, time_window_hours_requested no es directamente relevante ya que la ventana se basa en el último timestamp
        # pero se puede mantener si se desea para consistencia de logs.
        # analysis_metadata['time_window_hours_requested'] = params['time_window_hours']


        # Ejecutar análisis de IA para Nivel 2
        # Todos los análisis deben ser ejecutados con el 'analysis_level' forzado a 2
        # y con el DataFrame 'df' que ya contiene los datos históricos validados.

        # Aunque 'continuous_monitoring' se usa típicamente en Nivel 1, lo ejecutamos para Nivel 2 si se solicita
        # o si es parte de los tipos de análisis por defecto.
        if 'continuous_monitoring' in params['analysis_types']:
            results['continuous_monitoring'] = execute_continuous_monitoring(
                battery_id,
                df, models['continuous_engine'], battery_metadata,
                analysis_level, # Siempre 2 aquí
                params['include_explanation']
            )

        if 'fault_detection' in params['analysis_types']:
            results['fault_detection'] = execute_fault_detection(
                df, models['fault_model'], analysis_level, battery_metadata # Siempre 2 aquí
            )

        if 'health_prediction' in params['analysis_types']:
            results['health_prediction'] = execute_health_prediction(
                df, models['health_model'], analysis_level, battery_metadata # Siempre 2 aquí
            )

        if 'anomaly_detection' in params['analysis_types']:
            results['anomaly_detection'] = execute_anomaly_detection(
                df, models['continuous_engine'], battery_metadata, analysis_level # Siempre 2 aquí
            )

        # Generar explicaciones si se solicitan
        explanations_dict = {}
        if params['include_explanation']:
            try:
                explanations_dict = generate_comprehensive_explanations(
                    df, results, models['xai_explainer'], analysis_level
                )
            except Exception as e:
                logger.error(f"Error generando explicaciones para Nivel {analysis_level}: {str(e)}", exc_info=True)
                explanations_dict['error'] = f"No se pudieron generar explicaciones: {str(e)}"

        # Preparar los resultados para la respuesta JSON y guardar en DB
        # Convertir los objetos AIAnalysisResult a diccionarios
        results_output_json = {k: v.to_dict() for k, v in results.items()}
        results_output_json['explanations'] = explanations_dict

        # Guardar resultados completos en la base de datos
        save_analysis_results(battery_id, results_output_json, analysis_level)

        # Devolver la respuesta al cliente
        return jsonify({
            'success': True,
            'data': {
                **analysis_metadata,
                'results': results_output_json
            }
        })

    except Exception as e:
        logger.error(f"Error crítico en análisis de batería {battery_id} (Nivel 2): {str(e)}", exc_info=True)
        db.session.rollback() # Asegurar rollback en caso de error
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'analysis_critical_error'
        }), 500

# --- Funciones Auxiliares para Ejecutar Análisis Específicos ---
# Estas funciones envuelven las llamadas a los modelos de IA y manejan sus errores.
# Retornan objetos AIAnalysisResult.

def execute_continuous_monitoring(battery_id: int, df: pd.DataFrame, engine: ContinuousMonitoringEngine,
                                 metadata: Optional[BatteryMetadata], level: int, include_explanation: bool) -> AIAnalysisResult:
    """Ejecuta el monitoreo continuo para un DataFrame de datos."""
    try:
        result = engine.run_monitoring(df, metadata, level=level)
        return result
    except Exception as e:
        logger.error(f"Error en ejecución de monitoreo continuo (nivel {level}): {str(e)}", exc_info=True)
        return AIAnalysisResult(
            analysis_type='continuous_monitoring',
            status='error',
            error=str(e),
            predictions={},
            confidence=0.0,
            explanation=AIAnalysisResultExplanation(
                method=f"continuous_monitoring_level_{level}",
                summary=f"Error en monitoreo continuo: {str(e)}"
            ),
            metadata=AIAnalysisResultMetadata(level=level, models_used=[], processing_time_ms=0.0, data_points=len(df))
        )

def execute_fault_detection(df: pd.DataFrame, model: FaultDetectionModel,
                           level: int, metadata: Optional[BatteryMetadata]) -> AIAnalysisResult:
    """Ejecuta la detección de fallas para un DataFrame de datos."""
    try:
        result = model.predict_fault(df, level=level, battery_metadata=metadata)
        return result
    except Exception as e:
        logger.error(f"Error en ejecución de detección de fallas (nivel {level}): {str(e)}", exc_info=True)
        return AIAnalysisResult(
            analysis_type='fault_detection',
            status='error',
            error=str(e),
            predictions={},
            confidence=0.0,
            explanation=AIAnalysisResultExplanation(
                method=f"fault_detection_level_{level}",
                summary=f"Error en detección de fallas: {str(e)}"
            ),
            metadata=AIAnalysisResultMetadata(level=level, models_used=[], processing_time_ms=0.0, data_points=len(df))
        )

def execute_health_prediction(df: pd.DataFrame, model: HealthPredictionModel,
                             level: int, metadata: Optional[BatteryMetadata]) -> AIAnalysisResult:
    """Ejecuta la predicción de salud para un DataFrame de datos."""
    try:
        result = model.predict_health(df, level=level, battery_metadata=metadata)
        return result
    except Exception as e:
        logger.error(f"Error en ejecución de predicción de salud (nivel {level}): {str(e)}", exc_info=True)
        return AIAnalysisResult(
            analysis_type='health_prediction',
            status='error',
            error=str(e),
            predictions={},
            confidence=0.0,
            explanation=AIAnalysisResultExplanation(
                method=f"health_prediction_level_{level}",
                summary=f"Error en predicción de salud: {str(e)}"
            ),
            metadata=AIAnalysisResultMetadata(level=level, models_used=[], processing_time_ms=0.0, data_points=len(df))
        )

def execute_anomaly_detection(df: pd.DataFrame, engine: ContinuousMonitoringEngine,
                              metadata: Optional[BatteryMetadata], level: int) -> AIAnalysisResult:
    """Ejecuta una detección de anomalías específica, reutilizando el motor de monitoreo continuo."""
    try:
        result = engine.run_monitoring(df, metadata, level=level) # Puede que necesites un método específico en engine para AD
        return result
    except Exception as e:
        logger.error(f"Error en ejecución de detección de anomalías (nivel {level}): {str(e)}", exc_info=True)
        return AIAnalysisResult(
            analysis_type='anomaly_detection',
            status='error',
            error=str(e),
            predictions={},
            confidence=0.0,
            explanation=AIAnalysisResultExplanation(
                method=f"anomaly_detection_level_{level}",
                summary=f"Error en detección de anomalías: {str(e)}"
            ),
            metadata=AIAnalysisResultMetadata(level=level, models_used=[], processing_time_ms=0.0, data_points=len(df))
        )

# --- Funciones para Generación de Explicaciones y Resúmenes ---

def generate_comprehensive_explanations(df: pd.DataFrame, results: Dict[str, AIAnalysisResult],
                                        explainer: XAIExplainer, level: int) -> Dict[str, Any]:
    """
    Genera explicaciones comprensivas combinando resultados de diferentes modelos de IA.
    Espera objetos AIAnalysisResult para acceder a sus atributos.
    """
    explanations = {}
    try:
        fault_detection_result = results.get('fault_detection')
        fd_status = getattr(fault_detection_result, 'status', None)

        if fault_detection_result and fd_status == 'success':
            explanations['fault_detection'] = explainer.explain_fault_detection(
                df, fault_detection_result.predictions
            )
        else:
            error_msg = getattr(fault_detection_result, 'error', 'Error desconocido') if fault_detection_result else "Análisis de fallas no ejecutado o con errores."
            explanations['fault_detection'] = {"summary": f"Análisis de fallas no ejecutado o con errores: {error_msg}"}


        health_prediction_result = results.get('health_prediction')
        hp_status = getattr(health_prediction_result, 'status', None) # Default a None si no existe o es None
        
        if health_prediction_result and hp_status == 'success':
            explanations['health_prediction'] = explainer.explain_health_prediction(
                df, health_prediction_result.predictions
            )
        else:
            error_msg = getattr(health_prediction_result, 'error', 'Error desconocido') if health_prediction_result else "Análisis de salud no ejecutado o con errores."
            explanations['health_prediction'] = {"summary": f"Análisis de salud no ejecutado o con errores: {error_msg}"}

        explanations['system_summary'] = generate_system_summary(results, level)

    except Exception as e:
        logger.error(f"Error generando explicaciones generales: {str(e)}", exc_info=True)
        explanations['overall_error'] = f"Error al consolidar explicaciones: {str(e)}"

    return explanations

def generate_system_summary(results: Dict[str, AIAnalysisResult], level: int) -> Dict[str, Any]:
    """
    Genera un resumen general del estado del sistema basado en los resultados de IA.
    Accede directamente a los atributos de los objetos AIAnalysisResult.
    """
    summary = {
        'analysis_level': level,
        'overall_status': 'normal',
        'priority_alerts': [],
        'recommendations': []
    }

    critical_issues = 0
    warnings = 0

    if 'continuous_monitoring' in results:
        cm_result = results['continuous_monitoring']
        cm_status = getattr(cm_result, 'status', 'error')
        if cm_result.status == 'error':
            summary['priority_alerts'].append({
                'type': 'continuous_monitoring', 'severity': 'error', 'message': f"Error en monitoreo continuo: {getattr(cm_result, 'error', 'Error desconocido')}"
            })
        elif cm_result.predictions.get('issues_detected'):
            severity = cm_result.predictions.get('severity', 'low')
            if severity in ['high', 'critical']:
                critical_issues += 1
                summary['priority_alerts'].append({
                    'type': 'continuous_monitoring',
                    'severity': severity,
                    'message': 'Anomalías detectadas en monitoreo continuo'
                })
            else:
                warnings += 1

    if 'fault_detection' in results:
        fd_result = results['fault_detection']
        fd_status = getattr(fd_result, 'status', 'error')
        if fd_result.status == 'error':
            summary['priority_alerts'].append({
                'type': 'fault_detection', 'severity': 'error', 'message': f"Error en detección de fallas: {getattr(fd_result, 'error', 'Error desconocido')}"
            })
        elif fd_result.predictions.get('overall_fault_detected'):
            severity = fd_result.predictions.get('severity', 'low')
            if severity in ['high', 'critical']:
                critical_issues += 1
                summary['priority_alerts'].append({
                    'type': 'fault_detection',
                    'severity': severity,
                    'fault_type': fd_result.predictions.get('main_status'),
                    'message': f"Falla detectada: {fd_result.predictions.get('main_status', 'desconocida')}"
                })
            else:
                warnings += 1

    if 'health_prediction' in results:
        hp_result = results['health_prediction']
        hp_status = getattr(hp_result, 'status', 'error')
        if hp_result.status == 'error':
            summary['priority_alerts'].append({
                'type': 'health_prediction', 'severity': 'error', 'message': f"Error en predicción de salud: {getattr(hp_result, 'error', 'Error desconocido')}"
            })
        else:
            soh = hp_result.predictions.get('current_soh', 100)
            rul_days = hp_result.predictions.get('predicted_eol_days', 365)

            if soh < 70 or (isinstance(rul_days, (int, float)) and rul_days < 90):
                critical_issues += 1
                summary['priority_alerts'].append({
                    'type': 'health_prediction',
                    'severity': 'high',
                    'soh': soh,
                    'rul_days': rul_days,
                    'message': f"Estado de salud crítico: SOH {soh:.1f}%, RUL {rul_days} días"
                })
            elif soh < 80 or (isinstance(rul_days, (int, float)) and rul_days < 180):
                warnings += 1

    if critical_issues > 0:
        summary['overall_status'] = 'critical'
        summary['recommendations'].append("Requiere atención inmediata y revisión de Nivel 2.")
    elif warnings > 0:
        summary['overall_status'] = 'warning'
        summary['recommendations'].append("Monitoreo frecuente recomendado y posible revisión de Nivel 2.")
    else:
        summary['overall_status'] = 'normal'
        summary['recommendations'].append("Continuar con monitoreo rutinario. El sistema está operando dentro de los parámetros esperados.")

    return summary

# --- Funciones de Persistencia ---

def save_analysis_results(battery_id: int, results_output_json: Dict[str, Any], level: int):
    """
    Guarda los resultados del análisis en la base de datos.
    Recibe un diccionario `results_output_json` ya preparado para JSON.
    """
    try:
        # Guardar un resumen general del análisis
        summary_analysis = DBAnalysisResult(
            battery_id=battery_id,
            analysis_type='comprehensive_analysis',
            # `results_output_json` ya contiene los datos en formato de diccionario
            result=json.dumps({
                'level': level,
                'overall_status': results_output_json.get('explanations', {}).get('system_summary', {}).get('overall_status', 'unknown'),
                'results_summary': {k: v.get('status', 'unknown') for k, v in results_output_json.items() if k != 'explanations'},
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'priority_alerts': results_output_json.get('explanations', {}).get('system_summary', {}).get('priority_alerts', [])
            }),
            confidence=calculate_overall_confidence(results_output_json),
            explanation=json.dumps(results_output_json.get('explanations', {})),
            model_version=f'2.0-level{level}',
            level_of_analysis=level
        )
        db.session.add(summary_analysis)

        # Guardar cada tipo de análisis individualmente si está presente y no tiene errores de alto nivel
        for analysis_type, result_data in results_output_json.items():
            # Excluir 'explanations' ya que se guarda por separado en comprehensive_analysis
            if analysis_type in ['continuous_monitoring', 'fault_detection', 'health_prediction', 'anomaly_detection']:
                # Solo guardar si el estado es 'success' para registros detallados
                if result_data.get('status') == 'success':
                    db_analysis = DBAnalysisResult(
                        battery_id=battery_id,
                        analysis_type=analysis_type,
                        result=json.dumps(result_data.get('predictions', {})),
                        confidence=result_data.get('confidence', 0.0),
                        # Mapeo de campos específicos para la DB
                        fault_detected=result_data.get('predictions', {}).get('overall_fault_detected'),
                        fault_type=result_data.get('predictions', {}).get('main_status'),
                        severity=result_data.get('predictions', {}).get('severity'),
                        rul_prediction=result_data.get('predictions', {}).get('predicted_eol_days'),
                        explanation=json.dumps(result_data.get('explanation', {})),
                        model_version=f'2.0-level{level}',
                        level_of_analysis=level
                    )
                    db.session.add(db_analysis)
                else:
                    logger.warning(f"No se guardó el resultado detallado para '{analysis_type}' debido a su estado: {result_data.get('status')}")


        db.session.commit()
        logger.info(f"Resultados de análisis (Nivel {level}) guardados exitosamente para batería {battery_id}.")

    except Exception as e:
        logger.error(f"Error CRÍTICO guardando resultados del análisis para batería {battery_id}: {str(e)}", exc_info=True)
        db.session.rollback() # Asegura que la transacción se revierta si hay un error al guardar

def calculate_overall_confidence(results_output_json: Dict[str, Any]) -> float:
    """
    Calcula una confianza general ponderada basada en los resultados de los análisis individuales.
    Espera un diccionario con los resultados ya convertidos a dicts.
    """
    confidences = []
    # Definir pesos para cada tipo de análisis
    weights = {
        'fault_detection': 1.5,
        'health_prediction': 1.3,
        'continuous_monitoring': 1.0,
        'anomaly_detection': 0.8
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for analysis_type, result_data in results_output_json.items():
        if analysis_type in weights and isinstance(result_data, dict) and 'confidence' in result_data:
            confidence = result_data['confidence']
            weight = weights.get(analysis_type, 1.0) # Obtener el peso, default 1.0
            weighted_sum += confidence * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.5 # Default a 0.5 si no hay confianzas válidas

# Endpoints específicos mejorados

# Endpoints específicos mejorados
@ai_bp.route('/continuous-monitoring/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def continuous_monitoring(battery_id: int):
    """
    Endpoint específico para el monitoreo continuo de la batería (Nivel 1).
    Puede generar datos de ejemplo si los datos reales son insuficientes.
    """
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)

        request_data = request.get_json() or {}
        # time_window_hours del request, o un valor predeterminado si no se especifica
        time_window_hours = request_data.get('time_window_hours', 24) # Default 24 horas para monitoreo continuo

        # Obtener datos recientes para Nivel 1 (desde `now` hacia atrás)
        cutoff_time_level1 = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time_level1
        ).order_by(BatteryData.timestamp.desc()).all() # No limitar para asegurar todos los disponibles

        min_data_required_level1 = SYSTEM_CONFIG['min_data_points_level1']

        if len(recent_data) < min_data_required_level1:
            logger.info(f"Datos insuficientes para monitoreo Nivel 1 ({len(recent_data)}). Generando datos de ejemplo.")
            # Generar más datos de los mínimos para simular un stream
            recent_data = generate_enhanced_sample_data(battery_id, max(min_data_required_level1, 20), battery_metadata)

        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in recent_data
        ])

        if df.empty:
            error_message = "Error: No se pudieron procesar los datos para monitoreo continuo. El DataFrame resultante está vacío."
            logger.error(error_message)
            return jsonify({
                'success': False,
                'error': error_message,
                'error_type': 'dataframe_processing_error_monitoring'
            }), 500

        # --- AÑADIR PASOS DE PREPROCESAMIENTO DE DATOS AQUÍ (similares a analyze_battery) ---
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df = df.sort_values(by='timestamp').reset_index(drop=True)

        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in ['battery_id', 'cycles']]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    if pd.isna(mean_val):
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(mean_val)
        # ----------------------------------------------------------------------------------

        models = get_or_create_models()
        engine = models['continuous_engine']

        analysis_level = 1 # Este endpoint siempre es Nivel 1
        include_explanation = request_data.get('include_explanation', False)

        # Ejecutar el monitoreo continuo
        result_ai_analysis = execute_continuous_monitoring(
            battery_id, df, engine, battery_metadata, analysis_level, include_explanation
        )

        # Preparar la respuesta JSON
        response_data = result_ai_analysis.to_dict()

        # Intentar guardar los resultados en la base de datos
        try:
            # Dado que save_analysis_results espera un diccionario con todos los análisis,
            # lo envolvemos en un dict para que pueda ser procesado correctamente.
            # Solo guardamos el resultado específico de monitoreo aquí.
            db_analysis = DBAnalysisResult(
                battery_id=battery_id,
                analysis_type='continuous_monitoring',
                result=json.dumps(result_ai_analysis.predictions),
                confidence=result_ai_analysis.confidence,
                explanation=json.dumps(result_ai_analysis.explanation.to_dict()) if result_ai_analysis.explanation else None, # Convertir a dict
                model_version=f'2.0-level{analysis_level}',
                level_of_analysis=analysis_level
            )
            db.session.add(db_analysis)
            db.session.commit()
            logger.info(f"Resultados de monitoreo continuo (Nivel {analysis_level}) guardados para batería {battery_id}.")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error guardando resultados de monitoreo continuo para batería {battery_id}: {str(e)}", exc_info=True)
            # No se devuelve un error 500 al usuario final por un fallo de guardado,
            # el monitoreo se realizó correctamente pero la persistencia falló.

        return jsonify({
            'success': True,
            'data': response_data
        })
    except Exception as e:
        logger.error(f"Error en endpoint de monitoreo continuo para batería {battery_id}: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'continuous_monitoring_endpoint_error'
        }), 500
        
@ai_bp.route('/fault-detection/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def detect_faults(battery_id):
    """Detectar fallas específicas en una batería - Versión mejorada"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)

        # Obtener parámetros de la solicitud
        request_data = request.get_json() or {}
        analysis_level = request_data.get('analysis_level', 1)

        # Obtener datos recientes
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(100).all()
        if len(recent_data) < 10:
            recent_data = generate_enhanced_sample_data(battery_id, 30, battery_metadata)

        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in recent_data
        ])

        # Obtener modelo de detección de fallas
        models = get_or_create_models()
        fault_model = models['fault_model']

        # Ejecutar análisis (esto ya añade a la sesión pero no comitea)
        result = execute_fault_detection(df, fault_model, analysis_level, battery_metadata)

        # Agregar explicación
        if result.get('status') == 'success':
            explainer = models['xai_explainer']
            explanation = explainer.explain_fault_detection(df, result)
            result['explanation'] = explanation

        # Guardar en base de datos (Comitear la sesión)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error guardando resultados de detección de fallas: {str(e)}")
            # No se devuelve un error 500 para no ocultar el resultado del análisis en sí
            # pero se registra la falla en el guardado.

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Error en detección de fallas: {str(e)}")
        db.session.rollback() # Asegurar rollback en caso de error en el endpoint
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_bp.route('/health-prediction/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def predict_health(battery_id):
    """Predecir estado de salud y vida útil restante - Versión mejorada"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)

        # Obtener parámetros de la solicitud
        request_data = request.get_json() or {}
        analysis_level = request_data.get('analysis_level', 1)

        # Obtener datos históricos
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.asc()).all()
        if len(historical_data) < 20:
            historical_data = generate_enhanced_sample_data(battery_id, 50, battery_metadata)

        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in historical_data
        ])

        # Obtener modelo de predicción de salud
        models = get_or_create_models()
        health_model = models['health_model']

        # Ejecutar análisis (esto ya añade a la sesión pero no comitea)
        result = execute_health_prediction(df, health_model, analysis_level, battery_metadata)

        # Agregar explicación
        if result.get('status') == 'success':
            explainer = models['xai_explainer']
            explanation = explainer.explain_health_prediction(df, result)
            result['explanation'] = explanation

        # Guardar en base de datos (Comitear la sesión)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error guardando resultados de predicción de salud: {str(e)}")
            # No se devuelve un error 500 para no ocultar el resultado del análisis en sí
            # pero se registra la falla en el guardado.

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Error en predicción de salud: {str(e)}")
        db.session.rollback() # Asegurar rollback en caso de error en el endpoint
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_bp.route('/anomaly-detection/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def detect_anomalies(battery_id):
    """Detectar anomalías en tiempo real - Versión mejorada"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)

        # Obtener datos recientes para análisis de anomalías
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(50).all()

        if len(recent_data) < 10:
            recent_data = generate_enhanced_sample_data(battery_id, 30, battery_metadata)

        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in recent_data
        ])

        # Obtener motor de monitoreo continuo
        models = get_or_create_models()
        engine = models['continuous_engine']

        # Ejecutar detección de anomalías
        result = execute_anomaly_detection(df, engine, battery_metadata)

        # Guardar en base de datos
        # La detección de anomalías no se guarda directamente como un 'AnalysisResult' separado
        # en las funciones execute_X, sino que está implícita en el monitoreo continuo.
        # Si se desea un registro explícito de cada ejecución de este endpoint, se debería
        # crear un AnalysisResult aquí. Por ahora, se asume que su resultado es parte
        # del monitoreo continuo si se usa el endpoint '/analyze'.
        # Si se usa este endpoint de forma independiente, y se necesita registrar:
        # Aquí se podría añadir una lógica para guardar un AnalysisResult de tipo 'anomaly_detection'

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Error en detección de anomalías: {str(e)}")
        db.session.rollback() # Asegurar rollback en caso de error en el endpoint
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Endpoints de información y gestión

@ai_bp.route('/analyses/<int:battery_id>', methods=['GET'])
@cross_origin()
def get_analyses_history(battery_id):
    """Obtener historial de análisis de una batería - Versión mejorada"""
    try:
        battery = Battery.query.get_or_404(battery_id)

        # Parámetros de consulta
        analysis_type = request.args.get('analysis_type')
        limit = request.args.get('limit', 50, type=int)
        level = request.args.get('level', type=int)

        # Construir consulta
        query = AnalysisResult.query.filter_by(battery_id=battery_id)

        if analysis_type:
            query = query.filter_by(analysis_type=analysis_type)

        if level:
            # CORRECCIÓN: Filtrar por level_of_analysis directamente si el modelo lo tiene
            # Si model_version sigue siendo la única forma de filtrar por nivel, se mantiene
            # el 'like', pero si el campo 'level_of_analysis' ya existe y es Not Null,
            # lo ideal es filtrar directamente por él.
            # Asumo que 'level_of_analysis' ya está en el modelo y es preferible.
            query = query.filter_by(level_of_analysis=level)
        
        analyses = query.order_by(AnalysisResult.created_at.desc()).limit(limit).all()

        # Agregar estadísticas de resumen
        summary_stats = {
            'total_analyses': len(analyses),
            'fault_detections': len([a for a in analyses if a.analysis_type == 'fault_detection']),
            'health_predictions': len([a for a in analyses if a.analysis_type == 'health_prediction']),
            'comprehensive_analyses': len([a for a in analyses if a.analysis_type == 'comprehensive_analysis'])
        }

        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'summary_stats': summary_stats,
                'analyses_history': [
                    {
                        'id': a.id,
                        'analysis_type': a.analysis_type,
                        'result': json.loads(a.result) if a.result else {},
                        'confidence': a.confidence,
                        'created_at': a.created_at.isoformat(),
                        'model_version': a.model_version,
                        'level_of_analysis': a.level_of_analysis, # Incluir el nuevo campo
                        'fault_detected': a.fault_detected,
                        'fault_type': a.fault_type,
                        'severity': a.severity,
                        'rul_prediction': a.rul_prediction,
                        'explanation': json.loads(a.explanation) if a.explanation else {}
                    }
                    for a in analyses
                ]
            }
        })

    except Exception as e:
        logger.error(f"Error obteniendo historial de análisis para batería {battery_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_bp.route('/data/<int:battery_id>', methods=['GET'])
@cross_origin()
def get_battery_data(battery_id):
    """Obtener datos históricos de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Parámetros de consulta
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        limit = request.args.get('limit', 1000, type=int)

        query = BatteryData.query.filter_by(battery_id=battery_id)

        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
            query = query.filter(BatteryData.timestamp >= start_date)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str).replace(tzinfo=timezone.utc)
            query = query.filter(BatteryData.timestamp <= end_date)
        
        data_points = query.order_by(BatteryData.timestamp.desc()).limit(limit).all()

        return jsonify({
            'success': True,
            'data': [
                {
                    'id': dp.id,
                    'timestamp': dp.timestamp.isoformat(),
                    'voltage': dp.voltage,
                    'current': dp.current,
                    'temperature': dp.temperature,
                    'soc': dp.soc,
                    'soh': dp.soh,
                    'cycles': dp.cycles,
                    'internal_resistance': dp.internal_resistance
                }
                for dp in data_points
            ]
        })

    except Exception as e:
        logger.error(f"Error obteniendo datos para batería {battery_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_bp.route('/battery-metadata/<int:battery_id>', methods=['GET'])
@cross_origin()
def get_battery_metadata_endpoint(battery_id):
    """Obtener metadatos detallados de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        metadata = extract_battery_metadata(battery)

        if metadata:
            return jsonify({
                'success': True,
                'data': metadata.to_dict() # Assuming BatteryMetadata has a to_dict method
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Metadatos no disponibles para esta batería o incompletos'
            }), 404

    except Exception as e:
        logger.error(f"Error obteniendo metadatos para batería {battery_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
