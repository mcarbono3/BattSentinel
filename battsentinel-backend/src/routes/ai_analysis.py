"""
BattSentinel AI Analysis Routes - Versión Mejorada 2.0
Sistema de Monitoreo de Baterías de Clase Industrial

Implementa endpoints para sistema de doble nivel:
- Nivel 1: Monitoreo Continuo (Ligero y Eficiente)
- Nivel 2: Análisis Avanzado (Profundo y Preciso)

Autor: Manus AI
Fecha: 16 de Julio, 2025
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
from src.models.battery import db, Battery, BatteryData, AnalysisResult
from src.services.ai_models import (
    FaultDetectionModel,
    HealthPredictionModel,
    XAIExplainer,
    ContinuousMonitoringEngine,
    BatteryMetadata,
    DataPreprocessor
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
    'min_data_points_level2': 20
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
        'force_refresh': data.get('force_refresh', False)
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
def analyze_battery(battery_id):
    """Realizar análisis completo de IA en una batería - Versión mejorada"""
    try:
        battery = Battery.query.get_or_404(battery_id)

        # Validar y obtener parámetros
        request_data = request.get_json() or {}
        params = validate_analysis_request(request_data)

        # Extraer metadatos de la batería
        battery_metadata = extract_battery_metadata(battery)

        # Obtener datos recientes
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=params['time_window_hours'])
        recent_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time
        ).order_by(BatteryData.timestamp.desc()).all()

        # Verificar cantidad mínima de datos
        min_data_required = (SYSTEM_CONFIG['min_data_points_level2']
                           if params['analysis_level'] == 2
                           else SYSTEM_CONFIG['min_data_points_level1'])

        if len(recent_data) < min_data_required:
            logger.info(f"Datos insuficientes ({len(recent_data)}), generando datos de ejemplo")
            sample_count = max(min_data_required, 30)
            recent_data = generate_enhanced_sample_data(battery_id, sample_count, battery_metadata)

        # Convertir a DataFrame
        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in recent_data
        ])

        # Obtener modelos del cache
        models = get_or_create_models()

        results = {}
        analysis_metadata = {
            'battery_id': battery_id,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'data_points_analyzed': len(recent_data),
            'time_window_hours': params['time_window_hours'],
            'analysis_level': params['analysis_level'],
            'has_metadata': battery_metadata is not None
        }

        # Ejecutar análisis según tipos solicitados
        if 'continuous_monitoring' in params['analysis_types'] or params['analysis_level'] == 1:
            results['continuous_monitoring'] = execute_continuous_monitoring(
                df, models['continuous_engine'], battery_metadata
            )

        if 'fault_detection' in params['analysis_types']:
            results['fault_detection'] = execute_fault_detection(
                df, models['fault_model'], params['analysis_level'], battery_metadata
            )

        if 'health_prediction' in params['analysis_types']:
            results['health_prediction'] = execute_health_prediction(
                df, models['health_model'], params['analysis_level'], battery_metadata
            )

        if 'anomaly_detection' in params['analysis_types']:
            results['anomaly_detection'] = execute_anomaly_detection(
                df, models['continuous_engine'], battery_metadata
            )

        # Agregar explicaciones si se solicitan
        if params['include_explanation']:
            results['explanations'] = generate_comprehensive_explanations(
                df, results, models['xai_explainer'], params['analysis_level']
            )

        # Guardar resultados en base de datos
        save_analysis_results(battery_id, results, params['analysis_level'])

        return jsonify({
            'success': True,
            'data': {
                **analysis_metadata,
                'results': results
            }
        })

    except Exception as e:
        logger.error(f"Error en análisis de batería {battery_id}: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'analysis_error'
        }), 500

def execute_continuous_monitoring(df: pd.DataFrame, engine: ContinuousMonitoringEngine,
                                metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
    """Ejecutar monitoreo continuo (Nivel 1)"""
    try:
        result = engine.analyze_continuous(df, metadata)
        return {
            'status': 'success',
            'analysis_type': 'continuous_monitoring',
            'timestamp': result.timestamp.isoformat(),
            'confidence': result.confidence_score, # CORRECCIÓN: Usar confidence_score
            'predictions': result.predictions,
            'explanation': result.explanation,
            'metadata': result.metadata
        }
    except Exception as e:
        logger.error(f"Error en monitoreo continuo: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'analysis_type': 'continuous_monitoring'
        }

def execute_fault_detection(df: pd.DataFrame, model: FaultDetectionModel,
                          level: int, metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
    """Ejecutar detección de fallas"""
    try:
        # CORRECCIÓN: Cambiar model.analyze a model.predict_fault
        result = model.predict_fault(df, level=level)

        # Guardar en base de datos si es exitoso
        # Se asume que el `AnalysisResult` de `src.models.battery` es el que requiere `level_of_analysis`
        # y que se hará un solo `db.session.commit()` al final del `analyze_battery` endpoint.
        # Por lo tanto, aquí solo se añade al session, no se comitea.
        analysis = AnalysisResult(
            battery_id=df['battery_id'].iloc[0] if 'battery_id' in df.columns else 0,
            analysis_type='fault_detection',
            result=json.dumps(result.get('predictions', {})),
            confidence=result.get('confidence_score', 0.0), # CORRECCIÓN: Usar confidence_score
            fault_detected=result.get('fault_detected', False),
            fault_type=result.get('fault_type'),
            severity=result.get('severity'),
            explanation=json.dumps(result.get('explanation', {})),
            model_version=f'2.0-level{level}',
            level_of_analysis=level # CORRECCIÓN: Añadir level_of_analysis
        )
        db.session.add(analysis)

        return {
            'status': 'success',
            'analysis_type': 'fault_detection',
            'level': level,
            **result
        }
    except Exception as e:
        logger.error(f"Error en detección de fallas: {str(e)}")
        # NO hacer db.session.rollback() aquí, se maneja en el endpoint principal
        return {
            'status': 'error',
            'error': str(e),
            'analysis_type': 'fault_detection'
        }

def execute_health_prediction(df: pd.DataFrame, model: HealthPredictionModel,
                            level: int, metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
    """Ejecutar predicción de salud"""
    try:
        # CORRECCIÓN: Cambiar model.analyze a model.predict_health
        result = model.predict_health(df, level=level)

        # Guardar en base de datos si es exitoso
        # Se asume que el `AnalysisResult` de `src.models.battery` es el que requiere `level_of_analysis`
        # y que se hará un solo `db.session.commit()` al final del `analyze_battery` endpoint.
        # Por lo tanto, aquí solo se añade al session, no se comitea.
        if result.get('current_soh') is not None:
            analysis = AnalysisResult(
                battery_id=df['battery_id'].iloc[0] if 'battery_id' in df.columns else 0,
                analysis_type='health_prediction',
                result=json.dumps(result.get('predictions', {})),
                confidence=result.get('confidence_score', 0.0), # CORRECCIÓN: Usar confidence_score
                rul_prediction=result.get('rul_days'),
                explanation=json.dumps(result.get('explanation', {})),
                model_version=f'2.0-level{level}',
                level_of_analysis=level # CORRECCIÓN: Añadir level_of_analysis
            )
            db.session.add(analysis)

        return {
            'status': 'success',
            'analysis_type': 'health_prediction',
            'level': level,
            **result
        }
    except Exception as e:
        logger.error(f"Error en predicción de salud: {str(e)}")
        # NO hacer db.session.rollback() aquí, se maneja en el endpoint principal
        return {
            'status': 'error',
            'error': str(e),
            'analysis_type': 'health_prediction'
        }

def execute_anomaly_detection(df: pd.DataFrame, engine: ContinuousMonitoringEngine,
                            metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
    """Ejecutar detección de anomalías específica"""
    try:
        # Usar el motor de monitoreo continuo para detección de anomalías
        result = engine.analyze_continuous(df, metadata)

        # Extraer información específica de anomalías
        predictions = result.predictions
        anomaly_info = {
            'anomalies_detected': predictions.get('issues_detected', False),
            'anomaly_count': len(predictions.get('details', {}).get('anomalies', [])),
            'anomaly_score': predictions.get('anomaly_score', 0.0),
            'anomalies': predictions.get('details', {}).get('anomalies', []),
            'analysis_timestamp': result.timestamp.isoformat(),
            'data_points_analyzed': result.metadata.get('data_points', 0)
        }

        return {
            'status': 'success',
            'analysis_type': 'anomaly_detection',
            **anomaly_info
        }
    except Exception as e:
        logger.error(f"Error en detección de anomalías: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'analysis_type': 'anomaly_detection'
        }

def generate_comprehensive_explanations(df: pd.DataFrame, results: Dict[str, Any],
                                      explainer: XAIExplainer, level: int) -> Dict[str, Any]:
    """Generar explicaciones comprensivas para todos los análisis"""
    explanations = {}

    try:
        # Explicar detección de fallas
        if 'fault_detection' in results and results['fault_detection'].get('status') == 'success':
            explanations['fault_detection'] = explainer.explain_fault_detection(
                df, results['fault_detection']
            )

        # Explicar predicción de salud
        if 'health_prediction' in results and results['health_prediction'].get('status') == 'success':
            explanations['health_prediction'] = explainer.explain_health_prediction(
                df, results['health_prediction']
            )

        # Explicación general del sistema
        explanations['system_summary'] = generate_system_summary(results, level)

    except Exception as e:
        logger.error(f"Error generando explicaciones: {str(e)}")
        explanations['error'] = str(e)

    return explanations

def generate_system_summary(results: Dict[str, Any], level: int) -> Dict[str, Any]:
    """Generar resumen general del sistema"""
    summary = {
        'analysis_level': level,
        'overall_status': 'normal',
        'priority_alerts': [],
        'recommendations': []
    }

    # Analizar resultados para determinar estado general
    critical_issues = 0
    warnings = 0

    # Revisar monitoreo continuo
    if 'continuous_monitoring' in results:
        cm_result = results['continuous_monitoring']
        if cm_result.get('predictions', {}).get('issues_detected'):
            severity = cm_result.get('predictions', {}).get('severity', 'low')
            if severity in ['high', 'critical']:
                critical_issues += 1
                summary['priority_alerts'].append({
                    'type': 'continuous_monitoring',
                    'severity': severity,
                    'message': 'Anomalías detectadas en monitoreo continuo'
                })
            else:
                warnings += 1

    # Revisar detección de fallas
    if 'fault_detection' in results:
        fd_result = results['fault_detection']
        if fd_result.get('fault_detected'):
            severity = fd_result.get('severity', 'low')
            if severity in ['high', 'critical']:
                critical_issues += 1
                summary['priority_alerts'].append({
                    'type': 'fault_detection',
                    'severity': severity,
                    'fault_type': fd_result.get('fault_type'),
                    'message': f"Falla detectada: {fd_result.get('fault_type', 'desconocida')}"
                })
            else:
                warnings += 1

    # Revisar salud de la batería
    if 'health_prediction' in results:
        hp_result = results['health_prediction']
        soh = hp_result.get('current_soh', 100)
        rul_days = hp_result.get('rul_days', 365)

        if soh < 70 or rul_days < 90:
            critical_issues += 1
            summary['priority_alerts'].append({
                'type': 'health_prediction',
                'severity': 'high',
                'soh': soh,
                'rul_days': rul_days,
                'message': f"Estado de salud crítico: SOH {soh:.1f}%, RUL {rul_days} días"
            })
        elif soh < 80 or rul_days < 180:
            warnings += 1

    # Determinar estado general
    if critical_issues > 0:
        summary['overall_status'] = 'critical'
        summary['recommendations'].append("Requiere atención inmediata")
        if level == 1:
            summary['recommendations'].append("Se recomienda análisis de Nivel 2 para diagnóstico detallado")
    elif warnings > 0:
        summary['overall_status'] = 'warning'
        summary['recommendations'].append("Monitoreo frecuente recomendado")
    else:
        summary['overall_status'] = 'normal'
        summary['recommendations'].append("Continuar con monitoreo rutinario")

    return summary

def save_analysis_results(battery_id: int, results: Dict[str, Any], level: int):
    """Guardar resultados de análisis en base de datos"""
    try:
        # Crear entrada de resumen general
        summary_analysis = AnalysisResult(
            battery_id=battery_id,
            analysis_type='comprehensive_analysis',
            result=json.dumps({
                'level': level,
                'results_summary': {k: v.get('status', 'unknown') for k, v in results.items() if isinstance(v, dict)},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }),
            confidence=calculate_overall_confidence(results),
            explanation=json.dumps(results.get('explanations', {})),
            model_version=f'2.0-level{level}',
            level_of_analysis=level # CORRECCIÓN: Añadir level_of_analysis
        )
        db.session.add(summary_analysis)

        db.session.commit()
        logger.info(f"Resultados guardados para batería {battery_id}")

    except Exception as e:
        logger.error(f"Error guardando resultados: {str(e)}")
        db.session.rollback()

def calculate_overall_confidence(results: Dict[str, Any]) -> float:
    """Calcular confianza general basada en todos los análisis"""
    confidences = []

    for analysis_type, result in results.items():
        # CORRECCIÓN: Usar 'confidence_score'
        if isinstance(result, dict) and 'confidence_score' in result:
            confidences.append(result['confidence_score'])

    if not confidences:
        return 0.5

    # Promedio ponderado (dar más peso a análisis críticos)
    weights = {
        'fault_detection': 1.5,
        'health_prediction': 1.3,
        'continuous_monitoring': 1.0,
        'anomaly_detection': 0.8
    }

    weighted_sum = 0
    total_weight = 0

    for analysis_type, result in results.items():
        # CORRECCIÓN: Usar 'confidence_score'
        if isinstance(result, dict) and 'confidence_score' in result:
            weight = weights.get(analysis_type, 1.0)
            weighted_sum += result['confidence_score'] * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.5

# Endpoints específicos mejorados

@ai_bp.route('/continuous-monitoring/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def continuous_monitoring(battery_id):
    """Endpoint específico para monitoreo continuo (Nivel 1)"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)

        # Obtener datos recientes (ventana más pequeña para monitoreo continuo)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Solo última hora
        recent_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time
        ).order_by(BatteryData.timestamp.desc()).limit(50).all()

        if len(recent_data) < 5:
            recent_data = generate_enhanced_sample_data(battery_id, 20, battery_metadata)

        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point
            for point in recent_data
        ])

        # Obtener motor de monitoreo continuo
        models = get_or_create_models()
        engine = models['continuous_engine']

        # Ejecutar análisis
        result = execute_continuous_monitoring(df, engine, battery_metadata)
        # Comitear después de cada endpoint específico, si no es parte de un análisis completo.
        # Esto podría ser opcional y depender de si se desea guardar cada análisis individualmente.
        # Por simplicidad y consistencia con los otros endpoints específicos, se añade un commit aquí.
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error guardando resultados de monitoreo continuo: {str(e)}")
            # No se devuelve un error 500 para no ocultar el resultado del análisis en sí
            # pero se registra la falla en el guardado.

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"Error en monitoreo continuo: {str(e)}")
        db.session.rollback() # Asegurar rollback en caso de error en el endpoint
        return jsonify({
            'success': False,
            'error': str(e)
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
