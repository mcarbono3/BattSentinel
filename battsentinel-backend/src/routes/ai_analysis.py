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
from src.models.battery import db, Battery, BatteryData, AnalysisResult
from src.services.ai_models import (
    FaultDetectionModel, 
    HealthPredictionModel, 
    XAIExplainer,
    ContinuousMonitoringEngine,
    BatteryMetadata,
    DataPreprocessor
)

import inspect
logger.info(f"DEBUG: Atributos de AnalysisResult al cargar ai_analysis.py: {AnalysisResult.__dict__.keys()}")
logger.info(f"DEBUG: Archivo de AnalysisResult cargado: {inspect.getfile(AnalysisResult)}")

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
        # MODIFICACIÓN CLAVE: Usar getattr para acceder a 'confidence' de forma segura
        # Esto usa result.confidence si existe, de lo contrario, 0.0.
        # La solución robusta es que analyze_continuous en ai_models.py siempre retorne 'confidence'.
        confidence_value = getattr(result, 'confidence', 0.0) 
        
        return {
            'status': 'success',
            'analysis_type': 'continuous_monitoring',
            'timestamp': result.timestamp.isoformat(),
            'confidence_score': confidence_value, # Asignamos al campo confidence_score de la respuesta
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
        result = model.analyze(df, level=level)
        
        # Guardar en base de datos si es exitoso
        if result.get('fault_detected') is not None:
            analysis = AnalysisResult(
                # MODIFICACIÓN CLAVE: Asegurar que battery_id sea int
                battery_id=int(df['battery_id'].iloc[0]) if 'battery_id' in df.columns else 0,
                analysis_type='fault_detection',
                result=json.dumps(result.get('predictions', {})),
                confidence_score=float(result.get('confidence_score', 0.0)),
                fault_detected=result.get('fault_detected', False),
                fault_type=result.get('fault_type'),
                severity=result.get('severity'),
                explanation=json.dumps(result.get('explanation', {})),
                model_version=f'2.0-level{level}'
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
        return {
            'status': 'error',
            'error': str(e),
            'analysis_type': 'fault_detection'
        }

def execute_health_prediction(df: pd.DataFrame, model: HealthPredictionModel, level: int, metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
    """Ejecutar predicción de salud"""
    try:
        result = model.analyze(df, level=level)
        
        # Guardar en base de datos si es exitoso
        if result.get('current_soh') is not None:
            analysis = AnalysisResult(
                # MODIFICACIÓN CLAVE: Asegurar que battery_id sea int
                battery_id=int(df['battery_id'].iloc[0]) if 'battery_id' in df.columns else 0,
                analysis_type='health_prediction',
                result=json.dumps(result.get('predictions', {})),
                confidence_score=float(result.get('confidence_score', 0.0)),
                # MODIFICACIÓN CLAVE: Convertir rul_days a float para evitar numpy.int64
                rul_prediction=float(result.get('rul_days')) if result.get('rul_days') is not None else None,
                explanation=json.dumps(result.get('explanation', {})),
                model_version=f'2.0-level{level}'
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
        return {
            'status': 'error',
            'error': str(e),
            'analysis_type': 'health_prediction'
        }

def execute_anomaly_detection(df: pd.DataFrame, engine: ContinuousMonitoringEngine, metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
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

def save_analysis_results(battery_id: int, results: Dict[str, Any], analysis_level: int):
    """Guardar resultados de análisis en la base de datos"""
    try:
        # Solo guardar los resultados finales de comprehensive_analysis
        # y los específicos de fault_detection/health_prediction que ya se añaden individualmente.
        # Aquí consolidamos o creamos el registro de comprehensive_analysis
        
        comprehensive_result = results.get('comprehensive_analysis', {})
        if not comprehensive_result: # Si no se generó explícitamente, creamos uno resumido
            comprehensive_result = {
                "level": analysis_level,
                "results_summary": {
                    "continuous_monitoring": results.get('continuous_monitoring', {}).get('status', 'not_run'),
                    "fault_detection": results.get('fault_detection', {}).get('status', 'not_run'),
                    "health_prediction": results.get('health_prediction', {}).get('status', 'not_run'),
                    "anomaly_detection": results.get('anomaly_detection', {}).get('status', 'not_run'),
                    "explanations": "generated" if 'explanations' in results else "not_included"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            # Incluir un confidence score general si está disponible de algún análisis
            if 'fault_detection' in results and results['fault_detection'].get('confidence_score') is not None:
                comprehensive_result['confidence_score'] = float(results['fault_detection']['confidence_score'])
            elif 'health_prediction' in results and results['health_prediction'].get('confidence_score') is not None:
                comprehensive_result['confidence_score'] = float(results['health_prediction']['confidence_score'])
            elif 'continuous_monitoring' in results and results['continuous_monitoring'].get('confidence_score') is not None:
                comprehensive_result['confidence_score'] = float(results['continuous_monitoring']['confidence_score'])
            else:
                comprehensive_result['confidence_score'] = 0.5 # Valor por defecto si no se encuentra
                
            comprehensive_result['explanation'] = json.dumps(results.get('explanations', {}))
        
        # Siempre creamos un registro para el análisis completo
        analysis = AnalysisResult(
            battery_id=battery_id,
            analysis_type='comprehensive_analysis',
            result=json.dumps(comprehensive_result), # Guardar el resultado completo aquí
            confidence_score=float(comprehensive_result.get('confidence_score', 0.0)),
            model_version=f'2.0-level{analysis_level}',
            processing_time=None, # Se manejará a nivel de timing_decorator o se puede calcular aquí
            explanation=json.dumps(comprehensive_result.get('explanation', {})),
            level_of_analysis=analysis_level
        )
        db.session.add(analysis)
        db.session.commit()
        logger.info(f"Resultados del análisis completo para batería {battery_id} guardados exitosamente.")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error guardando resultados del análisis completo: {str(e)}")

def generate_comprehensive_explanations(df: pd.DataFrame, results: Dict[str, Any], explainer: XAIExplainer, level: int) -> Dict[str, Any]:
    """Generar explicaciones comprensivas para todos los análisis"""
    explanations = {}
    try:
        # Explicar detección de fallas
        if 'fault_detection' in results and results['fault_detection'].get('status') == 'success':
            # Asumimos que las explicaciones SHAP/LIME ya se agregaron en execute_fault_detection
            # o se generan aquí si el modelo lo permite
            fault_result = results['fault_detection']
            fault_type = fault_result.get('fault_type')
            if fault_type and explainer:
                try:
                    # Aquí asumo que fault_result ya tiene los datos necesarios para la explicación o se pueden reconstruir
                    # Esta parte puede necesitar más lógica si los modelos XAI requieren inputs específicos
                    explanation_text = explainer._generate_fault_explanation(fault_type, fault_result.get('feature_importances', {}))
                    explanations['fault_detection'] = {
                        'summary': explanation_text,
                        'details': fault_result.get('explanation', {}) # Mantener detalles originales si existen
                    }
                except Exception as ex:
                    logger.warning(f"Error generando explicación para detección de fallas: {str(ex)}")
                    explanations['fault_detection'] = {"summary": "Explicación no disponible"}

        # Explicar predicción de salud
        if 'health_prediction' in results and results['health_prediction'].get('status') == 'success':
            health_result = results['health_prediction']
            if explainer:
                try:
                    explanation_text = explainer._generate_health_explanation(health_result, health_result.get('feature_importances', {}))
                    explanations['health_prediction'] = {
                        'summary': explanation_text,
                        'details': health_result.get('explanation', {})
                    }
                except Exception as ex:
                    logger.warning(f"Error generando explicación para predicción de salud: {str(ex)}")
                    explanations['health_prediction'] = {"summary": "Explicación no disponible"}
                    
        # Generar resumen de resultados
        results_summary = results.get('comprehensive_analysis', {}).get('results_summary', {})
        strategic_recommendations = _generate_strategic_recommendations(results_summary, results)
        
        explanations['strategic_recommendations'] = strategic_recommendations

        return explanations

    except Exception as e:
        logger.error(f"Error generando explicaciones comprensivas: {str(e)}")
        return {"error": f"Error generando explicaciones: {str(e)}"}

def _generate_strategic_recommendations(results_summary: Dict[str, str], analysis_results: Dict[str, Any]) -> List[str]:
    """Generar recomendaciones estratégicas basadas en el resumen de resultados"""
    recommendations = []
    
    try:
        # Recomendaciones basadas en el estado general
        if results_summary.get('fault_detection') == 'error' or analysis_results.get('fault_detection', {}).get('fault_detected', False):
            recommendations.extend([
                "ATENCIÓN CRÍTICA: Fallas detectadas. Requiere inspección inmediata.",
                "Se recomienda análisis de Nivel 2 para diagnóstico detallado"
            ])
            if analysis_results.get('fault_detection', {}).get('severity') == 'critical':
                 recommendations.append("RIESGO DE FALLA MAYOR: Desconectar batería y realizar mantenimiento correctivo urgente.")
            
        elif results_summary.get('health_prediction') == 'error' or (analysis_results.get('health_prediction', {}).get('rul_prediction') is not None and analysis_results['health_prediction']['rul_prediction'] < 90): # RUL bajo
            recommendations.extend([
                "DEGRADACIÓN AVANZADA: Planificar reemplazo de batería a corto plazo.",
                "Considerar análisis de costo-beneficio para reemplazo temprano"
            ])
        
        else:
            recommendations.extend([
                "ESTRATEGIA DE MANTENIMIENTO: Continuar con programa regular",
                "Análisis de Nivel 2 mensual recomendado",
                "Mantener condiciones operacionales actuales"
            ])
        
        # Recomendaciones específicas basadas en análisis individuales
        if 'health_prediction' in analysis_results: # Cambiado de 'advanced_health_prediction' a 'health_prediction'
            health_result = analysis_results['health_prediction'] # Cambiado de 'advanced_health_prediction' a 'health_prediction'
            rul_days = health_result.get('rul_prediction', 365) # Usa 'rul_prediction' que es lo que guardamos
            
            if rul_days is not None: # Asegurar que rul_days no sea None
                if rul_days < 180:
                    recommendations.append(f"Planificar reemplazo en {int(rul_days)} días máximo")
                elif rul_days < 365:
                    recommendations.append(f"Considerar reemplazo en próximos {int(rul_days)} días")
        
        if 'continuous_monitoring' in analysis_results: # Aquí también puedes usar 'continuous_monitoring'
            monitoring_result = analysis_results['continuous_monitoring']
            # Asumo que 'prediction_uncertainty' o similar puede venir del monitoreo continuo si lo implementas
            uncertainty = monitoring_result.get('prediction_uncertainty', 0) # Asegúrate de que este campo exista
            
            if uncertainty > 0.15: # Ajusta el umbral de incertidumbre según tus modelos
                recommendations.append("Recopilar más datos para reducir incertidumbre en predicciones (monitoreo continuo)")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones estratégicas: {str(e)}")
        return ["Error generando recomendaciones - contactar soporte técnico"]

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
            confidence_score=calculate_overall_confidence(results),
            explanation=json.dumps(results.get('explanations', {})),
            model_version=f'2.0-level{level}'
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
    
    for i, (analysis_type, result) in enumerate(results.items()):
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
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error en monitoreo continuo: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@ai_bp.route('/fault-detection/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def detect_faults(battery_id):
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)
        
        request_data = request.get_json() or {}
        analysis_level = request_data.get('analysis_level', 1)
        
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(100).all()
        if len(recent_data) < 10:
            recent_data = generate_enhanced_sample_data(battery_id, 30, battery_metadata)
        
        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point 
            for point in recent_data
        ])
        
        models = get_or_create_models()
        fault_model = models['fault_model']
        explainer_instance = models['xai_explainer'] # Obtener la instancia del XAIExplainer
        
        result = execute_fault_detection(df, fault_model, analysis_level, battery_metadata)
        
        # --- AGREGAR EXPLICACIÓN XAI CORRECTAMENTE ---
        if result.get('status') == 'success':
            try:
                # 1. Crear el modelo dummy específico para la explicación de fallas
                # 'result' aquí es el diccionario devuelto por execute_fault_detection
                dummy_fault_model = fault_model._create_dummy_model_for_explanation(result)
                
                if dummy_fault_model:
                    # 2. Llamar al método _add_xai_explanation INYECTADO en el fault_model
                    # Este método modificará el diccionario 'result' in-place.
                    fault_model._add_xai_explanation(
                        explainer=explainer_instance,
                        # Asegúrate que el input_data sea un array NumPy con la forma correcta
                        # y solo las columnas que el modelo espera.
                        input_data=df.iloc[-1][fault_model.feature_columns].values.reshape(1, -1),
                        model_instance=dummy_fault_model, # ¡Usar el modelo dummy aquí!
                        feature_names=fault_model.feature_columns,
                        class_names=list(fault_model.fault_types.values()),
                        fault_result=result # El diccionario 'result' se actualiza con las explicaciones
                    )
                else:
                    logger.warning("No se pudo crear el modelo dummy de fallas para XAI. Las explicaciones no se generarán.")
                    result['explanation'] = {'error': 'Dummy model creation failed for XAI'}
            except Exception as e:
                logger.error(f"Error generando explicaciones XAI en detect_faults: {str(e)}")
                result['explanation'] = {'error': str(e)}
        # --- FIN AGREGAR EXPLICACIÓN XAI ---

        # Guardar en base de datos (asegúrate que tu función execute_fault_detection ya hace esto,
        # o que esta sección esté después de guardar el resultado principal si es una explicación adicional)
        # La línea `db.session.commit()` ya está en execute_fault_detection si la explicación se guarda allí.
        # Si 'explanation' se añade a 'result' y luego 'result' se usa para guardar en DB, esto es suficiente.
        try:
            db.session.commit() # Si el commit ocurre aquí, es importante que 'result' ya tenga las explicaciones
        except Exception as e:
            logger.error(f"Error en commit de DB en detect_faults: {str(e)}")
            db.session.rollback()

        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error en detección de fallas: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@ai_bp.route('/health-prediction/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def predict_health(battery_id):
    try:
        battery = Battery.query.get_or_404(battery_id)
        battery_metadata = extract_battery_metadata(battery)
        
        request_data = request.get_json() or {}
        analysis_level = request_data.get('analysis_level', 1)
        
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.asc()).all()
        if len(historical_data) < 20:
            historical_data = generate_enhanced_sample_data(battery_id, 50, battery_metadata)
        
        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point 
            for point in historical_data
        ])
        
        models = get_or_create_models()
        health_model = models['health_model']
        explainer_instance = models['xai_explainer'] # Obtener la instancia del XAIExplainer
        
        result = execute_health_prediction(df, health_model, analysis_level, battery_metadata)
        
        # --- AGREGAR EXPLICACIÓN XAI CORRECTAMENTE ---
        if result.get('status') == 'success':
            try:
                # 1. Crear el modelo dummy específico para la explicación de salud
                dummy_health_model = health_model._create_dummy_health_model_for_explanation(result)
                
                if dummy_health_model:
                    # 2. Llamar al método _add_xai_explanation INYECTADO en el health_model
                    health_model._add_xai_explanation(
                        explainer=explainer_instance,
                        input_data=df.iloc[-1][health_model.feature_columns].values.reshape(1, -1),
                        model_instance=dummy_health_model, # ¡Usar el modelo dummy aquí!
                        feature_names=health_model.feature_columns,
                        class_names=['SOH Prediction'], # Para regresión, puedes usar un nombre descriptivo
                        health_result=result # El diccionario 'result' se actualiza con las explicaciones
                    )
                else:
                    logger.warning("No se pudo crear el modelo dummy de salud para XAI. Las explicaciones no se generarán.")
                    result['explanation'] = {'error': 'Dummy model creation failed for XAI'}
            except Exception as e:
                logger.error(f"Error generando explicaciones XAI en predict_health: {str(e)}")
                result['explanation'] = {'error': str(e)}
        # --- FIN AGREGAR EXPLICACIÓN XAI ---

        # Guardar en base de datos (similar al caso de fault_detection)
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error en commit de DB en predict_health: {str(e)}")
            db.session.rollback()

        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error en predicción de salud: {str(e)}")
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
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error en detección de anomalías: {str(e)}")
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
            query = query.filter(AnalysisResult.model_version.like(f'%-level{level}'))
        
        analyses = query.order_by(AnalysisResult.created_at.desc()).limit(limit).all()
        
        # Agregar estadísticas de resumen
        summary_stats = {
            'total_analyses': len(analyses),
            'fault_detections': len([a for a in analyses if a.analysis_type == 'fault_detection']),
            'health_predictions': len([a for a in analyses if a.analysis_type == 'health_prediction']),
            'continuous_monitoring': len([a for a in analyses if a.analysis_type == 'continuous_monitoring']),
            'recent_faults': len([a for a in analyses if a.fault_detected and 
                                (datetime.now(timezone.utc) - a.created_at).days <= 7])
        }
        
        return jsonify({
            'success': True,
            'data': {
                'analyses': [analysis.to_dict() for analysis in analyses],
                'summary': summary_stats,
                'battery_id': battery_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@ai_bp.route('/model-info', methods=['GET'])
@cross_origin()
def get_model_info():
    """Obtener información sobre los modelos de IA disponibles - Versión mejorada"""
    try:
        model_info = {
            'system_version': '2.0',
            'architecture': 'dual_level',
            'level1_continuous_monitoring': {
                'name': 'Continuous Monitoring Engine',
                'version': '2.0',
                'description': 'Sistema ligero para monitoreo continuo y detección temprana',
                'techniques': ['isolation_forest', 'statistical_control', 'dynamic_thresholds'],
                'max_latency_ms': SYSTEM_CONFIG['max_processing_time_level1'] * 1000,
                'min_data_points': SYSTEM_CONFIG['min_data_points_level1'],
                'status': 'active'
            },
            'level1_fault_detection': {
                'name': 'Lightweight Fault Detection',
                'version': '2.0',
                'description': 'Detección rápida de fallas críticas y anomalías súbitas',
                'input_features': ['voltage', 'current', 'temperature', 'soc', 'soh'],
                'output_classes': ['normal', 'degradation', 'short_circuit', 'overcharge', 'overheat'],
                'confidence_threshold': 0.7,
                'status': 'active'
            },
            'level1_health_prediction': {
                'name': 'Basic Health Assessment',
                'version': '2.0',
                'description': 'Estimación rápida de SOH y RUL basada en tendencias',
                'outputs': ['current_soh', 'rul_days', 'health_status'],
                'confidence_range': [0.6, 0.8],
                'status': 'active'
            },
            'level2_advanced_analysis': {
                'name': 'Deep Learning Analysis Suite',
                'version': '2.0',
                'description': 'Análisis profundo con modelos de deep learning y XAI completo',
                'techniques': ['lstm_networks', 'autoencoders', 'gaussian_processes', 'shap_lime'],
                'max_processing_time_s': SYSTEM_CONFIG['max_processing_time_level2'],
                'min_data_points': SYSTEM_CONFIG['min_data_points_level2'],
                'status': 'development'  # Se implementará en la siguiente fase
            },
            'xai_explainability': {
                'name': 'Explainable AI Engine',
                'version': '2.0',
                'description': 'Sistema de explicabilidad con SHAP, LIME y generación de texto',
                'methods': ['feature_importance', 'rule_based', 'natural_language'],
                'languages': ['spanish', 'english'],
                'status': 'active'
            },
            'system_capabilities': {
                'real_time_monitoring': True,
                'batch_analysis': True,
                'metadata_integration': True,
                'multi_chemistry_support': True,
                'industrial_grade': True,
                'scalable_architecture': True
            }
        }
        
        return jsonify({
            'success': True,
            'data': model_info
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo información de modelos: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@ai_bp.route('/system-status', methods=['GET'])
@cross_origin()
def get_system_status():
    """Obtener estado del sistema de IA"""
    try:
        models = get_or_create_models()
        
        status = {
            'system_health': 'healthy',
            'models_loaded': {
                'continuous_engine': models['continuous_engine'] is not None,
                'fault_model': models['fault_model'] is not None,
                'health_model': models['health_model'] is not None,
                'xai_explainer': models['xai_explainer'] is not None
            },
            'cache_status': {
                'last_updated': models['last_updated'].isoformat() if models['last_updated'] else None,
                'cache_timeout_minutes': SYSTEM_CONFIG['level1_cache_timeout'] / 60
            },
            'performance_metrics': {
                'level1_max_latency_ms': SYSTEM_CONFIG['max_processing_time_level1'] * 1000,
                'level2_max_latency_s': SYSTEM_CONFIG['max_processing_time_level2']
            },
            'capabilities': {
                'tensorflow_available': 'tensorflow' in sys.modules,
                'shap_available': 'shap' in sys.modules,
                'lime_available': 'lime' in sys.modules
            }
        }
        
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del sistema: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500



# ============================================================================
# ENDPOINTS DE API DEDICADOS - IMPLEMENTACIÓN CRÍTICA
# ============================================================================

@ai_bp.route('/api/v2/level1/continuous-monitoring/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def level1_continuous_monitoring_dedicated(battery_id):
    """
    Endpoint dedicado para monitoreo continuo Nivel 1
    Optimizado para respuesta rápida con datos recientes
    Enfocado en anomalías y signos tempranos de degradación
    """
    start_time = time.time()
    
    try:
        # Validación rápida de entrada
        battery = Battery.query.get_or_404(battery_id)
        request_data = request.get_json() or {}
        
        # Parámetros específicos para Nivel 1
        time_window_minutes = request_data.get('time_window_minutes', 60)  # Última hora por defecto
        include_trends = request_data.get('include_trends', True)
        alert_threshold = request_data.get('alert_threshold', 'medium')  # low, medium, high
        
        # Extraer metadatos de batería
        battery_metadata = extract_battery_metadata(battery)
        
        # Obtener datos recientes optimizado para velocidad
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)
        recent_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time
        ).order_by(BatteryData.timestamp.desc()).limit(100).all()
        
        # Verificar cantidad mínima de datos
        if len(recent_data) < SYSTEM_CONFIG['min_data_points_level1']:
            logger.info(f"Datos insuficientes para Nivel 1 ({len(recent_data)}), generando datos sintéticos")
            recent_data = generate_enhanced_sample_data(
                battery_id, 
                max(SYSTEM_CONFIG['min_data_points_level1'], 20), 
                battery_metadata
            )
        
        # Convertir a DataFrame optimizado
        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point 
            for point in recent_data
        ])
        
        # Obtener motor de monitoreo continuo
        models = get_or_create_models()
        continuous_engine = models['continuous_engine']
        
        # Ejecutar análisis de Nivel 1 optimizado
        monitoring_result = continuous_engine.analyze_continuous(df, battery_metadata)
        
        # Análisis de tendencias si se solicita
        trend_analysis = {}
        if include_trends and len(df) > 10:
            trend_analysis = analyze_short_term_trends(df)
        
        # Evaluación de alertas basada en umbral
        alert_evaluation = evaluate_alert_level(monitoring_result, alert_threshold)
        
        # Calcular tiempo de procesamiento
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Verificar límite de tiempo para Nivel 1
        if processing_time_ms > SYSTEM_CONFIG['max_processing_time_level1'] * 1000:
            logger.warning(f"Nivel 1 excedió límite de tiempo: {processing_time_ms:.1f}ms")
        
        # Construir respuesta optimizada
        response_data = {
            'battery_id': battery_id,
            'analysis_level': 1,
            'analysis_type': 'continuous_monitoring_dedicated',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processing_time_ms': round(processing_time_ms, 2),
            'data_points_analyzed': len(df),
            'time_window_minutes': time_window_minutes,
            
            # Resultados principales
            'monitoring_result': {
                'status': 'success',
                'issues_detected': monitoring_result.predictions.get('issues_detected', False),
                'severity': monitoring_result.predictions.get('severity', 'low'),
                'confidence_score': monitoring_result.confidence,
                'anomaly_score': monitoring_result.predictions.get('anomaly_score', 0.0),
                'threshold_violations': monitoring_result.predictions.get('threshold_violations', 0),
                'control_chart_violations': monitoring_result.predictions.get('control_violations', 0)
            },
            
            # Análisis de tendencias
            'trend_analysis': trend_analysis,
            
            # Evaluación de alertas
            'alert_evaluation': alert_evaluation,
            
            # Recomendaciones inmediatas
            'immediate_actions': generate_immediate_actions(monitoring_result, alert_evaluation),
            
            # Metadatos del análisis
            'metadata': {
                'algorithm_version': '2.0-level1-dedicated',
                'has_battery_metadata': battery_metadata is not None,
                'data_quality_score': calculate_data_quality_score(df),
                'next_analysis_recommended_minutes': calculate_next_analysis_interval(monitoring_result)
            }
        }
        
        # Logging para monitoreo
        logger.info(f"Nivel 1 completado para batería {battery_id}: {processing_time_ms:.1f}ms, "
                   f"issues={response_data['monitoring_result']['issues_detected']}")
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error en monitoreo continuo Nivel 1 para batería {battery_id}: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'level1_monitoring_error',
            'processing_time_ms': round(processing_time_ms, 2),
            'battery_id': battery_id
        }), 500

@ai_bp.route('/api/v2/level2/advanced-analysis/<int:battery_id>', methods=['POST'])
@cross_origin()
@timing_decorator
def level2_advanced_analysis_dedicated(battery_id):
    """
    Endpoint dedicado para análisis avanzado Nivel 2
    Activado bajo demanda para datos históricos extensos
    Diagnósticos profundos y predicciones precisas con XAI detallado
    """
    start_time = time.time()
    
    try:
        # Validación de entrada
        battery = Battery.query.get_or_404(battery_id)
        request_data = request.get_json() or {}
        
        # Parámetros específicos para Nivel 2
        historical_days = request_data.get('historical_days', 30)  # 30 días por defecto
        analysis_types = request_data.get('analysis_types', [
            'deep_learning_fault_detection',
            'advanced_health_prediction', 
            'anomaly_detection_vae',
            'uncertainty_quantification',
            'survival_analysis'
        ])
        include_xai = request_data.get('include_xai', True)
        xai_detail_level = request_data.get('xai_detail_level', 'comprehensive')  # basic, detailed, comprehensive
        generate_report = request_data.get('generate_report', False)
        
        # Extraer metadatos de batería
        battery_metadata = extract_battery_metadata(battery)
        
        # Obtener datos históricos extensos
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=historical_days)
        historical_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time
        ).order_by(BatteryData.timestamp.asc()).all()
        
        # Verificar cantidad mínima de datos para Nivel 2
        if len(historical_data) < SYSTEM_CONFIG['min_data_points_level2']:
            logger.info(f"Datos insuficientes para Nivel 2 ({len(historical_data)}), generando datos sintéticos")
            historical_data = generate_enhanced_sample_data(
                battery_id, 
                max(SYSTEM_CONFIG['min_data_points_level2'], 100), 
                battery_metadata
            )
        
        # Convertir a DataFrame
        df = pd.DataFrame([
            point.to_dict() if hasattr(point, 'to_dict') else point 
            for point in historical_data
        ])
        
        # Obtener motores de análisis
        models = get_or_create_models()
        advanced_engine = models.get('advanced_engine')
        if not advanced_engine:
            # Crear motor avanzado si no existe
            from ai_models_improved import AdvancedAnalysisEngine
            advanced_engine = AdvancedAnalysisEngine()
            models['advanced_engine'] = advanced_engine
        
        # Ejecutar análisis avanzado
        logger.info(f"Iniciando análisis Nivel 2 para batería {battery_id} con {len(df)} puntos de datos")
        
        analysis_results = {}
        
        # 1. Deep Learning Fault Detection
        if 'deep_learning_fault_detection' in analysis_types:
            logger.info("Ejecutando detección de fallas con deep learning")
            dl_fault_result = execute_deep_learning_fault_detection(df, battery_metadata, models)
            analysis_results['deep_learning_fault_detection'] = dl_fault_result
        
        # 2. Advanced Health Prediction
        if 'advanced_health_prediction' in analysis_types:
            logger.info("Ejecutando predicción avanzada de salud")
            health_result = execute_advanced_health_prediction(df, battery_metadata, models)
            analysis_results['advanced_health_prediction'] = health_result
        
        # 3. VAE Anomaly Detection
        if 'anomaly_detection_vae' in analysis_types:
            logger.info("Ejecutando detección de anomalías con VAE")
            vae_result = execute_vae_anomaly_detection(df, battery_metadata, models)
            analysis_results['anomaly_detection_vae'] = vae_result
        
        # 4. Uncertainty Quantification
        if 'uncertainty_quantification' in analysis_types:
            logger.info("Ejecutando cuantificación de incertidumbre")
            uncertainty_result = execute_uncertainty_quantification(df, battery_metadata, models)
            analysis_results['uncertainty_quantification'] = uncertainty_result
        
        # 5. Survival Analysis
        if 'survival_analysis' in analysis_types:
            logger.info("Ejecutando análisis de supervivencia")
            survival_result = execute_survival_analysis(df, battery_metadata, models)
            analysis_results['survival_analysis'] = survival_result
        
        # Combinar resultados y generar insights
        combined_insights = combine_level2_insights(analysis_results, df, battery_metadata)
        
        # Generar explicaciones XAI si se solicita
        xai_explanations = {}
        if include_xai:
            logger.info(f"Generando explicaciones XAI nivel {xai_detail_level}")
            xai_explanations = generate_comprehensive_xai_explanations(
                df, analysis_results, models, xai_detail_level
            )
        
        # Generar reporte si se solicita
        report_data = {}
        if generate_report:
            logger.info("Generando reporte comprehensivo")
            report_data = generate_level2_report(
                battery_id, df, analysis_results, combined_insights, xai_explanations
            )
        
        # Calcular tiempo de procesamiento
        processing_time_s = time.time() - start_time
        
        # Verificar límite de tiempo para Nivel 2
        if processing_time_s > SYSTEM_CONFIG['max_processing_time_level2']:
            logger.warning(f"Nivel 2 excedió límite de tiempo: {processing_time_s:.1f}s")
        
        # Construir respuesta comprehensiva
        response_data = {
            'battery_id': battery_id,
            'analysis_level': 2,
            'analysis_type': 'advanced_analysis_dedicated',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processing_time_s': round(processing_time_s, 2),
            'data_points_analyzed': len(df),
            'historical_days': historical_days,
            'analysis_types_executed': list(analysis_results.keys()),
            
            # Resultados principales
            'analysis_results': analysis_results,
            
            # Insights combinados
            'combined_insights': combined_insights,
            
            # Explicaciones XAI
            'xai_explanations': xai_explanations if include_xai else {},
            
            # Reporte (si se generó)
            'report': report_data if generate_report else {},
            
            # Recomendaciones estratégicas
            'strategic_recommendations': generate_strategic_recommendations(
                combined_insights, analysis_results
            ),
            
            # Metadatos del análisis
            'metadata': {
                'algorithm_version': '2.0-level2-dedicated',
                'has_battery_metadata': battery_metadata is not None,
                'data_quality_score': calculate_data_quality_score(df),
                'confidence_score': calculate_overall_confidence_level2(analysis_results),
                'models_used': get_models_used_in_analysis(analysis_results),
                'next_level2_recommended_days': calculate_next_level2_interval(combined_insights)
            }
        }
        
        # Guardar resultados en base de datos
        save_level2_analysis_results(battery_id, response_data)
        
        # Logging para monitoreo
        logger.info(f"Nivel 2 completado para batería {battery_id}: {processing_time_s:.1f}s, "
                   f"análisis={len(analysis_results)}, insights={combined_insights.get('overall_status', 'unknown')}")
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        processing_time_s = time.time() - start_time
        logger.error(f"Error en análisis avanzado Nivel 2 para batería {battery_id}: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'level2_analysis_error',
            'processing_time_s': round(processing_time_s, 2),
            'battery_id': battery_id
        }), 500

# ============================================================================
# FUNCIONES DE SOPORTE PARA ENDPOINTS DEDICADOS
# ============================================================================

def analyze_short_term_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Analizar tendencias a corto plazo para Nivel 1"""
    try:
        trends = {}
        
        # Parámetros clave para análisis de tendencias
        key_params = ['voltage', 'current', 'temperature', 'soc', 'soh']
        
        for param in key_params:
            if param in df.columns and not df[param].isna().all():
                values = df[param].dropna()
                if len(values) > 5:
                    # Calcular tendencia lineal
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    # Clasificar tendencia
                    if abs(slope) < std_err:
                        trend_direction = 'stable'
                    elif slope > 0:
                        trend_direction = 'increasing'
                    else:
                        trend_direction = 'decreasing'
                    
                    trends[param] = {
                        'direction': trend_direction,
                        'slope': float(slope),
                        'r_squared': float(r_value ** 2),
                        'significance': 'significant' if p_value < 0.05 else 'not_significant',
                        'recent_value': float(values.iloc[-1]),
                        'change_rate_per_hour': float(slope * 60)  # Asumiendo datos cada minuto
                    }
        
        return {
            'trends': trends,
            'overall_stability': assess_overall_stability(trends),
            'concerning_trends': identify_concerning_trends(trends)
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de tendencias: {str(e)}")
        return {'error': str(e)}

def evaluate_alert_level(monitoring_result, alert_threshold: str) -> Dict[str, Any]:
    """Evaluar nivel de alerta basado en umbral especificado"""
    try:
        predictions = monitoring_result.predictions
        
        # Mapeo de umbrales
        threshold_configs = {
            'low': {'anomaly_threshold': 0.3, 'violation_threshold': 1},
            'medium': {'anomaly_threshold': 0.5, 'violation_threshold': 2},
            'high': {'anomaly_threshold': 0.7, 'violation_threshold': 3}
        }
        
        config = threshold_configs.get(alert_threshold, threshold_configs['medium'])
        
        # Evaluar condiciones de alerta
        anomaly_score = predictions.get('anomaly_score', 0.0)
        violations = predictions.get('threshold_violations', 0)
        
        alert_triggered = (
            anomaly_score > config['anomaly_threshold'] or 
            violations >= config['violation_threshold']
        )
        
        # Determinar severidad
        if alert_triggered:
            if anomaly_score > 0.8 or violations >= 5:
                severity = 'critical'
            elif anomaly_score > 0.6 or violations >= 3:
                severity = 'high'
            else:
                severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'alert_triggered': alert_triggered,
            'severity': severity,
            'threshold_used': alert_threshold,
            'anomaly_score': anomaly_score,
            'violations_count': violations,
            'alert_message': generate_alert_message(alert_triggered, severity, predictions)
        }
        
    except Exception as e:
        logger.error(f"Error evaluando nivel de alerta: {str(e)}")
        return {'error': str(e)}

def generate_immediate_actions(monitoring_result, alert_evaluation) -> List[str]:
    """Generar acciones inmediatas recomendadas"""
    actions = []
    
    try:
        if alert_evaluation.get('alert_triggered', False):
            severity = alert_evaluation.get('severity', 'low')
            
            if severity == 'critical':
                actions.extend([
                    "ACCIÓN INMEDIATA: Detener operación de la batería",
                    "Verificar sistemas de seguridad",
                    "Contactar personal técnico especializado",
                    "Documentar condiciones actuales"
                ])
            elif severity == 'high':
                actions.extend([
                    "Reducir carga operacional",
                    "Aumentar frecuencia de monitoreo",
                    "Verificar condiciones ambientales",
                    "Preparar para posible mantenimiento"
                ])
            elif severity == 'medium':
                actions.extend([
                    "Monitoreo continuo recomendado",
                    "Revisar tendencias en próximas horas",
                    "Verificar parámetros operacionales"
                ])
        else:
            actions.append("Continuar operación normal con monitoreo rutinario")
        
        # Acciones específicas basadas en tipo de problema
        predictions = monitoring_result.predictions
        if predictions.get('threshold_violations', 0) > 0:
            actions.append("Revisar parámetros que exceden umbrales normales")
        
        if predictions.get('anomaly_score', 0) > 0.5:
            actions.append("Investigar causa de comportamiento anómalo")
        
        return actions
        
    except Exception as e:
        logger.error(f"Error generando acciones inmediatas: {str(e)}")
        return ["Error generando recomendaciones - contactar soporte técnico"]

def calculate_data_quality_score(df: pd.DataFrame) -> float:
    """Calcular puntuación de calidad de datos"""
    try:
        if len(df) == 0:
            return 0.0
        
        # Factores de calidad
        completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Consistencia temporal (si hay timestamp)
        temporal_consistency = 1.0
        if 'timestamp' in df.columns:
            time_diffs = pd.to_datetime(df['timestamp']).diff().dropna()
            if len(time_diffs) > 1:
                cv = time_diffs.std() / time_diffs.mean()
                temporal_consistency = max(0.0, 1.0 - cv)
        
        # Rango de valores razonables
        range_validity = calculate_range_validity(df)
        
        # Puntuación combinada
        quality_score = (completeness * 0.4 + temporal_consistency * 0.3 + range_validity * 0.3)
        
        return float(min(1.0, max(0.0, quality_score)))
        
    except Exception as e:
        logger.error(f"Error calculando calidad de datos: {str(e)}")
        return 0.5

def calculate_next_analysis_interval(monitoring_result) -> int:
    """Calcular intervalo recomendado para próximo análisis (en minutos)"""
    try:
        predictions = monitoring_result.predictions
        
        # Intervalo base
        base_interval = 60  # 1 hora
        
        # Ajustar basado en severidad
        if predictions.get('issues_detected', False):
            severity = predictions.get('severity', 'low')
            if severity == 'critical':
                return 5  # 5 minutos
            elif severity == 'high':
                return 15  # 15 minutos
            elif severity == 'medium':
                return 30  # 30 minutos
        
        # Ajustar basado en anomaly score
        anomaly_score = predictions.get('anomaly_score', 0.0)
        if anomaly_score > 0.7:
            return 15
        elif anomaly_score > 0.5:
            return 30
        
        return base_interval
        
    except Exception as e:
        logger.error(f"Error calculando intervalo de análisis: {str(e)}")
        return 60

# Funciones de soporte para Nivel 2

def execute_deep_learning_fault_detection(df: pd.DataFrame, metadata, models) -> Dict[str, Any]:
    """Ejecutar detección de fallas con deep learning"""
    try:
        fault_model = models.get('fault_model')
        if not fault_model:
            return {'error': 'Modelo de detección de fallas no disponible'}
        
        # Ejecutar análisis de Nivel 2
        result = fault_model.analyze(df, level=2, battery_metadata=metadata)
        
        # Agregar detalles específicos de deep learning
        if 'predictions' in result and 'level2_details' in result['predictions']:
            dl_details = result['predictions']['level2_details']
            
            return {
                'status': 'success',
                'fault_detected': result.get('fault_detected', False),
                'fault_type': result.get('fault_type', 'normal'),
                'confidence_score': result.get('confidence_score', 0.0),
                'deep_learning_details': dl_details,
                'model_performance': {
                    'processing_time_s': result.get('analysis_details', {}).get('processing_time_s', 0),
                    'models_used': result.get('analysis_details', {}).get('models_used', [])
                }
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error en detección de fallas DL: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

def execute_advanced_health_prediction(df: pd.DataFrame, metadata, models) -> Dict[str, Any]:
    """Ejecutar predicción avanzada de salud"""
    try:
        health_model = models.get('health_model')
        if not health_model:
            return {'error': 'Modelo de predicción de salud no disponible'}
        
        # Ejecutar análisis de Nivel 2
        result = health_model.analyze(df, level=2, battery_metadata=metadata)
        
        # Agregar detalles específicos de análisis avanzado
        return {
            'status': 'success',
            'current_soh': result.get('current_soh', 0.0),
            'rul_days': float(result.get('rul_days', 0)),
            'health_status': result.get('health_status', 'unknown'),
            'degradation_rate': float(result.get('degradation_rate', 0.0)),
            'confidence_score': float(result.get('confidence_score', 0.0)),
            'advanced_predictions': result.get('predictions', {}),
            'model_performance': result.get('analysis_details', {})
        }
        
    except Exception as e:
        logger.error(f"Error en predicción de salud avanzada: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

def execute_vae_anomaly_detection(df: pd.DataFrame, metadata, models) -> Dict[str, Any]:
    """Ejecutar detección de anomalías con VAE"""
    try:
        advanced_engine = models.get('advanced_engine')
        if not advanced_engine:
            return {'error': 'Motor de análisis avanzado no disponible'}
        
        # Ejecutar detección de anomalías específica
        result = advanced_engine._autoencoder_anomaly_detection(df)
        
        return {
            'status': 'success' if 'error' not in result else 'failed',
            **result
        }
        
    except Exception as e:
        logger.error(f"Error en detección de anomalías VAE: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

def execute_uncertainty_quantification(df: pd.DataFrame, metadata, models) -> Dict[str, Any]:
    """Ejecutar cuantificación de incertidumbre"""
    try:
        advanced_engine = models.get('advanced_engine')
        if not advanced_engine:
            return {'error': 'Motor de análisis avanzado no disponible'}
        
        # Ejecutar análisis de incertidumbre
        result = advanced_engine._gaussian_process_prediction(df)
        
        return {
            'status': 'success' if 'error' not in result else 'failed',
            **result
        }
        
    except Exception as e:
        logger.error(f"Error en cuantificación de incertidumbre: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

def execute_survival_analysis(df: pd.DataFrame, metadata, models) -> Dict[str, Any]:
    """Ejecutar análisis de supervivencia"""
    try:
        advanced_engine = models.get('advanced_engine')
        if not advanced_engine:
            return {'error': 'Motor de análisis avanzado no disponible'}
        
        # Ejecutar análisis de supervivencia
        result = advanced_engine._survival_analysis(df, metadata)
        
        return {
            'status': 'success' if 'error' not in result else 'failed',
            **result
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de supervivencia: {str(e)}")
        return {'error': str(e), 'status': 'failed'}

def combine_level2_insights(analysis_results: Dict[str, Any], df: pd.DataFrame, metadata) -> Dict[str, Any]:
    """Combinar insights de todos los análisis de Nivel 2"""
    try:
        insights = {
            'overall_status': 'normal',
            'critical_findings': [],
            'risk_assessment': {},
            'predictive_insights': {},
            'operational_recommendations': []
        }
        
        # Analizar resultados de cada análisis
        critical_count = 0
        warning_count = 0
        
        for analysis_type, result in analysis_results.items():
            if result.get('status') == 'success':
                # Evaluar criticidad de cada resultado
                if analysis_type == 'deep_learning_fault_detection':
                    if result.get('fault_detected', False):
                        critical_count += 1
                        insights['critical_findings'].append({
                            'type': 'fault_detection',
                            'severity': 'high',
                            'description': f"Falla detectada: {result.get('fault_type', 'unknown')}"
                        })
                
                elif analysis_type == 'advanced_health_prediction':
                    soh = result.get('current_soh', 100)
                    rul = result.get('rul_days', 365)
                    
                    if soh < 70 or rul < 90:
                        critical_count += 1
                        insights['critical_findings'].append({
                            'type': 'health_critical',
                            'severity': 'high',
                            'description': f"SOH crítico: {soh:.1f}%, RUL: {rul} días"
                        })
                    elif soh < 80 or rul < 180:
                        warning_count += 1
                
                elif analysis_type == 'anomaly_detection_vae':
                    if result.get('anomalies_detected', False):
                        anomaly_count = result.get('anomaly_count', 0)
                        if anomaly_count > 5:
                            critical_count += 1
                        else:
                            warning_count += 1
        
        # Determinar estado general
        if critical_count > 0:
            insights['overall_status'] = 'critical'
        elif warning_count > 0:
            insights['overall_status'] = 'warning'
        else:
            insights['overall_status'] = 'normal'
        
        # Generar evaluación de riesgo
        insights['risk_assessment'] = generate_risk_assessment(analysis_results, critical_count, warning_count)
        
        # Generar insights predictivos
        insights['predictive_insights'] = generate_predictive_insights(analysis_results)
        
        # Generar recomendaciones operacionales
        insights['operational_recommendations'] = generate_operational_recommendations(
            insights['overall_status'], analysis_results
        )
        
        return insights
        
    except Exception as e:
        logger.error(f"Error combinando insights de Nivel 2: {str(e)}")
        return {'error': str(e)}

def generate_comprehensive_xai_explanations(df: pd.DataFrame, analysis_results: Dict[str, Any], 
                                          models: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
    """Generar explicaciones XAI comprehensivas"""
    try:
        xai_explainer = models.get('xai_explainer')
        if not xai_explainer:
            return {'error': 'Explicador XAI no disponible'}
        
        explanations = {
            'detail_level': detail_level,
            'explanations_by_analysis': {},
            'global_explanations': {},
            'feature_importance_summary': {},
            'natural_language_summary': {}
        }
        
        # Generar explicaciones para cada análisis
        for analysis_type, result in analysis_results.items():
            if result.get('status') == 'success':
                try:
                    if analysis_type == 'deep_learning_fault_detection':
                        explanation = xai_explainer.explain_fault_detection(df, result)
                        explanations['explanations_by_analysis']['fault_detection'] = explanation
                    
                    elif analysis_type == 'advanced_health_prediction':
                        explanation = xai_explainer.explain_health_prediction(df, result)
                        explanations['explanations_by_analysis']['health_prediction'] = explanation
                
                except Exception as e:
                    logger.warning(f"Error generando explicación para {analysis_type}: {str(e)}")
        
        # Generar explicaciones globales si hay explicador avanzado
        advanced_explainer = models.get('advanced_explainer')
        if advanced_explainer and detail_level in ['detailed', 'comprehensive']:
            try:
                # Crear mock AnalysisResult para el explicador avanzado
                mock_result = type('MockResult', (), {
                    'predictions': analysis_results,
                    'confidence_score': calculate_overall_confidence_level2(analysis_results),
                    'metadata': {'models_used': get_models_used_in_analysis(analysis_results)}
                })()
                
                global_explanations = advanced_explainer.explain_advanced_analysis(df, mock_result, models)
                explanations['global_explanations'] = global_explanations
                
            except Exception as e:
                logger.warning(f"Error generando explicaciones globales: {str(e)}")
        
        # Generar resumen de importancia de características
        explanations['feature_importance_summary'] = generate_feature_importance_summary(
            explanations['explanations_by_analysis']
        )
        
        # Generar resumen en lenguaje natural
        explanations['natural_language_summary'] = generate_natural_language_summary(
            analysis_results, explanations, detail_level
        )
        
        return explanations
        
    except Exception as e:
        logger.error(f"Error generando explicaciones XAI: {str(e)}")
        return {'error': str(e)}

def save_level2_analysis_results(battery_id: int, response_data: Dict[str, Any]):
    """Guardar resultados de análisis de Nivel 2 en base de datos"""
    try:
        # Crear entrada de análisis comprehensivo
        analysis = AnalysisResult(
            battery_id=int(battery_id),
            analysis_type='level2_comprehensive',
            result=json.dumps({
                'analysis_types': response_data.get('analysis_types_executed', []),
                'combined_insights': response_data.get('combined_insights', {}),
                'processing_time_s': response_data.get('processing_time_s', 0)
            }),
            confidence_score=response_data.get('metadata', {}).get('confidence_score', 0.0),
            explanation=json.dumps(response_data.get('xai_explanations', {})),
            model_version='2.0-level2-dedicated'
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        logger.info(f"Resultados de Nivel 2 guardados para batería {battery_id}")
        
    except Exception as e:
        logger.error(f"Error guardando resultados de Nivel 2: {str(e)}")
        db.session.rollback()

# Funciones auxiliares adicionales

def assess_overall_stability(trends: Dict[str, Any]) -> str:
    """Evaluar estabilidad general basada en tendencias"""
    if not trends:
        return 'unknown'
    
    concerning_count = 0
    total_count = len(trends)
    
    for param, trend_data in trends.items():
        if trend_data['direction'] != 'stable' and trend_data['significance'] == 'significant':
            concerning_count += 1
    
    if concerning_count == 0:
        return 'stable'
    elif concerning_count / total_count < 0.3:
        return 'mostly_stable'
    elif concerning_count / total_count < 0.7:
        return 'moderately_unstable'
    else:
        return 'unstable'

def identify_concerning_trends(trends: Dict[str, Any]) -> List[str]:
    """Identificar tendencias preocupantes"""
    concerning = []
    
    for param, trend_data in trends.items():
        if trend_data['significance'] == 'significant':
            if param == 'soh' and trend_data['direction'] == 'decreasing':
                concerning.append(f"SOH en declive significativo: {trend_data['slope']:.3f}/hora")
            elif param == 'temperature' and trend_data['direction'] == 'increasing':
                concerning.append(f"Temperatura en aumento: {trend_data['change_rate_per_hour']:.2f}°C/hora")
            elif param == 'voltage' and abs(trend_data['slope']) > 0.01:
                concerning.append(f"Voltaje inestable: cambio de {trend_data['change_rate_per_hour']:.3f}V/hora")
    
    return concerning

def generate_alert_message(alert_triggered: bool, severity: str, predictions: Dict[str, Any]) -> str:
    """Generar mensaje de alerta"""
    if not alert_triggered:
        return "Operación normal - No se requiere acción inmediata"
    
    messages = {
        'critical': "ALERTA CRÍTICA: Requiere atención inmediata",
        'high': "ALERTA ALTA: Monitoreo cercano requerido", 
        'medium': "ALERTA MEDIA: Revisar condiciones operacionales"
    }
    
    base_message = messages.get(severity, "Alerta activada")
    
    # Agregar detalles específicos
    details = []
    if predictions.get('threshold_violations', 0) > 0:
        details.append(f"{predictions['threshold_violations']} violaciones de umbral")
    
    anomaly_score = predictions.get('anomaly_score', 0.0)
    if anomaly_score > 0.5:
        details.append(f"Puntuación de anomalía: {anomaly_score:.2f}")
    
    if details:
        return f"{base_message} - {', '.join(details)}"
    
    return base_message

def calculate_range_validity(df: pd.DataFrame) -> float:
    """Calcular validez de rangos de valores"""
    try:
        validity_scores = []
        
        # Rangos esperados para parámetros de baterías
        expected_ranges = {
            'voltage': (2.5, 4.5),
            'current': (-100, 100),
            'temperature': (-20, 80),
            'soc': (0, 100),
            'soh': (0, 100)
        }
        
        for param, (min_val, max_val) in expected_ranges.items():
            if param in df.columns:
                values = df[param].dropna()
                if len(values) > 0:
                    valid_count = ((values >= min_val) & (values <= max_val)).sum()
                    validity_scores.append(valid_count / len(values))
        
        return float(np.mean(validity_scores)) if validity_scores else 1.0
        
    except Exception as e:
        logger.error(f"Error calculando validez de rangos: {str(e)}")
        return 0.5

def calculate_overall_confidence_level2(analysis_results: Dict[str, Any]) -> float:
    """Calcular confianza general para análisis de Nivel 2"""
    try:
        confidences = []
        
        for analysis_type, result in analysis_results.items():
            if result.get('status') == 'success' and 'confidence_score' in result:
                confidences.append(result['confidence_score'])
        
        if not confidences:
            return 0.5
        
        # Promedio ponderado con pesos específicos
        weights = {
            'deep_learning_fault_detection': 1.5,
            'advanced_health_prediction': 1.3,
            'uncertainty_quantification': 1.2,
            'survival_analysis': 1.0,
            'anomaly_detection_vae': 0.8
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for i, (analysis_type, result) in enumerate(analysis_results.items()):
            if result.get('status') == 'success' and 'confidence_score' in result:
                weight = weights.get(analysis_type, 1.0)
                weighted_sum += result['confidence_score'] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
        
    except Exception as e:
        logger.error(f"Error calculando confianza general: {str(e)}")
        return 0.5

def get_models_used_in_analysis(analysis_results: Dict[str, Any]) -> List[str]:
    """Obtener lista de modelos utilizados en el análisis"""
    models_used = set()
    
    for analysis_type, result in analysis_results.items():
        if result.get('status') == 'success':
            if 'model_performance' in result and 'models_used' in result['model_performance']:
                models_used.update(result['model_performance']['models_used'])
            elif 'method' in result:
                models_used.add(result['method'])
    
    return list(models_used)

def generate_strategic_recommendations(combined_insights: Dict[str, Any], 
                                     analysis_results: Dict[str, Any]) -> List[str]:
    """Generar recomendaciones estratégicas basadas en análisis comprehensivo"""
    recommendations = []
    
    try:
        overall_status = combined_insights.get('overall_status', 'normal')
        
        if overall_status == 'critical':
            recommendations.extend([
                "ESTRATEGIA CRÍTICA: Planificar reemplazo inmediato de la batería",
                "Implementar monitoreo continuo cada 5-15 minutos",
                "Reducir carga operacional al mínimo necesario",
                "Preparar batería de respaldo para transición",
                "Documentar todas las condiciones para análisis post-mortem"
            ])
        
        elif overall_status == 'warning':
            recommendations.extend([
                "ESTRATEGIA PREVENTIVA: Aumentar frecuencia de análisis de Nivel 2",
                "Planificar mantenimiento preventivo en próximas 2-4 semanas",
                "Optimizar condiciones operacionales para reducir estrés",
                "Considerar análisis de costo-beneficio para reemplazo temprano"
            ])
        
        else:
            recommendations.extend([
                "ESTRATEGIA DE MANTENIMIENTO: Continuar con programa regular",
                "Análisis de Nivel 2 mensual recomendado",
                "Mantener condiciones operacionales actuales"
            ])
        
        # Recomendaciones específicas basadas en análisis individuales
        if 'advanced_health_prediction' in analysis_results:
            health_result = analysis_results['advanced_health_prediction']
            rul_days = health_result.get('rul_days', 365)
            
            if rul_days < 180:
                recommendations.append(f"Planificar reemplazo en {rul_days} días máximo")
            elif rul_days < 365:
                recommendations.append(f"Considerar reemplazo en próximos {rul_days} días")
        
        if 'uncertainty_quantification' in analysis_results:
            uncertainty_result = analysis_results['uncertainty_quantification']
            uncertainty = uncertainty_result.get('prediction_uncertainty', 0)
            
            if uncertainty > 15:
                recommendations.append("Recopilar más datos para reducir incertidumbre en predicciones")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones estratégicas: {str(e)}")
        return ["Error generando recomendaciones - contactar soporte técnico"]
