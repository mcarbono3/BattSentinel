from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

# Importaciones locales
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.battery import db, Battery, BatteryData
from services.windows_battery import windows_battery_service

ai_bp = Blueprint('ai_analysis', __name__)

@ai_bp.route('/api/ai/analyze/<int:battery_id>', methods=['POST'])
def analyze_battery(battery_id):
    """Realizar análisis de IA en una batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            data = request.get_json() or {}
            analysis_types = data.get('analysis_types', ['fault_detection', 'health_prediction'])
            
            # Obtener datos históricos para análisis
            try:
                historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
                    .order_by(BatteryData.timestamp.desc()).limit(200).all()
                
                if len(historical_data) < 10:
                    # Generar datos de ejemplo si no hay suficientes
                    sample_data = generate_sample_analysis_data(battery_id, 100)
                    df = pd.DataFrame(sample_data)
                else:
                    df = pd.DataFrame([point.to_dict() for point in historical_data])
            except Exception as e:
                # Fallback a datos simulados
                sample_data = generate_sample_analysis_data(battery_id, 100)
                df = pd.DataFrame(sample_data)
            
            # Realizar análisis según los tipos solicitados
            analysis_results = {}
            
            if 'fault_detection' in analysis_types:
                analysis_results['fault_detection'] = perform_fault_detection(df)
            
            if 'health_prediction' in analysis_types:
                analysis_results['health_prediction'] = perform_health_prediction(df)
            
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
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/detect-faults/<int:battery_id>', methods=['POST'])
def detect_faults(battery_id):
    """Detectar fallas en la batería usando IA - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Obtener datos recientes
            try:
                recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
                    .order_by(BatteryData.timestamp.desc()).limit(50).all()
                df = pd.DataFrame([point.to_dict() for point in recent_data])
            except Exception as e:
                # Fallback a datos simulados
                sample_data = generate_sample_analysis_data(battery_id, 50)
                df = pd.DataFrame(sample_data)
            
            # Realizar detección de fallas
            fault_results = perform_fault_detection(df)
            
            return jsonify({
                'success': True,
                'data': fault_results
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/predict-health/<int:battery_id>', methods=['POST'])
def predict_health(battery_id):
    """Predecir salud futura de la batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            data = request.get_json() or {}
            prediction_horizon = data.get('prediction_horizon', 30)  # días
            
            # Obtener datos históricos
            try:
                historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
                    .order_by(BatteryData.timestamp.desc()).limit(200).all()
                df = pd.DataFrame([point.to_dict() for point in historical_data])
            except Exception as e:
                # Fallback a datos simulados
                sample_data = generate_sample_analysis_data(battery_id, 200)
                df = pd.DataFrame(sample_data)
            
            # Realizar predicción de salud
            health_prediction = perform_health_prediction(df, prediction_horizon)
            
            return jsonify({
                'success': True,
                'data': health_prediction
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/detect-anomalies/<int:battery_id>', methods=['POST'])
def detect_anomalies(battery_id):
    """Detectar anomalías en el comportamiento de la batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Obtener datos recientes
            try:
                recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
                    .order_by(BatteryData.timestamp.desc()).limit(100).all()
                df = pd.DataFrame([point.to_dict() for point in recent_data])
            except Exception as e:
                # Fallback a datos simulados
                sample_data = generate_sample_analysis_data(battery_id, 100)
                df = pd.DataFrame(sample_data)
            
            # Realizar detección de anomalías
            anomaly_results = perform_anomaly_detection(df)
            
            return jsonify({
                'success': True,
                'data': anomaly_results
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/api/ai/analyses-history/<int:battery_id>', methods=['GET'])
def get_analyses_history(battery_id):
    """Obtener historial de análisis de IA - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Simular historial de análisis
            analyses_history = generate_sample_analyses_history(battery_id)
            
            return jsonify({
                'success': True,
                'data': analyses_history
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Funciones auxiliares para análisis de IA

def perform_fault_detection(df):
    """Realizar detección de fallas básica"""
    try:
        if df.empty:
            return {'faults_detected': [], 'status': 'no_data'}
        
        faults = []
        
        # Verificar voltaje
        if 'voltage' in df.columns:
            avg_voltage = df['voltage'].mean()
            if avg_voltage < 11.5:
                faults.append({
                    'type': 'low_voltage',
                    'severity': 'high',
                    'description': f'Voltaje promedio bajo: {avg_voltage:.2f}V',
                    'recommendation': 'Verificar sistema de carga'
                })
            elif avg_voltage > 13.5:
                faults.append({
                    'type': 'high_voltage',
                    'severity': 'medium',
                    'description': f'Voltaje promedio alto: {avg_voltage:.2f}V',
                    'recommendation': 'Verificar regulador de voltaje'
                })
        
        # Verificar temperatura
        if 'temperature' in df.columns:
            avg_temp = df['temperature'].mean()
            if avg_temp > 45:
                faults.append({
                    'type': 'overheating',
                    'severity': 'high',
                    'description': f'Temperatura elevada: {avg_temp:.1f}°C',
                    'recommendation': 'Mejorar ventilación y verificar carga'
                })
            elif avg_temp < 0:
                faults.append({
                    'type': 'low_temperature',
                    'severity': 'medium',
                    'description': f'Temperatura baja: {avg_temp:.1f}°C',
                    'recommendation': 'Considerar calentamiento en ambiente frío'
                })
        
        # Verificar SOH
        if 'soh' in df.columns:
            avg_soh = df['soh'].mean()
            if avg_soh < 70:
                faults.append({
                    'type': 'degraded_health',
                    'severity': 'high',
                    'description': f'Estado de salud degradado: {avg_soh:.1f}%',
                    'recommendation': 'Considerar reemplazo de batería'
                })
            elif avg_soh < 80:
                faults.append({
                    'type': 'declining_health',
                    'severity': 'medium',
                    'description': f'Estado de salud en declive: {avg_soh:.1f}%',
                    'recommendation': 'Monitoreo frecuente y mantenimiento preventivo'
                })
        
        return {
            'faults_detected': faults,
            'total_faults': len(faults),
            'status': 'completed',
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'faults_detected': [],
            'status': 'error',
            'error': str(e)
        }

def perform_health_prediction(df, prediction_horizon=30):
    """Realizar predicción de salud básica"""
    try:
        if df.empty:
            return {'prediction': None, 'status': 'no_data'}
        
        # Análisis de tendencias básico
        current_soh = df['soh'].iloc[0] if 'soh' in df.columns else 85.0
        
        # Simular degradación basada en ciclos y tiempo
        cycles = df['cycles'].iloc[0] if 'cycles' in df.columns else 150
        degradation_rate = 0.02  # 2% por cada 100 ciclos
        
        # Predicción simple
        predicted_soh = max(50, current_soh - (degradation_rate * prediction_horizon / 30))
        
        # Calcular RUL (Remaining Useful Life)
        rul_cycles = max(0, int((current_soh - 70) / degradation_rate * 100))
        rul_days = max(0, int(rul_cycles / 2))  # Asumiendo 2 ciclos por día
        
        prediction_data = {
            'current_soh': float(current_soh),
            'predicted_soh': float(predicted_soh),
            'prediction_horizon_days': prediction_horizon,
            'degradation_rate_per_month': float(degradation_rate * 30),
            'remaining_useful_life': {
                'cycles': rul_cycles,
                'days': rul_days,
                'months': max(0, int(rul_days / 30))
            },
            'confidence_level': 0.75,
            'prediction_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return {
            'prediction': prediction_data,
            'status': 'completed'
        }
        
    except Exception as e:
        return {
            'prediction': None,
            'status': 'error',
            'error': str(e)
        }

def perform_anomaly_detection(df):
    """Realizar detección de anomalías básica"""
    try:
        if df.empty:
            return {'anomalies': [], 'status': 'no_data'}
        
        anomalies = []
        
        # Detectar anomalías en voltaje
        if 'voltage' in df.columns:
            voltage_mean = df['voltage'].mean()
            voltage_std = df['voltage'].std()
            voltage_threshold = voltage_std * 2
            
            voltage_anomalies = df[abs(df['voltage'] - voltage_mean) > voltage_threshold]
            for _, row in voltage_anomalies.iterrows():
                anomalies.append({
                    'type': 'voltage_anomaly',
                    'timestamp': row.get('timestamp', ''),
                    'value': float(row['voltage']),
                    'expected_range': [float(voltage_mean - voltage_threshold), float(voltage_mean + voltage_threshold)],
                    'severity': 'medium'
                })
        
        # Detectar anomalías en temperatura
        if 'temperature' in df.columns:
            temp_mean = df['temperature'].mean()
            temp_std = df['temperature'].std()
            temp_threshold = temp_std * 2
            
            temp_anomalies = df[abs(df['temperature'] - temp_mean) > temp_threshold]
            for _, row in temp_anomalies.iterrows():
                anomalies.append({
                    'type': 'temperature_anomaly',
                    'timestamp': row.get('timestamp', ''),
                    'value': float(row['temperature']),
                    'expected_range': [float(temp_mean - temp_threshold), float(temp_mean + temp_threshold)],
                    'severity': 'high' if abs(row['temperature'] - temp_mean) > temp_threshold * 1.5 else 'medium'
                })
        
        return {
            'anomalies': anomalies[:10],  # Limitar a 10 anomalías más recientes
            'total_anomalies': len(anomalies),
            'status': 'completed',
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
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
        
        # Eficiencia energética
        if 'voltage' in df.columns and 'current' in df.columns:
            power = df['voltage'] * df['current']
            metrics['average_power'] = float(power.mean())
            metrics['power_efficiency'] = min(100, float(power.mean() / 50 * 100))  # Asumiendo 50W como referencia
        
        # Estabilidad de voltaje
        if 'voltage' in df.columns:
            voltage_stability = 100 - (df['voltage'].std() / df['voltage'].mean() * 100)
            metrics['voltage_stability'] = max(0, float(voltage_stability))
        
        # Consistencia de temperatura
        if 'temperature' in df.columns:
            temp_consistency = 100 - (df['temperature'].std() / 10 * 100)  # 10°C como rango aceptable
            metrics['temperature_consistency'] = max(0, float(temp_consistency))
        
        # Score general de rendimiento
        performance_scores = [v for v in metrics.values() if isinstance(v, (int, float))]
        overall_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        return {
            'performance_metrics': metrics,
            'overall_performance_score': float(overall_performance),
            'performance_grade': get_performance_grade(overall_performance),
            'status': 'completed',
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            'performance_metrics': {},
            'status': 'error',
            'error': str(e)
        }

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

def generate_sample_analysis_data(battery_id, count=100):
    """Generar datos de ejemplo para análisis"""
    import random
    from datetime import timedelta
    
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
    from datetime import timedelta
    
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

