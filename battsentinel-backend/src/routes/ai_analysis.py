from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from src.models.battery import db, Battery, BatteryData, BatteryAnalysis
from src.services.ai_models import FaultDetectionModel, HealthPredictionModel, XAIExplainer

ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/analyze/<int:battery_id>', methods=['POST'])
def analyze_battery(battery_id):
    """Realizar análisis completo de IA en una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener parámetros del análisis
        data = request.get_json() or {}
        analysis_types = data.get('analysis_types', ['fault_detection', 'health_prediction'])
        time_window = data.get('time_window_hours', 24)  # Últimas 24 horas por defecto
        
        # Obtener datos recientes
        cutoff_time = datetime.now() - timedelta(hours=time_window)
        recent_data = BatteryData.query.filter(
            BatteryData.battery_id == battery_id,
            BatteryData.timestamp >= cutoff_time
        ).order_by(BatteryData.timestamp.desc()).all()
        
        if len(recent_data) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for analysis (minimum 10 data points required)'
            }), 400
        
        # Convertir a DataFrame para análisis
        df = pd.DataFrame([point.to_dict() for point in recent_data])
        
        results = {}
        
        # Análisis de detección de fallas
        if 'fault_detection' in analysis_types:
            fault_model = FaultDetectionModel()
            fault_result = fault_model.analyze(df)
            results['fault_detection'] = fault_result
            
            # Guardar resultado en BD
            analysis = BatteryAnalysis(
                battery_id=battery_id,
                analysis_type='fault_detection',
                result=json.dumps(fault_result['predictions']),
                confidence=fault_result['confidence'],
                fault_detected=fault_result['fault_detected'],
                fault_type=fault_result.get('fault_type'),
                severity=fault_result.get('severity'),
                explanation=json.dumps(fault_result.get('explanation', {})),
                model_version='1.0'
            )
            db.session.add(analysis)
        
        # Análisis de predicción de salud
        if 'health_prediction' in analysis_types:
            health_model = HealthPredictionModel()
            health_result = health_model.analyze(df)
            results['health_prediction'] = health_result
            
            # Guardar resultado en BD
            analysis = BatteryAnalysis(
                battery_id=battery_id,
                analysis_type='health_prediction',
                result=json.dumps(health_result['predictions']),
                confidence=health_result['confidence'],
                rul_prediction=health_result.get('rul_days'),
                explanation=json.dumps(health_result.get('explanation', {})),
                model_version='1.0'
            )
            db.session.add(analysis)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points_analyzed': len(recent_data),
                'time_window_hours': time_window,
                'results': results
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/fault-detection/<int:battery_id>', methods=['POST'])
def detect_faults(battery_id):
    """Detectar fallas específicas en una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos recientes
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(1000).all()
        
        if len(recent_data) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for fault detection'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in recent_data])
        
        # Inicializar modelo de detección de fallas
        fault_model = FaultDetectionModel()
        result = fault_model.analyze(df)
        
        # Explicabilidad con XAI
        explainer = XAIExplainer()
        explanation = explainer.explain_fault_detection(df, result)
        result['explanation'] = explanation
        
        # Guardar análisis
        analysis = BatteryAnalysis(
            battery_id=battery_id,
            analysis_type='fault_detection',
            result=json.dumps(result['predictions']),
            confidence=result['confidence'],
            fault_detected=result['fault_detected'],
            fault_type=result.get('fault_type'),
            severity=result.get('severity'),
            explanation=json.dumps(explanation),
            model_version='1.0'
        )
        db.session.add(analysis)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/health-prediction/<int:battery_id>', methods=['POST'])
def predict_health(battery_id):
    """Predecir estado de salud y vida útil restante"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos históricos
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.asc()).all()
        
        if len(historical_data) < 50:
            return jsonify({
                'success': False,
                'error': 'Insufficient historical data for health prediction (minimum 50 points)'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in historical_data])
        
        # Inicializar modelo de predicción de salud
        health_model = HealthPredictionModel()
        result = health_model.analyze(df)
        
        # Explicabilidad
        explainer = XAIExplainer()
        explanation = explainer.explain_health_prediction(df, result)
        result['explanation'] = explanation
        
        # Guardar análisis
        analysis = BatteryAnalysis(
            battery_id=battery_id,
            analysis_type='health_prediction',
            result=json.dumps(result['predictions']),
            confidence=result['confidence'],
            rul_prediction=result.get('rul_days'),
            explanation=json.dumps(explanation),
            model_version='1.0'
        )
        db.session.add(analysis)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/anomaly-detection/<int:battery_id>', methods=['POST'])
def detect_anomalies(battery_id):
    """Detectar anomalías en tiempo real"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos recientes para análisis de anomalías
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(100).all()
        
        if len(recent_data) < 20:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for anomaly detection'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in recent_data])
        
        # Detectar anomalías usando métodos estadísticos
        anomalies = self._detect_statistical_anomalies(df)
        
        # Clasificar severidad de anomalías
        classified_anomalies = self._classify_anomaly_severity(anomalies, df)
        
        result = {
            'anomalies_detected': len(classified_anomalies) > 0,
            'anomaly_count': len(classified_anomalies),
            'anomalies': classified_anomalies,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_points_analyzed': len(recent_data)
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/analyses/<int:battery_id>', methods=['GET'])
def get_analyses_history(battery_id):
    """Obtener historial de análisis de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Parámetros de consulta
        analysis_type = request.args.get('analysis_type')
        limit = request.args.get('limit', 50, type=int)
        
        query = BatteryAnalysis.query.filter_by(battery_id=battery_id)
        
        if analysis_type:
            query = query.filter_by(analysis_type=analysis_type)
        
        analyses = query.order_by(BatteryAnalysis.created_at.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'data': [analysis.to_dict() for analysis in analyses]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Obtener información sobre los modelos de IA disponibles"""
    try:
        model_info = {
            'fault_detection': {
                'name': 'Battery Fault Detection Model',
                'version': '1.0',
                'description': 'Detecta fallas como degradación acelerada, cortocircuito, sobrecarga, sobrecalentamiento',
                'input_features': ['voltage', 'current', 'temperature', 'soc', 'soh', 'internal_resistance'],
                'output_classes': ['normal', 'degradation', 'short_circuit', 'overcharge', 'overheat', 'thermal_runaway'],
                'accuracy': 0.92,
                'last_trained': '2024-01-01'
            },
            'health_prediction': {
                'name': 'Battery Health Prediction Model',
                'version': '1.0',
                'description': 'Predice estado de salud (SOH) y vida útil restante (RUL)',
                'input_features': ['voltage', 'current', 'temperature', 'cycles', 'capacity', 'internal_resistance'],
                'outputs': ['soh_prediction', 'rul_days', 'degradation_rate'],
                'mae': 2.5,  # Mean Absolute Error in days for RUL
                'r2_score': 0.89,
                'last_trained': '2024-01-01'
            },
            'anomaly_detection': {
                'name': 'Real-time Anomaly Detection',
                'version': '1.0',
                'description': 'Detecta patrones anómalos en tiempo real usando métodos estadísticos',
                'methods': ['isolation_forest', 'statistical_outliers', 'time_series_anomalies'],
                'sensitivity': 'configurable',
                'false_positive_rate': 0.05
            }
        }
        
        return jsonify({
            'success': True,
            'data': model_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _detect_statistical_anomalies(df):
    """Detectar anomalías usando métodos estadísticos"""
    anomalies = []
    
    # Columnas numéricas para análisis
    numeric_cols = ['voltage', 'current', 'temperature', 'soc', 'soh', 'internal_resistance']
    
    for col in numeric_cols:
        if col in df.columns and df[col].notna().sum() > 10:
            values = df[col].dropna()
            
            # Método Z-score
            z_scores = np.abs((values - values.mean()) / values.std())
            z_anomalies = df[z_scores > 3].index.tolist()
            
            # Método IQR
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_anomalies = df[(values < lower_bound) | (values > upper_bound)].index.tolist()
            
            # Combinar anomalías
            combined_anomalies = list(set(z_anomalies + iqr_anomalies))
            
            for idx in combined_anomalies:
                anomaly = {
                    'index': int(idx),
                    'parameter': col,
                    'value': float(df.loc[idx, col]),
                    'timestamp': df.loc[idx, 'timestamp'],
                    'z_score': float(z_scores.loc[idx]) if idx in z_scores.index else None,
                    'method': 'statistical'
                }
                anomalies.append(anomaly)
    
    return anomalies

def _classify_anomaly_severity(anomalies, df):
    """Clasificar severidad de anomalías"""
    classified = []
    
    for anomaly in anomalies:
        severity = 'low'
        
        # Clasificar basado en el parámetro y la desviación
        param = anomaly['parameter']
        z_score = anomaly.get('z_score', 0)
        
        if param in ['temperature'] and z_score > 5:
            severity = 'critical'
        elif param in ['voltage', 'current'] and z_score > 4:
            severity = 'high'
        elif z_score > 3.5:
            severity = 'medium'
        
        anomaly['severity'] = severity
        classified.append(anomaly)
    
    return classified

