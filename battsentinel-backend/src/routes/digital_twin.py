from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from src.models.battery import db, Battery, BatteryData
from src.services.digital_twin import DigitalTwinSimulator

twin_bp = Blueprint('twin', __name__)

@twin_bp.route('/create/<int:battery_id>', methods=['POST'])
def create_digital_twin(battery_id):
    """Crear gemelo digital para una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos históricos para inicializar el gemelo
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(1000).all()
        
        if len(historical_data) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient data to create digital twin (minimum 10 data points required)'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in historical_data])
        
        # Inicializar simulador de gemelo digital
        simulator = DigitalTwinSimulator()
        twin_model = simulator.create_twin(df, battery_id)
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'twin_id': twin_model['twin_id'],
                'model_parameters': twin_model['parameters'],
                'initialization_data': twin_model['initialization'],
                'created_at': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/simulate/<int:battery_id>', methods=['POST'])
def simulate_battery_response(battery_id):
    """Simular respuesta de la batería a cambios en variables"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No simulation parameters provided'}), 400
        
        # Parámetros de simulación
        simulation_params = {
            'temperature': data.get('temperature'),
            'load_current': data.get('load_current'),
            'ambient_conditions': data.get('ambient_conditions', {}),
            'simulation_duration': data.get('simulation_duration', 3600),  # segundos
            'time_step': data.get('time_step', 60)  # segundos
        }
        
        # Obtener estado actual de la batería
        latest_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).first()
        
        if not latest_data:
            return jsonify({
                'success': False,
                'error': 'No current battery state available'
            }), 400
        
        # Inicializar simulador
        simulator = DigitalTwinSimulator()
        
        # Ejecutar simulación
        simulation_result = simulator.simulate_response(
            latest_data.to_dict(),
            simulation_params
        )
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'simulation_parameters': simulation_params,
                'initial_state': latest_data.to_dict(),
                'simulation_results': simulation_result,
                'simulation_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/state/<int:battery_id>', methods=['GET'])
def get_twin_state(battery_id):
    """Obtener estado actual del gemelo digital"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos más recientes
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(100).all()
        
        if not recent_data:
            return jsonify({
                'success': False,
                'error': 'No data available for digital twin'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in recent_data])
        
        # Calcular estado actual del gemelo
        simulator = DigitalTwinSimulator()
        current_state = simulator.calculate_current_state(df)
        
        # Calcular parámetros derivados
        derived_params = simulator.calculate_derived_parameters(current_state)
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'current_state': current_state,
                'derived_parameters': derived_params,
                'last_updated': datetime.now().isoformat(),
                'data_points_used': len(recent_data)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/parameters/<int:battery_id>', methods=['GET'])
def get_twin_parameters(battery_id):
    """Obtener parámetros del gemelo digital"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos históricos
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(500).all()
        
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'No historical data available'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in historical_data])
        
        # Calcular parámetros del modelo
        simulator = DigitalTwinSimulator()
        model_parameters = simulator.extract_model_parameters(df)
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'model_parameters': model_parameters,
                'parameter_confidence': simulator.calculate_parameter_confidence(df),
                'last_calibration': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/predict/<int:battery_id>', methods=['POST'])
def predict_future_behavior(battery_id):
    """Predecir comportamiento futuro de la batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()
        
        # Parámetros de predicción
        prediction_horizon = data.get('prediction_horizon', 24)  # horas
        scenario = data.get('scenario', 'normal')  # normal, stress, optimal
        
        # Obtener datos históricos
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(1000).all()
        
        if len(historical_data) < 50:
            return jsonify({
                'success': False,
                'error': 'Insufficient historical data for prediction'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in historical_data])
        
        # Inicializar simulador
        simulator = DigitalTwinSimulator()
        
        # Realizar predicción
        prediction_result = simulator.predict_future_behavior(
            df, 
            prediction_horizon, 
            scenario
        )
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'prediction_horizon_hours': prediction_horizon,
                'scenario': scenario,
                'predictions': prediction_result,
                'prediction_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/optimize/<int:battery_id>', methods=['POST'])
def optimize_battery_usage(battery_id):
    """Optimizar uso de la batería basado en el gemelo digital"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()
        
        # Parámetros de optimización
        optimization_goal = data.get('goal', 'longevity')  # longevity, performance, efficiency
        constraints = data.get('constraints', {})
        time_horizon = data.get('time_horizon', 168)  # horas (1 semana)
        
        # Obtener datos históricos
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(1000).all()
        
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'No historical data available for optimization'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in historical_data])
        
        # Inicializar simulador
        simulator = DigitalTwinSimulator()
        
        # Realizar optimización
        optimization_result = simulator.optimize_usage(
            df,
            optimization_goal,
            constraints,
            time_horizon
        )
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'optimization_goal': optimization_goal,
                'constraints': constraints,
                'recommendations': optimization_result,
                'optimization_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/calibrate/<int:battery_id>', methods=['POST'])
def calibrate_twin_model(battery_id):
    """Calibrar modelo del gemelo digital con nuevos datos"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener todos los datos disponibles
        all_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.asc()).all()
        
        if len(all_data) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for model calibration'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in all_data])
        
        # Inicializar simulador
        simulator = DigitalTwinSimulator()
        
        # Calibrar modelo
        calibration_result = simulator.calibrate_model(df)
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'calibration_results': calibration_result,
                'data_points_used': len(all_data),
                'calibration_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/compare/<int:battery_id>', methods=['POST'])
def compare_scenarios(battery_id):
    """Comparar diferentes escenarios de uso"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()
        
        scenarios = data.get('scenarios', [])
        if not scenarios:
            return jsonify({
                'success': False,
                'error': 'No scenarios provided for comparison'
            }), 400
        
        # Obtener datos históricos
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(500).all()
        
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'No historical data available'
            }), 400
        
        # Convertir a DataFrame
        df = pd.DataFrame([point.to_dict() for point in historical_data])
        
        # Inicializar simulador
        simulator = DigitalTwinSimulator()
        
        # Comparar escenarios
        comparison_result = simulator.compare_scenarios(df, scenarios)
        
        return jsonify({
            'success': True,
            'data': {
                'battery_id': battery_id,
                'scenarios_compared': len(scenarios),
                'comparison_results': comparison_result,
                'comparison_timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

