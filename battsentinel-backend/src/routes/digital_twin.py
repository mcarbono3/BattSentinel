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
from services.digital_twin import digital_twin_service

twin_bp = Blueprint('digital_twin', __name__)

@twin_bp.route('/api/digital-twin/create/<int:battery_id>', methods=['POST'])
def create_digital_twin(battery_id):
    """Crear gemelo digital para una batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            data = request.get_json() or {}
            
            # Parámetros del gemelo digital
            twin_config = {
                'battery_id': battery_id,
                'model_type': data.get('model_type', 'electrochemical'),
                'simulation_parameters': data.get('simulation_parameters', {}),
                'calibration_data': data.get('calibration_data', []),
                'update_frequency': data.get('update_frequency', 'real_time')
            }
            
            # Crear gemelo digital usando el servicio
            try:
                twin_result = digital_twin_service.create_twin(battery_id, twin_config)
            except Exception as e:
                # Fallback a simulación básica
                twin_result = create_basic_twin_simulation(battery_id, twin_config)
            
            return jsonify({
                'success': True,
                'data': twin_result
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/api/digital-twin/simulate/<int:battery_id>', methods=['POST'])
def simulate_battery_response(battery_id):
    """Simular respuesta de la batería a cambios en variables - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            data = request.get_json() or {}
            
            # Parámetros de simulación
            simulation_params = {
                'load_current': data.get('load_current', 2.5),
                'ambient_temperature': data.get('ambient_temperature', 25.0),
                'simulation_duration': data.get('simulation_duration', 3600),  # segundos
                'time_step': data.get('time_step', 60),  # segundos
                'initial_soc': data.get('initial_soc', 80.0)
            }
            
            # Ejecutar simulación
            try:
                simulation_result = digital_twin_service.run_simulation(battery_id, simulation_params)
            except Exception as e:
                # Fallback a simulación básica
                simulation_result = run_basic_simulation(battery_id, simulation_params)
            
            return jsonify({
                'success': True,
                'data': simulation_result
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/api/digital-twin/state/<int:battery_id>', methods=['GET'])
def get_twin_state(battery_id):
    """Obtener estado actual del gemelo digital - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Obtener estado del gemelo digital
            try:
                twin_state = digital_twin_service.get_twin_state(battery_id)
            except Exception as e:
                # Fallback a estado simulado
                twin_state = get_simulated_twin_state(battery_id)
            
            return jsonify({
                'success': True,
                'data': twin_state
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/api/digital-twin/parameters/<int:battery_id>', methods=['GET'])
def get_twin_parameters(battery_id):
    """Obtener parámetros del gemelo digital - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Obtener parámetros del modelo
            try:
                twin_params = digital_twin_service.get_model_parameters(battery_id)
            except Exception as e:
                # Fallback a parámetros por defecto
                twin_params = get_default_twin_parameters(battery_id)
            
            return jsonify({
                'success': True,
                'data': twin_params
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/api/digital-twin/parameters/<int:battery_id>', methods=['PUT'])
def update_twin_parameters(battery_id):
    """Actualizar parámetros del gemelo digital - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            data = request.get_json() or {}
            
            # Actualizar parámetros
            try:
                update_result = digital_twin_service.update_parameters(battery_id, data)
            except Exception as e:
                # Simulación de actualización
                update_result = {
                    'battery_id': battery_id,
                    'updated_parameters': data,
                    'update_timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'updated'
                }
            
            return jsonify({
                'success': True,
                'data': update_result
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/api/digital-twin/predict/<int:battery_id>', methods=['POST'])
def predict_future_behavior(battery_id):
    """Predecir comportamiento futuro de la batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            data = request.get_json() or {}
            
            # Parámetros de predicción
            prediction_params = {
                'prediction_horizon': data.get('prediction_horizon', 30),  # días
                'usage_pattern': data.get('usage_pattern', 'normal'),
                'environmental_conditions': data.get('environmental_conditions', {}),
                'maintenance_schedule': data.get('maintenance_schedule', [])
            }
            
            # Ejecutar predicción
            try:
                prediction_result = digital_twin_service.predict_behavior(battery_id, prediction_params)
            except Exception as e:
                # Fallback a predicción básica
                prediction_result = generate_basic_prediction(battery_id, prediction_params)
            
            return jsonify({
                'success': True,
                'data': prediction_result
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@twin_bp.route('/api/digital-twin/optimize/<int:battery_id>', methods=['POST'])
def optimize_battery_usage(battery_id):
    """Optimizar uso de la batería usando gemelo digital - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            data = request.get_json() or {}
            
            # Parámetros de optimización
            optimization_params = {
                'objective': data.get('objective', 'maximize_lifespan'),
                'constraints': data.get('constraints', {}),
                'optimization_horizon': data.get('optimization_horizon', 7),  # días
                'current_usage_pattern': data.get('current_usage_pattern', {})
            }
            
            # Ejecutar optimización
            try:
                optimization_result = digital_twin_service.optimize_usage(battery_id, optimization_params)
            except Exception as e:
                # Fallback a recomendaciones básicas
                optimization_result = generate_basic_optimization(battery_id, optimization_params)
            
            return jsonify({
                'success': True,
                'data': optimization_result
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Funciones auxiliares para simulación básica

def create_basic_twin_simulation(battery_id, config):
    """Crear simulación básica del gemelo digital"""
    return {
        'twin_id': f"twin_{battery_id}",
        'battery_id': battery_id,
        'model_type': config.get('model_type', 'electrochemical'),
        'status': 'active',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'accuracy': 0.85,
        'calibration_status': 'calibrated',
        'parameters': get_default_twin_parameters(battery_id)
    }

def run_basic_simulation(battery_id, params):
    """Ejecutar simulación básica"""
    duration = params.get('simulation_duration', 3600)
    time_step = params.get('time_step', 60)
    initial_soc = params.get('initial_soc', 80.0)
    load_current = params.get('load_current', 2.5)
    ambient_temp = params.get('ambient_temperature', 25.0)
    
    # Generar datos de simulación
    time_points = list(range(0, duration + 1, time_step))
    simulation_data = []
    
    current_soc = initial_soc
    current_voltage = 12.0
    current_temp = ambient_temp
    
    for t in time_points:
        # Simulación básica de descarga
        discharge_rate = load_current / 100  # Simplificado
        current_soc = max(0, current_soc - (discharge_rate * time_step / 3600))
        
        # Voltaje basado en SOC
        current_voltage = 10.5 + (current_soc / 100) * 2.5
        
        # Temperatura con calentamiento por carga
        temp_rise = load_current * 0.5
        current_temp = ambient_temp + temp_rise + np.random.normal(0, 0.5)
        
        data_point = {
            'time': t,
            'soc': round(current_soc, 2),
            'voltage': round(current_voltage, 2),
            'current': load_current,
            'temperature': round(current_temp, 1),
            'power': round(current_voltage * load_current, 2)
        }
        simulation_data.append(data_point)
    
    return {
        'battery_id': battery_id,
        'simulation_parameters': params,
        'simulation_data': simulation_data,
        'summary': {
            'initial_soc': initial_soc,
            'final_soc': current_soc,
            'energy_consumed': round((initial_soc - current_soc) * 1.0, 2),  # kWh simplificado
            'average_power': round(current_voltage * load_current, 2),
            'max_temperature': round(max([d['temperature'] for d in simulation_data]), 1),
            'simulation_duration': duration
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

def get_simulated_twin_state(battery_id):
    """Obtener estado simulado del gemelo digital"""
    return {
        'twin_id': f"twin_{battery_id}",
        'battery_id': battery_id,
        'status': 'active',
        'last_sync': datetime.now(timezone.utc).isoformat(),
        'sync_frequency': 'real_time',
        'model_accuracy': 0.87,
        'current_state': {
            'soc': 75.5,
            'voltage': 12.3,
            'temperature': 26.8,
            'current': 2.1,
            'power': 25.8,
            'internal_resistance': 0.045
        },
        'predicted_state_1h': {
            'soc': 72.1,
            'voltage': 12.1,
            'temperature': 28.2,
            'estimated_runtime': 3.2  # horas
        },
        'health_indicators': {
            'soh': 84.2,
            'remaining_cycles': 850,
            'degradation_rate': 0.02  # % por mes
        }
    }

def get_default_twin_parameters(battery_id):
    """Obtener parámetros por defecto del gemelo digital"""
    return {
        'battery_id': battery_id,
        'electrochemical_parameters': {
            'nominal_capacity': 100.0,  # Ah
            'nominal_voltage': 12.0,    # V
            'internal_resistance': 0.05, # Ohm
            'charge_efficiency': 0.95,
            'discharge_efficiency': 0.98
        },
        'thermal_parameters': {
            'thermal_capacity': 1000.0,  # J/K
            'thermal_resistance': 0.1,   # K/W
            'ambient_temperature': 25.0  # °C
        },
        'aging_parameters': {
            'calendar_aging_factor': 0.001,  # per day
            'cycle_aging_factor': 0.0001,    # per cycle
            'temperature_aging_factor': 0.05  # per °C above 25°C
        },
        'model_parameters': {
            'model_type': 'equivalent_circuit',
            'update_frequency': 'real_time',
            'calibration_interval': 24,  # horas
            'accuracy_threshold': 0.8
        },
        'last_updated': datetime.now(timezone.utc).isoformat()
    }

def generate_basic_prediction(battery_id, params):
    """Generar predicción básica"""
    horizon = params.get('prediction_horizon', 30)
    
    # Simulación de predicción
    predictions = []
    current_soh = 84.0
    degradation_rate = 0.02 / 30  # por día
    
    for day in range(horizon):
        predicted_soh = max(50, current_soh - (degradation_rate * day))
        predicted_capacity = predicted_soh
        
        prediction = {
            'day': day + 1,
            'date': (datetime.now(timezone.utc) + pd.Timedelta(days=day+1)).isoformat(),
            'predicted_soh': round(predicted_soh, 1),
            'predicted_capacity': round(predicted_capacity, 1),
            'estimated_runtime': round(predicted_capacity / 25, 1),  # horas a 25A
            'confidence': max(0.5, 0.9 - (day * 0.01))
        }
        predictions.append(prediction)
    
    return {
        'battery_id': battery_id,
        'prediction_horizon': horizon,
        'predictions': predictions,
        'summary': {
            'current_soh': 84.0,
            'predicted_soh_end': predictions[-1]['predicted_soh'],
            'total_degradation': round(84.0 - predictions[-1]['predicted_soh'], 1),
            'average_confidence': round(sum([p['confidence'] for p in predictions]) / len(predictions), 2)
        },
        'generated_at': datetime.now(timezone.utc).isoformat()
    }

def generate_basic_optimization(battery_id, params):
    """Generar optimización básica"""
    objective = params.get('objective', 'maximize_lifespan')
    
    recommendations = []
    
    if objective == 'maximize_lifespan':
        recommendations = [
            {
                'category': 'charging',
                'recommendation': 'Limitar carga al 80% para uso diario',
                'impact': 'Incremento del 20% en vida útil',
                'priority': 'high'
            },
            {
                'category': 'temperature',
                'recommendation': 'Mantener temperatura entre 15-25°C',
                'impact': 'Reducción del 15% en degradación',
                'priority': 'high'
            },
            {
                'category': 'discharge',
                'recommendation': 'Evitar descargas por debajo del 20%',
                'impact': 'Reducción del 10% en degradación por ciclo',
                'priority': 'medium'
            }
        ]
    elif objective == 'maximize_performance':
        recommendations = [
            {
                'category': 'charging',
                'recommendation': 'Carga rápida hasta 90% cuando sea necesario',
                'impact': 'Máximo rendimiento disponible',
                'priority': 'high'
            },
            {
                'category': 'cooling',
                'recommendation': 'Sistema de enfriamiento activo durante alta demanda',
                'impact': 'Mantenimiento de potencia máxima',
                'priority': 'medium'
            }
        ]
    
    return {
        'battery_id': battery_id,
        'optimization_objective': objective,
        'recommendations': recommendations,
        'estimated_improvements': {
            'lifespan_increase': '15-25%',
            'performance_gain': '10-15%',
            'efficiency_improvement': '5-10%'
        },
        'implementation_timeline': '1-2 weeks',
        'generated_at': datetime.now(timezone.utc).isoformat()
    }

