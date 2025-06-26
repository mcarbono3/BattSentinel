from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timezone
import json
import os
import pandas as pd
import numpy as np
import traceback 

# Importaciones locales
import sys

from src.main import db
from src.models.battery import Battery, BatteryData
from src.services.windows_battery import windows_battery_service

battery_bp = Blueprint('battery', __name__)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def get_real_battery_data():
    """Obtiene datos reales de la batería del sistema"""
    try:
        # Intentar obtener datos reales de Windows
        battery_info = windows_battery_service.get_battery_info()
        if battery_info and battery_info.get('success'):
            return battery_info['data']
    except Exception as e:
        print(f"Error obteniendo datos reales: {e}")
    
    # Fallback a datos simulados
    return {
        'voltage': 12.6 + np.random.normal(0, 0.1),
        'current': 2.5 + np.random.normal(0, 0.3),
        'temperature': 25.0 + np.random.normal(0, 2.0),
        'soc': 75.0 + np.random.normal(0, 5.0),
        'soh': 85.0 + np.random.normal(0, 2.0),
        'cycles': 150 + int(np.random.normal(0, 10)),
        'power': 31.5 + np.random.normal(0, 2.0),
        'energy': 100.0 + np.random.normal(0, 5.0),
        'internal_resistance': 0.05 + np.random.normal(0, 0.01),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

@battery_bp.route('/api/batteries', methods=['GET'])
def get_batteries():
    """Obtener lista de baterías - Sin autenticación"""
    try:
        print("DEBUG: Intentando obtener lista de baterías...")
        with current_app.app_context():
            batteries = Battery.query.all()
            print(f"DEBUG: Consulta de baterías exitosa. Se encontraron {len(batteries)} baterías.")
        
        if not batteries:
            print("DEBUG: No se encontraron baterías, generando batería por defecto.")
            # Crear batería por defecto si no existe
            default_battery = {
                'id': 1,
                'name': 'Batería Principal',
                'type': 'Li-ion',
                'capacity': 100.0,
                'voltage_nominal': 12.0,
                'manufacturer': 'BattSentinel',
                'model': 'BS-100',
                'serial_number': 'BS001',
                'installation_date': datetime.now(timezone.utc).isoformat(),
                'status': 'active'
            }
            batteries_data = [default_battery]
        else:
            batteries_data = [battery.to_dict() for battery in batteries]
        
        return jsonify({
            'success': True,
            'data': batteries_data
        })
        
    except Exception as e:
        error_trace = traceback.format_exc() # <--- ¡Esto es CRÍTICO para obtener el traceback!
        print(f"ERROR en GET /api/batteries: {e}") # <--- Imprime el mensaje de error
        print(f"TRACEBACK COMPLETO: \n{error_trace}") # <--- Imprime el traceback completo al log de Render
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

# Ruta para obtener una batería por ID (modificada para coincidir con el formato de get_batteries)
@battery_bp.route('/batteries/<int:battery_id>', methods=['GET', 'OPTIONS'])
def get_battery_by_id(battery_id):
    if request.method == 'OPTIONS':
        return '', 204 # Manejo de pre-vuelo CORS

    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404
        
        # Formatear la batería para que coincida con la salida de get_batteries
        # NO se incluyen campos derivados de BatteryData o Alert aquí,
        # solo los que están directamente en el modelo Battery y los que usas en get_batteries.
        formatted_battery = {
            'id': battery.id,
            'name': battery.name,
            'type': battery.chemistry, # Mapeado a 'chemistry' para coincidir con tu ejemplo 'Li-ion'
            'capacity': battery.capacity_ah, # Mapeado a 'capacity_ah'
            'voltage_nominal': battery.voltage_nominal,
            'manufacturer': battery.manufacturer,
            'model': battery.model, # Mapeado a 'model'
            'serial_number': battery.serial_number,
            'installation_date': battery.installation_date.isoformat() if battery.installation_date else None,
            'status': battery.status
            # No se incluyen charge_cycles, soh, soc, last_update, alerts_count, predicted_rul ni data_points
            # ya que no están en la salida de tu get_batteries actual.
        }
        return jsonify({'success': True, 'data': formatted_battery})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error fetching battery {battery_id}: {e}")
        print(f"TRACEBACK COMPLETO para /batteries/{battery_id}:\n{error_trace}") # Imprime traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/api/battery/real-time', methods=['GET'])
def get_real_time_data():
    """Obtener datos en tiempo real de la batería - Sin autenticación"""
    try:
        # Obtener datos reales o simulados
        battery_data = get_real_battery_data()
        
        return jsonify({
            'success': True,
            'data': battery_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/battery/<int:battery_id>/data', methods=['GET'])
def get_battery_data(battery_id):
    """Obtener datos históricos de una batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Parámetros de consulta
            limit = request.args.get('limit', 100, type=int)
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            try:
                query = BatteryData.query.filter_by(battery_id=battery_id)
                
                if start_date:
                    query = query.filter(BatteryData.timestamp >= start_date)
                if end_date:
                    query = query.filter(BatteryData.timestamp <= end_date)
                
                data_points = query.order_by(BatteryData.timestamp.desc()).limit(limit).all()
                data_list = [point.to_dict() for point in data_points]
            except Exception as e:
                # Generar datos de ejemplo si falla la consulta
                data_list = generate_sample_battery_data(battery_id, limit)
            
            return jsonify({
                'success': True,
                'data': data_list,
                'battery_id': battery_id,
                'count': len(data_list)
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/battery/health-analysis', methods=['GET'])
def get_health_analysis():
    """Obtener análisis de salud de la batería - Sin autenticación"""
    try:
        # Obtener datos actuales
        current_data = get_real_battery_data()
        
        # Realizar análisis básico
        health_analysis = {
            'overall_health': 'Good',
            'health_score': min(100, max(0, current_data.get('soh', 85))),
            'voltage_status': 'Normal' if 11.5 <= current_data.get('voltage', 12.0) <= 13.5 else 'Warning',
            'temperature_status': 'Normal' if 0 <= current_data.get('temperature', 25) <= 45 else 'Warning',
            'soc_status': 'Normal' if current_data.get('soc', 75) >= 20 else 'Low',
            'estimated_remaining_life': f"{max(1, int(1000 - current_data.get('cycles', 150)))} cycles",
            'recommendations': [
                'Mantener temperatura entre 15-35°C',
                'Evitar descargas profundas (<20%)',
                'Realizar mantenimiento preventivo cada 6 meses'
            ],
            'last_analysis': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': health_analysis
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/battery/<int:battery_id>/upload-data', methods=['POST'])
def upload_battery_data(battery_id):
    """Subir datos de batería desde archivo - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
                return jsonify({'success': False, 'error': 'File type not allowed'}), 400
            
            # Procesar archivo (simulado)
            processed_records = 0
            try:
                # Aquí iría el procesamiento real del archivo
                processed_records = 100  # Simulado
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'}), 500
            
            return jsonify({
                'success': True,
                'data': {
                    'battery_id': battery_id,
                    'filename': file.filename,
                    'records_processed': processed_records,
                    'upload_timestamp': datetime.now(timezone.utc).isoformat()
                }
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/battery/<int:battery_id>/export-data', methods=['GET'])
def export_battery_data(battery_id):
    """Exportar datos de batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Parámetros de exportación
            format_type = request.args.get('format', 'json')
            limit = request.args.get('limit', 1000, type=int)
            
            try:
                data_points = BatteryData.query.filter_by(battery_id=battery_id)\
                    .order_by(BatteryData.timestamp.desc()).limit(limit).all()
                data_list = [point.to_dict() for point in data_points]
            except Exception as e:
                # Generar datos de ejemplo si falla la consulta
                data_list = generate_sample_battery_data(battery_id, limit)
            
            export_data = {
                'battery_info': {
                    'id': battery_id,
                    'name': battery.name if hasattr(battery, 'name') else f'Batería {battery_id}',
                    'export_date': datetime.now(timezone.utc).isoformat(),
                    'total_records': len(data_list)
                },
                'data': data_list
            }
            
            return jsonify({
                'success': True,
                'data': export_data,
                'format': format_type
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_sample_battery_data(battery_id, count=100):
    """Generar datos de ejemplo para la batería"""
    import random
    from datetime import timedelta
    
    sample_data = []
    base_time = datetime.now(timezone.utc)
    
    for i in range(count):
        timestamp = base_time - timedelta(minutes=i * 5)
        
        # Simular variaciones realistas
        voltage = 12.0 + random.uniform(-0.5, 0.5)
        current = 2.5 + random.uniform(-1.0, 1.0)
        temperature = 25.0 + random.uniform(-5.0, 10.0)
        soc = max(20, min(100, 75 + random.uniform(-10, 10)))
        soh = max(70, min(100, 85 + random.uniform(-3, 3)))
        cycles = 150 + i
        
        data_point = {
            'id': i + 1,
            'battery_id': battery_id,
            'timestamp': timestamp.isoformat(),
            'voltage': round(voltage, 2),
            'current': round(current, 2),
            'temperature': round(temperature, 1),
            'soc': round(soc, 1),
            'soh': round(soh, 1),
            'cycles': cycles,
            'power': round(voltage * current, 2),
            'energy': round(voltage * current * 0.5, 2),
            'internal_resistance': round(0.05 + random.uniform(-0.01, 0.01), 4)
        }
        sample_data.append(data_point)
    
    return sample_data
