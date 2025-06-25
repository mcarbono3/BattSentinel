from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import psutil
import platform
import subprocess

# Importaciones locales
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.battery import db, Battery, BatteryData
from services.windows_battery import windows_battery_service

battery_bp = Blueprint('battery', __name__)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def get_real_battery_data():
    """Obtiene datos reales de la batería del sistema"""
    try:
        battery_info = {}
        
        # Información básica usando psutil
        battery = psutil.sensors_battery()
        if battery:
            battery_info.update({
                'soc': battery.percent,  # State of Charge
                'power_plugged': battery.power_plugged,
                'seconds_left': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None,
                'voltage': 12.0,  # Valor por defecto, se actualizará si está disponible
                'current': 2.5 if not battery.power_plugged else -1.5,  # Estimación
                'temperature': 25.0  # Valor por defecto
            })
        
        # Información adicional en Windows
        if platform.system() == 'Windows':
            try:
                # Obtener voltaje de diseño
                voltage_cmd = 'wmic path Win32_Battery get DesignVoltage /value'
                result = subprocess.run(voltage_cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and '=' in result.stdout:
                    voltage_str = result.stdout.split('=')[1].strip()
                    if voltage_str and voltage_str.isdigit():
                        battery_info['voltage'] = float(voltage_str) / 1000  # Convertir de mV a V
                
                # Obtener capacidades
                design_cap_cmd = 'wmic path Win32_Battery get DesignCapacity /value'
                result = subprocess.run(design_cap_cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and '=' in result.stdout:
                    cap_str = result.stdout.split('=')[1].strip()
                    if cap_str and cap_str.isdigit():
                        battery_info['design_capacity'] = float(cap_str)
                
                full_cap_cmd = 'wmic path Win32_Battery get FullChargeCapacity /value'
                result = subprocess.run(full_cap_cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and '=' in result.stdout:
                    cap_str = result.stdout.split('=')[1].strip()
                    if cap_str and cap_str.isdigit():
                        battery_info['full_charge_capacity'] = float(cap_str)
                
                # Calcular SOH si tenemos ambas capacidades
                if 'design_capacity' in battery_info and 'full_charge_capacity' in battery_info:
                    design_cap = battery_info['design_capacity']
                    full_cap = battery_info['full_charge_capacity']
                    if design_cap > 0:
                        battery_info['soh'] = round((full_cap / design_cap) * 100, 2)
                
                # Obtener ciclos de batería
                cycle_cmd = 'powershell "Get-WmiObject -Class Win32_Battery | Select-Object -ExpandProperty CycleCount"'
                result = subprocess.run(cycle_cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    battery_info['cycles'] = int(result.stdout.strip())
                
                # Intentar obtener temperatura del sistema
                temp_cmd = 'wmic /namespace:\\\\root\\wmi PATH MSAcpi_ThermalZoneTemperature get CurrentTemperature /value'
                result = subprocess.run(temp_cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'CurrentTemperature=' in line:
                            temp_kelvin = int(line.split('=')[1].strip())
                            temp_celsius = (temp_kelvin / 10) - 273.15
                            battery_info['temperature'] = round(temp_celsius, 2)
                            break
                            
            except Exception as e:
                print(f"Error obteniendo información adicional: {e}")
        
        # Valores por defecto si no se pudieron obtener
        battery_info.setdefault('soh', 85.0)
        battery_info.setdefault('cycles', 150)
        battery_info.setdefault('voltage', 12.0)
        battery_info.setdefault('current', 2.5)
        battery_info.setdefault('temperature', 25.0)
        
        # Calcular RUL (Remaining Useful Life) estimado
        if 'soh' in battery_info and 'cycles' in battery_info:
            soh = battery_info['soh']
            cycles = battery_info['cycles']
            # Estimación simple: RUL basado en SOH y ciclos
            max_cycles = 1000  # Ciclos típicos para Li-ion
            remaining_cycles = max(0, max_cycles - cycles)
            rul_cycles = remaining_cycles * (soh / 100)
            battery_info['rul'] = round(rul_cycles, 0)
        
        battery_info['timestamp'] = datetime.now(timezone.utc).isoformat()
        return battery_info
        
    except Exception as e:
        print(f"Error obteniendo datos de batería: {e}")
        return {
            'soc': 75.0,
            'soh': 85.0,
            'voltage': 12.0,
            'current': 2.5,
            'temperature': 25.0,
            'cycles': 150,
            'rul': 500,
            'power_plugged': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e)
        }

@battery_bp.route('/api/batteries', methods=['GET'])
def get_batteries():
    """Obtener todas las baterías - Sin autenticación"""
    try:
        batteries = Battery.query.all()
        return jsonify({
            'success': True,
            'data': [battery.to_dict() for battery in batteries]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/batteries', methods=['POST'])
def create_battery():
    """Crear una nueva batería - Sin autenticación"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        
        battery = Battery(
            name=data['name'],
            battery_type=data.get('battery_type', 'Li-ion'),
            device_type=data.get('device_type', 'Laptop')
        )
        db.session.add(battery)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Battery created successfully',
            'data': battery.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/batteries/<int:battery_id>', methods=['GET'])
def get_battery(battery_id):
    """Obtener detalles de una batería específica - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        return jsonify({
            'success': True,
            'data': battery.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/batteries/<int:battery_id>/data', methods=['GET'])
def get_battery_data(battery_id):
    """Obtener datos históricos de una batería - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)

        data_points = BatteryData.query.filter_by(battery_id=battery_id)\
                                     .order_by(BatteryData.timestamp.desc())\
                                     .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'success': True,
            'data': [dp.to_dict() for dp in data_points.items],
            'pagination': {
                'total_items': data_points.total,
                'total_pages': data_points.pages,
                'current_page': data_points.page,
                'per_page': data_points.per_page,
                'has_next': data_points.has_next,
                'has_prev': data_points.has_prev
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/batteries/<int:battery_id>/add-data', methods=['POST'])
def add_battery_data(battery_id):
    """Añadir un punto de datos a una batería - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()

        if not data:
            # Si no se proporcionan datos, usar datos reales del sistema
            real_data = get_real_battery_data()
            data = real_data

        # Validar campos requeridos
        required_fields = ['voltage', 'current', 'temperature']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'{field} is required'}), 400
        
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if timestamp_str else datetime.now(timezone.utc)

        battery_data = BatteryData(
            battery_id=battery.id,
            timestamp=timestamp,
            voltage=float(data['voltage']),
            current=float(data['current']),
            temperature=float(data['temperature']),
            soc=float(data.get('soc', 75.0)),
            soh=float(data.get('soh', 85.0)),
            cycles=int(data.get('cycles', 150))
        )
        db.session.add(battery_data)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Data point added successfully',
            'data': battery_data.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/batteries/<int:battery_id>/summary', methods=['GET'])
def get_battery_summary(battery_id):
    """Obtener resumen de una batería - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos recientes
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
                                       .order_by(BatteryData.timestamp.desc())\
                                       .limit(100)\
                                       .all()
        
        if not recent_data:
            # Si no hay datos, crear algunos datos de ejemplo con datos reales
            real_data = get_real_battery_data()
            sample_data = BatteryData(
                battery_id=battery.id,
                timestamp=datetime.now(timezone.utc),
                voltage=real_data['voltage'],
                current=real_data['current'],
                temperature=real_data['temperature'],
                soc=real_data['soc'],
                soh=real_data['soh'],
                cycles=real_data['cycles']
            )
            db.session.add(sample_data)
            db.session.commit()
            recent_data = [sample_data]

        # Convertir a DataFrame para análisis
        df = pd.DataFrame([dp.to_dict() for dp in recent_data])
        
        stats = {
            'last_updated': df['timestamp'].max() if 'timestamp' in df else datetime.now(timezone.utc).isoformat(),
            'avg_voltage': float(df['voltage'].mean()) if 'voltage' in df else 0,
            'avg_current': float(df['current'].mean()) if 'current' in df else 0,
            'avg_temperature': float(df['temperature'].mean()) if 'temperature' in df else 0,
            'min_temperature': float(df['temperature'].min()) if 'temperature' in df else 0,
            'max_temperature': float(df['temperature'].max()) if 'temperature' in df else 0,
            'avg_soc': float(df['soc'].mean()) if 'soc' in df else 0,
            'avg_soh': float(df['soh'].mean()) if 'soh' in df else 0,
            'avg_cycles': float(df['cycles'].mean()) if 'cycles' in df else 0
        }
        
        # Detección de anomalías
        if 'voltage' in df and len(df) > 1:
            voltage_std = df['voltage'].std()
            stats['voltage_variability'] = float(voltage_std)
            stats['anomaly_detected'] = voltage_std > df['voltage'].mean() * 0.1
        else:
            stats['voltage_variability'] = 0.0
            stats['anomaly_detected'] = False

        # Estado de salud
        if 'soh' in df:
            min_soh = df['soh'].min()
            if min_soh < 70:
                stats['health_status_alert'] = 'Degradación significativa detectada'
            elif min_soh < 80:
                stats['health_status_alert'] = 'Salud de batería en observación'
            else:
                stats['health_status_alert'] = 'Salud de batería en niveles aceptables'
        else:
            stats['health_status_alert'] = 'No hay datos de SOH disponibles'

        # Últimas lecturas
        last_data_point = recent_data[0]
        stats['last_readings'] = {
            'voltage': last_data_point.voltage,
            'current': last_data_point.current,
            'temperature': last_data_point.temperature,
            'soc': last_data_point.soc,
            'soh': last_data_point.soh,
            'cycles': last_data_point.cycles,
            'timestamp': last_data_point.timestamp.isoformat()
        }

        stats['data_points_count'] = len(recent_data)
        
        return jsonify({
            'success': True,
            'data': {
                'battery': battery.to_dict(),
                'stats': stats
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/api/batteries/real-time', methods=['GET'])
def get_real_time_data():
    """Obtener datos de batería en tiempo real del sistema - Sin autenticación"""
    try:
        real_data = get_real_battery_data()
        return jsonify({
            'success': True,
            'data': real_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@battery_bp.route('/api/batteries/<int:battery_id>/update-real-time', methods=['POST'])
def update_with_real_time_data(battery_id):
    """Actualizar batería con datos en tiempo real - Sin autenticación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        real_data = get_real_battery_data()
        
        # Crear nuevo punto de datos con información real
        battery_data = BatteryData(
            battery_id=battery.id,
            timestamp=datetime.now(timezone.utc),
            voltage=real_data['voltage'],
            current=real_data['current'],
            temperature=real_data['temperature'],
            soc=real_data['soc'],
            soh=real_data.get('soh', 85.0),
            cycles=real_data.get('cycles', 150)
        )
        db.session.add(battery_data)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Battery updated with real-time data',
            'data': battery_data.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

