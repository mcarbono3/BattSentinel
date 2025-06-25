from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone # Importar timezone para consistencia
import json
import cv2 # Asegúrate de que opencv-python esté instalado
from PIL import Image # Asegúrate de que Pillow esté instalado
from src.models.battery import db, Battery, BatteryData, ThermalImage
from src.services.data_processor import DataProcessor
from src.services.thermal_analyzer import ThermalAnalyzer
# --- CAMBIO IMPORTANTE: Importar decoradores de seguridad ---
from src.routes.auth import require_token, require_role
# --- FIN CAMBIO IMPORTANTE ---


battery_bp = Blueprint('battery', __name__)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@battery_bp.route('/batteries', methods=['GET'])
@require_token # Requiere token para acceder
def get_batteries():
    """Obtener todas las baterías"""
    try:
        batteries = Battery.query.all()
        return jsonify({
            'success': True,
            'data': [battery.to_dict() for battery in batteries]
        })
    except Exception as e:
        # Mejorar la respuesta de error para ser más informativa
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries', methods=['POST'])
@require_token
@require_role('admin') # Solo administradores pueden crear baterías
def create_battery():
    """Crear una nueva batería"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        
        battery = Battery(
            name=data['name'],
            battery_type=data.get('battery_type', 'Li-ion'), # Default Li-ion
            device_type=data.get('device_type')
        )
        db.session.add(battery)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Battery created successfully',
            'data': battery.to_dict()
        }), 201 # 201 Created
        
    except Exception as e:
        db.session.rollback() # Revertir la transacción si falla
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['GET'])
@require_token # Requiere token para acceder
def get_battery(battery_id):
    """Obtener detalles de una batería específica"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        return jsonify({
            'success': True,
            'data': battery.to_dict()
        })
    except Exception as e:
        # get_or_404 ya lanza una excepción que se convierte en 404 por Flask.
        # Aquí capturamos cualquier otra excepción.
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['GET'])
@require_token # Requiere token para acceder
def get_battery_data(battery_id):
    """Obtener datos históricos de una batería con paginación"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)

        # Ordenar por timestamp descendente para obtener los más recientes primero
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

@battery_bp.route('/batteries/<int:battery_id>/upload-data', methods=['POST'])
@require_token
@require_role('technician') # Técnicos o administradores pueden subir datos
def upload_battery_data(battery_id):
    """Subir un archivo de datos para una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar el archivo (esto debería hacerse en un proceso en segundo plano para archivos grandes)
            # Por simplicidad, lo hago aquí. DataProcessor debe manejar diferentes tipos de archivo.
            try:
                processed_data_count = DataProcessor.process_data_file(filepath, battery_id)
                os.remove(filepath) # Eliminar el archivo después de procesarlo
                return jsonify({
                    'success': True,
                    'message': f'File processed successfully. Added {processed_data_count} data points.'
                }), 200
            except Exception as e:
                os.remove(filepath) # Asegurarse de eliminar el archivo incluso si falla el procesamiento
                return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'}), 500
        else:
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload-thermal', methods=['POST'])
@require_token
@require_role('technician') # Técnicos o administradores pueden subir imágenes térmicas
def upload_thermal_image(battery_id):
    """Subir una imagen térmica para una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            # Guardar la imagen temporalmente para análisis
            file.save(filepath)

            try:
                # Procesar la imagen (ej. análisis térmico)
                thermal_analyzer = ThermalAnalyzer(filepath)
                max_temp = thermal_analyzer.analyze_temperature()
                
                # Guardar en la base de datos
                thermal_image = ThermalImage(
                    battery_id=battery.id,
                    file_path=filepath, # Aquí puedes guardar la ruta completa
                    max_temperature=max_temp,
                    # Asegúrate de que upload_timestamp esté en tu modelo y se establezca automáticamente
                    # o pása `datetime.now(timezone.utc)`
                    upload_timestamp=datetime.now(timezone.utc) # Agrega esto si no es default en el modelo
                )
                db.session.add(thermal_image)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': 'Thermal image uploaded and analyzed successfully',
                    'data': thermal_image.to_dict()
                }), 201 # 201 Created
                
            except Exception as e:
                db.session.rollback()
                # Eliminar el archivo si el procesamiento falla
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'success': False, 'error': f'Error processing thermal image: {str(e)}'}), 500
        else:
            return jsonify({'success': False, 'error': 'Image file type not allowed'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/thermal-images', methods=['GET'])
@require_token # Requiere token para acceder
def get_thermal_images(battery_id):
    """Obtener imágenes térmicas de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        thermal_images = ThermalImage.query.filter_by(battery_id=battery_id)\
                                           .order_by(ThermalImage.upload_timestamp.desc())\
                                           .all()
        
        return jsonify({
            'success': True,
            'data': [img.to_dict() for img in thermal_images]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@battery_bp.route('/batteries/<int:battery_id>/add-data', methods=['POST'])
@require_token
@require_role('technician') # Técnicos o administradores pueden añadir datos
def add_battery_data(battery_id):
    """Añadir un punto de datos individual a una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()

        if not data or any(field not in data for field in ['voltage', 'current', 'temperature']):
            return jsonify({'success': False, 'error': 'Voltage, current, and temperature are required fields.'}), 400
        
        # Validar y convertir timestamp
        timestamp_str = data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now(timezone.utc)

        battery_data = BatteryData(
            battery_id=battery.id,
            timestamp=timestamp,
            voltage=data['voltage'],
            current=data['current'],
            temperature=data['temperature'],
            soc=data.get('soc'),
            soh=data.get('soh'),
            cycles=data.get('cycles')
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


@battery_bp.route('/batteries/<int:battery_id>/summary', methods=['GET'])
@require_token # Requiere token para acceder
def get_battery_summary(battery_id):
    """Obtener un resumen analítico de los datos más recientes de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener los 1000 puntos de datos más recientes
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
                                       .order_by(BatteryData.timestamp.desc())\
                                       .limit(1000)\
                                       .all()
        
        if not recent_data:
            return jsonify({
                'success': True,
                'data': {
                    'battery': battery.to_dict(),
                    'stats': {'message': 'No data points available for summary.'}
                }
            })

        # Convertir a DataFrame de Pandas para un análisis más fácil
        df = pd.DataFrame([dp.to_dict() for dp in recent_data])
        
        stats = {
            'last_updated': df['timestamp'].max().isoformat() if 'timestamp' in df else None,
            'avg_voltage': df['voltage'].mean() if 'voltage' in df else None,
            'avg_current': df['current'].mean() if 'current' in df else None,
            'avg_temperature': df['temperature'].mean() if 'temperature' in df else None,
            'min_temperature': df['temperature'].min() if 'temperature' in df else None,
            'max_temperature': df['temperature'].max() if 'temperature' in df else None,
            'avg_soc': df['soc'].mean() if 'soc' in df else None,
            'avg_soh': df['soh'].mean() if 'soh' in df else None,
            'avg_cycles': df['cycles'].mean() if 'cycles' in df else None
        }
        
        # Detección de anomalías simple (ejemplo)
        # Esto es muy básico y debe ser mejorado por tu módulo DataProcessor/AI
        voltage_std = df['voltage'].std() if 'voltage' in df else 0
        stats['voltage_variability'] = voltage_std
        stats['anomaly_detected'] = voltage_std > np.mean(df['voltage']) * 0.1 # Si la desviación es > 10% del promedio

        # Consideraciones de estado de salud (ejemplo muy simple)
        if 'soh' in df and df['soh'].min() is not None:
            if df['soh'].min() < 0.7: # Si cualquier SoH cae por debajo del 70%
                stats['health_status_alert'] = 'Degradación significativa de la salud de la batería detectada.'
            else:
                stats['health_status_alert'] = 'Salud de la batería en niveles aceptables.'
        else:
            stats['health_status_alert'] = 'No hay datos de SoH para evaluar la salud.'

        # Últimas lecturas importantes
        last_data_point = recent_data[0] # Ya está ordenado descendentemente
        stats['last_readings'] = {
                    'voltage': last_data_point.voltage,
                    'current': last_data_point.current,
                    'temperature': last_data_point.temperature,
                    'soc': last_data_point.soc,
                    'soh': last_data_point.soh,
                    'cycles': last_data_point.cycles,
                    'timestamp': last_data_point.timestamp.isoformat()
                } if last_data_point else None

        # Información adicional
        stats['data_points_count'] = len(recent_data)
        stats['date_range'] = {
            'start': recent_data[-1].timestamp.isoformat() if recent_data else None,
            'end': recent_data[0].timestamp.isoformat() if recent_data else None
        }
        
        return jsonify({
            'success': True,
            'data': {
                'battery': battery.to_dict(),
                'stats': stats
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['DELETE'])
@require_token
@require_role('admin') # Solo administradores pueden eliminar baterías
def delete_battery(battery_id):
    """Eliminar una batería y todos sus datos"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Eliminar archivos de imágenes térmicas
        thermal_images = ThermalImage.query.filter_by(battery_id=battery_id).all()
        for image in thermal_images:
            if os.path.exists(image.file_path):
                os.remove(image.file_path)
        
        # Eliminar batería (cascade eliminará datos relacionados si la relación está configurada con cascade="all, delete-orphan")
        db.session.delete(battery)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Battery deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
