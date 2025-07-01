from flask import Blueprint, request, jsonify, current_app, send_file
from datetime import datetime, timezone
import os
import pandas as pd
import numpy as np
import traceback
import io # Para manejar datos en memoria para archivos
import json # Para manejar campos JSON en los modelos

# Importaciones locales
import sys
# Si es necesario, descomenta la siguiente línea para que Python encuentre los módulos
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import db
from src.models.battery import Battery, BatteryData, Alert, AnalysisResult, ThermalImage, MaintenanceRecord # Asegúrate de importar todos los modelos necesarios
from src.services.windows_battery import windows_battery_service

battery_bp = Blueprint('battery', __name__)

# Extensiones permitidas para carga de datos y imágenes
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def get_real_battery_data():
    """Obtiene datos reales de la batería del sistema o genera simulados como fallback."""
    try:
        current_app.logger.debug("Intentando obtener datos reales de la batería del sistema...")
        battery_info = windows_battery_service.get_battery_info()

        if battery_info and battery_info.get('success'):
            current_app.logger.info("Datos de batería REALES obtenidos.")
            return {'success': True, 'data': battery_info.get('data')}
        else:
            # Fallback a datos simulados si no se pueden obtener datos reales
            current_app.logger.warning("No se pudieron obtener datos reales de la batería. Generando datos simulados.")
            return {'success': True, 'data': generate_mock_battery_data()}
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error general al obtener datos de la batería: {e}\n{error_trace}")
        return {'success': False, 'error': f'Error interno del servidor al obtener datos de la batería: {e}'}

def generate_mock_battery_data():
    """Genera datos de batería simulados para propósitos de demostración."""
    now = datetime.now(timezone.utc)
    return {
        'timestamp': now.isoformat(),
        'voltage': round(12.0 + (np.random.rand() - 0.5) * 0.5, 2), # 11.75V - 12.25V
        'current': round(2.5 + (np.random.rand() - 0.5) * 1.0, 2), # 2.0A - 3.0A
        'temperature': round(25 + (np.random.rand() - 0.5) * 10, 2), # 20°C - 30°C
        'soc': max(0, min(100, round(75 + (np.random.rand() - 0.5) * 50))), # 50% - 100%
        'soh': max(0, min(100, round(90 + (np.random.rand() - 0.5) * 10))), # 85% - 95%
        'cycles': int(100 + np.random.rand() * 500),
        'status': 'normal' if np.random.rand() > 0.1 else 'warning'
    }

# === Rutas existentes y mejoradas ===

@battery_bp.route('/batteries', methods=['GET'])
def get_batteries():
    """Obtener todas las baterías registradas."""
    try:
        batteries = Battery.query.all()
        batteries_list = [battery.to_dict() for battery in batteries]
        current_app.logger.debug(f"Obtenidas {len(batteries_list)} baterías.")
        return jsonify({'success': True, 'data': batteries_list})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener baterías: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries', methods=['POST'])
def create_battery():
    """Crear una nueva batería."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos en el cuerpo de la solicitud'}), 400

        name = data.get('name')
        if not name:
            return jsonify({'success': False, 'error': 'El nombre de la batería es requerido'}), 400

        new_battery = Battery(
            name=name,
            type=data.get('type'),
            nominal_voltage=data.get('nominal_voltage'),
            capacity_ah=data.get('capacity_ah')
        )
        db.session.add(new_battery)
        db.session.commit()
        current_app.logger.info(f"Batería '{name}' creada con ID: {new_battery.id}.")
        return jsonify({'success': True, 'data': new_battery.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al crear batería: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['GET'])
def get_battery(battery_id):
    """Obtener una batería específica por ID."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            current_app.logger.warning(f"Batería con ID {battery_id} no encontrada.")
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404
        current_app.logger.debug(f"Obtenida batería con ID: {battery_id}.")
        return jsonify({'success': True, 'data': battery.to_dict()})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['PUT'])
def update_battery(battery_id):
    """Actualizar una batería existente."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            current_app.logger.warning(f"Intento de actualizar batería con ID {battery_id} que no existe.")
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos para actualizar'}), 400

        battery.name = data.get('name', battery.name)
        battery.type = data.get('type', battery.type)
        battery.nominal_voltage = data.get('nominal_voltage', battery.nominal_voltage)
        battery.capacity_ah = data.get('capacity_ah', battery.capacity_ah)
        battery.last_update = datetime.now(timezone.utc) # Actualizar el timestamp

        db.session.commit()
        current_app.logger.info(f"Batería con ID {battery_id} actualizada.")
        return jsonify({'success': True, 'data': battery.to_dict()})
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al actualizar batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['DELETE'])
def delete_battery(battery_id):
    """Eliminar una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            current_app.logger.warning(f"Intento de eliminar batería con ID {battery_id} que no existe.")
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        db.session.delete(battery)
        db.session.commit()
        current_app.logger.info(f"Batería con ID {battery_id} eliminada.")
        return jsonify({'success': True, 'message': 'Batería eliminada correctamente'})
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al eliminar batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['GET'])
def get_battery_data(battery_id):
    """Obtener los últimos datos registrados para una batería específica."""
    try:
        battery_data = BatteryData.query.filter_by(battery_id=battery_id).order_by(BatteryData.timestamp.desc()).first()
        if not battery_data:
            current_app.logger.warning(f"No se encontraron datos para la batería {battery_id}.")
            return jsonify({'success': True, 'data': None, 'message': 'No hay datos disponibles para esta batería.'})
        current_app.logger.debug(f"Obtenidos los últimos datos para la batería {battery_id}.")
        return jsonify({'success': True, 'data': battery_data.to_dict()})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener datos de batería para {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['POST'])
def add_battery_data(battery_id):
    """Añadir nuevos datos a una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos'}), 400

        new_data = BatteryData(
            battery_id=battery_id,
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(timezone.utc),
            voltage=data.get('voltage'),
            current=data.get('current'),
            temperature=data.get('temperature'),
            soc=data.get('soc'),
            soh=data.get('soh'),
            cycles=data.get('cycles'),            
        )

        # Asignar 'status' por separado, de forma defensiva
        if 'status' in data and data['status'] is not None:
            new_data.status = data['status']
        # Si por alguna razón el cliente todavía envía 'batterystatus', lo mapeamos
        elif 'batterystatus' in data and data['batterystatus'] is not None:
            new_data.status = str(data['batterystatus']) # Convertir a string para el campo db.String
            
        db.session.add(new_data)
        db.session.commit()
        current_app.logger.info(f"Nuevos datos añadidos a la batería {battery_id}.")
        return jsonify({'success': True, 'data': new_data.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al añadir datos a la batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/summary', methods=['GET'])
def get_battery_summary(battery_id):
    """Obtener un resumen de datos clave para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        latest_data = BatteryData.query.filter_by(battery_id=battery_id).order_by(BatteryData.timestamp.desc()).first()
        summary = {
            'battery_id': battery.id,
            'name': battery.name,
            'type': battery.type,
            'latest_voltage': latest_data.voltage if latest_data else None,
            'latest_current': latest_data.current if latest_data else None,
            'latest_temperature': latest_data.temperature if latest_data else None,
            'latest_soc': latest_data.soc if latest_data else None,
            'latest_soh': latest_data.soh if latest_data else None,
            'latest_cycles': latest_data.cycles if latest_data else None,
            'latest_status': latest_data.status if latest_data else None,
            'last_update': latest_data.timestamp.isoformat() if latest_data else None
        }
        current_app.logger.debug(f"Resumen de batería {battery_id} generado.")
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener resumen de batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload_data', methods=['POST'])
def upload_battery_data(battery_id):
    """Subir un archivo con datos históricos de la batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nombre de archivo vacío'}), 400

        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            df = None
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file.stream)
            else:
                return jsonify({'success': False, 'error': 'Formato de archivo no soportado. Use CSV o Excel.'}), 400

            # Asumir que el archivo tiene columnas como: timestamp, voltage, current, temperature, soc, soh, cycles, status
            # Realizar validaciones básicas de columnas
            required_cols = ['timestamp', 'voltage', 'current', 'temperature', 'soc', 'soh', 'cycles', 'status']
            if not all(col in df.columns for col in required_cols):
                return jsonify({'success': False, 'error': f"El archivo debe contener las columnas: {', '.join(required_cols)}"}), 400

            # Convertir a formato adecuado e insertar en la base de datos
            data_to_insert = []
            for index, row in df.iterrows():
                try:
                    timestamp = pd.to_datetime(row['timestamp'], errors='coerce')
                    if pd.isna(timestamp):
                        current_app.logger.warning(f"Fila {index}: Timestamp inválido, omitiendo fila.")
                        continue

                    data_to_insert.append(BatteryData(
                        battery_id=battery_id,
                        timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
                        voltage=row.get('voltage'),
                        current=row.get('current'),
                        temperature=row.get('temperature'),
                        soc=row.get('soc'),
                        soh=row.get('soh'),
                        cycles=row.get('cycles'),
                        status=row.get('status')
                    ))
                except Exception as row_e:
                    current_app.logger.error(f"Error procesando fila {index} del archivo: {row_e}")
                    continue

            if data_to_insert:
                db.session.bulk_save_objects(data_to_insert)
                db.session.commit()
                current_app.logger.info(f"Datos del archivo '{file.filename}' cargados para la batería {battery_id}. {len(data_to_insert)} registros insertados.")
                return jsonify({'success': True, 'message': f'Archivo {file.filename} cargado y datos procesados correctamente.', 'records_inserted': len(data_to_insert)})
            else:
                return jsonify({'success': False, 'error': 'No se pudieron procesar registros válidos del archivo.'}), 400
        else:
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido o archivo no encontrado.'}), 400
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al cargar datos para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload_thermal_image', methods=['POST'])
def upload_thermal_image(battery_id):
    """Subir una imagen térmica para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró el archivo de imagen en la solicitud'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nombre de archivo de imagen vacío'}), 400

        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            # Guardar la imagen en un directorio configurado (ej. UPLOAD_FOLDER)
            # Asegúrate de que current_app.config['UPLOAD_FOLDER'] esté definido
            upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            filename = f"thermal_image_{battery_id}_{int(datetime.now().timestamp())}_{file.filename}"
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            # Guardar el registro en la base de datos
            new_image = ThermalImage(
                battery_id=battery_id,
                image_path=filepath,
                upload_timestamp=datetime.now(timezone.utc)
            )
            db.session.add(new_image)
            db.session.commit()
            current_app.logger.info(f"Imagen térmica cargada para batería {battery_id}: {filename}.")
            return jsonify({'success': True, 'message': 'Imagen térmica cargada correctamente', 'image_path': filepath}), 201
        else:
            return jsonify({'success': False, 'error': 'Tipo de archivo de imagen no permitido o archivo no encontrado.'}), 400
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al cargar imagen térmica para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/thermal_images', methods=['GET'])
def get_thermal_images(battery_id):
    """Obtener las imágenes térmicas registradas para una batería específica."""
    try:
        images = ThermalImage.query.filter_by(battery_id=battery_id).order_by(ThermalImage.upload_timestamp.desc()).all()
        images_list = [{'id': img.id, 'image_path': img.image_path, 'upload_timestamp': img.upload_timestamp.isoformat()} for img in images]
        current_app.logger.debug(f"Obtenidas {len(images_list)} imágenes térmicas para batería {battery_id}.")
        return jsonify({'success': True, 'data': images_list, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener imágenes térmicas para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

# === Nuevas funciones y rutas solicitadas ===

@battery_bp.route('/battery/real-time', methods=['GET'])
def get_real_time_battery_variables():
    """
    Nuevo endpoint: Obtiene y devuelve solo las variables en tiempo real de la batería.
    Omite el envío de alertas y análisis como solicitado.
    """
    try:
        current_app.logger.info("Solicitud de datos de batería en tiempo real recibida.")
        battery_data_response = get_real_battery_data() # Esta función ya maneja el fallback a mock data
        if battery_data_response['success']:
            current_app.logger.info("Datos de batería en tiempo real obtenidos y enviados.")
            return jsonify({'success': True, 'data': battery_data_response['data']})
        else:
            current_app.logger.error(f"Error al obtener datos en tiempo real: {battery_data_response['error']}")
            return jsonify({'success': False, 'error': battery_data_response['error']}), 500
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error inesperado en get_real_time_battery_variables: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(e)}'}), 500

@battery_bp.route('/batteries/<int:battery_id>/alerts', methods=['GET'])
def get_battery_alerts(battery_id):
    """
    Obtener alertas para una batería específica.
    Por ahora, devuelve una lista vacía como solicitado.
    """
    current_app.logger.info(f"Solicitud de alertas para batería {battery_id}. Devolviendo lista vacía (funcionalidad omitida).")
    return jsonify({'success': True, 'data': [], 'battery_id': battery_id})

@battery_bp.route('/batteries/<int:battery_id>/analysis_results', methods=['GET'])
def get_battery_analysis_results(battery_id):
    """
    Obtener resultados de análisis para una batería específica.
    Por ahora, devuelve una lista vacía como solicitado.
    """
    current_app.logger.info(f"Solicitud de resultados de análisis para batería {battery_id}. Devolviendo lista vacía (funcionalidad omitida).")
    return jsonify({'success': True, 'data': [], 'battery_id': battery_id})

@battery_bp.route('/batteries/<int:battery_id>/maintenance_records', methods=['GET'])
def get_battery_maintenance_records(battery_id):
    """
    Obtener registros de mantenimiento para una batería específica.
    Por ahora, devuelve una lista vacía como solicitado.
    """
    current_app.logger.info(f"Solicitud de registros de mantenimiento para batería {battery_id}. Devolviendo lista vacía (funcionalidad omitida).")
    return jsonify({'success': True, 'data': [], 'battery_id': battery_id})

@battery_bp.route('/batteries/<int:battery_id>/historical_data', methods=['GET'])
def get_battery_historical_data(battery_id):
    """
    Obtener datos históricos para una batería específica.
    Devuelve datos de la tabla BatteryData.
    """
    try:
        historical_data = BatteryData.query.filter_by(battery_id=battery_id).order_by(BatteryData.timestamp.asc()).all()
        data_list = [data.to_dict() for data in historical_data]
        current_app.logger.debug(f"Obtenidos {len(data_list)} puntos de datos históricos para batería {battery_id}.")
        return jsonify({'success': True, 'data': data_list, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener datos históricos para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/maintenance_records', methods=['POST'])
def add_maintenance_record(battery_id):
    """Añadir un nuevo registro de mantenimiento para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos para el registro de mantenimiento'}), 400

        record_type = data.get('record_type')
        description = data.get('description')
        performed_at = data.get('performed_at')

        if not all([record_type, description, performed_at]):
            return jsonify({'success': False, 'error': 'Tipo de registro, descripción y fecha de realización son requeridos'}), 400

        new_record = MaintenanceRecord(
            battery_id=battery_id,
            record_type=record_type,
            description=description,
            performed_at=datetime.fromisoformat(performed_at).replace(tzinfo=timezone.utc)
        )
        db.session.add(new_record)
        db.session.commit()
        current_app.logger.info(f"Registro de mantenimiento añadido para batería {battery_id}.")
        return jsonify({'success': True, 'data': new_record.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al añadir registro de mantenimiento para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500
