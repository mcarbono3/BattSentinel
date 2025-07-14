# src/routes/battery.py

from flask import Blueprint, request, jsonify, current_app, send_file
from datetime import datetime, timezone
# Asegúrate de importar 'and_' si planeas combinar múltiples filtros complejos,
# aunque para este caso chaining .filter() es suficiente.
from sqlalchemy import and_

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

BATCH_SIZE = 1000
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

def parse_iso_date(date_string):

    if not date_string:
        return None
    try:      
        if date_string.endswith('Z'):
            date_string = date_string[:-1] + '+00:00'
            
        dt_obj = datetime.fromisoformat(date_string)

        if dt_obj.tzinfo is None:
            # Si el string original tenía 'Z' o +00:00, asumimos UTC
            return dt_obj.replace(tzinfo=timezone.utc)
        return dt_obj
    except ValueError as e:
        current_app.logger.error(f"Error al parsear la cadena de fecha '{date_string}': {e}")
        raise ValueError(f"Formato de fecha ISO inválido: {date_string}") from e

def parse_float_or_none(value):
    """
    Convierte un valor a float o retorna None si es una cadena vacía, None,
    o no se puede convertir a float.
    """
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        current_app.logger.warning(f"No se pudo convertir '{value}' a float. Estableciendo a None.")
        return None

def parse_int_or_none(value):
    """
    Convierte un valor a int o retorna None si es una cadena vacía, None,
    o no se puede convertir a int.
    """
    if value is None or value == '':
        return None
    try:
        # Convertir a float primero para manejar números como "100.0"
        return int(float(value))
    except (ValueError, TypeError):
        current_app.logger.warning(f"No se pudo convertir '{value}' a entero. Estableciendo a None.")
        return None

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
            chemistry=data.get('chemistry'),
            designvoltage=parse_float_or_none(data.get('designvoltage')),
            full_charge_capacity=parse_float_or_none(data.get('full_charge_capacity')),
            full_charge_capacity_unit=data.get('full_charge_capacity_unit'),
            nominal_capacity=parse_float_or_none(data.get('nominal_capacity')),
            nominal_capacity_unit=data.get('nominal_capacity_unit'),
            model=data.get('model'),
            manufacturer=data.get('manufacturer'),
            serial_number=data.get('serial_number'),
            installation_date=parse_iso_date(data.get('installation_date')),
            location=data.get('location'),
            status=data.get('status', 'active'), # Default 'active' si no se provee
            last_maintenance_date=parse_iso_date(data.get('last_maintenance_date')),
            warranty_expiry_date=parse_iso_date(data.get('warranty_expiry_date')),
            cycles=parse_int_or_none(data.get('cycles')),
            monitoring_source=data.get('monitoring_source'),
            description=data.get('description')
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
        battery.chemistry = data.get('chemistry', battery.chemistry)
        battery.full_charge_capacity_unit = data.get('full_charge_capacity_unit', battery.full_charge_capacity_unit)
        battery.status = data.get('status', battery.status)
        battery.model = data.get('model', battery.model)
        battery.manufacturer = data.get('manufacturer', battery.manufacturer)
        battery.serial_number = data.get('serial_number', battery.serial_number)        
        battery.location = data.get('location', battery.location)
        battery.nominal_capacity_unit = data.get('nominal_capacity_unit', battery.nominal_capacity_unit)
        battery.monitoring_source = data.get('monitoring_source', battery.monitoring_source)
        battery.description = data.get('description', battery.description)

        # Manejo de campos numéricos usando las funciones auxiliares
        # Se comprueba si la clave existe en 'data' antes de intentar parsear
        if 'designvoltage' in data:
            battery.designvoltage = parse_float_or_none(data['designvoltage'])
        if 'full_charge_capacity' in data:
            battery.full_charge_capacity = parse_float_or_none(data['full_charge_capacity'])
        if 'nominal_capacity' in data:
            battery.nominal_capacity = parse_float_or_none(data['nominal_capacity'])
        if 'cycles' in data:
            battery.cycles = parse_int_or_none(data['cycles'])

        # Manejo de fechas usando la función parse_iso_date
        if 'installation_date' in data:
            battery.installation_date = parse_iso_date(data['installation_date'])
        if 'last_maintenance_date' in data:
            battery.last_maintenance_date = parse_iso_date(data['last_maintenance_date'])
        if 'warranty_expiry_date' in data:
            battery.warranty_expiry_date = parse_iso_date(data['warranty_expiry_date'])          

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
    """ Eliminar una batería y todos sus datos relacionados de manera optimizada y robusta."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            current_app.logger.warning(f"Intento de eliminar batería con ID {battery_id} que no existe.")
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        current_app.logger.info(f"Iniciando eliminación optimizada para batería con ID {battery_id}...")

        # 1. Eliminar registros de BatteryData por lotes (la tabla más probable con muchos datos)
        total_data_deleted = 0
        while True:

            data_to_delete = db.session.query(BatteryData.id).\
                                filter(BatteryData.battery_id == battery_id).\
                                limit(BATCH_SIZE).all()
            
            if not data_to_delete:
                break # No hay más datos para eliminar
            
            # Extraer solo los IDs
            data_ids = [data.id for data in data_to_delete]

            # Realiza la eliminación en lote
            deleted_count = db.session.query(BatteryData).\
                                filter(BatteryData.id.in_(data_ids)).\
                                delete(synchronize_session=False) # 'False' indica no sincronizar la sesión, más eficiente para bulk operations
            
            db.session.commit() # Confirma cada lote para liberar recursos y evitar transacciones largas
            total_data_deleted += deleted_count
            current_app.logger.info(f"Eliminados {deleted_count} registros de BatteryData para batería {battery_id}. Total: {total_data_deleted}")

            if deleted_count < BATCH_SIZE: # Si eliminamos menos que el tamaño del lote, significa que no quedan más
                break
        
        current_app.logger.info(f"Total de registros de BatteryData eliminados para batería {battery_id}: {total_data_deleted}")

        # Eliminar Alertas
        alerts_deleted = db.session.query(Alert).filter(Alert.battery_id == battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {alerts_deleted} registros de Alertas para batería {battery_id}.")

        # Eliminar Resultados de Análisis
        analysis_deleted = db.session.query(AnalysisResult).filter(AnalysisResult.battery_id == battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {analysis_deleted} registros de AnalysisResult para batería {battery_id}.")

        # Eliminar Imágenes Térmicas
        thermal_images_deleted = db.session.query(ThermalImage).filter(ThermalImage.battery_id == battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {thermal_images_deleted} registros de ThermalImage para batería {battery_id}.")

        # Eliminar Registros de Mantenimiento
        maintenance_deleted = db.session.query(MaintenanceRecord).filter(MaintenanceRecord.battery_id == battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {maintenance_deleted} registros de MaintenanceRecord para batería {battery_id}.")

        # Confirma todas las eliminaciones de tablas relacionadas
        db.session.commit() 
        current_app.logger.info(f"Todos los registros relacionados para batería {battery_id} han sido eliminados.")

        # 3. Finalmente, eliminar la batería principal
        db.session.delete(battery)
        db.session.commit()
        current_app.logger.info(f"Batería con ID {battery_id} eliminada correctamente.")
        
        return jsonify({'success': True, 'message': 'Batería y todos sus datos relacionados eliminados correctamente'})

    except Exception as e:
        db.session.rollback() # Asegura que si algo falla, se revierte todo lo hecho
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error crítico al eliminar batería {battery_id}: {e}\n{error_trace}")
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

# NUEVO ENDPOINT: Verificar si la batería tiene datos históricos
@battery_bp.route('/batteries/<int:battery_id>/has_historical_data', methods=['GET'])
def has_historical_data(battery_id):
    """
    Verifica de manera eficiente si una batería específica tiene algún dato histórico cargado.
    Devuelve True si hay datos, False en caso contrario.
    """
    try:
        # Usamos .first() para ver si existe al menos un registro.
        # No necesitamos cargar el objeto completo, solo verificar su existencia.
        exists = db.session.query(BatteryData.id).filter_by(battery_id=battery_id).first() is not None
        
        if exists:
            current_app.logger.debug(f"Batería {battery_id} TIENE datos históricos.")
            return jsonify({'success': True, 'has_data': True, 'message': 'La batería tiene datos históricos.'})
        else:
            current_app.logger.debug(f"Batería {battery_id} NO tiene datos históricos.")
            return jsonify({'success': True, 'has_data': False, 'message': 'La batería no tiene datos históricos.'})

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al verificar datos históricos para batería {battery_id}: {e}\n{error_trace}")
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
            power=data.get('power'), # Corrección: usa data.get()            
            # --- CAMPOS AÑADIDOS / MEJORADOS ---
            energy_rate=data.get('energy_rate'), 
            internal_resistance=data.get('internal_resistance'), 
            rul_days=data.get('rul_days'), 
            efficiency=data.get('efficiency'), 
            is_plugged=data.get('is_plugged'), 
            time_left=data.get('time_left'),       
        )

        # Asignar 'status' por separado, de forma defensiva
        if 'status' in data and data['status'] is not None:
            new_data.status = data['status']
        # Si por alguna razón el cliente todavía envía 'batterystatus', lo mapeamos
        elif 'batterystatus' in data and data['batterystatus'] is not None:
            new_data.status = str(data['batterystatus']) # Convertir a string para el campo db.String

        db.session.add(new_data)
        db.session.commit()
        current_app.logger.info(f"Nuevos datos añadidos a la batería {battery_id} y emitidos vía WebSocket.")
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
    """Subir un archivo con datos históricos de la batería, evitando duplicación."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            current_app.logger.warning(f"Intento de carga de datos para batería no encontrada: {battery_id}")
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if 'file' not in request.files:
            current_app.logger.warning(f"Solicitud de carga para batería {battery_id}: No se encontró el archivo.")
            return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud'}), 400

        file = request.files['file']
        if file.filename == '':
            current_app.logger.warning(f"Solicitud de carga para batería {battery_id}: Nombre de archivo vacío.")
            return jsonify({'success': False, 'error': 'Nombre de archivo vacío'}), 400

        if not (file and allowed_file(file.filename, ALLOWED_EXTENSIONS)):
            current_app.logger.warning(f"Solicitud de carga para batería {battery_id}: Tipo de archivo no permitido para '{file.filename}'.")
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido. Use CSV, TXT o Excel.'}), 400

        df = None
        # Leer el contenido del archivo con pandas según su extensión
        if file.filename.endswith(('.csv', '.txt')):
            # Se usa io.StringIO para leer el contenido del archivo en memoria como un string            
            df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            # Para Excel, pandas puede leer directamente del stream
            df = pd.read_excel(file.stream)
        else:
            # Esta condición debería ser cubierta por allowed_file, pero se deja como fallback de seguridad
            current_app.logger.error(f"Formato de archivo inesperado para '{file.filename}' después de la verificación de extensiones.")
            return jsonify({'success': False, 'error': 'Formato de archivo no soportado. Use CSV, TXT o Excel.'}), 400

        required_cols = ['timestamp', 'voltage', 'current', 'temperature', 'soc', 'soh', 'cycles', 'status']
        # Verifica si todas las columnas requeridas están presentes en el DataFrame
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            current_app.logger.warning(f"Archivo '{file.filename}' para batería {battery_id} carece de columnas requeridas: {', '.join(missing_cols)}")
            return jsonify({'success': False, 'error': f"El archivo debe contener las columnas: {', '.join(required_cols)}. Columnas faltantes: {', '.join(missing_cols)}. Columnas encontradas: {', '.join(df.columns)}"}), 400

        # --- LÓGICA DE SOBREESCRITURA COMPLETA ---
        # Paso 1: Eliminar ABSOLUTAMENTE TODOS los registros de BatteryData existentes para esta batería.
        deleted_old_data_count = BatteryData.query.filter_by(battery_id=battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {deleted_old_data_count} registros históricos de BatteryData existentes para la batería {battery_id} antes de la carga del nuevo archivo.")

        data_to_insert = []
        parsed_valid_rows_count = 0
        # Paso 2: Procesar cada fila del nuevo archivo para crear objetos BatteryData
        for index, row in df.iterrows():
            try:
                # Convertir el timestamp, manejando errores (errores='coerce' convertirá inválidos a NaT)
                timestamp = pd.to_datetime(row['timestamp'], errors='coerce')
                if pd.isna(timestamp): # Si el timestamp es inválido (NaT)
                    current_app.logger.warning(f"Fila {index} en '{file.filename}': Timestamp inválido, omitiendo esta fila.")
                    continue

                # Convertir a objeto datetime nativo de Python y asegurar que sea UTC
                formatted_timestamp = timestamp.to_pydatetime().replace(tzinfo=timezone.utc)

                data_to_insert.append(BatteryData(
                    battery_id=battery_id,
                    timestamp=formatted_timestamp,
                    voltage=row.get('voltage'),
                    current=row.get('current'),
                    temperature=row.get('temperature'),
                    soc=row.get('soc'),
                    soh=row.get('soh'),
                    cycles=row.get('cycles'),
                    status=row.get('status')
                    # Asegúrate de incluir aquí todos los campos de tu modelo BatteryData
                ))
                parsed_valid_rows_count += 1

            except Exception as row_e:
                current_app.logger.error(f"Error procesando fila {index} del archivo '{file.filename}': {row_e}. Fila omitida.")
                continue # Continuar con la siguiente fila si hay un error en esta

        if not data_to_insert:
            # Si después de procesar todo el archivo, no hay registros válidos para insertar.
            # En este caso, se revierte la eliminación de los datos antiguos,
            # ya que el archivo nuevo no aportó nada válido.
            db.session.rollback()
            current_app.logger.info(f"Archivo '{file.filename}' para batería {battery_id} no contenía registros válidos para insertar. Eliminación de datos antiguos revertida.")
            return jsonify({
                'success': False,
                'error': 'El archivo no contiene registros válidos para insertar.',
                'filename': file.filename,
                'note': 'No se insertaron nuevos datos, y los datos antiguos fueron restaurados.'
            }), 400

        # Paso 3: Insertar los nuevos registros en la base de datos
        # bulk_save_objects es muy eficiente para insertar múltiples objetos a la vez.
        db.session.bulk_save_objects(data_to_insert)
        db.session.commit() # Confirma tanto la eliminación previa como la inserción de los nuevos datos en una sola transacción atómica.

        current_app.logger.info(f"Carga y sobreescritura completadas para batería {battery_id} con archivo '{file.filename}'. Se insertaron {parsed_valid_rows_count} nuevos registros.")
        return jsonify({
            'success': True,
            'message': f'Archivo {file.filename} cargado exitosamente. Se han sobrescrito todos los datos históricos anteriores de la batería {battery_id} con {parsed_valid_rows_count} nuevos registros.',
            'records_inserted': parsed_valid_rows_count,
            'records_overwritten': deleted_old_data_count
        }), 201 # 201 Created es un código de estado apropiado cuando se reemplaza el contenido de un recurso

    except Exception as e:
        db.session.rollback() # Si ocurre cualquier error en el proceso, se revierte toda la transacción.
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error crítico durante la carga y sobreescritura de datos para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': 'Error interno del servidor al cargar o sobrescribir datos.', 'details': str(e), 'traceback': error_trace}), 500

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
    Obtener todos los registros de mantenimiento para una batería específica.
    """
    try:
        current_app.logger.info(f"Solicitud para obtener registros de mantenimiento para batería ID: {battery_id}")

        maintenance_records = MaintenanceRecord.query.filter_by(battery_id=battery_id).order_by(MaintenanceRecord.performed_at.desc()).all()

        records_data = [record.to_dict() for record in maintenance_records]

        current_app.logger.info(f"Se encontraron {len(records_data)} registros de mantenimiento para batería {battery_id}.")
        return jsonify({'success': True, 'data': records_data, 'battery_id': battery_id})

    except Exception as e:
        current_app.logger.error(f"Error al obtener registros de mantenimiento para batería {battery_id}: {str(e)}")
        return jsonify({'success': False, 'error': 'Error interno del servidor al obtener registros de mantenimiento.'}), 500

# --- INICIO DE LA MEJORA PARA FILTRADO DE FECHAS ---
@battery_bp.route('/batteries/<int:battery_id>/historical_data', methods=['GET'])
def get_battery_historical_data(battery_id):
    """
    Obtener datos históricos para una batería específica,
    opcionalmente filtrados por un rango de fechas (startDate y endDate).
    """
    try:
        # 1. Obtener la batería (si no existe, retornar error)
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        # 2. Obtener los parámetros de fecha de la URL
        start_date_str = request.args.get('startDate')
        end_date_str = request.args.get('endDate')

        # 3. Construir la consulta base
        query = BatteryData.query.filter_by(battery_id=battery_id)

        # 4. Aplicar filtros de fecha si están presentes
        if start_date_str:
            try:
                # Convertir la fecha string (yyyy-MM-dd) a objeto datetime
                # Asegurarse de que sea aware de la zona horaria UTC para comparación consistente
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                query = query.filter(BatteryData.timestamp >= start_date)
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato de startDate inválido. Usar YYYY-MM-DD'}), 400

        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                # IMPORTANTE: El frontend ajusta el endDate sumándole un día.
                # Para incluir todo el día final, filtraremos *antes* del inicio del día siguiente.
                query = query.filter(BatteryData.timestamp < end_date)
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato de endDate inválido. Usar YYYY-MM-DD'}), 400

        # 5. Ordenar los resultados y ejecutarlos
        filtered_data_points = query.order_by(BatteryData.timestamp.asc()).all()
        data_list = [data.to_dict() for data in filtered_data_points]

        current_app.logger.debug(f"Obtenidos {len(data_list)} puntos de datos históricos para batería {battery_id} con filtros.")
        return jsonify({'success': True, 'data': data_list, 'battery_id': battery_id})
    except Exception as e:
        db.session.rollback() # En caso de error en la DB, aunque no debería ser necesario aquí
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener datos históricos para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500
# --- FIN DE LA MEJORA PARA FILTRADO DE FECHAS ---

@battery_bp.route('/batteries/<int:battery_id>/historical_data_cleanup', methods=['DELETE'])
def cleanup_battery_historical_data(battery_id):
    """
    Realiza una limpieza COMPLETA de todos los datos históricos de monitoreo
    (BatteryData) y todos los registros de mantenimiento (MaintenanceRecord)
    para una batería específica.

    Esta acción eliminará ABSOLUTAMENTE TODOS los registros asociados
    a la battery_id proporcionada en ambas tablas.
    """
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            current_app.logger.warning(f"Intento de limpiar datos para batería no encontrada: {battery_id}")
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        # --- Limpieza COMPLETA de Datos Históricos de Monitoreo (BatteryData) ---
        # Eliminar TODOS los registros de BatteryData para esta batería
        deleted_data_count = BatteryData.query.filter_by(battery_id=battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {deleted_data_count} registros de BatteryData (todos) para la batería {battery_id}.")

        # --- Limpieza COMPLETA de Registros de Mantenimiento (MaintenanceRecord) ---
        # Eliminar TODOS los registros de MaintenanceRecord para esta batería
        deleted_maintenance_count = MaintenanceRecord.query.filter_by(battery_id=battery_id).delete(synchronize_session=False)
        current_app.logger.info(f"Eliminados {deleted_maintenance_count} registros de MaintenanceRecord (todos) para la batería {battery_id}.")

        # --- Confirmar todos los cambios en la base de datos ---
        db.session.commit()

        # --- Preparar la respuesta ---
        message = (
            f'Limpieza COMPLETA realizada para batería {battery_id}. '
            f'Se eliminaron {deleted_data_count} registros históricos de monitoreo. '
            f'Se eliminaron {deleted_maintenance_count} registros de mantenimiento.'
        )

        response_data = {
            'success': True,
            'message': message,
            'battery_id': battery_id,
            'deleted_historical_data_count': deleted_data_count,
            'deleted_maintenance_records_count': deleted_maintenance_count
        }

        return jsonify(response_data), 200

    except Exception as e:
        db.session.rollback() # Asegura que si algo falla, ningún cambio se guarda
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error crítico al realizar limpieza completa de datos para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': 'Error interno del servidor al realizar la limpieza COMPLETA.', 'details': str(e), 'traceback': error_trace}), 500

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

        maintenance_type = data.get('maintenance_type')
        description = data.get('description')
        performed_at = data.get('performed_at')
        performed_by = data.get('performed_by')
        cost = data.get('cost')
        parts_replaced = data.get('parts_replaced')
        notes = data.get('notes')
        next_maintenance_due = data.get('next_maintenance_due')

        if not all([maintenance_type, description, performed_at]):
            return jsonify({'success': False, 'error': 'Tipo de registro, descripción y fecha de realización son requeridos'}), 400

        new_record = MaintenanceRecord(
            battery_id=battery_id,
            maintenance_type=maintenance_type,
            description=description,
            performed_at=datetime.fromisoformat(performed_at).replace(tzinfo=timezone.utc),
            performed_by=performed_by,
            cost=cost,
            parts_replaced=parts_replaced,
            notes=notes,
            next_maintenance_due=next_maintenance_due
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
