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
            current_app.logger.info("Datos de batería REALES obtenidos ...")
            return jsonify({'success': True, 'data': battery_info.get('data')})
        else:
            current_app.logger.warning("No se pudieron obtener datos reales de la batería. Generando datos simulados.")
            # Generar datos simulados si no se pueden obtener datos reales
            simulated_data = {
                'voltage': round(np.random.uniform(3.5, 4.2), 2),
                'current': round(np.random.uniform(0.1, 2.0), 2),
                'temperature': round(np.random.uniform(20.0, 35.0), 2),
                'soc': round(np.random.uniform(10, 100)),
                'soh': round(np.random.uniform(70, 100)),
                'cycles': int(np.random.randint(50, 500)),
                'status': np.random.choice(['optimal', 'warning', 'critical']),
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            return jsonify({'success': True, 'data': simulated_data})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener datos de batería: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries', methods=['GET'])
def get_batteries():
    """Obtener todas las baterías."""
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
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Datos de entrada no proporcionados'}), 400

    required_fields = ['name', 'type', 'voltage', 'capacity', 'manufacturing_date']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Campo requerido faltante: {field}'}), 400

    try:
        # Convertir la fecha de fabricación a objeto datetime
        manufacturing_date = datetime.fromisoformat(data['manufacturing_date'].replace('Z', '+00:00')) if isinstance(data['manufacturing_date'], str) else data['manufacturing_date']

        new_battery = Battery(
            name=data['name'],
            type=data['type'],
            voltage=data['voltage'],
            capacity=data['capacity'],
            manufacturing_date=manufacturing_date,
            last_calibration_date=datetime.fromisoformat(data['last_calibration_date'].replace('Z', '+00:00')) if data.get('last_calibration_date') else None,
            installation_date=datetime.fromisoformat(data['installation_date'].replace('Z', '+00:00')) if data.get('installation_date') else None,
            threshold_voltage_min=data.get('threshold_voltage_min'),
            threshold_voltage_max=data.get('threshold_voltage_max'),
            threshold_temp_min=data.get('threshold_temp_min'),
            threshold_temp_max=data.get('threshold_temp_max'),
            threshold_soc_min=data.get('threshold_soc_min'),
            threshold_soh_min=data.get('threshold_soh_min'),
            critical_alerts_enabled=data.get('critical_alerts_enabled', True)
        )
        db.session.add(new_battery)
        db.session.commit()
        current_app.logger.info(f"Batería '{new_battery.name}' creada con ID: {new_battery.id}.")
        return jsonify({'success': True, 'data': new_battery.to_dict()}), 201
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Error en el formato de fecha: {e}'}), 400
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al crear batería: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['GET'])
def get_battery_by_id(battery_id):
    """Obtener una batería por su ID."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404
        current_app.logger.debug(f"Obtenida batería con ID: {battery_id}.")
        return jsonify({'success': True, 'data': battery.to_dict()})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener batería por ID {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['PUT'])
def update_battery(battery_id):
    """Actualizar una batería existente."""
    battery = Battery.query.get(battery_id)
    if not battery:
        return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Datos de entrada no proporcionados'}), 400

    try:
        for key, value in data.items():
            if hasattr(battery, key):
                # Manejar campos de fecha que pueden venir como strings ISO
                if key in ['manufacturing_date', 'last_calibration_date', 'installation_date'] and isinstance(value, str):
                    setattr(battery, key, datetime.fromisoformat(value.replace('Z', '+00:00')))
                else:
                    setattr(battery, key, value)
        db.session.commit()
        current_app.logger.info(f"Batería con ID: {battery_id} actualizada.")
        return jsonify({'success': True, 'data': battery.to_dict()})
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Error en el formato de fecha: {e}'}), 400
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al actualizar batería con ID {battery_id}: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['DELETE'])
def delete_battery(battery_id):
    """Eliminar una batería."""
    battery = Battery.query.get(battery_id)
    if not battery:
        return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

    try:
        # Eliminar datos relacionados (opcional, dependiendo de la relación en el modelo)
        BatteryData.query.filter_by(battery_id=battery_id).delete()
        Alert.query.filter_by(battery_id=battery_id).delete()
        AnalysisResult.query.filter_by(battery_id=battery_id).delete()
        ThermalImage.query.filter_by(battery_id=battery_id).delete()
        MaintenanceRecord.query.filter_by(battery_id=battery_id).delete()

        db.session.delete(battery)
        db.session.commit()
        current_app.logger.info(f"Batería con ID: {battery_id} eliminada.")
        return jsonify({'success': True, 'message': 'Batería eliminada correctamente'})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al eliminar batería con ID {battery_id}: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['POST'])
def add_battery_data(battery_id):
    """Añadir nuevos datos de monitoreo para una batería."""
    battery = Battery.query.get(battery_id)
    if not battery:
        return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Datos de entrada no proporcionados'}), 400

    required_fields = ['voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Campo requerido faltante: {field}'}), 400

    try:
        # Usar el timestamp proporcionado o el actual si no se proporciona
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')) if data.get('timestamp') else datetime.now(timezone.utc)

        new_data = BatteryData(
            battery_id=battery_id,
            timestamp=timestamp,
            voltage=data['voltage'],
            current=data['current'],
            temperature=data['temperature'],
            soc=data['soc'],
            soh=data['soh'],
            cycles=data['cycles'],
            status=data.get('status', 'optimal') # Valor por defecto 'optimal'
        )
        db.session.add(new_data)
        db.session.commit()
        current_app.logger.info(f"Datos añadidos para batería {battery_id}. SOC: {new_data.soc}%, Temp: {new_data.temperature}°C.")
        return jsonify({'success': True, 'data': new_data.to_dict()}), 201
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Error en el formato de fecha/hora: {e}'}), 400
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al añadir datos para batería {battery_id}: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['GET'])
def get_battery_data(battery_id):
    """Obtener datos de monitoreo históricos para una batería, con filtrado por rango de tiempo e intervalo."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        time_range = request.args.get('time_range', 'last_24_hours') # Ej: 'last_7_days', 'last_30_days', 'all'
        interval = request.args.get('interval', 'hourly') # Ej: 'hourly', 'daily', 'monthly'

        query = BatteryData.query.filter_by(battery_id=battery_id)

        # Filtrado por rango de tiempo
        end_time = datetime.now(timezone.utc)
        if time_range == 'last_24_hours':
            start_time = end_time - timedelta(hours=24)
        elif time_range == 'last_7_days':
            start_time = end_time - timedelta(days=7)
        elif time_range == 'last_30_days':
            start_time = end_time - timedelta(days=30)
        elif time_range == 'last_year':
            start_time = end_time - timedelta(days=365)
        else: # 'all'
            start_time = None # No filter by start time

        if start_time:
            query = query.filter(BatteryData.timestamp >= start_time)

        # Ordenar por timestamp
        query = query.order_by(BatteryData.timestamp.asc())
        all_data = query.all()

        # Agregación por intervalo (simplificado para ejemplo)
        # En una aplicación real, usarías Pandas o funciones de base de datos para una agregación más eficiente
        aggregated_data = []
        if interval == 'hourly':
            # Agrupar por hora
            df = pd.DataFrame([d.to_dict() for d in all_data])
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                # Seleccionar columnas numéricas para promediar
                numeric_cols = ['voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
                # Asegurarse de que solo las columnas existentes y numéricas se incluyan en el promedio
                cols_to_avg = [col for col in numeric_cols if col in df.columns]
                
                if not cols_to_avg:
                    # Si no hay columnas numéricas para promediar, simplemente devolver los datos crudos o manejar
                    current_app.logger.warning(f"No hay columnas numéricas para promediar en datos de batería para intervalo {interval}.")
                    aggregated_data = [d.to_dict() for d in all_data] # Devuelve datos crudos si no hay columnas numéricas
                else:
                    aggregated_df = df[cols_to_avg].resample('H').mean().reset_index()
                    aggregated_data = aggregated_df.to_dict(orient='records')

        elif interval == 'daily':
            # Agrupar por día
            df = pd.DataFrame([d.to_dict() for d in all_data])
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                numeric_cols = ['voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
                cols_to_avg = [col for col in numeric_cols if col in df.columns]
                
                if not cols_to_avg:
                    current_app.logger.warning(f"No hay columnas numéricas para promediar en datos de batería para intervalo {interval}.")
                    aggregated_data = [d.to_dict() for d in all_data]
                else:
                    aggregated_df = df[cols_to_avg].resample('D').mean().reset_index()
                    aggregated_data = aggregated_df.to_dict(orient='records')
        
        elif interval == 'monthly':
            # Agrupar por mes
            df = pd.DataFrame([d.to_dict() for d in all_data])
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                numeric_cols = ['voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
                cols_to_avg = [col for col in numeric_cols if col in df.columns]
                
                if not cols_to_avg:
                    current_app.logger.warning(f"No hay columnas numéricas para promediar en datos de batería para intervalo {interval}.")
                    aggregated_data = [d.to_dict() for d in all_data]
                else:
                    aggregated_df = df[cols_to_avg].resample('M').mean().reset_index()
                    aggregated_data = aggregated_df.to_dict(orient='records')
        else: # No agregación (intervalo 'raw' o desconocido)
            aggregated_data = [d.to_dict() for d in all_data]

        current_app.logger.debug(f"Obtenidos {len(aggregated_data)} puntos de datos para batería {battery_id} con rango '{time_range}' e intervalo '{interval}'.")
        return jsonify({'success': True, 'data': aggregated_data, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener datos de batería para ID {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/summary', methods=['GET'])
def get_battery_summary(battery_id):
    """Obtener un resumen del estado actual de una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        latest_data = BatteryData.query.filter_by(battery_id=battery_id).order_by(BatteryData.timestamp.desc()).first()

        summary = {
            'battery_info': battery.to_dict(),
            'latest_data': latest_data.to_dict() if latest_data else None,
            'status': latest_data.status if latest_data else 'unknown'
        }
        current_app.logger.debug(f"Obtenido resumen para batería {battery_id}.")
        return jsonify({'success': True, 'data': summary})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener resumen de batería para ID {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/upload/battery_data', methods=['POST'])
def upload_battery_data():
    """Cargar datos de batería desde un archivo."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nombre de archivo vacío'}), 400
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'}), 400

    battery_id = request.form.get('battery_id')
    if not battery_id:
        return jsonify({'success': False, 'error': 'ID de batería no proporcionado'}), 400

    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        df = pd.read_csv(file) # Asume CSV, podrías añadir lógica para xlsx
        
        # Validar columnas necesarias en el CSV
        required_cols = ['timestamp', 'voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'success': False, 'error': f'El archivo CSV debe contener las columnas: {", ".join(required_cols)}'}), 400

        for index, row in df.iterrows():
            # Convertir timestamp a objeto datetime
            timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')) if isinstance(row['timestamp'], str) else row['timestamp']

            new_data = BatteryData(
                battery_id=battery_id,
                timestamp=timestamp,
                voltage=row['voltage'],
                current=row['current'],
                temperature=row['temperature'],
                soc=row['soc'],
                soh=row['soh'],
                cycles=row['cycles'],
                status=row.get('status', 'optimal')
            )
            db.session.add(new_data)
        db.session.commit()
        current_app.logger.info(f"Datos cargados exitosamente desde el archivo para batería {battery_id}. Total de filas: {len(df)}.")
        return jsonify({'success': True, 'message': f'Datos de {len(df)} filas cargados exitosamente.'}), 200
    except ValueError as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Error en el formato de datos del archivo (ej. fecha/hora): {e}'}), 400
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al cargar datos desde el archivo para batería {battery_id}: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/upload/thermal_image', methods=['POST'])
def upload_thermal_image():
    """Cargar una imagen térmica para una batería."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No se encontró el archivo de imagen en la solicitud'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nombre de archivo vacío'}), 400
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'success': False, 'error': 'Tipo de archivo de imagen no permitido'}), 400

    battery_id = request.form.get('battery_id')
    if not battery_id:
        return jsonify({'success': False, 'error': 'ID de batería no proporcionado'}), 400

    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        # Guardar la imagen en un directorio configurado o en almacenamiento en la nube
        # Por simplicidad, aquí se usa un enfoque básico de guardar en el sistema de archivos
        # En producción, considera servicios como S3, Google Cloud Storage, etc.
        
        # Generar un nombre de archivo único
        filename = f"{battery_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        new_image = ThermalImage(
            battery_id=battery_id,
            image_path=filepath, # Guardar la ruta donde se almacenó la imagen
            timestamp=datetime.now(timezone.utc),
            temperature_max=request.form.get('temperature_max'), # Asume que estos datos se envían
            temperature_min=request.form.get('temperature_min'),
            temperature_avg=request.form.get('temperature_avg')
        )
        db.session.add(new_image)
        db.session.commit()
        current_app.logger.info(f"Imagen térmica cargada para batería {battery_id}: {filename}.")
        return jsonify({'success': True, 'message': 'Imagen térmica cargada correctamente', 'data': new_image.to_dict()}), 201
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al cargar imagen térmica para batería {battery_id}: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/thermal_images', methods=['GET'])
def get_thermal_images(battery_id):
    """Obtener las imágenes térmicas de una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404
        
        images = ThermalImage.query.filter_by(battery_id=battery_id).order_by(ThermalImage.timestamp.desc()).all()
        images_list = [image.to_dict() for image in images]
        current_app.logger.debug(f"Obtenidas {len(images_list)} imágenes térmicas para batería {battery_id}.")
        return jsonify({'success': True, 'data': images_list})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener imágenes térmicas para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Servir archivos cargados (imágenes térmicas)."""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    return send_file(os.path.join(upload_folder, filename))

@battery_bp.route('/batteries/<int:battery_id>/alerts', methods=['GET'])
def get_battery_alerts(battery_id):
    """Obtener alertas para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        # MODIFICACIÓN: Cambiado 'Alert.timestamp' a 'Alert.id' para solucionar AttributeError.
        # Se recomienda añadir una columna 'timestamp' o 'created_at' al modelo Alert en src/models/battery.py
        # y luego usar 'Alert.timestamp.desc()' o 'Alert.created_at.desc()' aquí.
        alerts = Alert.query.filter_by(battery_id=battery_id).order_by(Alert.id.desc()).all()
        alerts_list = [alert.to_dict() for alert in alerts]

        current_app.logger.debug(f"Obtenidas {len(alerts_list)} alertas para batería {battery_id}.")
        return jsonify({'success': True, 'data': alerts_list, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener alertas para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/analysis_results', methods=['GET'])
def get_battery_analysis_results(battery_id):
    """Obtener resultados de análisis para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        # MODIFICACIÓN: Cambiado 'AnalysisResult.timestamp' a 'AnalysisResult.id' para solucionar AttributeError.
        # Se recomienda añadir una columna 'timestamp' o 'created_at' al modelo AnalysisResult en src/models/battery.py
        # y luego usar 'AnalysisResult.timestamp.desc()' o 'AnalysisResult.created_at.desc()' aquí.
        analysis_results = AnalysisResult.query.filter_by(battery_id=battery_id).order_by(AnalysisResult.id.desc()).all()
        results_list = [result.to_dict() for result in analysis_results]

        current_app.logger.debug(f"Obtenidos {len(results_list)} resultados de análisis para batería {battery_id}.")
        return jsonify({'success': True, 'data': results_list, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener resultados de análisis para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/maintenance_records', methods=['GET'])
def get_battery_maintenance_records(battery_id):
    """Obtener registros de mantenimiento para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        maintenance_records = MaintenanceRecord.query.filter_by(battery_id=battery_id).order_by(MaintenanceRecord.performed_at.desc()).all()
        records_list = [record.to_dict() for record in maintenance_records]

        current_app.logger.debug(f"Obtenidos {len(records_list)} registros de mantenimiento para batería {battery_id}.")
        return jsonify({'success': True, 'data': records_list, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener registros de mantenimiento para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/maintenance_records', methods=['POST'])
def add_maintenance_record(battery_id):
    """Añadir un nuevo registro de mantenimiento para una batería."""
    battery = Battery.query.get(battery_id)
    if not battery:
        return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Datos de entrada no proporcionados'}), 400

    required_fields = ['maintenance_type', 'description', 'performed_by', 'performed_at']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Campo requerido faltante: {field}'}), 400

    try:
        # Convertir la fecha/hora a objeto datetime
        performed_at = datetime.fromisoformat(data['performed_at'].replace('Z', '+00:00')) if isinstance(data['performed_at'], str) else data['performed_at']

        new_record = MaintenanceRecord(
            battery_id=battery_id,
            maintenance_type=data['maintenance_type'],
            description=data['description'],
            performed_by=data['performed_by'],
            performed_at=performed_at,
            cost=data.get('cost')
        )
        db.session.add(new_record)
        db.session.commit()
        current_app.logger.info(f"Registro de mantenimiento añadido para batería {battery_id}. Tipo: {new_record.maintenance_type}.")
        return jsonify({'success': True, 'data': new_record.to_dict()}), 201
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Error en el formato de fecha/hora: {e}'}), 400
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al añadir registro de mantenimiento para batería {battery_id}: {e}\n{error_trace}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

# Nueva ruta para simular la generación de un reporte de batería
from datetime import timedelta
@battery_bp.route('/batteries/<int:battery_id>/generate_report', methods=['GET'])
def generate_battery_report(battery_id):
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        # Obtener datos de los últimos 30 días
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)
        
        historical_data = BatteryData.query.filter_by(battery_id=battery_id)\
                                        .filter(BatteryData.timestamp >= start_time)\
                                        .order_by(BatteryData.timestamp.asc()).all()
        
        alerts = Alert.query.filter_by(battery_id=battery_id)\
                            .filter(Alert.timestamp >= start_time if hasattr(Alert, 'timestamp') else True)\
                            .order_by(Alert.id.desc()).all() # Usar id si no hay timestamp
        
        analysis_results = AnalysisResult.query.filter_by(battery_id=battery_id)\
                                                .filter(AnalysisResult.timestamp >= start_time if hasattr(AnalysisResult, 'timestamp') else True)\
                                                .order_by(AnalysisResult.id.desc()).all() # Usar id si no hay timestamp

        maintenance_records = MaintenanceRecord.query.filter_by(battery_id=battery_id)\
                                                .filter(MaintenanceRecord.performed_at >= start_time)\
                                                .order_by(MaintenanceRecord.performed_at.desc()).all()

        report_data = {
            'battery_info': battery.to_dict(),
            'historical_data': [d.to_dict() for d in historical_data],
            'alerts': [a.to_dict() for a in alerts],
            'analysis_results': [ar.to_dict() for ar in analysis_results],
            'maintenance_records': [mr.to_dict() for mr in maintenance_records],
            'report_generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Opcional: Generar un archivo HTML o PDF en el servidor y luego enviarlo
        # Por ahora, simplemente devolver el JSON de los datos del reporte.
        current_app.logger.info(f"Reporte generado para batería {battery_id}.")
        return jsonify({'success': True, 'data': report_data})

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al generar reporte para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500
