from flask import Blueprint, request, jsonify, current_app, send_file
from datetime import datetime, timezone, timedelta
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
            current_app.logger.info("Datos de batería REALES obtenidos del sistema.")
            return battery_info.get('data')
        else:
            current_app.logger.warning("No se pudieron obtener datos reales de la batería. Generando datos simulados.")
            return generate_simulated_battery_data()
    except Exception as e:
        current_app.logger.error(f"Error al obtener datos reales de la batería: {e}")
        return generate_simulated_battery_data()

def generate_simulated_battery_data():
    """Genera datos de batería simulados para propósitos de demostración."""
    now = datetime.now(timezone.utc)
    # Valores aleatorios dentro de un rango razonable para simulación
    voltage = round(np.random.uniform(11.8, 12.5), 2)
    current = round(np.random.uniform(0.5, 5.0), 2)
    temperature = round(np.random.uniform(20.0, 35.0), 2)
    soc = round(np.random.uniform(20, 100)) # State of Charge
    soh = round(np.random.uniform(70, 100)) # State of Health
    cycles = np.random.randint(50, 500)

    return {
        'timestamp': now.isoformat(),
        'voltage': voltage,
        'current': current,
        'temperature': temperature,
        'soc': soc,
        'soh': soh,
        'cycles': cycles
    }

def generate_sample_battery_data(battery_id, count=100):
    """Genera una lista de datos de batería simulados para una batería específica."""
    sample_data = []
    # Genera datos para los últimos 'count' días
    for i in range(count):
        timestamp = datetime.now(timezone.utc) - timedelta(days=count - 1 - i)
        voltage = round(np.random.uniform(11.8, 12.5), 2)
        current = round(np.random.uniform(0.5, 5.0), 2)
        temperature = round(np.random.uniform(20.0, 35.0), 2)
        soc = round(np.random.uniform(50, 95))
        soh = round(np.random.uniform(80, 98))
        cycles = np.random.randint(100, 500)

        sample_data.append({
            'battery_id': battery_id,
            'timestamp': timestamp.isoformat(),
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc': soc,
            'soh': soh,
            'cycles': cycles,
            'internal_resistance': round(np.random.uniform(0.01, 0.1), 3),
            'power': round(voltage * current, 2),
            'efficiency': round(np.random.uniform(85, 99), 2),
            'charge_rate': round(np.random.uniform(0, 1), 2),
            'discharge_rate': round(np.random.uniform(0, 1), 2),
            'ambient_temperature': round(np.random.uniform(15, 30), 2),
            'humidity': round(np.random.uniform(30, 80), 2),
        })
    return sample_data


@battery_bp.route('/batteries', methods=['GET'])
def get_batteries():
    """Obtener una lista de todas las baterías."""
    try:
        batteries = Battery.query.all()
        if not batteries:
            current_app.logger.info("No se encontraron baterías. Creando una batería principal por defecto.")
            try:
                # Crear una batería por defecto si no hay ninguna
                default_battery = Battery(
                    name='Batería Principal #001',
                    model='Li-ion 18650',
                    manufacturer='Default Mfg',
                    serial_number='DEFAULT-001',
                    capacity_ah=200,
                    voltage_nominal=12,
                    chemistry='LiFePO4',
                    installation_date=datetime.now(timezone.utc)
                )
                db.session.add(default_battery)
                db.session.commit()
                current_app.logger.info("Batería principal por defecto creada con éxito.")
                batteries = [default_battery] # Usar la batería recién creada
            except Exception as create_e:
                db.session.rollback()
                error_trace = traceback.format_exc()
                current_app.logger.error(f"Error al crear batería por defecto: {create_e}\n{error_trace}")
                return jsonify({'success': False, 'error': 'Error al inicializar baterías', 'traceback': error_trace}), 500

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
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON en la solicitud'}), 400

        required_fields = ['name', 'model', 'manufacturer', 'capacity_ah', 'voltage_nominal', 'chemistry']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Falta el campo requerido: {field}'}), 400

        # Validación y conversión de la fecha de instalación
        installation_date = None
        if 'installation_date' in data and data['installation_date']:
            try:
                # Añadir 'Z' a la cadena si falta y la zona horaria no está presente para ISO 8601
                date_str = data['installation_date']
                if not (date_str.endswith('Z') or '+' in date_str or '-' in date_str[len(date_str)-6:]):
                    date_str += 'Z'
                installation_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato de fecha de instalación inválido. Use ISO 8601.'}), 400
        else:
            installation_date = datetime.now(timezone.utc) # Por defecto a la fecha actual si no se proporciona

        new_battery = Battery(
            name=data['name'],
            model=data['model'],
            manufacturer=data['manufacturer'],
            serial_number=data.get('serial_number', None), # serial_number es opcional al crear, pero debe ser único
            capacity_ah=data['capacity_ah'],
            voltage_nominal=data['voltage_nominal'],
            chemistry=data['chemistry'],
            installation_date=installation_date,
            manufacturing_date=data.get('manufacturing_date', None),
            last_maintenance_date=data.get('last_maintenance_date', None),
            status=data.get('status', 'active'), # Estado por defecto
            location=data.get('location', None)
        )

        db.session.add(new_battery)
        db.session.commit()
        current_app.logger.info(f"Batería '{new_battery.name}' creada con éxito.")
        return jsonify({'success': True, 'data': new_battery.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al crear batería: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/batteries/<int:battery_id>', methods=['GET', 'OPTIONS'])
def get_battery(battery_id):
    """Obtener los detalles de una batería específica."""
    if request.method == 'OPTIONS':
        return '', 200 # Manejar la solicitud OPTIONS/preflight

    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404
        
        current_app.logger.debug(f"Detalles de batería {battery_id} obtenidos con éxito.")
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
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON en la solicitud'}), 400

        # Actualizar campos
        for key, value in data.items():
            if hasattr(battery, key):
                if key in ['installation_date', 'manufacturing_date', 'last_maintenance_date'] and value:
                    try:
                        # Convertir la fecha a formato datetime
                        date_str = value
                        if not (date_str.endswith('Z') or '+' in date_str or '-' in date_str[len(date_str)-6:]):
                            date_str += 'Z'
                        setattr(battery, key, datetime.fromisoformat(date_str.replace('Z', '+00:00')))
                    except ValueError:
                        return jsonify({'success': False, 'error': f'Formato de fecha inválido para {key}. Use ISO 8601.'}), 400
                else:
                    setattr(battery, key, value)

        db.session.commit()
        current_app.logger.info(f"Batería '{battery_id}' actualizada con éxito.")
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
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        db.session.delete(battery)
        db.session.commit()
        current_app.logger.info(f"Batería '{battery_id}' eliminada con éxito.")
        return jsonify({'success': True, 'message': 'Batería eliminada con éxito'}), 200
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al eliminar batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/battery/real-time', methods=['GET'])
def get_battery_real_time_data():
    """Obtener datos de batería en tiempo real (reales o simulados)."""
    try:
        data = get_real_battery_data()
        current_app.logger.debug("Datos de batería en tiempo real obtenidos.")
        return jsonify({'success': True, 'data': data, 'timestamp': datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener datos de batería en tiempo real: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/batteries/<int:battery_id>/data', methods=['POST', 'GET'])
def handle_battery_data(battery_id):
    """
    Gestiona la carga de nuevos datos de batería (POST)
    y la obtención de datos históricos para gráficos (GET) para una batería específica.
    """
    battery = Battery.query.get(battery_id)
    if not battery:
        return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

    if request.method == 'POST':
        current_app.logger.debug(f"Recibida solicitud POST para datos de batería {battery_id}.")
        try:
            data = request.json
            if not data:
                return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON'}), 400

            # Lógica para obtener datos reales si se solicita
            if data.get('get_real_data', False):
                real_data = windows_battery_service.get_battery_info()
                if real_data and real_data.get('success'):
                    data.update(real_data['data'])
                else:
                    current_app.logger.warning("No se pudieron obtener datos reales, utilizando simulados para POST.")
                    simulated_data = generate_simulated_battery_data()
                    data.update(simulated_data)
            
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')) if 'timestamp' in data else datetime.now(timezone.utc)

            new_data_point = BatteryData(
                battery_id=battery_id,
                timestamp=timestamp,
                voltage=data.get('voltage'),
                current=data.get('current'),
                temperature=data.get('temperature'),
                soc=data.get('soc'),
                soh=data.get('soh'),
                cycles=data.get('cycles'),
                internal_resistance=data.get('internal_resistance'),
                power=data.get('power'),
                efficiency=data.get('efficiency'),
                charge_rate=data.get('charge_rate'),
                discharge_rate=data.get('discharge_rate'),
                ambient_temperature=data.get('ambient_temperature'),
                humidity=data.get('humidity')
            )
            db.session.add(new_data_point)
            db.session.commit()
            current_app.logger.info(f"Dato añadido a batería {battery_id} en {new_data_point.timestamp}")
            return jsonify({'success': True, 'data': new_data_point.to_dict()}), 201
        except Exception as e:
            db.session.rollback()
            error_trace = traceback.format_exc()
            current_app.logger.error(f"Error al añadir datos a la batería {battery_id}: {e}\n{error_trace}")
            return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

    elif request.method == 'GET':
        current_app.logger.debug(f"Recibida solicitud GET para datos históricos de batería {battery_id}.")
        time_range = request.args.get('time_range', 'all_time') # 'last_24_hours', 'last_7_days', 'last_30_days', 'all_time'
        interval = request.args.get('interval', 'hourly') # 'hourly', 'daily', 'monthly'

        query = BatteryData.query.filter_by(battery_id=battery_id)

        # Filtrar por rango de tiempo
        now_utc = datetime.now(timezone.utc)
        if time_range == 'last_24_hours':
            query = query.filter(BatteryData.timestamp >= now_utc - timedelta(hours=24))
        elif time_range == 'last_7_days':
            query = query.filter(BatteryData.timestamp >= now_utc - timedelta(days=7))
        elif time_range == 'last_30_days':
            query = query.filter(BatteryData.timestamp >= now_utc - timedelta(days=30))
        # 'all_time' no necesita filtro adicional de tiempo

        historical_data = query.order_by(BatteryData.timestamp.asc()).all()
        
        # Si no hay datos en la base de datos, generar algunos datos de ejemplo.
        if not historical_data:
            current_app.logger.info(f"No se encontraron datos históricos para la batería {battery_id}. Generando datos de ejemplo.")
            sample_data_list = generate_sample_battery_data(battery_id, count=100)
            return jsonify({'success': True, 'data': sample_data_list, 'battery_id': battery_id})

        data_list = [data.to_dict() for data in historical_data]

        current_app.logger.debug(f"Obtenidos {len(data_list)} puntos de datos históricos para batería {battery_id} con rango '{time_range}' e intervalo '{interval}'.")
        return jsonify({'success': True, 'data': data_list, 'battery_id': battery_id})
    
    return jsonify({'success': False, 'error': 'Método no permitido para esta ruta.'}), 405


@battery_bp.route('/battery/upload_data', methods=['POST'])
def upload_battery_data():
    """Cargar datos de batería desde un archivo CSV/Excel."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud'}), 400

        file = request.files['file']
        battery_id = request.form.get('battery_id')

        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nombre de archivo vacío'}), 400

        if not battery_id:
            return jsonify({'success': False, 'error': 'Se requiere el ID de la batería para cargar datos'}), 400

        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            df = None

            if file_extension in ['csv', 'txt']:
                df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file.stream)
            else:
                return jsonify({'success': False, 'error': 'Formato de archivo no soportado'}), 400

            # Asegúrate de que las columnas esperadas están presentes
            expected_columns = ['timestamp', 'voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
            if not all(col in df.columns for col in expected_columns):
                missing_cols = [col for col in expected_columns if col not in df.columns]
                return jsonify({'success': False, 'error': f'Faltan columnas requeridas en el archivo: {", ".join(missing_cols)}'}), 400

            data_to_add = []
            for index, row in df.iterrows():
                try:
                    timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')) if isinstance(row['timestamp'], str) else row['timestamp']
                    # Asegurarse de que el timestamp sea un objeto datetime
                    if not isinstance(timestamp, datetime):
                        # Intentar convertir si es un timestamp numérico (ej. de Excel)
                        if isinstance(timestamp, (int, float)):
                            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        else:
                            raise ValueError("Timestamp no es un formato de fecha reconocido.")

                    data_to_add.append(BatteryData(
                        battery_id=battery_id,
                        timestamp=timestamp,
                        voltage=row['voltage'],
                        current=row['current'],
                        temperature=row['temperature'],
                        soc=row['soc'],
                        soh=row['soh'],
                        cycles=row['cycles'],
                        # Incluye otros campos si existen en el archivo y en tu modelo
                        internal_resistance=row.get('internal_resistance'),
                        power=row.get('power'),
                        efficiency=row.get('efficiency'),
                        charge_rate=row.get('charge_rate'),
                        discharge_rate=row.get('discharge_rate'),
                        ambient_temperature=row.get('ambient_temperature'),
                        humidity=row.get('humidity')
                    ))
                except Exception as row_e:
                    current_app.logger.warning(f"Error al procesar fila {index+1}: {row_e}. Fila omitida.")
                    continue

            if not data_to_add:
                return jsonify({'success': False, 'error': 'No se pudieron extraer datos válidos del archivo o el archivo está vacío'}), 400

            db.session.bulk_save_objects(data_to_add)
            db.session.commit()
            current_app.logger.info(f"Se cargaron {len(data_to_add)} puntos de datos para la batería {battery_id}.")
            return jsonify({'success': True, 'message': f'Se cargaron {len(data_to_add)} puntos de datos con éxito'}), 200
        else:
            return jsonify({'success': False, 'error': 'Tipo de archivo no permitido'}), 400
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al cargar datos de batería: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/batteries/<int:battery_id>/alerts', methods=['GET'])
def get_battery_alerts(battery_id):
    """Obtener alertas para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        alerts = Alert.query.filter_by(battery_id=battery_id).order_by(Alert.timestamp.desc()).all()
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

        analysis_results = AnalysisResult.query.filter_by(battery_id=battery_id).order_by(AnalysisResult.timestamp.desc()).all()
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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON'}), 400

        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        required_fields = ['description', 'performed_by', 'cost', 'next_due_date']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Falta el campo requerido: {field}'}), 400

        # Convertir fechas
        performed_at = datetime.fromisoformat(data.get('performed_at', datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
        next_due_date = datetime.fromisoformat(data['next_due_date'].replace('Z', '+00:00'))

        new_record = MaintenanceRecord(
            battery_id=battery_id,
            description=data['description'],
            performed_at=performed_at,
            performed_by=data['performed_by'],
            cost=data['cost'],
            next_due_date=next_due_date,
            notes=data.get('notes')
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


@battery_bp.route('/thermal_images', methods=['POST'])
def upload_thermal_image():
    """Cargar una imagen térmica para una batería."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró la imagen en la solicitud'}), 400

        file = request.files['file']
        battery_id = request.form.get('battery_id')

        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nombre de archivo de imagen vacío'}), 400

        if not battery_id:
            return jsonify({'success': False, 'error': 'Se requiere el ID de la batería para cargar la imagen térmica'}), 400

        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            # Guardar la imagen en una ubicación accesible (ej. /static/thermal_images)
            # Asegúrate de que el directorio exista
            upload_folder = os.path.join(current_app.root_path, 'static', 'thermal_images')
            os.makedirs(upload_folder, exist_ok=True)

            filename = f"battery_{battery_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            # Guardar la ruta en la base de datos
            new_image = ThermalImage(
                battery_id=battery_id,
                timestamp=datetime.now(timezone.utc),
                image_url=f"/static/thermal_images/{filename}",
                temperature_data=json.loads(request.form.get('temperature_data', '{}')) # Datos de temperatura como JSON string
            )
            db.session.add(new_image)
            db.session.commit()
            current_app.logger.info(f"Imagen térmica cargada para batería {battery_id}.")
            return jsonify({'success': True, 'message': 'Imagen térmica cargada con éxito', 'data': new_image.to_dict()}), 201
        else:
            return jsonify({'success': False, 'error': 'Tipo de archivo de imagen no permitido'}), 400
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al cargar imagen térmica: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/batteries/<int:battery_id>/thermal_images', methods=['GET'])
def get_thermal_images(battery_id):
    """Obtener lista de imágenes térmicas para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        thermal_images = ThermalImage.query.filter_by(battery_id=battery_id).order_by(ThermalImage.timestamp.desc()).all()
        images_list = [img.to_dict() for img in thermal_images]

        current_app.logger.debug(f"Obtenidas {len(images_list)} imágenes térmicas para batería {battery_id}.")
        return jsonify({'success': True, 'data': images_list, 'battery_id': battery_id})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener imágenes térmicas para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/static/thermal_images/<filename>')
def serve_thermal_image(filename):
    """Servir imágenes térmicas estáticas."""
    try:
        return send_file(os.path.join(current_app.root_path, 'static', 'thermal_images', filename))
    except Exception as e:
        current_app.logger.error(f"Error al servir imagen térmica {filename}: {e}")
        return jsonify({'success': False, 'error': 'Imagen no encontrada o error del servidor'}), 404
