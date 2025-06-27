Aquí tienes el código mejorado para tu archivo `src/routes/battery.py`. He añadido los endpoints que faltaban para alertas, resultados de análisis específicos por batería, y registros de mantenimiento, además de mantener toda la funcionalidad existente.

Este código asume que tus modelos (`Alert`, `AnalysisResult`, `MaintenanceRecord`) están correctamente definidos en `src.models.battery` y que la instancia `db` se importa correctamente desde `src.main`.

```python
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
            current_app.logger.info("Datos de batería REALES obtenidos exitosamente del sistema Windows.")
            return battery_info['data']
        else:
            error_message = battery_info.get('error', 'Error desconocido al obtener datos reales.')
            current_app.logger.warning(f"No se pudieron obtener datos reales de la batería: {error_message}. Usando datos simulados.")
            current_app.logger.debug(f"Detalle del fallback: {battery_info.get('traceback', 'No traceback provided.')}")

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Excepción crítica al intentar obtener datos reales del sistema: {e}\n{error_trace}")
        current_app.logger.info("Recurriendo a la generación de datos de batería SIMULADOS debido a una excepción.")
        
    current_app.logger.info("Generando y usando datos de batería SIMULADOS.")
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

def generate_sample_battery_data(battery_id, count=100):
    """Generar datos de ejemplo para la batería."""
    import random
    from datetime import timedelta

    sample_data = []
    base_time = datetime.now(timezone.utc)

    for i in range(count):
        timestamp = base_time - timedelta(minutes=i * 5)

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


# === RUTAS DE API ===

@battery_bp.route('/batteries', methods=['GET'])
def get_batteries():
    """Obtener lista de baterías."""
    try:
        current_app.logger.debug("Intentando obtener lista de baterías...")
        batteries = Battery.query.all()

        if not batteries:
            current_app.logger.info("No se encontraron baterías. Creando una batería por defecto persistente.")
            try:
                existing_default = Battery.query.get(1)
                if not existing_default:
                    default_battery = Battery(
                        name='Batería Principal',
                        model='BS-100',
                        manufacturer='BattSentinel',
                        serial_number='BS001',
                        capacity_ah=100.0,
                        voltage_nominal=12.0,
                        chemistry='Li-ion',
                        installation_date=datetime.now(timezone.utc),
                        location='Ubicación por Defecto',
                        status='active'
                    )
                    db.session.add(default_battery)
                    db.session.commit()
                    current_app.logger.info("Batería por defecto creada exitosamente.")
                else:
                    current_app.logger.info("Batería por defecto (ID 1) ya existe, no se crea una nueva.")
                batteries = Battery.query.all()
            except Exception as e_db:
                db.session.rollback()
                error_trace_db = traceback.format_exc()
                current_app.logger.error(f"Error al crear batería por defecto: {e_db}\n{error_trace_db}")
                return jsonify({'success': False, 'error': f'Error al inicializar baterías: {str(e_db)}'}), 500

        batteries_data = [battery.to_dict() for battery in batteries]

        return jsonify({
            'success': True,
            'data': batteries_data
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error en GET /batteries: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries', methods=['POST'])
def create_battery():
    """Crear una nueva batería."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON'}), 400

        required_fields = ['name', 'model', 'manufacturer', 'capacity_ah', 'voltage_nominal', 'chemistry']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Falta el campo requerido: {field}'}), 400

        installation_date = None
        if 'installation_date' in data and data['installation_date']:
            try:
                installation_date = datetime.fromisoformat(data['installation_date'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato de fecha de instalación inválido. Use ISO 8601.'}), 400

        new_battery = Battery(
            name=data['name'],
            model=data['model'],
            manufacturer=data['manufacturer'],
            serial_number=data.get('serial_number'),
            capacity_ah=data['capacity_ah'],
            voltage_nominal=data['voltage_nominal'],
            chemistry=data['chemistry'],
            installation_date=installation_date,
            location=data.get('location'),
            status=data.get('status', 'active')
        )
        db.session.add(new_battery)
        db.session.commit()
        current_app.logger.info(f"Batería '{new_battery.name}' creada con ID: {new_battery.id}")
        return jsonify({'success': True, 'data': new_battery.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al crear batería: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['GET', 'OPTIONS'])
def get_battery_by_id(battery_id):
    """Obtener una batería específica por ID."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

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

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON'}), 400

        battery.name = data.get('name', battery.name)
        battery.model = data.get('model', battery.model)
        battery.manufacturer = data.get('manufacturer', battery.manufacturer)
        battery.serial_number = data.get('serial_number', battery.serial_number)
        battery.capacity_ah = data.get('capacity_ah', battery.capacity_ah)
        battery.voltage_nominal = data.get('voltage_nominal', battery.voltage_nominal)
        battery.chemistry = data.get('chemistry', battery.chemistry)
        if 'installation_date' in data and data['installation_date'] is not None:
            try:
                battery.installation_date = datetime.fromisoformat(data['installation_date'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'success': False, 'error': 'Formato de fecha de instalación inválido. Use ISO 8601.'}), 400
        elif 'installation_date' in data and data['installation_date'] is None:
            battery.installation_date = None
        battery.location = data.get('location', battery.location)
        battery.status = data.get('status', battery.status)
        battery.updated_at = datetime.utcnow()

        db.session.commit()
        current_app.logger.info(f"Batería '{battery.name}' (ID: {battery.id}) actualizada.")
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
        current_app.logger.info(f"Batería (ID: {battery_id}) eliminada exitosamente.")
        return jsonify({'success': True, 'message': 'Batería eliminada exitosamente'}), 200
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al eliminar batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/battery/real-time', methods=['GET'])
def get_real_time_data():
    """Obtener datos en tiempo real de la batería o simulados."""
    try:
        battery_data = get_real_battery_data()

        return jsonify({
            'success': True,
            'data': battery_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error en GET /battery/real-time: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/battery/<int:battery_id>/data', methods=['GET'])
def get_battery_data(battery_id):
    """Obtener datos históricos de una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        limit = request.args.get('limit', 100, type=int)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        query = BatteryData.query.filter_by(battery_id=battery_id)

        try:
            if start_date_str:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                query = query.filter(BatteryData.timestamp >= start_date)
            if end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                query = query.filter(BatteryData.timestamp <= end_date)

            data_points = query.order_by(BatteryData.timestamp.desc()).limit(limit).all()
            data_list = [point.to_dict() for point in data_points]
            current_app.logger.debug(f"Obtenidos {len(data_list)} puntos de datos para batería {battery_id}.")
        except Exception as e_query:
            current_app.logger.warning(f"Error al consultar datos históricos para batería {battery_id}: {e_query}. Generando datos de ejemplo.")
            data_list = generate_sample_battery_data(battery_id, limit)

        return jsonify({
            'success': True,
            'data': data_list,
            'battery_id': battery_id,
            'count': len(data_list)
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error en GET /battery/{battery_id}/data: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['POST'])
def add_battery_data(battery_id):
    """Agrega un nuevo punto de dato a una batería."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No se proporcionaron datos JSON'}), 400

        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if data.get('get_real_data', False):
            real_data = get_real_battery_data()
            if real_data:
                data.update(real_data)
            else:
                return jsonify({'success': False, 'error': 'No se pudieron obtener datos reales de la batería'}), 500

        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')) if 'timestamp' in data else datetime.utcnow()

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

@battery_bp.route('/batteries/<int:battery_id>/summary', methods=['GET'])
def get_battery_summary(battery_id):
    """Obtener un resumen de los datos de una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        data_points = BatteryData.query.filter_by(battery_id=battery_id).order_by(BatteryData.timestamp.desc()).all()
        if not data_points:
            return jsonify({
                'success': True,
                'battery_id': battery_id,
                'message': 'No hay datos disponibles para esta batería',
                'summary': {}
            })

        df = pd.DataFrame([dp.to_dict() for dp in data_points])

        summary = {
            'latest_timestamp': df['timestamp'].max().isoformat() if not df['timestamp'].empty and 'timestamp' in df.columns else None,
            'latest_voltage': df['voltage'].iloc[0] if 'voltage' in df.columns and not df.empty else None,
            'latest_current': df['current'].iloc[0] if 'current' in df.columns and not df.empty else None,
            'latest_temperature': df['temperature'].iloc[0] if 'temperature' in df.columns and not df.empty else None,
            'latest_soc': df['soc'].iloc[0] if 'soc' in df.columns and not df.empty else None,
            'latest_soh': df['soh'].iloc[0] if 'soh' in df.columns and not df.empty else None,
            'latest_cycles': df['cycles'].iloc[0] if 'cycles' in df.columns and not df.empty else None,
            'avg_voltage': df['voltage'].mean() if 'voltage' in df.columns else None,
            'avg_temperature': df['temperature'].mean() if 'temperature' in df.columns else None,
            'min_soc': df['soc'].min() if 'soc' in df.columns else None,
            'max_soc': df['soc'].max() if 'soc' in df.columns else None,
            'total_data_points': len(df)
        }
        return jsonify({'success': True, 'battery_id': battery_id, 'summary': summary})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener resumen de la batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/battery/health-analysis', methods=['GET'])
def get_health_analysis():
    """Obtener análisis de salud general de la batería."""
    try:
        current_data = get_real_battery_data()

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
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error en GET /battery/health-analysis: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload-data', methods=['POST'])
def upload_battery_data(battery_id):
    """Subir datos de batería desde archivo."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó ningún archivo'}), 400

        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Tipo de archivo no soportado'}), 400

        df = None
        if file.filename.lower().endswith(('.csv', '.txt')):
            df = pd.read_csv(file)
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': 'Tipo de archivo no soportado. Se esperaba CSV, TXT, XLSX o XLS.'}), 400

        expected_columns = ['timestamp', 'voltage', 'current', 'temperature', 'soc', 'soh', 'cycles', 'internal_resistance', 'power', 'efficiency', 'charge_rate', 'discharge_rate', 'ambient_temperature', 'humidity']
        for col in expected_columns:
            if col not in df.columns:
                current_app.logger.warning(f"Columna '{col}' no encontrada en el archivo. Se intentará continuar sin ella.")

        records_added = 0
        for index, row in df.iterrows():
            try:
                timestamp_raw = row.get('timestamp')
                timestamp = None
                if pd.notna(timestamp_raw):
                    try:
                        timestamp = pd.to_datetime(timestamp_raw, errors='coerce', utc=True).to_pydatetime()
                        if timestamp and not timestamp.tzinfo:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except Exception as e_ts:
                        current_app.logger.warning(f"Fila {index}: Timestamp '{timestamp_raw}' inválido o no convertible. Error: {e_ts}. Saltando fila.")
                        continue

                if timestamp is None:
                    current_app.logger.warning(f"Fila {index}: Timestamp es None después de procesamiento. Saltando fila.")
                    continue

                data_point = BatteryData(
                    battery_id=battery_id,
                    timestamp=timestamp,
                    voltage=row.get('voltage'),
                    current=row.get('current'),
                    temperature=row.get('temperature'),
                    soc=row.get('soc'),
                    soh=row.get('soh'),
                    cycles=row.get('cycles'),
                    internal_resistance=row.get('internal_resistance'),
                    power=row.get('power'),
                    efficiency=row.get('efficiency'),
                    charge_rate=row.get('charge_rate'),
                    discharge_rate=row.get('discharge_rate'),
                    ambient_temperature=row.get('ambient_temperature'),
                    humidity=row.get('humidity')
                )
                db.session.add(data_point)
                records_added += 1
            except Exception as ex:
                current_app.logger.error(f"Error procesando la fila {index} del archivo '{file.filename}': {ex}. Saltando esta fila.")
                continue

        db.session.commit()
        current_app.logger.info(f"Se agregaron {records_added} registros de datos para batería {battery_id}.")
        return jsonify({'success': True, 'message': f'Se agregaron {records_added} registros de datos'}), 200
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al subir datos para la batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/battery/<int:battery_id>/export-data', methods=['GET'])
def export_battery_data(battery_id):
    """Exportar datos de batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        format_type = request.args.get('format', 'json').lower()
        limit = request.args.get('limit', 1000, type=int)

        data_points = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(limit).all()
        data_list = [point.to_dict() for point in data_points]

        export_data = {
            'battery_info': {
                'id': battery_id,
                'name': battery.name,
                'model': battery.model,
                'serial_number': battery.serial_number,
                'export_date': datetime.now(timezone.utc).isoformat(),
                'total_records': len(data_list)
            },
            'data': data_list
        }

        if format_type == 'json':
            return jsonify({'success': True, 'data': export_data, 'format': 'json'})
        elif format_type == 'csv':
            df = pd.DataFrame(data_list)
            if not df.empty:
                for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
                    df[col] = df[col].dt.isoformat()

                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                return send_file(io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
                                 mimetype='text/csv',
                                 as_attachment=True,
                                 download_name=f'battery_data_{battery_id}.csv')
            else:
                return jsonify({'success': False, 'error': 'No hay datos para exportar en CSV'}), 404
        else:
            return jsonify({'success': False, 'error': 'Formato de exportación no soportado. Use "json" o "csv"'}), 400

    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al exportar datos para la batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload-thermal-image', methods=['POST'])
def upload_thermal_image(battery_id):
    """Sube una imagen térmica para una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se encontró el archivo en la solicitud'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó ninguna imagen'}), 400

        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'success': False, 'error': 'Formato de imagen no permitido'}), 400
        
        # Guardar la imagen en un directorio local
        # Asegúrate de que este directorio exista y sea accesible
        upload_folder = os.path.join(current_app.root_path, 'static', 'thermal_images')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Generar un nombre de archivo único para evitar colisiones
        unique_filename = f"{battery_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{file.filename}"
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        
        # Determinar el mimetype si no viene en request.files
        mimetype = file.mimetype if file.mimetype else 'application/octet-stream'

        new_image = ThermalImage(
            battery_id=battery_id,
            filename=unique_filename,
            file_path=file_path, # Guardar la ruta relativa o absoluta
            file_size=os.path.getsize(file_path),
            image_format=os.path.splitext(unique_filename)[1].lower().replace('.', ''),
            # Los metadatos de temperatura y análisis se añadirían aquí si se procesa la imagen
            uploaded_at=datetime.utcnow()
        )
        db.session.add(new_image)
        db.session.commit()
        current_app.logger.info(f"Imagen térmica '{unique_filename}' subida para batería {battery_id}. ID: {new_image.id}")
        return jsonify({'success': True, 'message': 'Imagen térmica subida exitosamente', 'image_id': new_image.id, 'image_url': f'/static/thermal_images/{unique_filename}'}), 201
    except Exception as e:
        db.session.rollback()
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al subir imagen térmica para la batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

@battery_bp.route('/batteries/<int:battery_id>/thermal-images/<int:image_id>/view', methods=['GET'])
def view_thermal_image(battery_id, image_id):
    """Sirve una imagen térmica específica."""
    try:
        image = ThermalImage.query.filter_by(id=image_id, battery_id=battery_id).first()
        if not image:
            return jsonify({'success': False, 'error': 'Imagen térmica no encontrada'}), 404
        
        # Asumiendo que file_path es la ruta completa al archivo
        # O la ruta relativa desde la carpeta 'static'
        # Asegúrate de que el STATIC_FOLDER esté configurado correctamente en Flask
        return send_file(image.file_path, mimetype=image.mimetype)
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al servir imagen térmica {image_id} para batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


@battery_bp.route('/batteries/<int:battery_id>/thermal-images', methods=['GET'])
def get_thermal_images(battery_id):
    """Obtiene metadatos de imágenes térmicas para una batería."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        images = ThermalImage.query.filter_by(battery_id=battery_id).order_by(ThermalImage.uploaded_at.desc()).all()
        if not images:
            return jsonify({'success': True, 'images': [], 'message': 'No hay imágenes térmicas para esta batería'})

        image_list = []
        for img in images:
            image_list.append({
                'id': img.id,
                'filename': img.filename,
                'mimetype': img.image_format, # Usar image_format del modelo
                'uploaded_at': img.uploaded_at.isoformat(),
                'width': img.width,
                'height': img.height,
                'min_temperature': img.min_temperature,
                'max_temperature': img.max_temperature,
                'avg_temperature': img.avg_temperature,
                'hotspots_detected': img.hotspots_detected,
                'anomalies_detected': img.anomalies_detected,
                'analysis_result': json.loads(img.analysis_result) if img.analysis_result else {},
                'captured_at': img.captured_at.isoformat() if img.captured_at else None,
                'processed_at': img.processed_at.isoformat() if img.processed_at else None,
                # URL para obtener la imagen real. Asegúrate de que /static sea accesible
                'url': f'/static/thermal_images/{img.filename}' # Asume que el filename es único y se guarda en static/thermal_images
            })
        current_app.logger.debug(f"Obtenidas {len(image_list)} imágenes térmicas para batería {battery_id}.")
        return jsonify({'success': True, 'images': image_list})
    except Exception as e:
        error_trace = traceback.format_exc()
        current_app.logger.error(f"Error al obtener imágenes térmicas para la batería {battery_id}: {e}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500

# === NUEVOS ENDPOINTS SOLICITADOS ===

@battery_bp.route('/batteries/<int:battery_id>/alerts', methods=['GET'])
def get_battery_alerts(battery_id):
    """Obtener alertas para una batería específica."""
    try:
        battery = Battery.query.get(battery_id)
        if not battery:
            return jsonify({'success': False, 'error': 'Batería no encontrada'}), 404

        alerts = Alert.query.filter_by(battery_id=battery_id).order_by(Alert.created_at.desc()).all()
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

        analysis_results = AnalysisResult.query.filter_by(battery_id=battery_id).order_by(AnalysisResult.created_at.desc()).all()
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
