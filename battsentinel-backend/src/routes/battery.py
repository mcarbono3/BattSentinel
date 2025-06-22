from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import cv2
from PIL import Image
from src.models.battery import db, Battery, BatteryData, ThermalImage
from src.services.data_processor import DataProcessor
from src.services.thermal_analyzer import ThermalAnalyzer

battery_bp = Blueprint('battery', __name__)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'xls'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@battery_bp.route('/batteries', methods=['GET'])
def get_batteries():
    """Obtener todas las baterías"""
    try:
        batteries = Battery.query.all()
        return jsonify({
            'success': True,
            'data': [battery.to_dict() for battery in batteries]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries', methods=['POST'])
def create_battery():
    """Crear una nueva batería"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        
        battery = Battery(
            name=data['name'],
            battery_type=data.get('battery_type', 'Li-ion'),
            device_type=data.get('device_type')
        )
        
        db.session.add(battery)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': battery.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>', methods=['GET'])
def get_battery(battery_id):
    """Obtener una batería específica"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        return jsonify({
            'success': True,
            'data': battery.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/data', methods=['GET'])
def get_battery_data(battery_id):
    """Obtener datos de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Parámetros de consulta
        limit = request.args.get('limit', 1000, type=int)
        offset = request.args.get('offset', 0, type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = BatteryData.query.filter_by(battery_id=battery_id)
        
        # Filtros de fecha
        if start_date:
            query = query.filter(BatteryData.timestamp >= datetime.fromisoformat(start_date))
        if end_date:
            query = query.filter(BatteryData.timestamp <= datetime.fromisoformat(end_date))
        
        # Ordenar por timestamp descendente
        query = query.order_by(BatteryData.timestamp.desc())
        
        # Paginación
        data_points = query.offset(offset).limit(limit).all()
        total_count = query.count()
        
        return jsonify({
            'success': True,
            'data': [point.to_dict() for point in data_points],
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload-data', methods=['POST'])
def upload_battery_data(battery_id):
    """Cargar datos de batería desde archivo"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({
                'success': False, 
                'error': 'Invalid file type. Allowed: CSV, TXT, XLSX'
            }), 400
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Procesar archivo
            processor = DataProcessor()
            data_points = processor.process_file(temp_path, battery_id)
            
            # Guardar en base de datos
            for point_data in data_points:
                data_point = BatteryData(**point_data)
                db.session.add(data_point)
            
            db.session.commit()
            
            # Limpiar archivo temporal
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded {len(data_points)} data points',
                'count': len(data_points)
            })
            
        except Exception as e:
            # Limpiar archivo temporal en caso de error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/upload-thermal', methods=['POST'])
def upload_thermal_image(battery_id):
    """Cargar imagen térmica"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                'success': False, 
                'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, TIFF'
            }), 400
        
        # Generar nombre único
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Guardar archivo
        file.save(file_path)
        
        try:
            # Crear registro en base de datos
            thermal_image = ThermalImage(
                battery_id=battery_id,
                filename=unique_filename,
                original_filename=filename,
                file_path=file_path
            )
            
            db.session.add(thermal_image)
            db.session.commit()
            
            # Analizar imagen térmica en segundo plano
            analyzer = ThermalAnalyzer()
            analysis_result = analyzer.analyze_image(file_path)
            
            # Actualizar registro con resultados del análisis
            thermal_image.max_temperature = analysis_result.get('max_temperature')
            thermal_image.min_temperature = analysis_result.get('min_temperature')
            thermal_image.avg_temperature = analysis_result.get('avg_temperature')
            thermal_image.hotspot_detected = analysis_result.get('hotspot_detected', False)
            thermal_image.hotspot_coordinates = json.dumps(analysis_result.get('hotspot_coordinates', []))
            thermal_image.analysis_completed = True
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': thermal_image.to_dict(),
                'analysis': analysis_result
            })
            
        except Exception as e:
            # Limpiar archivo en caso de error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/thermal-images', methods=['GET'])
def get_thermal_images(battery_id):
    """Obtener imágenes térmicas de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        images = ThermalImage.query.filter_by(battery_id=battery_id).order_by(ThermalImage.upload_timestamp.desc()).all()
        
        return jsonify({
            'success': True,
            'data': [image.to_dict() for image in images]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/add-data', methods=['POST'])
def add_battery_data(battery_id):
    """Agregar datos de batería manualmente o desde API"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Crear punto de datos
        data_point = BatteryData(
            battery_id=battery_id,
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            voltage=data.get('voltage'),
            current=data.get('current'),
            power=data.get('power'),
            soc=data.get('soc'),
            soh=data.get('soh'),
            capacity=data.get('capacity'),
            cycles=data.get('cycles'),
            temperature=data.get('temperature'),
            internal_resistance=data.get('internal_resistance'),
            data_source=data.get('data_source', 'api')
        )
        
        db.session.add(data_point)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': data_point.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@battery_bp.route('/batteries/<int:battery_id>/summary', methods=['GET'])
def get_battery_summary(battery_id):
    """Obtener resumen estadístico de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Obtener datos recientes
        recent_data = BatteryData.query.filter_by(battery_id=battery_id)\
            .order_by(BatteryData.timestamp.desc()).limit(1000).all()
        
        if not recent_data:
            return jsonify({
                'success': True,
                'data': {
                    'battery': battery.to_dict(),
                    'stats': {},
                    'message': 'No data available'
                }
            })
        
        # Calcular estadísticas
        df = pd.DataFrame([point.to_dict() for point in recent_data])
        
        stats = {}
        numeric_columns = ['voltage', 'current', 'power', 'soc', 'soh', 'capacity', 'temperature', 'internal_resistance']
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().any():
                stats[col] = {
                    'current': float(df[col].iloc[0]) if pd.notna(df[col].iloc[0]) else None,
                    'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
                    'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
                    'std': float(df[col].std()) if pd.notna(df[col].std()) else None
                }
        
        # Información adicional
        stats['data_points'] = len(recent_data)
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
def delete_battery(battery_id):
    """Eliminar una batería y todos sus datos"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Eliminar archivos de imágenes térmicas
        thermal_images = ThermalImage.query.filter_by(battery_id=battery_id).all()
        for image in thermal_images:
            if os.path.exists(image.file_path):
                os.remove(image.file_path)
        
        # Eliminar batería (cascade eliminará datos relacionados)
        db.session.delete(battery)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Battery deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

