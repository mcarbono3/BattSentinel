import os
import sys
import psutil
import platform
import subprocess
from datetime import datetime, timezone

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# Importaciones de modelos y rutas
from models.battery import db, Battery, BatteryData
from models.user import User
from routes.battery import battery_bp
from routes.ai_analysis import ai_bp
from routes.digital_twin import twin_bp
from routes.notifications import notifications_bp
from services.windows_battery import windows_battery_service

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'BattSentinel#2024$SecureKey!AI'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///batt_sentinel.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Desactivar seguimiento de modificaciones para reducir sobrecarga

# Inicializar db con la aplicación
db.init_app(app)

# Enable CORS for all routes - Sin restricciones
CORS(app, origins="*")

# Register blueprints - Sin auth_bp
app.register_blueprint(battery_bp)
app.register_blueprint(ai_bp)
app.register_blueprint(twin_bp)
app.register_blueprint(notifications_bp)

# Función para obtener datos reales de batería de Windows
# Endpoint para obtener datos reales de batería
@app.route('/api/battery/real-time', methods=['GET'])
def get_real_time_battery():
    """Endpoint para obtener datos de batería en tiempo real"""
    try:
        battery_data = windows_battery_service.get_battery_info()
        return jsonify({
            'success': True,
            'data': battery_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/battery/health-analysis', methods=['GET'])
def get_battery_health_analysis():
    """Endpoint para obtener análisis de salud de la batería"""
    try:
        analysis = windows_battery_service.get_battery_health_analysis()
        return jsonify({
            'success': True,
            'data': analysis
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Endpoint para obtener información del sistema"""
    try:
        import platform
        import psutil
        
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'hostname': platform.node(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': system_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Endpoint de salud del sistema
@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud del sistema"""
    return jsonify({
        'success': True,
        'message': 'BattSentinel Backend is running',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '2.0.0-no-auth'
    })

# Ensure database tables are created
with app.app_context():
    db.create_all()
    print("Tablas de la base de datos verificadas/creadas.")
    
    # Crear usuario admin por defecto (opcional, ya que no hay autenticación)
    if not User.query.filter_by(username='admin').first():
        try:
            admin_user = User(username='admin', email='admin@battsentinel.com', role='admin')
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            print("Usuario 'admin' creado en la base de datos.")
        except Exception as e:
            db.session.rollback()
            print(f"Error al crear el usuario 'admin': {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

