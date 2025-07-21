import os
import sys
import psutil
import platform
import subprocess
from datetime import datetime, timezone

print("DEBUG: PYTHONPATH al inicio:", os.environ.get('PYTHONPATH'))
print("DEBUG: sys.path al inicio:", sys.path)
print("DEBUG: Directorio de trabajo actual:", os.getcwd())

from flask import Flask, send_from_directory, jsonify, current_app, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# === IMPORTANTE: db se creará e inicializará AQUÍ ===

print("DEBUG (main.py): Iniciando la aplicación Flask...")
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'BattSentinel#2024$SecureKey!AI'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max file size

# === CONFIGURACIÓN DE CORS ===
# **IMPORTANTE:** Usamos SOLO esta configuración para manejar CORS.
# Permite CORS para tu frontend específico en todas las rutas bajo /api/
CORS(app, resources={r"/api/*": {"origins": [
    "https://mcarbono3.github.io",
    "https://mcarbono3.github.io/BattSentinel",
    "https://mcarbono3.github.io/BattSentinel/"
]},
r"/*": {"origins": [
    "https://mcarbono3.github.io",
    "https://mcarbono3.github.io/BattSentinel",
    "https://mcarbono3.github.io/BattSentinel/"
]}})
# Si en algún momento necesitas permitir CUALQUIER origen para depuración (menos seguro en producción):
# CORS(app, origins="*")
# Si solo necesitas habilitar CORS globalmente sin restricciones específicas de ruta:
# CORS(app)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///batt_sentinel.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Desactivar seguimiento de modificaciones para reducir sobrecarga

# === PASO 1: Crear la instancia de SQLAlchemy AQUÍ, después de que 'app' ha sido creada ===
db = SQLAlchemy()

print(f"DEBUG (main.py): ID del objeto 'app' antes de init_app: {id(app)}")
print(f"DEBUG (main.py): ID del objeto 'db' antes de init_app: {id(db)}")

# Importar Modelos
from src.models.user import User # Importación única y correcta
from src.models.battery import Battery, BatteryData, Alert, AnalysisResult, ThermalImage, MaintenanceRecord

# Inicializar db con la aplicación
print("DEBUG (main.py): Llamando a db.init_app(app)...")
db.init_app(app)
print("DEBUG (main.py): db.init_app(app) ha sido llamado.")
print(f"DEBUG (main.py): ID del objeto 'db' después de init_app: {id(db)}")

# === PASO 2: Importar modelos y rutas DESPUÉS de que 'db' ha sido inicializada con 'app' ===
# Esto asegura que los modelos usen la instancia de 'db' que ya está asociada con la aplicación.
# ¡Asegúrate de que tus modelos y rutas importen 'db' desde este archivo 'main.py' ahora!
from src.routes.battery import battery_bp
from src.routes.ai_analysis import ai_bp
from src.routes.digital_twin import twin_bp
from src.routes.notifications import notifications_bp
from src.services.windows_battery import windows_battery_service

# Register blueprints - ¡IMPORTANTE: Añade url_prefix='/api' a los blueprints que pertenecen a la API!
print("DEBUG (main.py): Registrando Blueprints...")
app.register_blueprint(battery_bp, url_prefix='/api')
app.register_blueprint(ai_bp, url_prefix='/api')
app.register_blueprint(twin_bp, url_prefix='/api')
app.register_blueprint(notifications_bp, url_prefix='/api')
print("DEBUG (main.py): Blueprints registrados.")

# Función para obtener datos reales del sistema (ya usa /api/ en su ruta, se mantiene)
@app.route('/api/real-time-data', methods=['GET'])
def get_real_time_data():
    try:
        data = windows_battery_service.get_battery_info()
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint de información del sistema (ya usa /api/ en su ruta, se mantiene)
@app.route('/api/system-info', methods=['GET'])
def system_info():
    try:
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
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

# Endpoint de salud del sistema (ya usa /api/ en su ruta, se mantiene)
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
print("DEBUG (main.py): Entrando en app context para db.create_all()...")
with app.app_context():
    print("DEBUG (main.py): Dentro del app context. Llamando a db.create_all()...")
    db.create_all()
    print("DEBUG (main.py): Tablas de la base de datos verificadas/creadas.")
    
    # Crear usuario admin por defecto (opcional, ya que no hay autenticación)
    print("DEBUG (main.py): Verificando si existe usuario 'admin'...")
    if not User.query.filter_by(username='admin').first():
        try:
            print("DEBUG (main.py): Creando usuario 'admin'...")
            # IMPORTANTE: Asegúrate de que tu modelo User no requiera 'password_hash' si no lo estableces aquí
            admin_user = User(username='admin', email='admin@battsentinel.com', role='admin')
            db.session.add(admin_user)
            db.session.commit()
            print("DEBUG (main.py): Usuario 'admin' creado en la base de datos.")
        except Exception as e:
            db.session.rollback()
            print(f"DEBUG (main.py): Error al crear el usuario 'admin': {e}")
    else:
        print("DEBUG (main.py): Usuario 'admin' ya existe.")

if __name__ == '__main__':
    print("DEBUG (main.py): La aplicación se está ejecutando en el bloque __main__.")
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
