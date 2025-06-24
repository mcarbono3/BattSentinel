import os
import sys

# DON'T CHANGE THIS !!!
# La siguiente línea puede causar problemas en Render si la estructura de carpetas
# es diferente a la esperada por sys.path.insert.
# Si tus imports como 'from src.models.battery import db' funcionan sin ella,
# es mejor dejarla comentada para un entorno de despliegue como Render.
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.battery import db
# IMPORTANTE: Asegúrate de importar el modelo User desde user.py
# Es crucial que src/models/user.py defina la clase User
# y que esa User tenga los campos password_hash y role, y los métodos set_password, check_password.
from src.models.user import User 

from src.routes.battery import battery_bp
from src.routes.ai_analysis import ai_bp
from src.routes.digital_twin import twin_bp
from src.routes.notifications import notifications_bp
from src.routes.auth import auth_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'BattSentinel#2024$SecureKey!AI'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for all routes
CORS(app, origins="*")

# Register blueprints
app.register_blueprint(battery_bp, url_prefix='/api/battery')
app.register_blueprint(ai_bp, url_prefix='/api/ai')
app.register_blueprint(twin_bp, url_prefix='/api/twin')
app.register_blueprint(notifications_bp, url_prefix='/api/notifications')
app.register_blueprint(auth_bp, url_prefix='/api/auth')

# ================================================================
# CONFIGURACIÓN DE LA BASE DE DATOS - ¡CRÍTICO PARA RENDER!
# ================================================================
# Render inyecta automáticamente la variable de entorno DATABASE_URL
# cuando conectas una base de datos PostgreSQL a tu servicio.
# Este es el método correcto para conectarse en producción.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')

# Si DATABASE_URL no está configurada (ej. en desarrollo local),
# usamos un archivo SQLite local para facilitar el desarrollo.
if not app.config['SQLALCHEMY_DATABASE_URI']:
    # Asegúrate de que el directorio 'database' exista si usas SQLite local
    sqlite_dir = os.path.join(os.path.dirname(__file__), 'database')
    os.makedirs(sqlite_dir, exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(sqlite_dir, 'app.db')}"
    print("ADVERTENCIA: DATABASE_URL no encontrada. Usando SQLite local para desarrollo.")
else:
    print("Conectando a la base de datos PostgreSQL de Render (DATABASE_URL detectada).")


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create upload directory (para archivos subidos, no DB)
upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(upload_dir, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_dir

# ================================================================
# INICIALIZACIÓN DE LA BASE DE DATOS Y CREACIÓN DEL USUARIO ADMIN
# Este bloque se ejecuta una vez al iniciar la aplicación.
# ================================================================
with app.app_context():
    # Crea todas las tablas definidas en tus modelos (Battery, BatteryData, User, etc.)
    # Esto es idempotente: si las tablas ya existen, no hace nada.
    db.create_all() 
    print("Tablas de la base de datos verificadas/creadas.")

    # Crear usuario administrador si no existe
    # Asume que tu modelo User tiene campos `username`, `email`, `role`, `password_hash`
    # y el método `set_password`.
    if not User.query.filter_by(username='admin').first():
        try:
            admin_user = User(username='admin', email='admin@battsentinel.com', role='admin')
            admin_user.set_password('admin123') # Usa una contraseña fuerte para producción
            db.session.add(admin_user)
            db.session.commit()
            print("Usuario 'admin' creado/inicializado en la base de datos.")
        except Exception as e:
            db.session.rollback() # Si algo falla, revierte la transacción
            print(f"Error al crear el usuario 'admin': {e}")
    else:
        print("El usuario 'admin' ya existe en la base de datos.")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "Frontend not found", 404

if __name__ == '__main__':
    # Cuando se ejecuta localmente (python main.py), se usa debug=True.
    # En Render, Gunicorn o similar ejecutará la aplicación, no esta línea.
    app.run(debug=True)
