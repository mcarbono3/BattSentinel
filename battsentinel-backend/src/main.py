import os
import sys

# DON'T CHANGE THIS !!!
# La siguiente línea puede causar problemas en Render si la estructura de carpetas
# es diferente a la esperada por sys.path.insert.
# Si tus imports como 'from src.models.battery import db' funcionan sin ella,
# es mejor dejarla comentada para un entorno de despliegue como Render.
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS # Asegúrate de que Flask-CORS esté importado
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
from src.routes.user import user_bp # Asegúrate de importar el blueprint de usuario si existe

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'BattSentinel#2024$SecureKey!AI'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# --- CAMBIO IMPORTANTE: CONFIGURACIÓN DE UPLOAD_FOLDER ---
# Define la carpeta donde se guardarán los archivos subidos.
# Asegúrate de que esta carpeta exista.
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# --- FIN CAMBIO IMPORTANTE ---

# Enable CORS for all routes
# Para desarrollo, origins="*" es práctico. Para producción, especifica tus dominios:
# CORS(app, origins=["https://battsentinel-frontend.onrender.com", "http://localhost:5173"])
CORS(app, origins="*") # Mantengo '*' para que funcione fácilmente durante el desarrollo local y el despliegue

# Register blueprints
app.register_blueprint(battery_bp, url_prefix='/api')
app.register_blueprint(ai_bp, url_prefix='/api')
app.register_blueprint(twin_bp, url_prefix='/api')
app.register_blueprint(notifications_bp, url_prefix='/api')
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(user_bp, url_prefix='/api/users') # Asumiendo que tienes un blueprint para usuarios

# Inicializa SQLAlchemy con la aplicación Flask
db.init_app(app)

# --- INICIO DE LA MODIFICACIÓN: Función 'serve' comentada para evitar conflictos ---
# Función 'serve' comentada para evitar conflictos de ruteo cuando el frontend
# se despliega de forma independiente.
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve(path):
#     static_folder_path = app.static_folder
#     if static_folder_path is None:
#         return "Static folder not configured", 404
#
#     if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
#         return send_from_directory(static_folder_path, path)
#     else:
#         index_path = os.path.join(static_folder_path, 'index.html')
#         if os.path.exists(index_path):
#             return send_from_directory(static_folder_path, 'index.html')
#         else:
#             return "Frontend not found", 404
# --- FIN DE LA MODIFICACIÓN ---

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Crea las tablas de la base de datos si no existen
        print("Tablas de la base de datos verificadas/creadas.")

        # Inicializa el usuario 'admin' si no existe
        # Asegúrate de que tu modelo 'User' tenga un campo `password_hash`
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

    # Obtén el puerto del entorno (Render lo proveerá) o usa 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    # Para despliegue, asegúrate de que host='0.0.0.0' para que sea accesible externamente
    app.run(host='0.0.0.0', port=port, debug=False) # En producción, debug=False
