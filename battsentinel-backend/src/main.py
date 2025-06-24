import os
import sys

# DON'T CHANGE THIS !!!
# Esta línea parece ser específica de tu configuración de rutas de importación.
# Si el proyecto funciona localmente con ella, mantenla. Si causa problemas,
# y tus imports funcionan sin ella, podrías comentarla.
# Sin embargo, dado el comentario original "DON'T CHANGE THIS !!!", la mantengo.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
# Importa la instancia 'db' y el modelo 'User' desde donde están definidos.
# 'db' se define en src.models.battery y 'User' en src.models.user.
from src.models.battery import db
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

# Database configuration
# USAR VARIABLE DE ENTORNO DATABASE_URL PARA PRODUCCIÓN (RENDER/POSTGRESQL)
# Fallback a SQLite para desarrollo local si DATABASE_URL no está definida.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create upload directory
upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(upload_dir, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_dir

# Bloque para inicializar la base de datos y crear el usuario 'admin'
with app.app_context():
    # Intentar crear todas las tablas.
    # Si ya existen, SQLAlchemy simplemente ignorará la creación.
    # Esto es seguro para despliegues repetidos.
    db.create_all()

    # Crear usuario administrador si no existe
    # Asegúrate de que el rol 'admin' sea el deseado
    if not User.query.filter_by(username='admin').first():
        admin_user = User(username='admin', email='admin@battsentinel.com', role='admin')
        admin_user.set_password('admin123') # Usa tu contraseña deseada
        db.session.add(admin_user)
        db.session.commit()
        print("Usuario 'admin' creado/inicializado en la base de datos.")
    else:
        print("El usuario 'admin' ya existe.")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    # Sirve archivos estáticos directamente desde la carpeta 'static'
    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        # Sirve index.html para todas las demás rutas (para aplicaciones de una sola página)
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "Index HTML not found", 404

if __name__ == '__main__':
    # Usar la variable de entorno 'PORT' en Render para el puerto de escucha,
    # o 5000 como default para desarrollo local.
    port = int(os.environ.get('PORT', 5000))
    # Escuchar en todas las interfaces disponibles (0.0.0.0) para que Render pueda acceder.
    app.run(debug=True, host='0.0.0.0', port=port)
