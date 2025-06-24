import os
import sys
# DON'T CHANGE THIS !!!
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) # Línea comentada

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.battery import db
# IMPORTANTE: Asegúrate de importar el modelo User desde user.py
from src.models.user import User #
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
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create upload directory
upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(upload_dir, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_dir

# Bloque para inicializar la base de datos y crear el usuario 'admin'
with app.app_context():
    db.create_all() # Crea todas las tablas, incluyendo la nueva columna password_hash

    # Crear usuario administrador si no existe
    if not User.query.filter_by(username='admin').first():
        admin_user = User(username='admin', email='admin@battsentinel.com')
        admin_user.set_password('admin123') # Usa tu contraseña deseada
        db.session.add(admin_user) #
        db.session.commit() #
        print("Usuario 'admin' creado/inicializado en la base de datos.") #
    else:
        print("El usuario 'admin' ya existe.") #

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
            return "Frontend static files not found.", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
