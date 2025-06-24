from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash # Importa estas funciones

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    # Nuevo campo para almacenar el hash de la contraseña
    password_hash = db.Column(db.String(128), nullable=False) # String más largo para el hash

    def __repr__(self):
        return f'<User {self.username}>'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
            # No incluir password_hash aquí por seguridad
        }

    # Método para establecer la contraseña (la hashea antes de guardarla)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Método para verificar la contraseña
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
