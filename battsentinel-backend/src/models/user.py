from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone

# IMPORTANTE: No importes SQLAlchemy aquí y no definas 'db = SQLAlchemy()'.
# La instancia 'db' DEBE importarse desde donde ya está inicializada globalmente,
# que en tu proyecto es src.models.battery.
from src.models.battery import db # <-- ¡Solo importa la instancia 'db' aquí!

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    # Campo para almacenar el hash de la contraseña
    # ¡Aumentado a 255 para acomodar hashes más largos!
    password_hash = db.Column(db.String(255), nullable=False) 

    # Campos adicionales para roles y estado
    role = db.Column(db.String(50), default='user', nullable=False) # 'admin', 'technician', 'user'
    active = db.Column(db.Boolean, default=True, nullable=False) # Para activar/desactivar cuentas

    # Campos para restablecimiento de contraseña
    reset_token = db.Column(db.String(128), unique=True, nullable=True)
    reset_token_expiration = db.Column(db.DateTime, nullable=True)

    # Campo para activación de cuenta (por ejemplo, vía email)
    activation_token = db.Column(db.String(128), unique=True, nullable=True)

    # Campos para preferencias de usuario y contacto
    email_notifications = db.Column(db.Boolean, default=True)
    whatsapp_number = db.Column(db.String(50), nullable=True)
    sms_number = db.Column(db.String(50), nullable=True)

    # Campo para auditoría (último login)
    last_login = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) # Ajustado a timezone.utc
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False) # Ajustado a timezone.utc

    def __repr__(self):
        return f'<User {self.username}>'

    # Método para establecer la contraseña (la hashea antes de guardarla)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Método para verificar la contraseña
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """
        Convierte el objeto User a un diccionario para respuestas JSON.
        NO incluye campos sensibles como password_hash o tokens.
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'active': self.active,
            'email_notifications': self.email_notifications,
            'whatsapp_number': self.whatsapp_number,
            'sms_number': self.sms_number,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
