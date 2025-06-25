from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from werkzeug.security import generate_password_hash, check_password_hash

# Usar la misma instancia de db que battery.py
from .battery import db

class User(db.Model):
    """Modelo de Usuario Simplificado - Sin autenticación estricta"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255))
    
    # Información del perfil
    first_name = Column(String(100))
    last_name = Column(String(100))
    role = Column(String(50), default='user')  # admin, technician, user
    department = Column(String(100))
    phone = Column(String(20))
    
    # Configuración de notificaciones
    email_notifications = Column(Boolean, default=True)
    sms_notifications = Column(Boolean, default=False)
    whatsapp_number = Column(String(20))
    sms_number = Column(String(20))
    
    # Estado de la cuenta
    active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Preferencias
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='es')
    theme = Column(String(20), default='light')
    
    def set_password(self, password):
        """Establecer contraseña hasheada"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verificar contraseña"""
        if not self.password_hash:
            return True  # Sin autenticación estricta
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'department': self.department,
            'phone': self.phone,
            'email_notifications': self.email_notifications,
            'sms_notifications': self.sms_notifications,
            'whatsapp_number': self.whatsapp_number,
            'sms_number': self.sms_number,
            'active': self.active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'timezone': self.timezone,
            'language': self.language,
            'theme': self.theme
        }
    
    def get_full_name(self):
        """Obtener nombre completo"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        else:
            return self.username
    
    def is_admin(self):
        """Verificar si es administrador"""
        return self.role == 'admin'
    
    def is_technician(self):
        """Verificar si es técnico"""
        return self.role in ['admin', 'technician']
    
    def can_access_battery(self, battery_id):
        """Verificar si puede acceder a una batería específica - Siempre True sin autenticación"""
        return True
    
    def get_notification_preferences(self):
        """Obtener preferencias de notificación"""
        return {
            'email': self.email_notifications,
            'sms': self.sms_notifications,
            'whatsapp': bool(self.whatsapp_number),
            'email_address': self.email,
            'sms_number': self.sms_number,
            'whatsapp_number': self.whatsapp_number
        }

class UserSession(db.Model):
    """Modelo de Sesión de Usuario - Simplificado"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, db.ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    active = Column(Boolean, default=True)
    
    # Relación con usuario
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_token': self.session_token,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'active': self.active
        }
    
    def is_expired(self):
        """Verificar si la sesión ha expirado"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

class UserPreferences(db.Model):
    """Modelo de Preferencias de Usuario"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, db.ForeignKey('users.id'), nullable=False)
    preference_key = Column(String(100), nullable=False)
    preference_value = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relación con usuario
    user = db.relationship('User', backref=db.backref('preferences', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'preference_key': self.preference_key,
            'preference_value': self.preference_value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Función para crear usuarios por defecto
def create_default_users():
    """Crear usuarios por defecto para el sistema"""
    try:
        # Usuario administrador por defecto
        if not User.query.filter_by(username='admin').first():
            admin_user = User(
                username='admin',
                email='admin@battsentinel.com',
                first_name='Administrador',
                last_name='Sistema',
                role='admin',
                department='IT',
                email_notifications=True,
                active=True
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)
        
        # Usuario técnico por defecto
        if not User.query.filter_by(username='technician').first():
            tech_user = User(
                username='technician',
                email='tech@battsentinel.com',
                first_name='Técnico',
                last_name='Principal',
                role='technician',
                department='Mantenimiento',
                email_notifications=True,
                sms_notifications=True,
                active=True
            )
            tech_user.set_password('tech123')
            db.session.add(tech_user)
        
        # Usuario demo por defecto
        if not User.query.filter_by(username='demo').first():
            demo_user = User(
                username='demo',
                email='demo@battsentinel.com',
                first_name='Usuario',
                last_name='Demo',
                role='user',
                department='Demo',
                email_notifications=True,
                active=True
            )
            demo_user.set_password('demo123')
            db.session.add(demo_user)
        
        db.session.commit()
        print("Usuarios por defecto creados exitosamente")
        
    except Exception as e:
        db.session.rollback()
        print(f"Error creando usuarios por defecto: {e}")

# Función para obtener usuario actual (simulado sin autenticación)
def get_current_user():
    """Obtener usuario actual - Simulado para demo sin autenticación"""
    # En un sistema real, esto obtendría el usuario de la sesión/token
    # Para demo, devolvemos el usuario admin por defecto
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        # Crear usuario admin temporal si no existe
        admin_user = User(
            username='admin',
            email='admin@battsentinel.com',
            first_name='Admin',
            last_name='Demo',
            role='admin',
            active=True
        )
        try:
            db.session.add(admin_user)
            db.session.commit()
        except:
            db.session.rollback()
    
    return admin_user

# Función para verificar permisos (simplificada)
def check_permission(user, action, resource=None):
    """Verificar permisos - Simplificado para demo"""
    if not user:
        return True  # Sin autenticación estricta
    
    if user.role == 'admin':
        return True  # Admin tiene todos los permisos
    
    # Permisos básicos para otros roles
    basic_permissions = ['read', 'view', 'list']
    if action in basic_permissions:
        return True
    
    # Técnicos pueden hacer más acciones
    if user.role == 'technician':
        tech_permissions = ['create', 'update', 'analyze', 'maintain']
        if action in tech_permissions:
            return True
    
    return True  # Por defecto permitir para demo
