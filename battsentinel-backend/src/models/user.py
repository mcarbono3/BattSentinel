from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app # Importar current_app

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

    def __init__(self, username, email, password=None, first_name=None, last_name=None, role='user', department=None, phone=None, email_notifications=True, sms_notifications=False, whatsapp_number=None, sms_number=None, active=True):
        self.username = username
        self.email = email
        if password:
            self.set_password(password)
        self.first_name = first_name
        self.last_name = last_name
        self.role = role
        self.department = department
        self.phone = phone
        self.email_notifications = email_notifications
        self.sms_notifications = sms_notifications
        self.whatsapp_number = whatsapp_number
        self.sms_number = sms_number
        self.active = active

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
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
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Función para obtener usuario actual (simulado sin autenticación)
def get_current_user():
    """Obtener usuario actual - Simulado para demo sin autenticación"""
    with current_app.app_context(): # <--- AÑADIDA ESTA LÍNEA
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
                active=True,
                password='default_admin_password' # Puedes establecer una contraseña predeterminada aquí si es necesario
            )
            try:
                db.session.add(admin_user)
                db.session.commit()
            except Exception as e: # <--- CAMBIADO 'except:' por 'except Exception as e:' para mejor manejo
                db.session.rollback()
                print(f"Error al crear el usuario admin por defecto: {e}") # <--- Añadido un print para depuración
        
        return admin_user

# Función para verificar permisos (simplificada)
def check_permission(user, action, resource=None):
    """Verificar permisos - Simplificado para demo"""
    if not user:
        return True  # Sin autenticación estricta (o manejar como 'no autorizado' en un sistema real)
    
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
    
    return True  # Por defecto permitir (ajustar según la política de seguridad real)
