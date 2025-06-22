from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Battery(db.Model):
    """Modelo para almacenar información de baterías"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    battery_type = db.Column(db.String(50), nullable=False, default='Li-ion')
    device_type = db.Column(db.String(50), nullable=True)  # laptop, phone, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    data_points = db.relationship('BatteryData', backref='battery', lazy=True, cascade='all, delete-orphan')
    analyses = db.relationship('BatteryAnalysis', backref='battery', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'battery_type': self.battery_type,
            'device_type': self.device_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class BatteryData(db.Model):
    """Modelo para almacenar datos de sensores de batería"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    
    # Parámetros eléctricos
    voltage = db.Column(db.Float, nullable=True)  # Voltios
    current = db.Column(db.Float, nullable=True)  # Amperios
    power = db.Column(db.Float, nullable=True)    # Watts
    
    # Estado de la batería
    soc = db.Column(db.Float, nullable=True)      # State of Charge (%)
    soh = db.Column(db.Float, nullable=True)      # State of Health (%)
    capacity = db.Column(db.Float, nullable=True) # Capacidad actual (mAh)
    cycles = db.Column(db.Integer, nullable=True) # Número de ciclos
    
    # Parámetros térmicos
    temperature = db.Column(db.Float, nullable=True)  # Celsius
    
    # Resistencia interna
    internal_resistance = db.Column(db.Float, nullable=True)  # Ohms
    
    # Metadatos
    data_source = db.Column(db.String(50), nullable=False, default='manual')  # manual, api, real_time
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'voltage': self.voltage,
            'current': self.current,
            'power': self.power,
            'soc': self.soc,
            'soh': self.soh,
            'capacity': self.capacity,
            'cycles': self.cycles,
            'temperature': self.temperature,
            'internal_resistance': self.internal_resistance,
            'data_source': self.data_source,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class BatteryAnalysis(db.Model):
    """Modelo para almacenar resultados de análisis de IA"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # fault_detection, health_prediction, etc.
    
    # Resultados del análisis
    result = db.Column(db.Text, nullable=False)  # JSON string con resultados
    confidence = db.Column(db.Float, nullable=True)  # Confianza del modelo (0-1)
    
    # Detección de fallas
    fault_detected = db.Column(db.Boolean, default=False)
    fault_type = db.Column(db.String(100), nullable=True)  # degradation, short_circuit, etc.
    severity = db.Column(db.String(20), nullable=True)     # low, medium, high, critical
    
    # Predicciones
    rul_prediction = db.Column(db.Float, nullable=True)    # Remaining Useful Life (days)
    
    # Explicabilidad (XAI)
    explanation = db.Column(db.Text, nullable=True)        # JSON string con explicación SHAP/LIME
    
    # Metadatos
    model_version = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'analysis_type': self.analysis_type,
            'result': json.loads(self.result) if self.result else None,
            'confidence': self.confidence,
            'fault_detected': self.fault_detected,
            'fault_type': self.fault_type,
            'severity': self.severity,
            'rul_prediction': self.rul_prediction,
            'explanation': json.loads(self.explanation) if self.explanation else None,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ThermalImage(db.Model):
    """Modelo para almacenar imágenes térmicas"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    
    # Análisis de imagen térmica
    max_temperature = db.Column(db.Float, nullable=True)
    min_temperature = db.Column(db.Float, nullable=True)
    avg_temperature = db.Column(db.Float, nullable=True)
    hotspot_detected = db.Column(db.Boolean, default=False)
    hotspot_coordinates = db.Column(db.Text, nullable=True)  # JSON string
    
    # Metadatos
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_completed = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'max_temperature': self.max_temperature,
            'min_temperature': self.min_temperature,
            'avg_temperature': self.avg_temperature,
            'hotspot_detected': self.hotspot_detected,
            'hotspot_coordinates': json.loads(self.hotspot_coordinates) if self.hotspot_coordinates else None,
            'upload_timestamp': self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            'analysis_completed': self.analysis_completed
        }

class User(db.Model):
    """Modelo para usuarios del sistema"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')  # admin, technician, user
    
    # Configuración de notificaciones
    email_notifications = db.Column(db.Boolean, default=True)
    whatsapp_number = db.Column(db.String(20), nullable=True)
    sms_number = db.Column(db.String(20), nullable=True)
    
    # Metadatos
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    active = db.Column(db.Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'email_notifications': self.email_notifications,
            'whatsapp_number': self.whatsapp_number,
            'sms_number': self.sms_number,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'active': self.active
        }

class Alert(db.Model):
    """Modelo para alertas y notificaciones"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Información de la alerta
    alert_type = db.Column(db.String(50), nullable=False)  # fault, warning, info
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)    # low, medium, high, critical
    
    # Estado de la alerta
    status = db.Column(db.String(20), default='active')    # active, acknowledged, resolved
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    
    # Notificaciones enviadas
    email_sent = db.Column(db.Boolean, default=False)
    whatsapp_sent = db.Column(db.Boolean, default=False)
    sms_sent = db.Column(db.Boolean, default=False)
    
    # Metadatos
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'user_id': self.user_id,
            'alert_type': self.alert_type,
            'title': self.title,
            'message': self.message,
            'severity': self.severity,
            'status': self.status,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'email_sent': self.email_sent,
            'whatsapp_sent': self.whatsapp_sent,
            'sms_sent': self.sms_sent,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

