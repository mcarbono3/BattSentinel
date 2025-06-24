from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone # Importar timezone para fechas conscientes de UTC
import json # Dejar si se usa en alguna parte no visible, si no, se puede remover

db = SQLAlchemy()

# IMPORTANTE: No importes el modelo User aquí si ya lo importas en main.py y lo relacionas.
# Sin embargo, si vas a definir relaciones explícitas (como en Alert con User),
# el modelo 'User' debe ser conocido por SQLAlchemy.
# Normalmente, se importa en main.py para db.create_all() y en cualquier blueprint que lo use.
# Aquí lo importamos para la relación en Alert.
from src.models.user import User # Asegúrate de que este path sea correcto para tu User model

class Battery(db.Model):
    """Modelo para almacenar información de baterías"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    battery_type = db.Column(db.String(50), nullable=False, default='Li-ion')
    device_type = db.Column(db.String(50), nullable=True)  # laptop, phone, etc.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc)) # Usar timezone.utc
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)) # Usar timezone.utc

    # Relaciones con datos y análisis de la batería
    data_points = db.relationship('BatteryData', backref='battery', lazy=True, cascade='all, delete-orphan')
    analyses = db.relationship('BatteryAnalysis', backref='battery', lazy=True, cascade='all, delete-orphan')
    alerts = db.relationship('Alert', backref='battery', lazy=True, cascade='all, delete-orphan') # Añadida relación con Alert

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
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) # Usar timezone.utc

    # Parámetros de la batería
    voltage = db.Column(db.Float, nullable=True) # Voltaje actual (V)
    current = db.Column(db.Float, nullable=True) # Corriente actual (A)
    temperature = db.Column(db.Float, nullable=True) # Temperatura (°C)
    soc = db.Column(db.Float, nullable=True) # State of Charge (%)
    soh = db.Column(db.Float, nullable=True) # State of Health (%)
    cycles = db.Column(db.Integer, default=0) # Número de ciclos de carga/descarga

    # Campos opcionales para información más detallada
    power = db.Column(db.Float, nullable=True) # Potencia (W)
    energy_consumed = db.Column(db.Float, nullable=True) # Energía consumida (kWh)
    pressure = db.Column(db.Float, nullable=True) # Presión (si aplica)
    humidity = db.Column(db.Float, nullable=True) # Humedad (si aplica)
    # Otros parámetros que puedan ser relevantes (ej. status, cell_voltages, etc.)
    status = db.Column(db.String(50), nullable=True) # e.g., 'charging', 'discharging', 'idle', 'error'
    # Almacenar datos complejos como JSON (ej. voltajes de celdas individuales)
    extra_data = db.Column(db.Text, nullable=True) # Almacenar como JSON string

    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'soc': self.soc,
            'soh': self.soh,
            'cycles': self.cycles,
            'power': self.power,
            'energy_consumed': self.energy_consumed,
            'pressure': self.pressure,
            'humidity': self.humidity,
            'status': self.status,
            'extra_data': json.loads(self.extra_data) if self.extra_data else None # Parse JSON string
        }

class BatteryAnalysis(db.Model):
    """Modelo para almacenar resultados de análisis de IA sobre datos de batería"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    analysis_type = db.Column(db.String(100), nullable=False) # e.g., 'predictive_maintenance', 'anomaly_detection', 'performance_evaluation'
    analysis_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) # Usar timezone.utc
    result = db.Column(db.Text, nullable=False) # Almacenar el resultado del análisis (puede ser JSON string)
    status = db.Column(db.String(50), default='completed') # e.g., 'pending', 'completed', 'failed'

    # Relación con resultados predictivos específicos (si es un análisis predictivo)
    predictive_results = db.relationship('PredictiveAnalysisResult', backref='analysis', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'analysis_type': self.analysis_type,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'result': json.loads(self.result) if self.result else self.result, # Asume que el resultado puede ser JSON
            'status': self.status
        }

class PredictiveAnalysisResult(db.Model):
    """Modelo para almacenar resultados específicos de análisis predictivos"""
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('battery_analysis.id'), nullable=False)
    prediction_type = db.Column(db.String(100), nullable=False) # e.g., 'soh_prediction', 'eol_prediction', 'failure_prediction'
    prediction_value = db.Column(db.Float, nullable=True) # Valor de la predicción (ej. SOH%)
    prediction_unit = db.Column(db.String(20), nullable=True) # Unidad (e.g., '%', 'days', 'cycles')
    confidence_score = db.Column(db.Float, nullable=True) # Nivel de confianza del modelo (0-1)
    predicted_date = db.Column(db.DateTime, nullable=True) # Fecha o momento predicho (ej. fin de vida útil)
    details = db.Column(db.Text, nullable=True) # Más detalles o JSON de la predicción

    def to_dict(self):
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'prediction_type': self.prediction_type,
            'prediction_value': self.prediction_value,
            'prediction_unit': self.prediction_unit,
            'confidence_score': self.confidence_score,
            'predicted_date': self.predicted_date.isoformat() if self.predicted_date else None,
            'details': json.loads(self.details) if self.details else self.details # Asume que detalles puede ser JSON
        }

class Alert(db.Model):
    """Modelo para almacenar alertas generadas por el sistema"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Asociar a un usuario específico, si aplica

    # Nueva relación para acceder al objeto User
    user = db.relationship('User', backref='alerts', lazy=True) # <-- ¡Añadido aquí!

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
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc)) # Usar timezone.utc

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
