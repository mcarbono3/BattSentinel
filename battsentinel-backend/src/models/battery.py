from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone 
import json 

db = SQLAlchemy()

# Es crucial que el modelo User se importe si hay relaciones con él en este archivo.
from src.models.user import User 

class Battery(db.Model):
    """Modelo para almacenar información de baterías"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    battery_type = db.Column(db.String(50), nullable=False, default='Li-ion')
    device_type = db.Column(db.String(50), nullable=True)  # laptop, phone, etc.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc)) 
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)) 

    # Relaciones con datos y análisis de la batería
    data_points = db.relationship('BatteryData', backref='battery', lazy=True, cascade='all, delete-orphan')
    analyses = db.relationship('BatteryAnalysis', backref='battery', lazy=True, cascade='all, delete-orphan')
    alerts = db.relationship('Alert', backref='battery', lazy=True, cascade='all, delete-orphan') 
    thermal_images = db.relationship('ThermalImage', backref='battery', lazy=True, cascade='all, delete-orphan') # <-- ¡NUEVA RELACIÓN AÑADIDA!

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
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) 

    # Parámetros de la batería
    voltage = db.Column(db.Float, nullable=True) 
    current = db.Column(db.Float, nullable=True) 
    temperature = db.Column(db.Float, nullable=True) 
    soc = db.Column(db.Float, nullable=True) 
    soh = db.Column(db.Float, nullable=True) 
    cycles = db.Column(db.Integer, default=0) 

    # Campos opcionales para información más detallada
    power = db.Column(db.Float, nullable=True) 
    energy_consumed = db.Column(db.Float, nullable=True) 
    pressure = db.Column(db.Float, nullable=True) 
    humidity = db.Column(db.Float, nullable=True) 
    status = db.Column(db.String(50), nullable=True) 
    extra_data = db.Column(db.Text, nullable=True) 

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
            'extra_data': json.loads(self.extra_data) if self.extra_data else None 
        }

class BatteryAnalysis(db.Model):
    """Modelo para almacenar resultados de análisis de IA sobre datos de batería"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    analysis_type = db.Column(db.String(100), nullable=False) 
    analysis_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) 
    result = db.Column(db.Text, nullable=False) 
    status = db.Column(db.String(50), default='completed') 

    # Relación con resultados predictivos específicos (si es un análisis predictivo)
    predictive_results = db.relationship('PredictiveAnalysisResult', backref='analysis', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'analysis_type': self.analysis_type,
            'analysis_date': self.analysis_date.isoformat() if self.analysis_date else None,
            'result': json.loads(self.result) if self.result else self.result, 
            'status': self.status
        }

class PredictiveAnalysisResult(db.Model):
    """Modelo para almacenar resultados específicos de análisis predictivos"""
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('battery_analysis.id'), nullable=False)
    prediction_type = db.Column(db.String(100), nullable=False) 
    prediction_value = db.Column(db.Float, nullable=True) 
    prediction_unit = db.Column(db.String(20), nullable=True) 
    confidence_score = db.Column(db.Float, nullable=True) 
    predicted_date = db.Column(db.DateTime, nullable=True) 
    details = db.Column(db.Text, nullable=True) 

    def to_dict(self):
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'prediction_type': self.prediction_type,
            'prediction_value': self.prediction_value,
            'prediction_unit': self.prediction_unit,
            'confidence_score': self.confidence_score,
            'predicted_date': self.predicted_date.isoformat() if self.predicted_date else None,
            'details': json.loads(self.details) if self.details else self.details 
        }

class Alert(db.Model):
    """Modelo para almacenar alertas generadas por el sistema"""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) 

    user = db.relationship('User', backref='alerts', lazy=True) 

    alert_type = db.Column(db.String(50), nullable=False)  
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)    

    status = db.Column(db.String(20), default='active')    
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)

    email_sent = db.Column(db.Boolean, default=False)
    whatsapp_sent = db.Column(db.Boolean, default=False)
    sms_sent = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc)) 

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

class ThermalImage(db.Model):
    """Modelo para almacenar información de imágenes térmicas de baterías."""
    id = db.Column(db.Integer, primary_key=True)
    battery_id = db.Column(db.Integer, db.ForeignKey('battery.id'), nullable=False)
    
    # Ruta o URL de la imagen térmica almacenada
    image_url = db.Column(db.String(255), nullable=False)
    
    # Metadatos de la imagen
    captured_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    max_temp = db.Column(db.Float, nullable=True) # Temperatura máxima detectada en la imagen
    min_temp = db.Column(db.Float, nullable=True) # Temperatura mínima detectada en la imagen
    avg_temp = db.Column(db.Float, nullable=True) # Temperatura promedio detectada
    
    # Posibles anomalías detectadas en la imagen térmica
    anomaly_detected = db.Column(db.Boolean, default=False)
    anomaly_description = db.Column(db.Text, nullable=True) # Descripción de la anomalía

    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'image_url': self.image_url,
            'captured_at': self.captured_at.isoformat() if self.captured_at else None,
            'max_temp': self.max_temp,
            'min_temp': self.min_temp,
            'avg_temp': self.avg_temp,
            'anomaly_detected': self.anomaly_detected,
            'anomaly_description': self.anomaly_description
        }
