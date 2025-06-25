from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class Battery(db.Model):
    """Modelo de Batería"""
    __tablename__ = 'batteries'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    model = Column(String(100))
    manufacturer = Column(String(100))
    serial_number = Column(String(100), unique=True)
    capacity_ah = Column(Float, default=100.0)
    voltage_nominal = Column(Float, default=12.0)
    chemistry = Column(String(50), default='Li-ion')
    installation_date = Column(DateTime, default=datetime.utcnow)
    location = Column(String(200))
    status = Column(String(50), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    data_points = relationship('BatteryData', backref='battery', lazy=True, cascade='all, delete-orphan')
    alerts = relationship('Alert', backref='battery', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'model': self.model,
            'manufacturer': self.manufacturer,
            'serial_number': self.serial_number,
            'capacity_ah': self.capacity_ah,
            'voltage_nominal': self.voltage_nominal,
            'chemistry': self.chemistry,
            'installation_date': self.installation_date.isoformat() if self.installation_date else None,
            'location': self.location,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class BatteryData(db.Model):
    """Modelo de Datos de Batería"""
    __tablename__ = 'battery_data'
    
    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    voltage = Column(Float)
    current = Column(Float)
    temperature = Column(Float)
    soc = Column(Float)  # State of Charge
    soh = Column(Float)  # State of Health
    cycles = Column(Integer, default=0)
    internal_resistance = Column(Float)
    power = Column(Float)
    energy = Column(Float)
    efficiency = Column(Float)
    
    # Campos adicionales para análisis
    charge_rate = Column(Float)
    discharge_rate = Column(Float)
    ambient_temperature = Column(Float)
    humidity = Column(Float)
    
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
            'internal_resistance': self.internal_resistance,
            'power': self.power,
            'energy': self.energy,
            'efficiency': self.efficiency,
            'charge_rate': self.charge_rate,
            'discharge_rate': self.discharge_rate,
            'ambient_temperature': self.ambient_temperature,
            'humidity': self.humidity
        }

class Alert(db.Model):
    """Modelo de Alertas"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)
    alert_type = Column(String(100), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    status = Column(String(20), default='active')  # active, acknowledged, resolved
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Notificaciones enviadas
    email_sent = Column(Boolean, default=False)
    sms_sent = Column(Boolean, default=False)
    whatsapp_sent = Column(Boolean, default=False)
    
    # Datos adicionales
    threshold_value = Column(Float)
    actual_value = Column(Float)
    alert_metadata = Column(Text)  # JSON string para datos adicionales
    
    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'alert_type': self.alert_type,
            'title': self.title,
            'message': self.message,
            'severity': self.severity,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'email_sent': self.email_sent,
            'sms_sent': self.sms_sent,
            'whatsapp_sent': self.whatsapp_sent,
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value,
            'alert_metadata': self.alert_metadata
        }

class AnalysisResult(db.Model):
    """Modelo de Resultados de Análisis"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)
    analysis_type = Column(String(100), nullable=False)  # fault_detection, health_prediction, etc.
    result = Column(Text, nullable=False)  # JSON string con resultados
    confidence_score = Column(Float)
    model_version = Column(String(50))
    processing_time = Column(Float)  # segundos
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        import json
        try:
            result_data = json.loads(self.result) if self.result else {}
        except:
            result_data = {'raw_result': self.result}
            
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'analysis_type': self.analysis_type,
            'result': result_data,
            'confidence_score': self.confidence_score,
            'model_version': self.model_version,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ThermalImage(db.Model):
    """Modelo de Imágenes Térmicas"""
    __tablename__ = 'thermal_images'
    
    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    image_format = Column(String(10))
    
    # Metadatos de la imagen
    width = Column(Integer)
    height = Column(Integer)
    min_temperature = Column(Float)
    max_temperature = Column(Float)
    avg_temperature = Column(Float)
    
    # Análisis de la imagen
    hotspots_detected = Column(Integer, default=0)
    anomalies_detected = Column(Boolean, default=False)
    analysis_result = Column(Text)  # JSON string
    
    # Timestamps
    captured_at = Column(DateTime)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    def to_dict(self):
        import json
        try:
            analysis_data = json.loads(self.analysis_result) if self.analysis_result else {}
        except:
            analysis_data = {}
            
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'image_format': self.image_format,
            'width': self.width,
            'height': self.height,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
            'avg_temperature': self.avg_temperature,
            'hotspots_detected': self.hotspots_detected,
            'anomalies_detected': self.anomalies_detected,
            'analysis_result': analysis_data,
            'captured_at': self.captured_at.isoformat() if self.captured_at else None,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }

class MaintenanceRecord(db.Model):
    """Modelo de Registros de Mantenimiento"""
    __tablename__ = 'maintenance_records'
    
    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)
    maintenance_type = Column(String(100), nullable=False)
    description = Column(Text)
    performed_by = Column(String(100))
    performed_at = Column(DateTime, default=datetime.utcnow)
    next_maintenance_due = Column(DateTime)
    cost = Column(Float)
    parts_replaced = Column(Text)  # JSON string
    notes = Column(Text)
    
    def to_dict(self):
        import json
        try:
            parts_data = json.loads(self.parts_replaced) if self.parts_replaced else []
        except:
            parts_data = []
            
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'maintenance_type': self.maintenance_type,
            'description': self.description,
            'performed_by': self.performed_by,
            'performed_at': self.performed_at.isoformat() if self.performed_at else None,
            'next_maintenance_due': self.next_maintenance_due.isoformat() if self.next_maintenance_due else None,
            'cost': self.cost,
            'parts_replaced': parts_data,
            'notes': self.notes
        }

# Función auxiliar para crear datos de ejemplo
def create_sample_data():
    """Crear datos de ejemplo para desarrollo"""
    try:
        # Crear batería de ejemplo si no existe
        if not Battery.query.first():
            sample_battery = Battery(
                name='Batería Principal',
                model='Li-ion 100Ah',
                manufacturer='BattTech',
                serial_number='BT-2024-001',
                capacity_ah=100.0,
                voltage_nominal=12.0,
                chemistry='Li-ion',
                location='Sala de Servidores',
                status='active'
            )
            db.session.add(sample_battery)
            db.session.commit()
            
            # Crear algunos datos de ejemplo
            from datetime import timedelta
            import random
            
            battery_id = sample_battery.id
            now = datetime.utcnow()
            
            for i in range(50):
                timestamp = now - timedelta(hours=i)
                
                # Simular degradación gradual
                degradation = i * 0.01
                
                data_point = BatteryData(
                    battery_id=battery_id,
                    timestamp=timestamp,
                    voltage=12.0 - (degradation * 0.1) + random.uniform(-0.2, 0.2),
                    current=2.5 + random.uniform(-0.5, 0.5),
                    temperature=25.0 + random.uniform(-3, 8),
                    soc=max(20, 85 - (i * 0.5) + random.uniform(-5, 5)),
                    soh=max(70, 95 - degradation + random.uniform(-2, 2)),
                    cycles=150 + i,
                    internal_resistance=0.05 + (degradation * 0.001),
                    power=30.0 + random.uniform(-5, 5),
                    efficiency=0.92 - (degradation * 0.001)
                )
                db.session.add(data_point)
            
            db.session.commit()
            print("Datos de ejemplo creados exitosamente")
            
    except Exception as e:
        db.session.rollback()
        print(f"Error creando datos de ejemplo: {e}")

# Función para inicializar la base de datos
def init_database(app):
    """Inicializar base de datos con la aplicación"""
    with app.app_context():
        db.create_all()
        create_sample_data()
        print("Base de datos inicializada")
