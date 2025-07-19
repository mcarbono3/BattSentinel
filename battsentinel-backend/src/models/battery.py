from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import json
from src.main import db

class Battery(db.Model):
    """Modelo de Batería"""
    __tablename__ = 'batteries'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    model = Column(String(100))
    manufacturer = Column(String(100))
    serial_number = Column(String(100))
    full_charge_capacity = Column(Float)
    full_charge_capacity_unit = Column(String(10)) # Nueva columna
    nominal_capacity = Column(Float) # Nueva columna
    nominal_capacity_unit = Column(String(10)) # Nueva columna
    designvoltage = Column(Float)  
    chemistry = Column(String(50))
    installation_date = Column(DateTime, default=datetime.utcnow)
    location = Column(String(200))
    status = Column(String(50), default='active')
    last_maintenance_date = Column(DateTime) # Nueva columna
    warranty_expiry_date = Column(DateTime) # Nueva columna
    cycles = Column(Integer, nullable=True) # Nueva columna
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    monitoring_source = db.Column(db.String(100))
    description = db.Column(db.Text)
    
    # Relaciones
    data_points = relationship('BatteryData', backref='battery', lazy=True, cascade='all, delete-orphan')
    alerts = relationship('Alert', backref='battery', lazy=True, cascade='all, delete-orphan')
    analysis_results = db.relationship('AnalysisResult', backref='battery', lazy=True, cascade='all, delete-orphan')
    thermal_images = db.relationship('ThermalImage', backref='battery', lazy=True, cascade='all, delete-orphan')
    maintenance_records = db.relationship('MaintenanceRecord', backref='battery', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        # 1. Inicializar el diccionario con los campos básicos de la batería
        data = {
            'id': self.id,
            'name': self.name,
            'model': self.model,
            'manufacturer': self.manufacturer,
            'serial_number': self.serial_number,
            'full_charge_capacity': self.full_charge_capacity,
            'full_charge_capacity_unit': self.full_charge_capacity_unit, # Añadido
            'nominal_capacity': self.nominal_capacity, # Añadido
            'nominal_capacity_unit': self.nominal_capacity_unit, # Añadido
            'designvoltage': self.designvoltage,
            'chemistry': self.chemistry,
            'installation_date': self.installation_date.isoformat() if self.installation_date else None,
            'location': self.location,
            'status': self.status,
            'last_maintenance_date': self.last_maintenance_date.isoformat() if self.last_maintenance_date else None, # Añadido
            'warranty_expiry_date': self.warranty_expiry_date.isoformat() if self.warranty_expiry_date else None, # Añadido
            'cycles': self.cycles, # Añadido
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'monitoring_source': self.monitoring_source,
            'description': self.description
        }

        # 2. LÓGICA PARA OBTENER EL ÚLTIMO PUNTO DE DATOS
        latest_data_point = db.session.query(BatteryData) \
                                .filter(BatteryData.battery_id == self.id) \
                                .order_by(BatteryData.timestamp.desc()) \
                                .first()
        # 3. Añadir solo los campos deseados de BatteryData al diccionario 'data'
        if latest_data_point:
            data['soc'] = latest_data_point.soc
            data['soh'] = latest_data_point.soh
            data['temperature'] = latest_data_point.temperature
            data['is_plugged'] = latest_data_point.is_plugged
            # Puedes añadir también la marca de tiempo si es útil para el frontend
            data['latest_timestamp'] = latest_data_point.timestamp.isoformat() if latest_data_point.timestamp else None
        else:
            # Si no hay datos, inicializa los campos con None o '--'
            data['soc'] = None
            data['soh'] = None
            data['temperature'] = None
            data['is_plugged'] = None
            data['latest_timestamp'] = None

        # 4. Finalmente, retornar el diccionario 'data' completo
        return data

class BatteryData(db.Model):
    """Modelo de Datos de Batería"""
    __tablename__ = 'battery_data'
    
    id = Column(Integer, primary_key=True)
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False) # Usar lambda para UTC        
    voltage = Column(Float, nullable=True)
    current = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    capacity = Column(Float, nullable=True)
    soc = Column(Float, nullable=True)  # State of Charge
    soh = Column(Float, nullable=True)  # State of Health
    cycles = Column(Integer, default=0, nullable=True)    
    # Campos que ya tenías definidos o se han unificado
    internal_resistance = Column(Float, nullable=True)
    power = Column(Float, nullable=True)
    efficiency = Column(Float, nullable=True)
    # --- CAMPOS AÑADIDOS / CORREGIDOS (incluyendo los del JSON de monitoreo) ---
    energy_rate = Column(Float, nullable=True) # Corregido: antes energy, ahora energy_rate
    rul_days = Column(Integer, nullable=True)
    is_plugged = Column(Boolean, nullable=True)
    time_left = Column(Integer, nullable=True) # En segundos o minutos
    status = Column(String(50), nullable=True) # Añadido: para manejar 'status' o 'batterystatus'
    energy_consumed = Column(Float, nullable=True) # Añadido si lo envías desde el cliente
    pressure = Column(Float, nullable=True) # Añadido si lo envías desde el cliente
    humidity = Column(Float, nullable=True) # Ya estaba

    # --- ¡NUEVOS CAMPOS AÑADIDOS DESDE TUS DATOS DE MONITOREO! ---
    estimatedchargeremaining = Column(Float, nullable=True)
    estimatedruntime = Column(Float, nullable=True)         
    source = Column(String(100), nullable=True) # Fuente de los datos
    
    def to_dict(self):
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'capacity': self.capacity, 
            'soc': self.soc,
            'soh': self.soh,
            'cycles': self.cycles,
            'internal_resistance': self.internal_resistance,
            'power': self.power,
            'efficiency': self.efficiency,
            'energy_rate': self.energy_rate,
            'rul_days': self.rul_days,
            'is_plugged': self.is_plugged,
            'time_left': self.time_left,
            'status': self.status,
            'energy_consumed': self.energy_consumed,
            'pressure': self.pressure,
            'humidity': self.humidity,
            # --- NUEVOS CAMPOS EN to_dict ---
            'estimatedchargeremaining': self.estimatedchargeremaining,
            'estimatedruntime': self.estimatedruntime,            
            'source': self.source,
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
    result = Column(JSONB, nullable=False)  # Usa JSONB para PostgreSQL
    confidence_score = Column(Float)
    model_version = Column(String(50))
    processing_time = Column(Float)  # segundos
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # --- CAMPOS ACTUALES (CONFIRMADOS O YA PRESENTES) ---
    fault_detected = Column(Boolean, nullable=True)
    fault_type = Column(String(100), nullable=True)
    severity = Column(String(50), nullable=True) # low, medium, high, critical
    rul_prediction = Column(Float, nullable=True) # Predicción de días de vida útil restante
    explanation = Column(JSONB, nullable=True) # Usa JSONB para PostgreSQL
    
    # --- NUEVOS CAMPOS A AÑADIR ---
    level_of_analysis = Column(Integer, nullable=False) # 1 para Monitoreo Continuo, 2 para Análisis Avanzado
    system_summary = Column(JSONB, nullable=True) # Almacena overall_status, priority_alerts, recommendations (Usa JSONB para PostgreSQL)

    # Opcional: relación de vuelta a la batería si no la tienes ya definida
    # battery = relationship('Battery', backref='analysis_results', lazy=True)
    
    def to_dict(self):
        # SQLAlchemy con JSONB generalmente deserializa automáticamente a dict/list.
        # Estas comprobaciones son más para seguridad o si se usa Text en algún caso.
        result_data = self.result if self.result is not None else {}
        explanation_data = self.explanation if self.explanation is not None else {}
        system_summary_data = self.system_summary if self.system_summary is not None else {}
            
        return {
            'id': self.id,
            'battery_id': self.battery_id,
            'analysis_type': self.analysis_type,
            'result': result_data,
            'confidence_score': self.confidence_score,
            'model_version': self.model_version,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'fault_detected': self.fault_detected,
            'fault_type': self.fault_type,
            'severity': self.severity,
            'rul_prediction': self.rul_prediction,
            'explanation': explanation_data,
            'level_of_analysis': self.level_of_analysis,
            'system_summary': system_summary_data
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

class DigitalTwin(db.Model):
    """Modelo para almacenar los gemelos digitales de las baterías."""
    __tablename__ = 'digital_twins'

    id = Column(Integer, primary_key=True)
    twin_id = Column(String(255), unique=True, nullable=False) # UUID para el gemelo
    battery_id = Column(Integer, ForeignKey('batteries.id'), nullable=False)

    # Almacenar el modelo complejo como JSON/Texto
    parameters = Column(JSONB, nullable=True) # Usa Text si no usas PostgreSQL y no quieres JSONB
    initial_state = Column(JSONB, nullable=True) # Usa Text si no usas PostgreSQL y no quieres JSONB
    initialization_info = Column(JSONB, nullable=True) # Usa Text si no usas PostgreSQL y no quieres JSONB

    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'twin_id': self.twin_id,
            'battery_id': self.battery_id,
            'parameters': self.parameters,
            'initial_state': self.initial_state,
            'initialization_info': self.initialization_info,
            'created_at': self.created_at.isoformat() if self.created_at else None
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
                full_charge_capacity=100.0,
                designvoltage=12.0,
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
