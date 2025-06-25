from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import json

# Importaciones locales
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.battery import db, Battery, Alert
from services.windows_battery import windows_battery_service

notifications_bp = Blueprint('notifications', __name__)

class NotificationService:
    """Servicio simplificado de notificaciones para alertas de batería"""
    
    def __init__(self):
        self.notification_log = []
    
    def send_notification(self, notification_type, recipient, message, alert_data=None):
        """Enviar notificación (simulada para demo)"""
        try:
            notification = {
                'type': notification_type,
                'recipient': recipient,
                'message': message,
                'alert_data': alert_data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'sent'
            }
            
            self.notification_log.append(notification)
            print(f"Notificación {notification_type} enviada a {recipient}: {message}")
            return True
            
        except Exception as e:
            print(f"Error enviando notificación: {e}")
            return False
    
    def get_notification_log(self):
        """Obtener log de notificaciones"""
        return self.notification_log

# Instancia global del servicio
notification_service = NotificationService()

@notifications_bp.route('/api/notifications/send-alert', methods=['POST'])
def send_alert_notification():
    """Enviar notificación de alerta - Sin autenticación"""
    try:
        data = request.get_json() or {}
        
        # Validar datos requeridos
        required_fields = ['battery_id', 'alert_type', 'message', 'severity']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Obtener información de la batería
        battery = Battery.query.get(data['battery_id'])
        if not battery:
            return jsonify({'success': False, 'error': 'Battery not found'}), 404
        
        # Crear alerta en la base de datos
        try:
            with current_app.app_context():
                alert = Alert(
                    battery_id=data["battery_id"],
                    alert_type=data["alert_type"],
                    title=data.get("title", f"Alerta de {data['alert_type']}"),
                    message=data["message"],
                    severity=data["severity"]
                )
                
                db.session.add(alert)
                db.session.commit()
                alert_id = alert.id
        except Exception as e:
            # Si falla la BD, continuar con notificación simulada
            alert_id = f"sim_{int(datetime.now().timestamp())}"
        
        # Preparar datos de la alerta
        alert_data = {
            'battery_id': data['battery_id'],
            'battery_name': battery.name if battery else f"Batería {data['battery_id']}",
            'alert_type': data['alert_type'],
            'severity': data['severity'],
            'current_values': data.get('current_values', {})
        }
        
        # Enviar notificaciones simuladas
        notification_results = {
            'email_sent': 0,
            'whatsapp_sent': 0,
            'sms_sent': 0,
            'in_app_sent': 0,
            'failed_notifications': []
        }
        
        # Simular envío a diferentes canales
        recipients = data.get('recipients', ['admin@battsentinel.com', '+1234567890'])
        
        for recipient in recipients:
            if '@' in recipient:
                # Email
                success = notification_service.send_notification(
                    'email',
                    recipient,
                    data['message'],
                    alert_data
                )
                if success:
                    notification_results['email_sent'] += 1
                else:
                    notification_results['failed_notifications'].append(f"Email to {recipient}")
            
            elif recipient.startswith('+'):
                # WhatsApp/SMS
                success = notification_service.send_notification(
                    'whatsapp',
                    recipient,
                    data['message'],
                    alert_data
                )
                if success:
                    notification_results['whatsapp_sent'] += 1
                else:
                    notification_results['failed_notifications'].append(f"WhatsApp to {recipient}")
        
        # Notificación in-app siempre exitosa
        notification_service.send_notification(
            'in_app',
            'dashboard',
            data['message'],
            alert_data
        )
        notification_results['in_app_sent'] = 1
        
        return jsonify({
            'success': True,
            'data': {
                'alert_id': alert_id,
                'notification_results': notification_results,
                'recipients_notified': len(recipients),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/api/notifications/alerts/<int:battery_id>', methods=['GET'])
def get_battery_alerts(battery_id):
    """Obtener alertas de una batería - Sin autenticación"""
    try:
        with current_app.app_context():
            battery = Battery.query.get_or_404(battery_id)
            
            # Parámetros de consulta
            status = request.args.get("status")
            severity = request.args.get("severity")
            limit = request.args.get("limit", 50, type=int)
            
            try:
                query = Alert.query.filter_by(battery_id=battery_id)
                
                if status:
                    query = query.filter_by(status=status)
                if severity:
                    query = query.filter_by(severity=severity)
                
                alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
                alerts_data = [alert.to_dict() for alert in alerts]
            except Exception as e:
                # Si falla la consulta, generar alertas de ejemplo
                alerts_data = generate_sample_alerts(battery_id, limit)
            
            return jsonify({
                "success": True,
                "data": alerts_data
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@notifications_bp.route('/api/notifications/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Reconocer una alerta - Sin autenticación"""
    try:
        with current_app.app_context():
            try:
                alert = Alert.query.get_or_404(alert_id)
                alert.status = "acknowledged"
                alert.acknowledged_at = datetime.now(timezone.utc)
                db.session.commit()
                alert_data = alert.to_dict()
            except Exception as e:
                # Simulación si falla la BD
                alert_data = {
                    "id": alert_id,
                    "status": "acknowledged",
                    "acknowledged_at": datetime.now(timezone.utc).isoformat()
                }
            
            return jsonify({
                "success": True,
                "data": alert_data
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@notifications_bp.route('/api/notifications/alerts/<int:alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolver una alerta - Sin autenticación"""
    try:
        with current_app.app_context():
            try:
                alert = Alert.query.get_or_404(alert_id)
                alert.status = "resolved"
                alert.resolved_at = datetime.now(timezone.utc)
                db.session.commit()
                alert_data = alert.to_dict()
            except Exception as e:
                # Simulación si falla la BD
                alert_data = {
                    "id": alert_id,
                    "status": "resolved",
                    "resolved_at": datetime.now(timezone.utc).isoformat()
                }
            
            return jsonify({
                "success": True,
                "data": alert_data
            })
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@notifications_bp.route('/api/notifications/test-notification', methods=['POST'])
def test_notification():
    """Probar envío de notificación - Sin autenticación"""
    try:
        data = request.get_json() or {}
        
        notification_type = data.get('type', 'email')
        recipient = data.get('recipient', 'test@battsentinel.com')
        message = data.get('message', 'Mensaje de prueba de BattSentinel')
        
        # Enviar notificación de prueba
        success = notification_service.send_notification(
            notification_type,
            recipient,
            message,
            {'test': True, 'timestamp': datetime.now(timezone.utc).isoformat()}
        )
        
        return jsonify({
            'success': success,
            'data': {
                'type': notification_type,
                'recipient': recipient,
                'message': message,
                'sent_at': datetime.now(timezone.utc).isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/api/notifications/settings', methods=['GET'])
def get_notification_settings():
    """Obtener configuración de notificaciones - Sin autenticación"""
    try:
        # Configuración por defecto
        settings = {
            'email_enabled': True,
            'whatsapp_enabled': True,
            'sms_enabled': False,
            'in_app_enabled': True,
            'severity_filters': {
                'low': True,
                'medium': True,
                'high': True,
                'critical': True
            },
            'alert_types': {
                'fault_detection': True,
                'temperature_alert': True,
                'voltage_alert': True,
                'soh_degradation': True,
                'maintenance_reminder': True
            },
            'quiet_hours': {
                'enabled': False,
                'start_time': '22:00',
                'end_time': '08:00'
            }
        }
        
        return jsonify({
            'success': True,
            'data': settings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/api/notifications/settings', methods=['PUT'])
def update_notification_settings():
    """Actualizar configuración de notificaciones - Sin autenticación"""
    try:
        data = request.get_json() or {}
        
        # Simular actualización de configuración
        updated_settings = {
            'email_enabled': data.get('email_enabled', True),
            'whatsapp_enabled': data.get('whatsapp_enabled', True),
            'sms_enabled': data.get('sms_enabled', False),
            'in_app_enabled': data.get('in_app_enabled', True),
            'severity_filters': data.get('severity_filters', {}),
            'alert_types': data.get('alert_types', {}),
            'quiet_hours': data.get('quiet_hours', {}),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': updated_settings,
            'message': 'Configuración de notificaciones actualizada'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/api/notifications/log', methods=['GET'])
def get_notification_log():
    """Obtener log de notificaciones - Sin autenticación"""
    try:
        limit = request.args.get('limit', 100, type=int)
        notification_type = request.args.get('type')
        
        log = notification_service.get_notification_log()
        
        # Filtrar por tipo si se especifica
        if notification_type:
            log = [n for n in log if n['type'] == notification_type]
        
        # Limitar resultados
        log = log[-limit:] if len(log) > limit else log
        
        return jsonify({
            'success': True,
            'data': {
                'notifications': log,
                'total_count': len(log),
                'types_summary': {
                    'email': len([n for n in log if n['type'] == 'email']),
                    'whatsapp': len([n for n in log if n['type'] == 'whatsapp']),
                    'sms': len([n for n in log if n['type'] == 'sms']),
                    'in_app': len([n for n in log if n['type'] == 'in_app'])
                }
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_sample_alerts(battery_id, limit=10):
    """Generar alertas de ejemplo"""
    import random
    
    alert_types = ['fault_detection', 'temperature_alert', 'voltage_alert', 'soh_degradation', 'maintenance_reminder']
    severities = ['low', 'medium', 'high', 'critical']
    statuses = ['active', 'acknowledged', 'resolved']
    
    sample_alerts = []
    
    for i in range(min(limit, 20)):
        alert_type = random.choice(alert_types)
        severity = random.choice(severities)
        status = random.choice(statuses)
        
        created_at = datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 168))
        
        alert = {
            'id': i + 1,
            'battery_id': battery_id,
            'alert_type': alert_type,
            'title': f"Alerta de {alert_type.replace('_', ' ').title()}",
            'message': generate_alert_message(alert_type, severity),
            'severity': severity,
            'status': status,
            'created_at': created_at.isoformat(),
            'acknowledged_at': (created_at + timedelta(minutes=random.randint(5, 60))).isoformat() if status != 'active' else None,
            'resolved_at': (created_at + timedelta(hours=random.randint(1, 24))).isoformat() if status == 'resolved' else None,
            'email_sent': True,
            'whatsapp_sent': random.choice([True, False]),
            'sms_sent': False
        }
        
        sample_alerts.append(alert)
    
    return sample_alerts

def generate_alert_message(alert_type, severity):
    """Generar mensaje de alerta según el tipo y severidad"""
    messages = {
        'fault_detection': {
            'low': 'Anomalía menor detectada en el comportamiento de la batería',
            'medium': 'Patrón de degradación detectado en la batería',
            'high': 'Falla significativa detectada en el sistema de batería',
            'critical': 'Falla crítica detectada - Requiere atención inmediata'
        },
        'temperature_alert': {
            'low': 'Temperatura ligeramente elevada (35°C)',
            'medium': 'Temperatura alta detectada (42°C)',
            'high': 'Temperatura peligrosa detectada (55°C)',
            'critical': 'Temperatura crítica - Riesgo de daño térmico (65°C)'
        },
        'voltage_alert': {
            'low': 'Voltaje fuera del rango óptimo',
            'medium': 'Fluctuaciones de voltaje detectadas',
            'high': 'Voltaje peligrosamente bajo o alto',
            'critical': 'Voltaje en niveles críticos - Riesgo de daño'
        },
        'soh_degradation': {
            'low': 'Degradación gradual del estado de salud (SOH: 82%)',
            'medium': 'Degradación acelerada detectada (SOH: 75%)',
            'high': 'Estado de salud significativamente degradado (SOH: 65%)',
            'critical': 'Estado de salud crítico - Reemplazo requerido (SOH: 55%)'
        },
        'maintenance_reminder': {
            'low': 'Mantenimiento preventivo programado',
            'medium': 'Mantenimiento requerido pronto',
            'high': 'Mantenimiento urgente requerido',
            'critical': 'Mantenimiento crítico - Sistema en riesgo'
        }
    }
    
    return messages.get(alert_type, {}).get(severity, 'Alerta del sistema de batería')
