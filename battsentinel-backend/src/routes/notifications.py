from flask import Blueprint, request, jsonify
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from src.models.battery import db, Alert, User, Battery

notifications_bp = Blueprint('notifications', __name__)

class NotificationService:
    """Servicio de notificaciones para alertas de batería"""
    
    def __init__(self):
        # Configuración de email (usar variables de entorno en producción)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = "battsentinel@gmail.com"  # Configurar en producción
        self.email_password = "app_password"  # Configurar en producción
        
        # Configuración de WhatsApp (usar WhatsApp Business API)
        self.whatsapp_api_url = "https://graph.facebook.com/v17.0/YOUR_PHONE_NUMBER_ID/messages"
        self.whatsapp_token = "YOUR_WHATSAPP_TOKEN"  # Configurar en producción
        
        # Configuración de SMS (Twilio)
        self.twilio_account_sid = "YOUR_TWILIO_SID"  # Configurar en producción
        self.twilio_auth_token = "YOUR_TWILIO_TOKEN"  # Configurar en producción
        self.twilio_phone_number = "+1234567890"  # Configurar en producción
    
    def send_email_notification(self, to_email, subject, message, alert_data=None):
        """Enviar notificación por email"""
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Crear cuerpo del mensaje HTML
            html_body = self._create_email_template(message, alert_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Enviar email (simulado para desarrollo)
            # En producción, descomentar las siguientes líneas:
            # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            # server.starttls()
            # server.login(self.email_user, self.email_password)
            # text = msg.as_string()
            # server.sendmail(self.email_user, to_email, text)
            # server.quit()
            
            print(f"Email enviado a {to_email}: {subject}")
            return True
            
        except Exception as e:
            print(f"Error enviando email: {e}")
            return False
    
    def send_whatsapp_notification(self, phone_number, message, alert_data=None):
        """Enviar notificación por WhatsApp"""
        try:
            # Formatear número de teléfono
            if not phone_number.startswith('+'):
                phone_number = '+' + phone_number
            
            # Crear mensaje para WhatsApp
            whatsapp_message = self._create_whatsapp_message(message, alert_data)
            
            # Payload para WhatsApp API
            payload = {
                "messaging_product": "whatsapp",
                "to": phone_number,
                "type": "text",
                "text": {
                    "body": whatsapp_message
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.whatsapp_token}",
                "Content-Type": "application/json"
            }
            
            # Enviar mensaje (simulado para desarrollo)
            # En producción, descomentar las siguientes líneas:
            # response = requests.post(self.whatsapp_api_url, json=payload, headers=headers)
            # response.raise_for_status()
            
            print(f"WhatsApp enviado a {phone_number}: {whatsapp_message}")
            return True
            
        except Exception as e:
            print(f"Error enviando WhatsApp: {e}")
            return False
    
    def send_sms_notification(self, phone_number, message, alert_data=None):
        """Enviar notificación por SMS"""
        try:
            # Crear mensaje SMS
            sms_message = self._create_sms_message(message, alert_data)
            
            # Enviar SMS usando Twilio (simulado para desarrollo)
            # En producción, usar la biblioteca de Twilio:
            # from twilio.rest import Client
            # client = Client(self.twilio_account_sid, self.twilio_auth_token)
            # message = client.messages.create(
            #     body=sms_message,
            #     from_=self.twilio_phone_number,
            #     to=phone_number
            # )
            
            print(f"SMS enviado a {phone_number}: {sms_message}")
            return True
            
        except Exception as e:
            print(f"Error enviando SMS: {e}")
            return False
    
    def _create_email_template(self, message, alert_data):
        """Crear plantilla HTML para email"""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        severity = alert_data.get('severity', 'medium') if alert_data else 'medium'
        color = severity_colors.get(severity, '#6c757d')
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>BattSentinel Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px; }}
                .alert-info {{ background-color: #f8f9fa; border-left: 4px solid {color}; padding: 15px; margin: 20px 0; }}
                .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #6c757d; }}
                .button {{ display: inline-block; background-color: {color}; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔋 BattSentinel Alert</h1>
                    <p>Sistema de Monitoreo de Baterías</p>
                </div>
                <div class="content">
                    <h2>Alerta de Batería</h2>
                    <p>{message}</p>
                    
                    {self._format_alert_details(alert_data) if alert_data else ''}
                    
                    <div class="alert-info">
                        <strong>Severidad:</strong> {severity.upper()}<br>
                        <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    
                    <p>Por favor, revise el estado de la batería en el dashboard de BattSentinel.</p>
                    
                    <a href="#" class="button">Ver Dashboard</a>
                </div>
                <div class="footer">
                    <p>Este es un mensaje automático de BattSentinel. No responda a este email.</p>
                    <p>© 2024 BattSentinel - Sistema Inteligente de Monitoreo de Baterías</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_whatsapp_message(self, message, alert_data):
        """Crear mensaje para WhatsApp"""
        severity = alert_data.get('severity', 'medium') if alert_data else 'medium'
        battery_id = alert_data.get('battery_id', 'N/A') if alert_data else 'N/A'
        
        severity_emojis = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }
        
        emoji = severity_emojis.get(severity, '⚠️')
        
        whatsapp_message = f"""
🔋 *BattSentinel Alert* {emoji}

*Batería ID:* {battery_id}
*Severidad:* {severity.upper()}
*Mensaje:* {message}

*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Revise el dashboard para más detalles.
        """.strip()
        
        return whatsapp_message
    
    def _create_sms_message(self, message, alert_data):
        """Crear mensaje para SMS"""
        severity = alert_data.get('severity', 'medium') if alert_data else 'medium'
        battery_id = alert_data.get('battery_id', 'N/A') if alert_data else 'N/A'
        
        sms_message = f"BattSentinel Alert - Batería {battery_id}: {message} (Severidad: {severity.upper()})"
        
        # Limitar a 160 caracteres para SMS
        if len(sms_message) > 160:
            sms_message = sms_message[:157] + "..."
        
        return sms_message
    
    def _format_alert_details(self, alert_data):
        """Formatear detalles de la alerta para email"""
        if not alert_data:
            return ""
        
        details = "<div class='alert-info'><strong>Detalles de la Alerta:</strong><br>"
        
        if 'battery_id' in alert_data:
            details += f"<strong>ID de Batería:</strong> {alert_data['battery_id']}<br>"
        
        if 'fault_type' in alert_data:
            details += f"<strong>Tipo de Falla:</strong> {alert_data['fault_type']}<br>"
        
        if 'current_values' in alert_data:
            values = alert_data['current_values']
            details += "<strong>Valores Actuales:</strong><br>"
            for key, value in values.items():
                details += f"&nbsp;&nbsp;• {key}: {value}<br>"
        
        details += "</div>"
        return details

# Instancia global del servicio
notification_service = NotificationService()

@notifications_bp.route('/send-alert', methods=['POST'])
def send_alert_notification():
    """Enviar notificación de alerta"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
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
        alert = Alert(
            battery_id=data['battery_id'],
            alert_type=data['alert_type'],
            title=data.get('title', f"Alerta de {data['alert_type']}"),
            message=data['message'],
            severity=data['severity']
        )
        
        db.session.add(alert)
        db.session.commit()
        
        # Obtener usuarios para notificar
        users_to_notify = data.get('users', [])
        if not users_to_notify:
            # Notificar a todos los usuarios activos si no se especifican
            users_to_notify = User.query.filter_by(active=True).all()
        else:
            users_to_notify = User.query.filter(User.id.in_(users_to_notify)).all()
        
        # Preparar datos de la alerta para las notificaciones
        alert_data = {
            'battery_id': data['battery_id'],
            'battery_name': battery.name,
            'alert_type': data['alert_type'],
            'severity': data['severity'],
            'current_values': data.get('current_values', {})
        }
        
        # Enviar notificaciones
        notification_results = {
            'email_sent': 0,
            'whatsapp_sent': 0,
            'sms_sent': 0,
            'failed_notifications': []
        }
        
        for user in users_to_notify:
            # Email
            if user.email_notifications and user.email:
                success = notification_service.send_email_notification(
                    user.email,
                    f"BattSentinel Alert - {data['severity'].upper()}",
                    data['message'],
                    alert_data
                )
                if success:
                    notification_results['email_sent'] += 1
                    alert.email_sent = True
                else:
                    notification_results['failed_notifications'].append(f"Email to {user.email}")
            
            # WhatsApp
            if user.whatsapp_number:
                success = notification_service.send_whatsapp_notification(
                    user.whatsapp_number,
                    data['message'],
                    alert_data
                )
                if success:
                    notification_results['whatsapp_sent'] += 1
                    alert.whatsapp_sent = True
                else:
                    notification_results['failed_notifications'].append(f"WhatsApp to {user.whatsapp_number}")
            
            # SMS
            if user.sms_number:
                success = notification_service.send_sms_notification(
                    user.sms_number,
                    data['message'],
                    alert_data
                )
                if success:
                    notification_results['sms_sent'] += 1
                    alert.sms_sent = True
                else:
                    notification_results['failed_notifications'].append(f"SMS to {user.sms_number}")
        
        # Actualizar alerta con resultados de notificación
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'alert_id': alert.id,
                'notification_results': notification_results,
                'users_notified': len(users_to_notify)
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/alerts/<int:battery_id>', methods=['GET'])
def get_battery_alerts(battery_id):
    """Obtener alertas de una batería"""
    try:
        battery = Battery.query.get_or_404(battery_id)
        
        # Parámetros de consulta
        status = request.args.get('status')
        severity = request.args.get('severity')
        limit = request.args.get('limit', 50, type=int)
        
        query = Alert.query.filter_by(battery_id=battery_id)
        
        if status:
            query = query.filter_by(status=status)
        if severity:
            query = query.filter_by(severity=severity)
        
        alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'data': [alert.to_dict() for alert in alerts]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Reconocer una alerta"""
    try:
        alert = Alert.query.get_or_404(alert_id)
        
        alert.status = 'acknowledged'
        alert.acknowledged_at = datetime.now()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': alert.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/alerts/<int:alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolver una alerta"""
    try:
        alert = Alert.query.get_or_404(alert_id)
        
        alert.status = 'resolved'
        alert.resolved_at = datetime.now()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': alert.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/test-notification', methods=['POST'])
def test_notification():
    """Probar envío de notificaciones"""
    try:
        data = request.get_json()
        
        notification_type = data.get('type', 'email')
        recipient = data.get('recipient')
        message = data.get('message', 'Mensaje de prueba de BattSentinel')
        
        if not recipient:
            return jsonify({'success': False, 'error': 'Recipient required'}), 400
        
        success = False
        
        if notification_type == 'email':
            success = notification_service.send_email_notification(
                recipient,
                "BattSentinel - Prueba de Notificación",
                message
            )
        elif notification_type == 'whatsapp':
            success = notification_service.send_whatsapp_notification(
                recipient,
                message
            )
        elif notification_type == 'sms':
            success = notification_service.send_sms_notification(
                recipient,
                message
            )
        else:
            return jsonify({'success': False, 'error': 'Invalid notification type'}), 400
        
        return jsonify({
            'success': success,
            'message': f'Test {notification_type} notification {"sent" if success else "failed"}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@notifications_bp.route('/settings/<int:user_id>', methods=['GET', 'PUT'])
def notification_settings(user_id):
    """Obtener o actualizar configuración de notificaciones de usuario"""
    try:
        user = User.query.get_or_404(user_id)
        
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'data': {
                    'user_id': user.id,
                    'email_notifications': user.email_notifications,
                    'whatsapp_number': user.whatsapp_number,
                    'sms_number': user.sms_number
                }
            })
        
        elif request.method == 'PUT':
            data = request.get_json()
            
            if 'email_notifications' in data:
                user.email_notifications = data['email_notifications']
            if 'whatsapp_number' in data:
                user.whatsapp_number = data['whatsapp_number']
            if 'sms_number' in data:
                user.sms_number = data['sms_number']
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': {
                    'user_id': user.id,
                    'email_notifications': user.email_notifications,
                    'whatsapp_number': user.whatsapp_number,
                    'sms_number': user.sms_number
                }
            })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

