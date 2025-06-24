from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import jwt
import os
import secrets # Para generar tokens de restablecimiento seguros
import string # Para caracteres alfanuméricos

# IMPORTANTE: Asegúrate de importar db y User correctamente.
# db viene de src.models.battery porque allí está la instancia compartida de SQLAlchemy.
# User viene de src.models.user, donde definimos el modelo actualizado.
from src.models.battery import db
from src.models.user import User # Asegúrate de que User tenga password_hash y sus métodos

auth_bp = Blueprint('auth', __name__)

# Función para generar un token JWT
def generate_jwt_token(user):
    # La clave secreta para JWT debe ser la misma que la de tu app principal (SECRET_KEY)
    # y se obtiene de current_app.config dentro de un contexto de aplicación.
    secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')

    # Define el tiempo de expiración para los tokens (ej. 24 horas)
    jwt_expiration_delta = timedelta(hours=24) # Puedes ajustar esto

    payload = {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role, # Asegúrate de que tu modelo User tenga un campo 'role'
        'exp': datetime.utcnow() + jwt_expiration_delta
    }
    return jwt.encode(payload, secret_key, algorithm='HS256')

# Función para verificar un token JWT
def verify_jwt_token(token):
    secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token ha expirado
    except jwt.InvalidTokenError:
        return None  # Token inválido

# Decorador para requerir autenticación
def require_token(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({'success': False, 'error': 'Token required'}), 401

        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401

        # Agregar información del usuario a la request
        request.current_user = payload

        return f(*args, **kwargs)

    return decorated_function

def require_role(required_role):
    """Decorador para requerir rol específico"""
    def decorator(f):
        from functools import wraps

        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'success': False, 'error': 'Authentication required'}), 401

            user_role = request.current_user.get('role')

            # Jerarquía de roles: admin > technician > user
            role_hierarchy = {'admin': 3, 'technician': 2, 'user': 1}

            user_level = role_hierarchy.get(user_role, 0)
            required_level = role_hierarchy.get(required_role, 0)

            if user_level < required_level:
                return jsonify({'success': False, 'error': 'Insufficient permissions'}), 403

            return f(*args, **kwargs)

        return decorated_function
    return decorator

@auth_bp.route('/register', methods=['POST'])
def register():
    """Registrar nuevo usuario"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Validar campos requeridos
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        # Verificar si el usuario ya existe
        existing_user = User.query.filter(
            (User.username == data['username']) | (User.email == data['email'])
        ).first()

        if existing_user:
            return jsonify({'success': False, 'error': 'Username or email already exists'}), 409

        # Crear nuevo usuario y hashear la contraseña usando el método del modelo
        user = User(
            username=data['username'],
            email=data['email'],
            # Los campos `role`, `email_notifications`, `whatsapp_number`, `sms_number`
            # deben existir en tu modelo User o serán ignorados/causarán error si no son válidos.
            # Asegúrate de que el modelo User en user.py los contenga.
            role=data.get('role', 'user'),
            email_notifications=data.get('email_notifications', True),
            whatsapp_number=data.get('whatsapp_number'),
            sms_number=data.get('sms_number')
        )
        user.set_password(data['password']) # ¡Aquí usamos el método set_password!

        db.session.add(user)
        db.session.commit()

        # Generar token JWT para el nuevo usuario
        token = generate_jwt_token(user)

        return jsonify({
            'success': True,
            'data': {
                'user': user.to_dict(),
                'token': token
            }
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Iniciar sesión"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Validar campos requeridos
        if 'username' not in data or 'password' not in data:
            return jsonify({'success': False, 'error': 'Username and password required'}), 400

        # Buscar usuario (puede ser username o email)
        user = User.query.filter(
            (User.username == data['username']) | (User.email == data['username'])
        ).first()

        if not user:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

        # Verificar contraseña usando el método del modelo
        if not user.check_password(data['password']): # ¡Aquí usamos el método check_password!
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

        # Verificar si el usuario está activo (asume un campo 'active' en el modelo User)
        if hasattr(user, 'active') and not user.active:
            return jsonify({'success': False, 'error': 'Account is deactivated'}), 401

        # Actualizar último login (asume un campo 'last_login' en el modelo User)
        if hasattr(user, 'last_login'):
            user.last_login = datetime.now()
            db.session.commit()

        # Generar token JWT
        token = generate_jwt_token(user)

        return jsonify({
            'success': True,
            'data': {
                'user': user.to_dict(),
                'token': token
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token_route(): # Renombrado para evitar conflicto con la función verify_jwt_token
    """Verificar token JWT"""
    try:
        data = request.get_json()

        if not data or 'token' not in data:
            return jsonify({'success': False, 'error': 'Token required'}), 400

        # Decodificar token
        payload = verify_jwt_token(data['token'])
        if not payload:
            return jsonify({'success': False, 'error': 'Invalid or expired token'}), 401

        user_id = payload['user_id']

        # Buscar usuario
        user = User.query.get(user_id)
        if not user or (hasattr(user, 'active') and not user.active):
            return jsonify({'success': False, 'error': 'User not found or inactive'}), 401

        return jsonify({
            'success': True,
            'data': {
                'user': user.to_dict(),
                'valid': True
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/refresh-token', methods=['POST'])
def refresh_token():
    """Renovar token JWT"""
    try:
        data = request.get_json()

        if not data or 'token' not in data:
            return jsonify({'success': False, 'error': 'Token required'}), 400

        # Decodificar token (permitir tokens expirados para renovación)
        secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')
        try:
            payload = jwt.decode(data['token'], secret_key, algorithms=['HS256'], options={"verify_exp": False})
            user_id = payload['user_id']
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Invalid token for refresh'}), 401

        # Buscar usuario
        user = User.query.get(user_id)
        if not user or (hasattr(user, 'active') and not user.active):
            return jsonify({'success': False, 'error': 'User not found or inactive'}), 401

        # Generar un nuevo token
        new_token = generate_jwt_token(user)

        return jsonify({
            'success': True,
            'data': {
                'token': new_token
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/request-password-reset', methods=['POST'])
def request_password_reset():
    """Solicitar token de restablecimiento de contraseña"""
    try:
        data = request.get_json()
        email = data.get('email')

        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400

        user = User.query.filter_by(email=email).first()
        if user:
            # Generar un token de restablecimiento seguro y con tiempo limitado
            reset_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(32)) # Token de 32 caracteres alfanuméricos
            user.reset_token = reset_token # Asume que el modelo User tiene 'reset_token'
            user.reset_token_expiration = datetime.utcnow() + timedelta(hours=1) # Token válido por 1 hora
            db.session.commit()

            # Aquí deberías enviar un email al usuario con el reset_token.
            # Por ahora, solo lo imprimimos para depuración.
            print(f"Password reset token for {user.email}: {reset_token}")

        # Siempre devuelve un éxito para evitar enumeración de usuarios
        return jsonify({'success': True, 'message': 'If your email is in our system, you will receive a password reset link.'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Restablecer contraseña usando el token"""
    try:
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('new_password')

        if not token or not new_password:
            return jsonify({'success': False, 'error': 'Token and new password required'}), 400

        user = User.query.filter_by(reset_token=token).first()

        if not user or user.reset_token_expiration < datetime.utcnow():
            return jsonify({'success': False, 'error': 'Invalid or expired reset token'}), 400

        user.set_password(new_password) # Usa el método set_password
        user.reset_token = None # Invalida el token
        user.reset_token_expiration = None
        db.session.commit()

        return jsonify({'success': True, 'message': 'Password has been reset successfully.'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/change-password', methods=['POST'])
@require_token
def change_password():
    """Cambiar contraseña (requiere login)"""
    try:
        data = request.get_json()
        old_password = data.get('old_password')
        new_password = data.get('new_password')

        if not old_password or not new_password:
            return jsonify({'success': False, 'error': 'Old and new passwords are required'}), 400

        user_id = request.current_user['user_id']
        user = User.query.get(user_id)

        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        if not user.check_password(old_password): # Verifica la contraseña antigua
            return jsonify({'success': False, 'error': 'Invalid old password'}), 401

        user.set_password(new_password) # Establece la nueva contraseña
        db.session.commit()

        return jsonify({'success': True, 'message': 'Password changed successfully'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/activate-account/<string:token>', methods=['GET'])
def activate_account(token):
    """Activar cuenta de usuario (por token de email)"""
    try:
        user = User.query.filter_by(activation_token=token).first()

        if not user:
            return jsonify({'success': False, 'error': 'Invalid activation token'}), 400

        if hasattr(user, 'active'): # Solo si el campo 'active' existe
            user.active = True
        user.activation_token = None
        db.session.commit()

        return jsonify({'success': True, 'message': 'Account activated successfully'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# Rutas de Gestión de Usuarios (requieren rol de administrador)
@auth_bp.route('/users', methods=['GET'])
@require_token
@require_role('admin')
def get_users():
    """Obtener todos los usuarios"""
    try:
        users = User.query.all()
        return jsonify({'success': True, 'data': [user.to_dict() for user in users]}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/users/<int:user_id>', methods=['GET'])
@require_token
@require_role('admin')
def get_user(user_id):
    """Obtener usuario por ID"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        return jsonify({'success': True, 'data': user.to_dict()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/users/<int:user_id>', methods=['PUT'])
@require_token
@require_role('admin')
def update_user(user_id):
    """Actualizar datos de usuario por ID"""
    try:
        data = request.get_json()
        user = User.query.get(user_id)

        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        if 'role' in data: # Asegúrate de que el modelo User tenga el campo 'role'
            user.role = data['role']
        if 'active' in data and hasattr(user, 'active'): # Solo si el campo 'active' existe
            user.active = data['active']
        if 'email_notifications' in data and hasattr(user, 'email_notifications'):
            user.email_notifications = data['email_notifications']
        if 'whatsapp_number' in data and hasattr(user, 'whatsapp_number'):
            user.whatsapp_number = data['whatsapp_number']
        if 'sms_number' in data and hasattr(user, 'sms_number'):
            user.sms_number = data['sms_number']
        if 'password' in data and data['password']: # Si se proporciona una nueva contraseña
            user.set_password(data['password']) # Usa el método set_password

        db.session.commit()
        return jsonify({'success': True, 'message': 'User updated successfully', 'data': user.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/users/<int:user_id>', methods=['DELETE'])
@require_token
@require_role('admin')
def delete_user(user_id):
    """Eliminar usuario por ID"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User deleted successfully'}), 204
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# Ruta para el perfil del usuario autenticado
@auth_bp.route('/profile', methods=['GET'])
@require_token
def get_user_profile():
    """Obtener el perfil del usuario actualmente autenticado"""
    user_id = request.current_user['user_id']
    user = User.query.get(user_id)
    if user:
        return jsonify({'success': True, 'data': user.to_dict()}), 200
    return jsonify({'success': False, 'error': 'User not found'}), 404

@auth_bp.route('/profile', methods=['PUT'])
@require_token
def update_user_profile():
    """Actualizar el perfil del usuario actualmente autenticado"""
    try:
        data = request.get_json()
        user_id = request.current_user['user_id']
        user = User.query.get(user_id)

        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        # Solo permite actualizar ciertos campos del propio perfil
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        # Los usuarios no deben poder cambiar su rol directamente aquí
        # if 'role' in data:
        #     user.role = data['role']
        if 'email_notifications' in data and hasattr(user, 'email_notifications'):
            user.email_notifications = data['email_notifications']
        if 'whatsapp_number' in data and hasattr(user, 'whatsapp_number'):
            user.whatsapp_number = data['whatsapp_number']
        if 'sms_number' in data and hasattr(user, 'sms_number'):
            user.sms_number = data['sms_number']
        # Si se proporciona una nueva contraseña para el perfil, se usa set_password
        if 'password' in data and data['password']:
            user.set_password(data['password'])

        db.session.commit()
        return jsonify({'success': True, 'message': 'Profile updated successfully', 'data': user.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
