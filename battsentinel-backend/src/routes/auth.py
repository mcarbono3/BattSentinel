from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone # Importa timezone
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
        'exp': datetime.now(timezone.utc) + jwt_expiration_delta # Usa datetime.now(timezone.utc)
    }
    return jwt.encode(payload, secret_key, algorithm='HS256')

# Middleware para proteger rutas que requieren JWT
def jwt_required():
    def decorator(f):
        def wrapper(*args, **kwargs):
            token = None
            if 'Authorization' in request.headers:
                token = request.headers['Authorization'].split(" ")[1]

            if not token:
                return jsonify({'success': False, 'error': 'Token is missing!'}), 401
            
            secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')

            try:
                # Intenta decodificar el token con el algoritmo correcto
                data = jwt.decode(token, secret_key, algorithms=['HS256'])
                current_user = User.query.get(data['user_id'])
                if not current_user:
                    return jsonify({'success': False, 'error': 'User not found'}), 401
                
                # Adjunta el objeto de usuario al objeto request para que esté disponible en la ruta
                request.current_user = current_user.to_dict() # Pasa el diccionario del usuario

            except jwt.ExpiredSignatureError:
                return jsonify({'success': False, 'error': 'Token has expired!'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'success': False, 'error': 'Token is invalid!'}), 401
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__ # Esto es importante para Flask
        return wrapper
    return decorator


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user') # Por defecto 'user'

    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Missing username, email, or password'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'error': 'Username already exists'}), 409
    
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'Email already exists'}), 409

    try:
        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User registered successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    if not user.active:
        return jsonify({'success': False, 'error': 'Account is inactive. Please contact support.'}), 403

    token = generate_jwt_token(user)
    
    # Actualiza la última fecha de login
    user.last_login = datetime.now(timezone.utc)
    db.session.commit()

    # Devuelve los datos del usuario y el token
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'data': {
            'user': user.to_dict(), # Asegúrate de que to_dict() no exponga datos sensibles
            'token': token
        }
    }), 200

# Endpoint para verificar la validez del token
@auth_bp.route('/verify-token', methods=['GET', 'POST']) # <--- Aquí está la mejora
@jwt_required()
def verify_token():
    """
    Verifica la validez del token JWT y devuelve la información del usuario.
    El decorador @jwt_required() ya se encarga de la lógica de verificación.
    """
    try:
        # Si llegamos aquí, el token es válido y request.current_user contiene los datos.
        user_data = request.current_user
        return jsonify({'success': True, 'message': 'Token is valid', 'data': user_data}), 200
    except Exception as e:
        # Esto es un catch-all, pero @jwt_required() debería manejar la mayoría de los errores.
        current_app.logger.error(f"Error inesperado en verify_token: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    # En un sistema sin listas negras de JWT (blacklist), el logout en el servidor
    # es puramente simbólico y se basa en que el cliente elimine el token.
    # Si tuvieras un sistema de blacklist, aquí es donde invalidarías el token.
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

# Ruta para solicitar restablecimiento de contraseña
@auth_bp.route('/request-password-reset', methods=['POST'])
def request_password_reset():
    data = request.get_json()
    email = data.get('email')
    user = User.query.filter_by(email=email).first()

    if not user:
        # Por seguridad, no reveles si el email existe o no
        return jsonify({'success': True, 'message': 'If an account with that email exists, a password reset link has been sent.'}), 200

    # Generar un token de restablecimiento único y de corta duración
    reset_token = secrets.token_urlsafe(32) # Genera un token seguro y amigable para URL
    user.reset_token = reset_token
    user.reset_token_expiration = datetime.now(timezone.utc) + timedelta(hours=1) # Token válido por 1 hora
    db.session.commit()

    # TODO: Enviar el email con el enlace de restablecimiento
    # En un entorno real, usarías una librería de envío de correos electrónicos.
    # El enlace al frontend sería algo como: `http://localhost:5173/reset-password?token={reset_token}`
    reset_link = f"{request.url_root.replace('/api/auth/', '')}reset-password?token={reset_token}"
    print(f"Password reset link for {user.email}: {reset_link}") # SOLO PARA DESARROLLO

    return jsonify({'success': True, 'message': 'Password reset link sent to your email.'}), 200

# Ruta para restablecer la contraseña usando el token
@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    token = data.get('token')
    new_password = data.get('new_password')

    if not token or not new_password:
        return jsonify({'success': False, 'error': 'Missing token or new password'}), 400

    user = User.query.filter_by(reset_token=token).first()

    if not user or user.reset_token_expiration < datetime.now(timezone.utc):
        return jsonify({'success': False, 'error': 'Invalid or expired reset token'}), 400

    user.set_password(new_password)
    user.reset_token = None # Invalida el token después de usarlo
    user.reset_token_expiration = None
    db.session.commit()

    return jsonify({'success': True, 'message': 'Password has been reset successfully.'}), 200

@auth_bp.route('/users/<int:user_id>/profile', methods=['PUT'])
@jwt_required()
def update_user_profile(user_id):
    """Actualizar el perfil del usuario actualmente autenticado"""
    try:
        data = request.get_json()
        
        # Asegurarse de que el usuario solo pueda actualizar su propio perfil
        # A menos que sea un admin y se esté actualizando otro usuario
        # Puedes añadir más lógica aquí si los administradores pueden editar perfiles de otros
        if request.current_user['id'] != user_id and request.current_user['role'] != 'admin':
            return jsonify({'success': False, 'error': 'Unauthorized to update this profile'}), 403

        user = User.query.get(user_id)

        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        # Solo permite actualizar ciertos campos del propio perfil
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        # Los usuarios no deben poder cambiar su rol directamente aquí
        # if 'role' in data: # Solo admin debería cambiar roles
        #     if request.current_user['role'] == 'admin':
        #         user.role = data['role']
        #     else:
        #         return jsonify({'success': False, 'error': 'Unauthorized to change role'}), 403

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
        current_app.logger.error(f"Error updating user profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
