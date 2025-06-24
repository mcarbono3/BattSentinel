# src/routes/auth.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
import jwt
import os
import secrets # Para generar tokens de restablecimiento seguros
from functools import wraps # Para usar en el decorador jwt_required

# IMPORTANTE: Asegúrate de importar db y User correctamente.
# db viene de src.models.battery porque allí está la instancia compartida de SQLAlchemy.
# User viene de src.models.user, donde definimos el modelo actualizado.
from src.models.battery import db
from src.models.user import User

auth_bp = Blueprint('auth', __name__)

# Función para generar un token JWT
def generate_jwt_token(user):
    # La clave secreta para JWT debe ser la misma que la de tu app principal (SECRET_KEY)
    # y se obtiene de current_app.config dentro de un contexto de aplicación.
    secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')

    # Define el tiempo de expiración para los tokens (ej. 24 horas)
    # Asegúrate de que datetime.now() use el mismo timezone que 'exp' en el payload.
    # Se recomienda usar UTC para consistencia.
    jwt_expiration_delta = timedelta(hours=24)

    payload = {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role, # Asegúrate de que tu modelo User tenga un campo 'role'
        'exp': datetime.now(timezone.utc) + jwt_expiration_delta # Usar datetime.now(timezone.utc)
    }
    # Asegúrate de especificar el algoritmo, comúnmente 'HS256'
    return jwt.encode(payload, secret_key, algorithm='HS256')


# Decorador para proteger rutas con JWT (¡Reemplaza tu versión existente!)
def jwt_required():
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                current_app.logger.warning("Intento de acceso sin token o token mal formado.")
                return jsonify({'success': False, 'error': 'Token de autenticación faltante o inválido'}), 401

            token = auth_header.split(' ')[1]
            secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')

            try:
                # Decodificar el token con la misma clave secreta y algoritmo
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                
                # Adjunta la información del usuario del token a request.current_user
                # Esto hace que los datos del usuario estén disponibles en la función de la ruta
                request.current_user = payload

            except jwt.ExpiredSignatureError:
                current_app.logger.warning("Token expirado.")
                return jsonify({'success': False, 'error': 'Token expirado'}), 401
            except jwt.InvalidTokenError:
                current_app.logger.warning("Token inválido.")
                return jsonify({'success': False, 'error': 'Token inválido'}), 401
            except Exception as e:
                current_app.logger.error(f"Error inesperado en jwt_required: {e}")
                return jsonify({'success': False, 'error': 'Error de autenticación'}), 500
            
            return fn(*args, **kwargs)
        return decorator
    return wrapper

# --- Rutas de autenticación ---

@auth_bp.route('/register', methods=['POST'])
def register():
    """Registra un nuevo usuario en el sistema."""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user') # Rol por defecto 'user'

    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Faltan campos obligatorios (username, email, password)'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'error': 'El nombre de usuario ya existe'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'El email ya está registrado'}), 409

    try:
        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        # Opcional: Podrías generar un token para iniciar sesión automáticamente o
        # solo devolver éxito y dejar que el usuario inicie sesión después.
        # Por ahora, solo confirmamos el registro.
        return jsonify({
            'success': True,
            'message': 'Usuario registrado exitosamente',
            'data': new_user.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error durante el registro: {e}")
        return jsonify({'success': False, 'error': f'Error interno del servidor al registrar usuario: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """Permite a los usuarios iniciar sesión y obtener un token JWT."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'error': 'Faltan credenciales (username, password)'}), 400

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        # Si la contraseña es correcta, genera un JWT
        token = generate_jwt_token(user)
        
        # Actualizar el campo last_login del usuario
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Login exitoso',
            'data': {
                'user': user.to_dict(), # Devuelve un diccionario del usuario (sin password_hash)
                'token': token
            }
        }), 200
    else:
        # Credenciales inválidas
        return jsonify({'success': False, 'error': 'Credenciales inválidas'}), 401

@auth_bp.route('/logout', methods=['POST'])
# @jwt_required() # Opcional: puedes proteger el logout si quieres invalidar tokens en el servidor
def logout():
    """Endpoint para cerrar sesión (principalmente para limpieza en el cliente)."""
    # Para JWTs, el "logout" en el lado del cliente es simplemente borrar el token.
    # Si quisieras invalidar el token en el servidor, necesitarías un mecanismo de lista negra.
    return jsonify({'success': True, 'message': 'Sesión cerrada exitosamente'}), 200


@auth_bp.route('/verify-token', methods=['GET'])
@jwt_required() # Protege esta ruta para que solo los tokens válidos puedan acceder
def verify_token():
    """Verifica la validez del token JWT y devuelve los datos básicos del usuario."""
    # Si el decorador jwt_required pasó, significa que el token es válido y no expiró.
    # request.current_user ya contiene el payload decodificado del token.
    user_id_from_token = request.current_user.get('user_id')

    user = User.query.get(user_id_from_token)
    if user:
        # Devuelve los datos del usuario del token/DB, y el token actual (o uno refrescado si se desea)
        return jsonify({
            'success': True,
            'message': 'Token válido',
            'data': {
                'user': user.to_dict(), # Asegúrate de que to_dict() no exponga datos sensibles
                'token': request.headers.get('Authorization').split(' ')[1] # Devuelve el mismo token que se usó
            }
        }), 200
    else:
        # Esto no debería ocurrir si el token es válido y el usuario existe en DB.
        current_app.logger.error(f"Usuario con ID {user_id_from_token} no encontrado a pesar de token válido.")
        return jsonify({'success': False, 'error': 'Usuario no encontrado para el token'}), 404


@auth_bp.route('/request-password-reset', methods=['POST'])
def request_password_reset():
    """Solicita un enlace para restablecer la contraseña a través de un email."""
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'success': False, 'error': 'Se requiere el email para restablecer la contraseña'}), 400

    user = User.query.filter_by(email=email).first()

    if user:
        # Generar un token de reseteo seguro y de longitud limitada
        reset_token = secrets.token_urlsafe(32) # Genera un token de 32 bytes URL-safe
        user.reset_token = reset_token
        # Token válido por 1 hora, usando timezone.utc para consistencia
        user.reset_token_expiration = datetime.now(timezone.utc) + timedelta(hours=1) 
        db.session.commit()

        # Enviar email con el enlace de reseteo.
        # En un entorno real, usarías un servicio de email (SendGrid, Mailgun, etc.).
        # Por ahora, lo imprimimos en la consola para depuración.
        reset_link = f"http://localhost:5173/reset-password?token={reset_token}" # Ajusta a la URL de tu frontend
        current_app.logger.info(f"DEBUG: Enlace de restablecimiento para {user.email}: {reset_link}")

        # Siempre devuelve un mensaje genérico para evitar la enumeración de usuarios.
        return jsonify({'success': True, 'message': 'Si tu email está registrado, recibirás un enlace de restablecimiento de contraseña.'}), 200
    else:
        # Para evitar la enumeración de usuarios, siempre devuelve un mensaje genérico.
        current_app.logger.warning(f"Intento de restablecimiento de contraseña para email no registrado: {email}")
        return jsonify({'success': True, 'message': 'Si tu email está registrado, recibirás un enlace de restablecimiento de contraseña.'}), 200


@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Restablece la contraseña de un usuario usando un token válido."""
    data = request.get_json()
    token = data.get('token')
    new_password = data.get('new_password')

    if not token or not new_password:
        return jsonify({'success': False, 'error': 'Faltan token o nueva contraseña'}), 400

    user = User.query.filter_by(reset_token=token).first()

    # Verifica si el usuario existe y si el token es válido y no ha expirado
    if user and user.reset_token_expiration and user.reset_token_expiration > datetime.now(timezone.utc):
        user.set_password(new_password)
        user.reset_token = None # Invalida el token después de usarlo
        user.reset_token_expiration = None
        db.session.commit()
        return jsonify({'success': True, 'message': 'Contraseña restablecida exitosamente'}), 200
    else:
        return jsonify({'success': False, 'error': 'Token inválido o expirado. Solicita un nuevo restablecimiento de contraseña.'}), 400


@auth_bp.route('/profile', methods=['GET', 'PUT'])
@jwt_required() # Aplica el decorador para proteger esta ruta
def profile():
    """Obtener o actualizar el perfil del usuario autenticado."""
    # Los datos del usuario ya están en request.current_user gracias a jwt_required
    user_id = request.current_user.get('user_id')
    user = User.query.get(user_id)

    if not user:
        current_app.logger.error(f"Perfil solicitado para usuario con ID {user_id} no encontrado.")
        return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404

    if request.method == 'GET':
        return jsonify({'success': True, 'data': user.to_dict()}), 200
    
    elif request.method == 'PUT':
        try:
            data = request.get_json()
            
            # Permite actualizar ciertos campos del propio perfil
            if 'username' in data and data['username'] != user.username:
                # Opcional: Añadir validación para username duplicado
                if User.query.filter_by(username=data['username']).first():
                    return jsonify({'success': False, 'error': 'El nuevo nombre de usuario ya está en uso.'}), 409
                user.username = data['username']
            
            if 'email' in data and data['email'] != user.email:
                # Opcional: Añadir validación para email duplicado
                if User.query.filter_by(email=data['email']).first():
                    return jsonify({'success': False, 'error': 'El nuevo email ya está en uso.'}), 409
                user.email = data['email']
            
            # Los usuarios no deben poder cambiar su rol directamente desde esta ruta
            # if 'role' in data:
            #     user.role = data['role'] 

            # Actualizar preferencias de notificaciones si existen en el modelo User
            if 'email_notifications' in data and hasattr(user, 'email_notifications'):
                user.email_notifications = bool(data['email_notifications'])
            if 'whatsapp_number' in data and hasattr(user, 'whatsapp_number'):
                user.whatsapp_number = data['whatsapp_number']
            if 'sms_number' in data and hasattr(user, 'sms_number'):
                user.sms_number = data['sms_number']
            
            # Si se proporciona una nueva contraseña, la hashea y la guarda
            if 'password' in data and data['password']:
                # Aquí podrías añadir validación de la contraseña actual si lo deseas
                user.set_password(data['password'])

            db.session.commit()
            return jsonify({'success': True, 'message': 'Perfil actualizado exitosamente', 'data': user.to_dict()}), 200
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error al actualizar perfil del usuario {user_id}: {e}")
            return jsonify({'success': False, 'error': f'Error al actualizar el perfil: {str(e)}'}), 500
