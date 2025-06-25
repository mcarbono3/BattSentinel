from functools import wraps
from flask import request, jsonify, current_app
import jwt
from src.models.user import User # Asegúrate de que User esté importado correctamente
from datetime import datetime, timedelta, timezone # Importar timezone


# Asumiendo que esta es la estructura de tu decorador require_token
def require_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # *** CAMBIO CLAVE: Permitir que las solicitudes OPTIONS pasen ***
        if request.method == 'OPTIONS':
            # Los navegadores envían OPTIONS como un preflight CORS.
            # No necesitan autenticación y deben recibir un 200 OK.
            return '', 200
        # ************************************************************

        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]

        if not token:
            return jsonify({'success': False, 'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.filter_by(id=data['user_id']).first()
            if not current_user:
                return jsonify({'success': False, 'error': 'User not found'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'error': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Token is invalid!'}), 401
        except Exception as e:
            return jsonify({'success': False, 'error': f'Token error: {str(e)}'}), 401

        request.current_user = current_user # Adjuntar el usuario al objeto request
        return f(*args, **kwargs)
    return decorated_function

# Asumiendo que esta es la estructura de tu decorador require_role
def require_role(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # *** CAMBIO CLAVE: Permitir que las solicitudes OPTIONS pasen ***
            if request.method == 'OPTIONS':
                # Similar a require_token, OPTIONS no necesita validación de rol
                return '', 200
            # ************************************************************

            # Asegúrate de que request.current_user esté disponible (proviene de require_token)
            if not hasattr(request, 'current_user') or not request.current_user:
                # Esto no debería ocurrir si require_token se aplica primero
                return jsonify({'success': False, 'error': 'Authentication required for role check.'}), 401

            # Si el rol del usuario no coincide y no es 'admin', deniega el acceso
            if request.current_user.role != required_role and request.current_user.role != 'admin':
                return jsonify({'success': False, 'error': 'Unauthorized role.'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Decorador para requerir un rol específico
def require_role(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'current_user') or not request.current_user:
                return jsonify({'success': False, 'error': 'Autenticación requerida para verificar rol'}), 401
            
            if request.current_user['role'] != role:
                return jsonify({'success': False, 'error': f'Rol {role} requerido. Acceso denegado.'}), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- FUNCIONES DE AUTENTICACIÓN ---
# Función para generar un token JWT
def generate_jwt_token(user):
    secret_key = current_app.config.get('SECRET_KEY', 'default-fallback-secret-for-jwt-if-not-set')
    jwt_expiration_delta = timedelta(hours=24) # Puedes ajustar esto

    payload = {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role, # Asegúrate de que tu modelo User tenga un campo 'role'
        'exp': datetime.now(timezone.utc) + jwt_expiration_delta
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")

# --- RUTAS DEL BLUEPRINT auth_bp ---

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user') # Por defecto 'user', solo un admin debería poder establecer 'admin'

    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Faltan campos obligatorios'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'error': 'El nombre de usuario ya existe'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'El email ya está registrado'}), 409

    try:
        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        # Generar token para el nuevo usuario registrado
        token = generate_jwt_token(new_user)

        return jsonify({
            'success': True,
            'message': 'Usuario registrado exitosamente',
            'data': {'user': new_user.to_dict(), 'token': token}
        }), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error durante el registro: {e}")
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        if not user.active:
            return jsonify({'success': False, 'error': 'Cuenta de usuario inactiva'}), 403
            
        # Actualizar last_login
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()

        token = generate_jwt_token(user)
        return jsonify({'success': True, 'message': 'Inicio de sesión exitoso', 'data': {'user': user.to_dict(), 'token': token}}), 200
    else:
        return jsonify({'success': False, 'error': 'Credenciales inválidas'}), 401

@auth_bp.route('/logout', methods=['POST'])
@require_token # Requiere token para "logout" efectivo (invalidación en el lado del cliente)
def logout():
    # En un sistema JWT puro, el logout es principalmente del lado del cliente (eliminar token).
    # Aquí podríamos añadir lógica de "lista negra" para tokens si es necesario,
    # pero para una implementación básica, es suficiente con que el cliente elimine el token.
    return jsonify({'success': True, 'message': 'Sesión cerrada exitosamente'}), 200

@auth_bp.route('/verify-token', methods=['GET'])
@require_token # Utiliza el decorador para verificar el token
def verify_token():
    # Si el decorador @require_token no lanzó un error, el token es válido
    # y la información del usuario está en request.current_user
    return jsonify({'success': True, 'message': 'Token válido', 'data': request.current_user}), 200

@auth_bp.route('/password-reset-request', methods=['POST'])
def password_reset_request():
    data = request.get_json()
    email = data.get('email')
    user = User.query.filter_by(email=email).first()

    if user:
        # Generar un token de restablecimiento seguro
        reset_token = secrets.token_urlsafe(32) # Token de 32 bytes (aprox 43 caracteres url-safe)
        reset_token_expiration = datetime.now(timezone.utc) + timedelta(hours=1) # Expira en 1 hora

        user.reset_token = reset_token
        user.reset_token_expiration = reset_token_expiration
        db.session.commit()

        # En una aplicación real, aquí enviarías un email al usuario con el link
        # que contiene el reset_token. Por ejemplo:
        # "Por favor, visita: http://tu-frontend.com/reset-password?token=" + reset_token
        current_app.logger.info(f"Token de restablecimiento para {user.email}: {reset_token}")
        return jsonify({'success': True, 'message': 'Si tu email está registrado, recibirás un enlace para restablecer tu contraseña.'}), 200
    else:
        # Devuelve el mismo mensaje para evitar enumeración de usuarios
        return jsonify({'success': True, 'message': 'Si tu email está registrado, recibirás un enlace para restablecer tu contraseña.'}), 200

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    token = data.get('token')
    new_password = data.get('new_password')

    if not token or not new_password:
        return jsonify({'success': False, 'error': 'Faltan token o nueva contraseña'}), 400

    user = User.query.filter_by(reset_token=token).first()

    if not user or user.reset_token_expiration < datetime.now(timezone.utc):
        return jsonify({'success': False, 'error': 'Token inválido o expirado'}), 400

    try:
        user.set_password(new_password)
        user.reset_token = None # Invalida el token después de usarlo
        user.reset_token_expiration = None
        db.session.commit()
        return jsonify({'success': True, 'message': 'Contraseña restablecida exitosamente.'}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error al restablecer contraseña: {e}")
        return jsonify({'success': False, 'error': f'Error interno del servidor: {str(e)}'}), 500

@auth_bp.route('/profile/<int:user_id>', methods=['PUT'])
@require_token # Protege esta ruta
def update_profile(user_id):
    # Asegúrate de que el usuario solo pueda actualizar su propio perfil a menos que sea admin
    if request.current_user['id'] != user_id and request.current_user['role'] != 'admin':
        return jsonify({'success': False, 'error': 'No autorizado para actualizar este perfil'}), 403

    data = request.get_json()
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404

    try:
        # Solo permite actualizar ciertos campos del propio perfil o si es admin
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        
        # Solo admin puede cambiar roles
        if 'role' in data:
            if request.current_user['role'] == 'admin':
                user.role = data['role']
            else:
                return jsonify({'success': False, 'error': 'No autorizado para cambiar el rol'}), 403

        if 'email_notifications' in data and hasattr(user, 'email_notifications'):
            user.email_notifications = data['email_notifications']
        if 'whatsapp_number' in data and hasattr(user, 'whatsapp_number'):
            user.whatsapp_number = data['whatsapp_number']
        if 'sms_number' in data and hasattr(user, 'sms_number'):
            user.sms_number = data['sms_number']
        
        # Si se proporciona una nueva contraseña, se usa set_password
        if 'password' in data and data['password']:
            user.set_password(data['password'])

        db.session.commit()
        return jsonify({'success': True, 'message': 'Perfil actualizado exitosamente', 'data': user.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error al actualizar perfil de usuario: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
