from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import jwt
import os
from src.models.battery import db, User

auth_bp = Blueprint('auth', __name__)

# Clave secreta para JWT (usar variable de entorno en producción)
JWT_SECRET = os.environ.get('JWT_SECRET', 'battsentinel-secret-key-2024')

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
        
        # Crear nuevo usuario
        password_hash = generate_password_hash(data['password'])
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=password_hash,
            role=data.get('role', 'user'),  # Por defecto 'user'
            email_notifications=data.get('email_notifications', True),
            whatsapp_number=data.get('whatsapp_number'),
            sms_number=data.get('sms_number')
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generar token JWT
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
        
        # Verificar contraseña
        if not check_password_hash(user.password_hash, data['password']):
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        # Verificar si el usuario está activo
        if not user.active:
            return jsonify({'success': False, 'error': 'Account is deactivated'}), 401
        
        # Actualizar último login
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
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    """Verificar token JWT"""
    try:
        data = request.get_json()
        
        if not data or 'token' not in data:
            return jsonify({'success': False, 'error': 'Token required'}), 400
        
        # Decodificar token
        try:
            payload = jwt.decode(data['token'], JWT_SECRET, algorithms=['HS256'])
            user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        
        # Buscar usuario
        user = User.query.get(user_id)
        if not user or not user.active:
            return jsonify({'success': False, 'error': 'User not found or inactive'}), 401
        
        return jsonify({
            'success': True,
            'data': {
                'user': user.to_dict(),
                'valid': True
            }
        })
        
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
        try:
            payload = jwt.decode(data['token'], JWT_SECRET, algorithms=['HS256'], options={"verify_exp": False})
            user_id = payload['user_id']
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        
        # Buscar usuario
        user = User.query.get(user_id)
        if not user or not user.active:
            return jsonify({'success': False, 'error': 'User not found or inactive'}), 401
        
        # Generar nuevo token
        new_token = generate_jwt_token(user)
        
        return jsonify({
            'success': True,
            'data': {
                'token': new_token,
                'user': user.to_dict()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/change-password', methods=['POST'])
def change_password():
    """Cambiar contraseña"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validar campos requeridos
        required_fields = ['user_id', 'current_password', 'new_password']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Buscar usuario
        user = User.query.get(data['user_id'])
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Verificar contraseña actual
        if not check_password_hash(user.password_hash, data['current_password']):
            return jsonify({'success': False, 'error': 'Current password is incorrect'}), 401
        
        # Validar nueva contraseña
        if len(data['new_password']) < 6:
            return jsonify({'success': False, 'error': 'New password must be at least 6 characters'}), 400
        
        # Actualizar contraseña
        user.password_hash = generate_password_hash(data['new_password'])
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/profile/<int:user_id>', methods=['GET', 'PUT'])
def user_profile(user_id):
    """Obtener o actualizar perfil de usuario"""
    try:
        user = User.query.get_or_404(user_id)
        
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'data': user.to_dict()
            })
        
        elif request.method == 'PUT':
            data = request.get_json()
            
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            # Campos actualizables
            updatable_fields = ['email', 'role', 'email_notifications', 'whatsapp_number', 'sms_number']
            
            for field in updatable_fields:
                if field in data:
                    setattr(user, field, data[field])
            
            # Verificar email único si se actualiza
            if 'email' in data:
                existing_user = User.query.filter(
                    User.email == data['email'],
                    User.id != user_id
                ).first()
                
                if existing_user:
                    return jsonify({'success': False, 'error': 'Email already exists'}), 409
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': user.to_dict()
            })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/users', methods=['GET'])
def get_users():
    """Obtener lista de usuarios (solo para administradores)"""
    try:
        # En una implementación real, verificar permisos de administrador aquí
        
        # Parámetros de consulta
        role = request.args.get('role')
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        limit = request.args.get('limit', 50, type=int)
        
        query = User.query
        
        if role:
            query = query.filter_by(role=role)
        if active_only:
            query = query.filter_by(active=True)
        
        users = query.limit(limit).all()
        
        return jsonify({
            'success': True,
            'data': [user.to_dict() for user in users]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/users/<int:user_id>/activate', methods=['POST'])
def activate_user(user_id):
    """Activar usuario"""
    try:
        user = User.query.get_or_404(user_id)
        
        user.active = True
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/users/<int:user_id>/deactivate', methods=['POST'])
def deactivate_user(user_id):
    """Desactivar usuario"""
    try:
        user = User.query.get_or_404(user_id)
        
        user.active = False
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Cerrar sesión (invalidar token del lado del cliente)"""
    try:
        # En una implementación real, se podría mantener una lista negra de tokens
        # Por ahora, simplemente confirmamos el logout
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_jwt_token(user):
    """Generar token JWT para usuario"""
    from datetime import datetime, timedelta
    
    payload = {
        'user_id': user.id,
        'username': user.username,
        'role': user.role,
        'exp': datetime.utcnow() + timedelta(hours=24),  # Token válido por 24 horas
        'iat': datetime.utcnow()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token

def verify_jwt_token(token):
    """Verificar token JWT"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorador para requerir autenticación"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'success': False, 'error': 'Token required'}), 401
        
        # Remover 'Bearer ' del token si está presente
        if token.startswith('Bearer '):
            token = token[7:]
        
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

