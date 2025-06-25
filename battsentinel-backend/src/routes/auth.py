from functools import wraps
from flask import request, jsonify, current_app, Blueprint # ¡Asegúrate de importar Blueprint aquí!
import jwt
from src.models.user import User
from datetime import datetime, timedelta, timezone


# Define el Blueprint 'auth_bp' aquí, antes de usarlo
auth_bp = Blueprint('auth', __name__)


# Asumiendo que esta es la estructura de tu decorador require_token
def require_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200

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

        request.current_user = current_user
        return f(*args, **kwargs)
    return decorated_function

# Asumiendo que esta es la estructura de tu decorador require_role
def require_role(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method == 'OPTIONS':
                return '', 200

            if not hasattr(request, 'current_user') or not request.current_user:
                return jsonify({'success': False, 'error': 'Authentication required for role check.'}), 401

            if request.current_user.role != required_role and request.current_user.role != 'admin':
                return jsonify({'success': False, 'error': 'Unauthorized role.'}), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator


# --- A PARTIR DE AQUÍ IRÍAN LAS RUTAS DE AUTENTICACIÓN QUE USAN auth_bp ---

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'user') # Default role is 'user'

        if not username or not email or not password:
            return jsonify({'success': False, 'error': 'Username, email, and password are required'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already exists'}), 409

        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password) # set_password maneja el hashing
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'success': True, 'message': 'User registered successfully'}), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error registering user: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        # Generar token JWT
        token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'exp': datetime.now(timezone.utc) + timedelta(hours=24) # Token válido por 24 horas
        }, current_app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            'success': True,
            'message': 'Logged in successfully',
            'token': token,
            'user': user.to_dict() # Devuelve los datos del usuario (sin password)
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error during login: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@auth_bp.route('/verify-token', methods=['GET'])
@require_token
def verify_token():
    """Endpoint para verificar la validez del token y obtener los datos del usuario."""
    # Si la ejecución llega aquí, require_token ya ha validado el token.
    # El usuario actual está disponible en request.current_user
    user_data = {
        'id': request.current_user.id,
        'username': request.current_user.username,
        'email': request.current_user.email,
        'role': request.current_user.role
        # No incluir la contraseña u otra información sensible
    }
    return jsonify({'success': True, 'message': 'Token is valid', 'user': user_data}), 200

# Endpoint para obtener el perfil del usuario autenticado
@auth_bp.route('/profile', methods=['GET'])
@require_token
def get_user_profile():
    try:
        user = request.current_user
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        return jsonify({'success': True, 'data': user.to_dict()}), 200
    except Exception as e:
        current_app.logger.error(f"Error fetching user profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint para actualizar el perfil del usuario autenticado
@auth_bp.route('/profile', methods=['PUT'])
@require_token
def update_user_profile():
    try:
        user_id = request.current_user.id
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided for update'}), 400

        # Solo permite actualizar ciertos campos del propio perfil
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        
        # Los usuarios no deben poder cambiar su rol directamente aquí
        # if 'role' in data: # Solo admin debería cambiar roles
        #     if request.current_user.role == 'admin': # Corrected access
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
