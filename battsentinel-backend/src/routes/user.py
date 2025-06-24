from flask import Blueprint, jsonify, request
# Importa también werkzeug.security para el caso de que la contraseña se pase aquí
from werkzeug.security import generate_password_hash, check_password_hash # Asegúrate de que estén importados

# IMPORTANTE: Confirma que la importación de User y db sea la correcta.
# user_bp.py suele importar el modelo User desde src.models.user
# y la instancia db desde src.models.battery (donde la inicializaste en main.py)
from src.models.user import User # El modelo User que tiene set_password/check_password
from src.models.battery import db # La instancia de SQLAlchemy compartida

# Importa también los decoradores de auth.py si se van a usar aquí
# Asegúrate de que las rutas que usan estos decoradores realmente los importen
# Por ahora, los importo para que el código sea completo, pero si no se usan, puedes quitarlos.
from src.routes.auth import require_token, require_role

user_bp = Blueprint('user', __name__)

@user_bp.route('/users', methods=['GET'])
@require_token # Protege esta ruta, requiere token JWT
@require_role('admin') # Solo administradores pueden ver todos los usuarios
def get_users():
    """Obtener todos los usuarios."""
    try:
        users = User.query.all()
        # Asegúrate de que to_dict() no exponga password_hash
        return jsonify({'success': True, 'data': [user.to_dict() for user in users]}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@user_bp.route('/users', methods=['POST'])
@require_token # Protege esta ruta, puede que solo admins o con token específico
@require_role('admin') # Solo administradores pueden crear usuarios directamente
def create_user():
    """Crear un nuevo usuario."""
    try:
        data = request.json

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Validar campos requeridos para la creación de un usuario
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        # Verificar si el usuario o email ya existen
        existing_user = User.query.filter(
            (User.username == data['username']) | (User.email == data['email'])
        ).first()
        if existing_user:
            return jsonify({'success': False, 'error': 'Username or email already exists'}), 409

        # Crear instancia de usuario
        user = User(
            username=data['username'],
            email=data['email'],
            # Incluye otros campos si son parte de la creación inicial
            role=data.get('role', 'user'), # Asume que 'role' puede venir en la data
            email_notifications=data.get('email_notifications', True),
            whatsapp_number=data.get('whatsapp_number'),
            sms_number=data.get('sms_number'),
            # Otros campos del modelo User, si aplican y se pasan en la creación
            # active=data.get('active', True),
            # activation_token=secrets.token_urlsafe(16) if not data.get('active', True) else None
        )

        # Establecer la contraseña usando el método del modelo User
        user.set_password(data['password'])

        db.session.add(user)
        db.session.commit()

        return jsonify({'success': True, 'data': user.to_dict()}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@user_bp.route('/users/<int:user_id>', methods=['GET'])
@require_token # Protege esta ruta
@require_role('admin') # Solo administradores pueden ver usuarios específicos
def get_user(user_id):
    """Obtener un usuario por su ID."""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        return jsonify({'success': True, 'data': user.to_dict()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@user_bp.route('/users/<int:user_id>', methods=['PUT'])
@require_token # Protege esta ruta
@require_role('admin') # Solo administradores pueden actualizar usuarios
def update_user(user_id):
    """Actualizar un usuario existente."""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided for update'}), 400

        # Actualizar campos
        if 'username' in data:
            user.username = data['username']
        if 'email' in data:
            user.email = data['email']
        # Si se proporciona una nueva contraseña, usa set_password
        if 'password' in data and data['password']:
            user.set_password(data['password']) # ¡Aquí usamos el método set_password!
        
        # Actualiza otros campos si están presentes en la data y en el modelo User
        if 'role' in data:
            user.role = data['role']
        if 'active' in data and hasattr(user, 'active'):
            user.active = data['active']
        if 'email_notifications' in data and hasattr(user, 'email_notifications'):
            user.email_notifications = data['email_notifications']
        if 'whatsapp_number' in data and hasattr(user, 'whatsapp_number'):
            user.whatsapp_number = data['whatsapp_number']
        if 'sms_number' in data and hasattr(user, 'sms_number'):
            user.sms_number = data['sms_number']

        db.session.commit()
        return jsonify({'success': True, 'message': 'User updated successfully', 'data': user.to_dict()}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@user_bp.route('/users/<int:user_id>', methods=['DELETE'])
@require_token # Protege esta ruta
@require_role('admin') # Solo administradores pueden eliminar usuarios
def delete_user(user_id):
    """Eliminar un usuario."""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User deleted successfully'}), 204 # 204 No Content

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
