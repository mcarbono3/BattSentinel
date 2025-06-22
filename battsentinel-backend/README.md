# BattSentinel Backend

Sistema backend para la plataforma de monitoreo inteligente de baterías de ion de litio.

## Características

- **API REST**: Endpoints completos para gestión de baterías y análisis
- **Inteligencia Artificial**: Modelos de ML/DL para detección de fallas
- **Explicabilidad**: Implementación de SHAP y LIME para XAI
- **Procesamiento de Datos**: Soporte para CSV, TXT, XLSX e imágenes térmicas
- **Gemelo Digital**: Simulación en tiempo real de comportamiento de baterías
- **Sistema de Alertas**: Notificaciones automáticas y preventivas

## Tecnologías

- **Framework**: Flask 3.1.0
- **IA/ML**: scikit-learn, TensorFlow, SHAP, LIME
- **Procesamiento**: pandas, numpy, OpenCV
- **Base de Datos**: SQLAlchemy ORM
- **CORS**: Flask-CORS

## Instalación

### Requisitos
- Python 3.11+
- pip

### Pasos

1. **Clonar repositorio**:
```bash
git clone <repository-url>
cd battsentinel-backend
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**:
```bash
cp .env.example .env
# Editar .env con tu configuración
```

5. **Inicializar base de datos**:
```bash
python init_db.py
```

6. **Ejecutar servidor**:
```bash
python src/main.py
```

El servidor estará disponible en `http://localhost:5000`

## Estructura del Proyecto

```
src/
├── main.py                 # Punto de entrada principal
├── models/
│   ├── battery.py          # Modelo de batería
│   └── user.py             # Modelo de usuario
├── routes/
│   ├── auth.py             # Autenticación
│   ├── battery.py          # Gestión de baterías
│   ├── ai_analysis.py      # Análisis con IA
│   ├── digital_twin.py     # Gemelo digital
│   └── notifications.py    # Notificaciones
└── services/
    ├── ai_models.py        # Modelos de IA
    ├── data_processor.py   # Procesamiento de datos
    ├── digital_twin.py     # Simulador
    └── thermal_analyzer.py # Análisis térmico
```

## API Endpoints

### Autenticación
- `POST /api/auth/login` - Iniciar sesión
- `POST /api/auth/register` - Registrar usuario
- `POST /api/auth/logout` - Cerrar sesión

### Baterías
- `GET /api/batteries` - Listar baterías
- `POST /api/batteries` - Crear batería
- `GET /api/batteries/{id}` - Obtener batería
- `PUT /api/batteries/{id}` - Actualizar batería
- `DELETE /api/batteries/{id}` - Eliminar batería

### Datos
- `POST /api/batteries/{id}/data` - Cargar datos
- `GET /api/batteries/{id}/data` - Obtener datos históricos

### Análisis IA
- `POST /api/ai/analyze/{battery_id}` - Ejecutar análisis
- `GET /api/ai/explain/{analysis_id}` - Obtener explicaciones

### Gemelo Digital
- `POST /api/digital-twin/{battery_id}/simulate` - Iniciar simulación
- `GET /api/digital-twin/{battery_id}/status` - Estado de simulación

### Alertas
- `GET /api/alerts` - Listar alertas
- `POST /api/alerts` - Crear alerta
- `PUT /api/alerts/{id}/read` - Marcar como leída

## Configuración

### Variables de Entorno

```bash
FLASK_ENV=development
DATABASE_URL=sqlite:///battsentinel.db
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:5173
DEBUG=True
```

### Configuración de Producción

```bash
FLASK_ENV=production
DATABASE_URL=postgresql://user:password@localhost/battsentinel
SECRET_KEY=your-super-secret-key
CORS_ORIGINS=https://yourdomain.com
```

## Modelos de IA

### Detección de Fallas
- **Random Forest**: Clasificación de estados
- **SVM**: Detección de anomalías
- **Neural Networks**: Predicción de vida útil
- **Isolation Forest**: Detección de outliers

### Explicabilidad
- **SHAP**: Contribución de características
- **LIME**: Explicaciones locales

## Desarrollo

### Ejecutar en modo desarrollo:
```bash
export FLASK_ENV=development
python src/main.py
```

### Ejecutar tests:
```bash
python -m pytest tests/
```

### Linting:
```bash
flake8 src/
black src/
```

## Despliegue

### Render
1. Conectar repositorio
2. Configurar variables de entorno
3. Comando de inicio: `python src/main.py`

### Heroku
```bash
heroku create battsentinel-backend
git push heroku main
```

### Docker
```bash
docker build -t battsentinel-backend .
docker run -p 5000:5000 battsentinel-backend
```

## Contribuir

1. Fork el proyecto
2. Crear rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.

## Contacto

- Email: support@battsentinel.com
- Proyecto: https://github.com/battsentinel/backend

