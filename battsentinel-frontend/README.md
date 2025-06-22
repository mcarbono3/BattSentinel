# BattSentinel Frontend

Interfaz de usuario moderna y profesional para la plataforma de monitoreo inteligente de baterías de ion de litio.

## Características

- **Dashboard Interactivo**: Vista general del estado de las baterías
- **Gemelo Digital**: Simulación interactiva en tiempo real
- **Análisis con IA**: Visualización de resultados de machine learning
- **Gestión de Alertas**: Centro de notificaciones y alertas
- **Diseño Responsivo**: Compatible con desktop y móvil
- **Tema Oscuro/Claro**: Interfaz adaptable
- **Múltiples Roles**: Administrador, técnico, usuario final

## Tecnologías

- **Framework**: React 19.1.0
- **Build Tool**: Vite 6.3.5
- **Styling**: Tailwind CSS 3.4.17
- **UI Components**: shadcn/ui
- **Charts**: Recharts 2.15.0
- **Icons**: Lucide React 0.468.0
- **Routing**: React Router DOM 7.1.1

## Instalación

### Requisitos
- Node.js 20.18.0+
- npm o pnpm

### Pasos

1. **Clonar repositorio**:
```bash
git clone <repository-url>
cd battsentinel-frontend
```

2. **Instalar dependencias**:
```bash
npm install
# o
pnpm install
```

3. **Configurar variables de entorno**:
```bash
cp .env.example .env
# Editar .env con tu configuración
```

4. **Ejecutar servidor de desarrollo**:
```bash
npm run dev
# o
pnpm dev
```

La aplicación estará disponible en `http://localhost:5173`

## Estructura del Proyecto

```
src/
├── App.jsx                 # Componente principal
├── main.jsx                # Punto de entrada
├── components/
│   ├── ui/                 # Componentes UI base
│   │   ├── button.jsx
│   │   ├── card.jsx
│   │   ├── input.jsx
│   │   └── ...
│   └── layout/             # Componentes de layout
│       ├── Header.jsx
│       └── Sidebar.jsx
├── pages/
│   ├── Dashboard.jsx       # Dashboard principal
│   ├── LoginPage.jsx       # Página de login
│   ├── BatteriesPage.jsx   # Gestión de baterías
│   ├── DigitalTwinPage.jsx # Gemelo digital
│   ├── AnalyticsPage.jsx   # Análisis con IA
│   ├── AlertsPage.jsx      # Alertas
│   └── SettingsPage.jsx    # Configuración
├── contexts/
│   ├── AuthContext.jsx     # Contexto de autenticación
│   └── BatteryContext.jsx  # Contexto de baterías
├── lib/
│   ├── api.js              # Cliente API
│   └── utils.js            # Utilidades
└── assets/
    └── BattSentinel_Logo.png
```

## Funcionalidades Principales

### Dashboard
- Métricas clave del sistema
- Estado en tiempo real de las baterías
- Alertas recientes
- Acciones rápidas

### Gemelo Digital
- Simulación interactiva de parámetros
- Gráficos dinámicos en tiempo real
- Control de velocidad de simulación
- Predicciones y métricas de salud

### Análisis con IA
- Visualización de resultados de ML
- Explicaciones de decisiones (XAI)
- Detección de fallas y anomalías
- Recomendaciones automáticas

### Gestión de Baterías
- Lista completa de baterías
- Detalles individuales
- Carga de datos (CSV, TXT, XLSX)
- Análisis de imágenes térmicas

### Sistema de Alertas
- Centro de notificaciones
- Filtros por prioridad y tipo
- Historial de alertas
- Configuración personalizable

## Configuración

### Variables de Entorno

```bash
VITE_API_BASE_URL=http://localhost:5000
VITE_APP_NAME=BattSentinel
VITE_APP_VERSION=1.0.0
VITE_ENVIRONMENT=development
```

### Configuración de Producción

```bash
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_APP_NAME=BattSentinel
VITE_APP_VERSION=1.0.0
VITE_ENVIRONMENT=production
```

## Scripts Disponibles

### Desarrollo
```bash
npm run dev          # Servidor de desarrollo
npm run build        # Build para producción
npm run preview      # Preview del build
npm run lint         # Linting con ESLint
```

### Producción
```bash
npm run build        # Generar build optimizado
npm run serve        # Servir build localmente
```

## Componentes UI

### Componentes Base (shadcn/ui)
- `Button` - Botones con variantes
- `Card` - Tarjetas de contenido
- `Input` - Campos de entrada
- `Badge` - Etiquetas y estados
- `Progress` - Barras de progreso
- `Slider` - Controles deslizantes
- `Tabs` - Pestañas de navegación
- `Toast` - Notificaciones temporales

### Componentes Personalizados
- `LoadingScreen` - Pantalla de carga
- `Header` - Cabecera de la aplicación
- `Sidebar` - Barra lateral de navegación
- `ThemeProvider` - Proveedor de tema

## Gráficos y Visualizaciones

### Recharts
- `LineChart` - Gráficos de líneas
- `AreaChart` - Gráficos de área
- `BarChart` - Gráficos de barras
- `ResponsiveContainer` - Contenedor responsivo

### Métricas Visualizadas
- Voltaje vs Tiempo
- Corriente vs Tiempo
- Temperatura vs Tiempo
- Estado de Carga (SOC)
- Estado de Salud (SOH)
- Resistencia interna

## Autenticación

### Contexto de Autenticación
- Login/Logout
- Gestión de tokens JWT
- Roles de usuario
- Protección de rutas

### Roles Soportados
- **Admin**: Acceso completo
- **Técnico**: Análisis y configuración
- **Usuario**: Solo lectura

## Responsive Design

### Breakpoints
- `sm`: 640px+
- `md`: 768px+
- `lg`: 1024px+
- `xl`: 1280px+
- `2xl`: 1536px+

### Características Móviles
- Navegación adaptable
- Gráficos responsivos
- Touch-friendly controls
- Optimización de rendimiento

## Desarrollo

### Ejecutar en modo desarrollo:
```bash
npm run dev
```

### Hot Reload
Vite proporciona hot reload automático para desarrollo rápido.

### Linting y Formateo:
```bash
npm run lint         # ESLint
npm run lint:fix     # Corregir automáticamente
```

## Build y Despliegue

### Build Local
```bash
npm run build
```

### GitHub Pages
1. Configurar GitHub Actions
2. Push a rama main
3. Deploy automático

### Netlify
1. Conectar repositorio
2. Build command: `npm run build`
3. Publish directory: `dist`

### Vercel
```bash
vercel --prod
```

## Optimización

### Performance
- Code splitting automático
- Lazy loading de componentes
- Optimización de imágenes
- Minificación de assets

### SEO
- Meta tags dinámicos
- Structured data
- Sitemap automático
- Open Graph tags

## Testing

### Unit Tests
```bash
npm run test
```

### E2E Tests
```bash
npm run test:e2e
```

### Coverage
```bash
npm run test:coverage
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
- Proyecto: https://github.com/battsentinel/frontend

## Capturas de Pantalla

### Dashboard Principal
![Dashboard](./screenshots/dashboard.png)

### Gemelo Digital
![Digital Twin](./screenshots/digital-twin.png)

### Análisis con IA
![AI Analysis](./screenshots/ai-analysis.png)

---

*Desarrollado con ❤️ usando React y tecnologías modernas*

