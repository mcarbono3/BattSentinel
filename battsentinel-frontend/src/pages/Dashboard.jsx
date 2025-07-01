// src/pages/Dashboard.jsx
import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom' // Añadido Link
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useBattery } from '@/contexts/BatteryContext'
import { useAuth } from '@/contexts/AuthContext'
import { cn, formatNumber, formatPercentage, getBatteryStatusColor, getRelativeTime } from '@/lib/utils'

import {
  Battery,
  Zap,
  Activity,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Clock,
  Thermometer,
  BarChart3,
  Plus,
  Eye,
  Settings,
  RefreshCw,
  Home,
  PlusCircle // Añadido PlusCircle
} from 'lucide-react'

// Mock data (se mantienen como fallback si no hay baterías reales o para props no disponibles)
const mockBatteries = [
  {
    id: 1,
    name: 'Batería Principal #001',
    type: 'Li-ion 18650',
    voltage: 3.7,
    current: 2.1,
    temperature: 25.3,
    soc: 85,
    soh: 92,
    cycles: 245,
    status: 'good',
    last_update: new Date(Date.now() - 5 * 60 * 1000),
    alerts: [],
    // Añadimos Model y serial_number a los mocks por consistencia
    model: 'BS-100',
    serial_number: 'BS001MOCK'
  },
  {
    id: 2,
    name: 'Batería Secundaria #002',
    type: 'Li-ion 21700',
    voltage: 3.6,
    current: 1.8,
    temperature: 28.7,
    soc: 67,
    soh: 88,
    cycles: 312,
    status: 'fair',
    last_update: new Date(Date.now() - 12 * 60 * 1000),
    alerts: [{ severity: 'medium', message: 'Temperatura elevada' }],
    model: 'BS-200',
    serial_number: 'BS002MOCK'
  },
  {
    id: 3,
    name: 'Batería de Respaldo #003',
    type: 'Li-ion 26650',
    voltage: 3.2,
    current: 0.5,
    temperature: 32.1,
    soc: 23,
    soh: 76,
    cycles: 567,
    status: 'poor',
    last_update: new Date(Date.now() - 3 * 60 * 1000),
    alerts: [
      { severity: 'high', message: 'SOC bajo' },
      { severity: 'medium', message: 'Degradación acelerada' }
    ],
    model: 'BS-300',
    serial_number: 'BS003MOCK'
  }
]

const mockSystemStats = {
  totalBatteries: 3,
  activeBatteries: 3,
  criticalAlerts: 1,
  averageHealth: 85.3,
  totalCycles: 1124,
  energyConsumed: 2847.5,
  uptime: '99.8%'
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { user } = useAuth()
  
  const { 
    // Añadimos getVisibleBatteries para el filtrado en el dashboard
    getVisibleBatteries, 
    // Aseguramos que 'error' sea desestructurado, ya que se usa más abajo
    batteries, loading, loadBatteries, error 
  } = useBattery()
  
  const [refreshing, setRefreshing] = useState(false)
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')

  // NUEVO ESTADO: `displayBatteries` contendrá solo las baterías visibles
  const [displayBatteries, setDisplayBatteries] = useState([]);
  
  // Calcular estadísticas del sistema basadas en las baterías REALES (no filtradas)
  const stats = {
    totalBatteries: batteries.length, // Usar 'batteries' directamente del contexto
    activeBatteries: batteries.filter(b => b.status === 'active').length, // Usar 'batteries'
    criticalAlerts: batteries.reduce((acc, battery) => { // Usar 'batteries'
      // Usar la propiedad 'alerts' si existe y tiene contenido.
      // Recuerda que el backend actualmente NO envía 'alerts' en la batería por defecto.
      return acc + (battery.alerts ? battery.alerts.filter(alert => alert.severity === 'high').length : 0);
    }, 0),
    // Esto es un placeholder. Para un valor real de averageHealth, necesitarías el SOH de cada batería del backend.
    // Si la batería no tiene 'soh', se usará 0 en el promedio, o el valor mock si no hay baterías.
    averageHealth: batteries.length > 0 // Usar 'batteries'
                     ? (batteries.reduce((sum, b) => sum + (b.soh || 0), 0) / batteries.length) // Si el SOH viene del backend, úsalo
                     : mockSystemStats.averageHealth, // Si no hay baterías o SOH, usa el mock
    
    // Estos valores (totalCycles, energyConsumed, uptime)
    // normalmente vendrían de un endpoint de estadísticas global del backend.
    // Por ahora, se mantendrán los valores mock o placeholders hasta que el backend los provea.
    totalCycles: mockSystemStats.totalCycles, 
    energyConsumed: mockSystemStats.energyConsumed,
    uptime: mockSystemStats.uptime 
  };

  // useEffect existente para cargar todas las baterías al montar el componente.
  useEffect(() => {
    // La lógica para cargar baterías ya está en BatteryContext y se ejecuta al autenticarse.
    // Esta parte asegura que si de alguna manera no se cargaron, se intenten cargar.
    if (batteries.length === 0) { // Opcional: podrías remover esta condición si confías en el useEffect de BatteryContext.
      loadBatteries()
    }
  }, [batteries.length, loadBatteries])

  // NUEVO useEffect: Se encarga de actualizar la lista de baterías a mostrar (displayBatteries)
  // cada vez que la lista completa de baterías o el estado de ocultamiento cambie en BatteryContext.
  useEffect(() => {
    // getVisibleBatteries() ya devuelve la lista filtrada de las baterías NO ocultas.
    setDisplayBatteries(getVisibleBatteries());
  }, [getVisibleBatteries]); // Dependencia de useCallback, es estable.

  const handleRefresh = async () => {
    setRefreshing(true)
    await loadBatteries()
    setTimeout(() => setRefreshing(false), 1000)
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'excellent':
      case 'good':
      case 'active': 
        return <Activity className="h-4 w-4 text-green-500" />
      case 'fair':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'poor':
      case 'critical':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      default:
        return <Battery className="h-4 w-4 text-gray-500" />
    }
  }

  const getTrendIcon = (value, threshold = 0) => {
    if (value > threshold) {
      return <TrendingUp className="h-4 w-4 text-green-500" />
    } else {
      return <TrendingDown className="h-4 w-4 text-red-500" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Header del Dashboard */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">
            Bienvenido, {user?.username}. Aquí tienes un resumen del estado de tus baterías.
          </p>
        </div>
        
        {/* Contenedor de botones de acción */}
        <div className="flex items-center space-x-2">
          {/* Botón de Regresar al Inicio */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate('/')} 
          >
            <Home className="h-4 w-4 mr-2" />
            Regresar al Inicio
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")} />
            Actualizar
          </Button>
          
          <Button
            onClick={() => navigate('/batteries')}
            size="sm"
          >
            <Plus className="h-4 w-4 mr-2" />
            Nueva Batería
          </Button>
        </div>
      </div>

      {/* System Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total de Baterías</CardTitle>
            <Battery className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {/* stats ya se calcula con los datos reales */}
            <div className="text-2xl font-bold">{stats.totalBatteries}</div>
            <p className="text-xs text-muted-foreground">
              {stats.activeBatteries} activas
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Salud Promedio</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {/* stats ya se calcula con los datos reales */}
            <div className="text-2xl font-bold">{formatPercentage(stats.averageHealth)}</div>
            <p className="text-xs text-muted-foreground">
              {getTrendIcon(stats.averageHealth - 80)} vs. mes anterior
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Alertas Críticas</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {/* stats ya se calcula con los datos reales */}
            <div className="text-2xl font-bold text-red-600">{stats.criticalAlerts}</div>
            <p className="text-xs text-muted-foreground">
              Requieren atención inmediata
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tiempo Activo</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {/* stats ya se calcula con los datos reales */}
            <div className="text-2xl font-bold text-green-600">{stats.uptime}</div>
            <p className="text-xs text-muted-foreground">
              Últimos 30 días
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Battery Status List */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Estado de las Baterías</CardTitle>
                  <CardDescription>
                    Monitoreo en tiempo real de todas las baterías del sistema
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate('/batteries')}
                >
                  Ver Todas
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Ahora usamos displayBatteries que contiene solo las baterías visibles */}
                {displayBatteries.length === 0 ? (
                    <div className="text-center py-4">
                        <Battery className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                        <p className="text-sm text-muted-foreground">
                          No hay baterías visibles en este momento. Si has ocultado alguna, puedes gestionarlas desde la página de Baterías.
                        </p>
                        <Button onClick={() => navigate('/batteries')} className="mt-4">Agregar Batería</Button>
                    </div>
                ) : (
                  displayBatteries.map((battery) => (
                    <div
                      key={battery.id}
                      className="flex flex-col sm:flex-row sm:items-center justify-between p-4 border border-border rounded-lg hover:bg-accent/50 transition-colors cursor-pointer gap-4"
                      onClick={() => navigate(`/batteries/${battery.id}`)}
                    >
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(battery.status)}
                          <div>
                            {/* Mostrar Nombre, Modelo y Número de Serie */}
                            <h4 className="font-medium text-foreground">{battery.name}</h4> {/* Usa 'Name' del backend */}
                            <p className="text-sm text-muted-foreground">
                              {battery.type} {battery.model && `(${battery.model})`}
                            </p>
                            {battery.serial_number && (
                              <p className="text-xs text-muted-foreground">SN: {battery.serial_number}</p>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Responsive metrics grid */}
                      <div className="grid grid-cols-3 sm:flex sm:items-center gap-4 sm:gap-6">
                        <div className="text-center">
                          <p className="text-xs sm:text-sm font-medium">SOC</p>
                          {/* Mostrar battery.soc si existe, sino "--" */}
                          <p className="text-base sm:text-lg font-bold text-blue-600">
                            {battery.soc ? formatPercentage(battery.soc, 0) : '--'}
                          </p>
                        </div>

                        <div className="text-center">
                          <p className="text-xs sm:text-sm font-medium">SOH</p>
                          {/* Mostrar battery.soh si existe, sino "--" */}
                          <p className="text-base sm:text-lg font-bold text-green-600">
                            {battery.soh ? formatPercentage(battery.soh, 0) : '--'}
                          </p>
                        </div>

                        <div className="text-center">
                          <p className="text-xs sm:text-sm font-medium">Temp</p>
                          {/* Mostrar battery.temperature si existe, sino "--°C" */}
                          <p className="text-base sm:text-lg font-bold text-orange-600">
                            {battery.temperature ? `${formatNumber(battery.temperature, 1)}°C` : '--°C'}
                          </p>
                        </div>

                        <div className="col-span-3 sm:col-span-1 flex items-center justify-center sm:justify-start space-x-2">
                          {battery.alerts && battery.alerts.length > 0 && (
                            <Badge variant="destructive">
                              {battery.alerts.length}
                            </Badge>
                          )}
                          <Badge variant="outline" className={getBatteryStatusColor(battery.status)}>
                            {battery.status || 'unknown'}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions & Alerts */}
        <div className="space-y-6">
          
          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Acciones Rápidas</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => navigate('/batteries')}
              >
                <Plus className="h-4 w-4 mr-2" />
                Agregar Nueva Batería
              </Button>
              
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => navigate('/analytics')}
              >
                <BarChart3 className="h-4 w-4 mr-2" />
                Análisis con IA
              </Button>
              
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => navigate('/digital-twin/1')}
              >
                <Zap className="h-4 w-4 mr-2" />
                Gemelo Digital
              </Button>
              
              <Button
                variant="outline"
                className="w-full justify-start"
                onClick={() => navigate('/settings')}
              >
                <Settings className="h-4 w-4 mr-2" />
                Configuración
              </Button>
            </CardContent>
          </Card>

          {/* Recent Alerts */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Alertas Recientes</CardTitle>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate('/alerts')}
                >
                  Ver Todas
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {displayBatteries // Usar displayBatteries para alertas si solo quieres alertas de las visibles
                  .filter(battery => battery.alerts && battery.alerts.length > 0)
                  .slice(0, 3)
                  .map((battery) => (
                    // Asegúrate de que la batería tenga una propiedad 'alerts' y que sea un array para poder mapear.
                    // Esto maneja el caso de que 'alerts' sea undefined o null
                    battery.alerts && Array.isArray(battery.alerts) && battery.alerts.map((alert, index) => (
                      <div
                        key={`${battery.id}-${index}`}
                        className="flex items-start space-x-3 p-3 border border-border rounded-lg"
                      >
                        <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground">
                            {battery.Name || battery.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {alert.message}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {getRelativeTime(battery.last_update)}
                          </p>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {alert.severity}
                        </Badge>
                      </div>
                    ))
                  ))}
                
                {/* Mostrar mensaje si no hay alertas */}
                {displayBatteries.every(battery => !battery.alerts || battery.alerts.length === 0) && (
                  <div className="text-center py-4">
                    <Activity className="h-8 w-8 text-green-500 mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">
                      No hay alertas activas
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* System Health */}
          <Card>
            <CardHeader>
              <CardTitle>Salud del Sistema</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Salud General</span>
                  <span className="text-sm text-muted-foreground">
                    {formatPercentage(stats.averageHealth)}
                  </span>
                </div>
                <Progress value={stats.averageHealth} className="h-2" />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Eficiencia Energética</span>
                  <span className="text-sm text-muted-foreground">94%</span>
                </div>
                <Progress value={94} className="h-2" />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Disponibilidad</span>
                  <span className="text-sm text-muted-foreground">99.8%</span>
                </div>
                <Progress value={99.8} className="h-2" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Métricas de Rendimiento</CardTitle>
          <CardDescription>
            Estadísticas del sistema en las últimas 24 horas
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-center p-4 border border-border rounded-lg">
              <Zap className="h-8 w-8 text-blue-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-foreground">{formatNumber(stats.energyConsumed, 1)}</p>
              <p className="text-sm text-muted-foreground">kWh Consumidos</p>
            </div>

            <div className="text-center p-4 border border-border rounded-lg">
              <Activity className="h-8 w-8 text-green-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-foreground">{stats.totalCycles}</p>
              <p className="text-sm text-muted-foreground">Ciclos Totales</p>
            </div>

            <div className="text-center p-4 border border-border rounded-lg">
              <Thermometer className="h-8 w-8 text-orange-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-foreground">26.8°C</p>
              <p className="text-sm text-muted-foreground">Temp. Promedio</p>
            </div>

            <div className="text-center p-4 border border-border rounded-lg">
              <Clock className="h-8 w-8 text-purple-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-foreground">2.3h</p>
              <p className="text-sm text-muted-foreground">Tiempo Restante</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
