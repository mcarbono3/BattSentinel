import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { cn, formatNumber, formatPercentage } from '@/lib/utils'

import {
  Battery,
  Zap,
  Thermometer,
  Activity,
  TrendingUp,
  TrendingDown,
  Play,
  Pause,
  RotateCcw,
  Settings,
  Download,
  Upload,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3
} from 'lucide-react'

// Simulación de datos del gemelo digital
const generateBatteryData = (params) => {
  const { voltage, current, temperature, soc, cycles } = params
  const data = []
  
  for (let i = 0; i < 100; i++) {
    const time = i * 0.1 // 0.1 horas
    const voltageVariation = voltage + (Math.sin(i * 0.1) * 0.1) + (Math.random() - 0.5) * 0.05
    const currentVariation = current + (Math.cos(i * 0.15) * 0.2) + (Math.random() - 0.5) * 0.1
    const tempVariation = temperature + (Math.sin(i * 0.05) * 2) + (Math.random() - 0.5) * 1
    const socVariation = Math.max(0, soc - (i * 0.8) + (Math.random() - 0.5) * 2)
    
    data.push({
      time: time.toFixed(1),
      voltage: Math.max(2.5, voltageVariation).toFixed(2),
      current: Math.max(0, currentVariation).toFixed(2),
      temperature: tempVariation.toFixed(1),
      soc: Math.max(0, Math.min(100, socVariation)).toFixed(1),
      power: (voltageVariation * currentVariation).toFixed(2),
      resistance: (voltageVariation / Math.max(0.1, currentVariation)).toFixed(3)
    })
  }
  
  return data
}

export default function DigitalTwinPage() {
  const { id } = useParams()
  const [isSimulating, setIsSimulating] = useState(false)
  const [simulationSpeed, setSimulationSpeed] = useState(1)
  const [currentStep, setCurrentStep] = useState(0)
  const intervalRef = useRef(null)
  
  // Parámetros del gemelo digital
  const [twinParams, setTwinParams] = useState({
    voltage: 3.7,
    current: 2.0,
    temperature: 25,
    soc: 85,
    soh: 92,
    cycles: 245,
    capacity: 3000, // mAh
    resistance: 0.05 // Ohms
  })
  
  // Datos de simulación
  const [simulationData, setSimulationData] = useState([])
  const [predictions, setPredictions] = useState({
    rul: 1250, // Remaining Useful Life in cycles
    degradationRate: 0.02, // % per 100 cycles
    optimalTemp: 23,
    maxCurrent: 3.0,
    efficiency: 94.5
  })

  // Estado del sistema
  const [systemStatus, setSystemStatus] = useState({
    status: 'normal',
    alerts: [],
    lastUpdate: new Date(),
    calibrated: true
  })

  useEffect(() => {
    // Generar datos iniciales
    const initialData = generateBatteryData(twinParams)
    setSimulationData(initialData)
  }, [twinParams])

  useEffect(() => {
    if (isSimulating) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + simulationSpeed
          return next >= simulationData.length ? 0 : next
        })
      }, 100)
    } else {
      clearInterval(intervalRef.current)
    }

    return () => clearInterval(intervalRef.current)
  }, [isSimulating, simulationSpeed, simulationData.length])

  const handleParameterChange = (param, value) => {
    setTwinParams(prev => ({
      ...prev,
      [param]: value
    }))
    
    // Regenerar datos con nuevos parámetros
    const newData = generateBatteryData({
      ...twinParams,
      [param]: value
    })
    setSimulationData(newData)
    setCurrentStep(0)
  }

  const toggleSimulation = () => {
    setIsSimulating(!isSimulating)
  }

  const resetSimulation = () => {
    setIsSimulating(false)
    setCurrentStep(0)
  }

  const getCurrentData = () => {
    if (simulationData.length === 0) return null
    return simulationData[Math.floor(currentStep)] || simulationData[0]
  }

  const getVisibleData = () => {
    const endIndex = Math.floor(currentStep) + 1
    return simulationData.slice(0, Math.max(1, endIndex))
  }

  const currentData = getCurrentData()
  const visibleData = getVisibleData()

  const getStatusColor = (status) => {
    switch (status) {
      case 'normal': return 'text-green-600 bg-green-50 border-green-200'
      case 'warning': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'critical': return 'text-red-600 bg-red-50 border-red-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const calculateHealth = () => {
    if (!currentData) return twinParams.soh
    
    const voltageHealth = Math.max(0, Math.min(100, (parseFloat(currentData.voltage) / 4.2) * 100))
    const tempHealth = Math.max(0, Math.min(100, 100 - Math.abs(parseFloat(currentData.temperature) - 25) * 2))
    const socHealth = Math.max(0, Math.min(100, parseFloat(currentData.soc)))
    
    return ((voltageHealth + tempHealth + socHealth) / 3).toFixed(1)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Gemelo Digital</h1>
          <p className="text-muted-foreground">
            Simulación interactiva de la batería #{id || '001'}
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className={getStatusColor(systemStatus.status)}>
            {systemStatus.status === 'normal' ? 'Sistema Normal' : 
             systemStatus.status === 'warning' ? 'Advertencia' : 'Crítico'}
          </Badge>
          
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Exportar
          </Button>
          
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Calibrar
          </Button>
        </div>
      </div>

      {/* Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Panel de Control de Simulación</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <Button
                onClick={toggleSimulation}
                variant={isSimulating ? "destructive" : "default"}
                size="sm"
              >
                {isSimulating ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Pausar
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Iniciar
                  </>
                )}
              </Button>
              
              <Button onClick={resetSimulation} variant="outline" size="sm">
                <RotateCcw className="h-4 w-4 mr-2" />
                Reiniciar
              </Button>
              
              <div className="flex items-center space-x-2">
                <Label htmlFor="speed">Velocidad:</Label>
                <Slider
                  id="speed"
                  min={0.1}
                  max={5}
                  step={0.1}
                  value={[simulationSpeed]}
                  onValueChange={(value) => setSimulationSpeed(value[0])}
                  className="w-24"
                />
                <span className="text-sm text-muted-foreground">{simulationSpeed}x</span>
              </div>
            </div>
            
            <div className="text-sm text-muted-foreground">
              Paso: {Math.floor(currentStep)} / {simulationData.length}
              {currentData && (
                <span className="ml-4">
                  Tiempo: {currentData.time}h
                </span>
              )}
            </div>
          </div>
          
          <Progress 
            value={(currentStep / simulationData.length) * 100} 
            className="h-2"
          />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Parameters Panel */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Parámetros de Entrada</CardTitle>
              <CardDescription>
                Ajusta los parámetros para simular diferentes condiciones
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="voltage">Voltaje (V)</Label>
                <div className="flex items-center space-x-2 mt-1">
                  <Slider
                    id="voltage"
                    min={2.5}
                    max={4.2}
                    step={0.1}
                    value={[twinParams.voltage]}
                    onValueChange={(value) => handleParameterChange('voltage', value[0])}
                    className="flex-1"
                  />
                  <span className="w-12 text-sm">{twinParams.voltage}V</span>
                </div>
              </div>

              <div>
                <Label htmlFor="current">Corriente (A)</Label>
                <div className="flex items-center space-x-2 mt-1">
                  <Slider
                    id="current"
                    min={0}
                    max={5}
                    step={0.1}
                    value={[twinParams.current]}
                    onValueChange={(value) => handleParameterChange('current', value[0])}
                    className="flex-1"
                  />
                  <span className="w-12 text-sm">{twinParams.current}A</span>
                </div>
              </div>

              <div>
                <Label htmlFor="temperature">Temperatura (°C)</Label>
                <div className="flex items-center space-x-2 mt-1">
                  <Slider
                    id="temperature"
                    min={-10}
                    max={60}
                    step={1}
                    value={[twinParams.temperature]}
                    onValueChange={(value) => handleParameterChange('temperature', value[0])}
                    className="flex-1"
                  />
                  <span className="w-12 text-sm">{twinParams.temperature}°C</span>
                </div>
              </div>

              <div>
                <Label htmlFor="soc">SOC (%)</Label>
                <div className="flex items-center space-x-2 mt-1">
                  <Slider
                    id="soc"
                    min={0}
                    max={100}
                    step={1}
                    value={[twinParams.soc]}
                    onValueChange={(value) => handleParameterChange('soc', value[0])}
                    className="flex-1"
                  />
                  <span className="w-12 text-sm">{twinParams.soc}%</span>
                </div>
              </div>

              <Separator />

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <Label>Capacidad</Label>
                  <p className="font-medium">{twinParams.capacity} mAh</p>
                </div>
                <div>
                  <Label>Ciclos</Label>
                  <p className="font-medium">{twinParams.cycles}</p>
                </div>
                <div>
                  <Label>Resistencia</Label>
                  <p className="font-medium">{twinParams.resistance} Ω</p>
                </div>
                <div>
                  <Label>SOH</Label>
                  <p className="font-medium">{twinParams.soh}%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Predictions */}
          <Card>
            <CardHeader>
              <CardTitle>Predicciones</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Vida Útil Restante</span>
                <span className="font-medium">{predictions.rul} ciclos</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Tasa de Degradación</span>
                <span className="font-medium">{predictions.degradationRate}%/100 ciclos</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Temperatura Óptima</span>
                <span className="font-medium">{predictions.optimalTemp}°C</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Corriente Máxima</span>
                <span className="font-medium">{predictions.maxCurrent}A</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm">Eficiencia</span>
                <span className="font-medium">{predictions.efficiency}%</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Display */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Current Status */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <Zap className="h-8 w-8 text-blue-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-foreground">
                  {currentData ? currentData.voltage : twinParams.voltage}V
                </p>
                <p className="text-sm text-muted-foreground">Voltaje</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4 text-center">
                <Activity className="h-8 w-8 text-green-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-foreground">
                  {currentData ? currentData.current : twinParams.current}A
                </p>
                <p className="text-sm text-muted-foreground">Corriente</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4 text-center">
                <Thermometer className="h-8 w-8 text-orange-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-foreground">
                  {currentData ? currentData.temperature : twinParams.temperature}°C
                </p>
                <p className="text-sm text-muted-foreground">Temperatura</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4 text-center">
                <Battery className="h-8 w-8 text-purple-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-foreground">
                  {currentData ? currentData.soc : twinParams.soc}%
                </p>
                <p className="text-sm text-muted-foreground">SOC</p>
              </CardContent>
            </Card>
          </div>

          {/* Charts */}
          <Tabs defaultValue="voltage" className="space-y-4">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="voltage">Voltaje</TabsTrigger>
              <TabsTrigger value="current">Corriente</TabsTrigger>
              <TabsTrigger value="temperature">Temperatura</TabsTrigger>
              <TabsTrigger value="soc">SOC</TabsTrigger>
            </TabsList>

            <TabsContent value="voltage">
              <Card>
                <CardHeader>
                  <CardTitle>Voltaje vs Tiempo</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={visibleData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={['dataMin - 0.1', 'dataMax + 0.1']} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="voltage" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="current">
              <Card>
                <CardHeader>
                  <CardTitle>Corriente vs Tiempo</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={visibleData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="current" 
                        stroke="#10b981" 
                        fill="#10b981" 
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="temperature">
              <Card>
                <CardHeader>
                  <CardTitle>Temperatura vs Tiempo</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={visibleData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="temperature" 
                        stroke="#f59e0b" 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="soc">
              <Card>
                <CardHeader>
                  <CardTitle>Estado de Carga vs Tiempo</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={visibleData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="soc" 
                        stroke="#8b5cf6" 
                        fill="#8b5cf6" 
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Health Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Métricas de Salud</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Salud General</span>
                    <span className="text-sm text-muted-foreground">
                      {calculateHealth()}%
                    </span>
                  </div>
                  <Progress value={parseFloat(calculateHealth())} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Eficiencia</span>
                    <span className="text-sm text-muted-foreground">
                      {predictions.efficiency}%
                    </span>
                  </div>
                  <Progress value={predictions.efficiency} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Vida Útil</span>
                    <span className="text-sm text-muted-foreground">
                      {Math.round((predictions.rul / 2000) * 100)}%
                    </span>
                  </div>
                  <Progress value={(predictions.rul / 2000) * 100} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

