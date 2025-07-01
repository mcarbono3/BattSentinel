// src/pages/BatteryDetailPage.jsx
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
// Se han añadido Bolt, Gauge, SlidersHorizontal, Power, Calendar, EyeOff, Trash2, Eye
import { Activity, Battery, Thermometer, Zap, Clock, TrendingUp, AlertTriangle, ChevronLeft, Wrench, BarChart2, Bolt, Gauge, SlidersHorizontal, Power, Calendar, EyeOff, Trash2, Eye } from 'lucide-react';
import { useBattery } from '@/contexts/BatteryContext';
import { cn, formatNumber, formatPercentage, getBatteryStatusColor, formatDate, formatDateTime } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
// Importar componentes de AlertDialog
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useToast } from '@/hooks/use-toast'; // Asumiendo que tienes un componente de Toast para notificaciones

export default function BatteryDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast(); // Inicializar toast

  const {
    getBatteryById,
    getAlertsByBatteryId,
    getAnalysisResultsByBatteryId,
    getMaintenanceRecordsByBatteryId,
    getBatteryHistoricalData,
    deleteBattery, // Añadido
    toggleBatteryVisibility, // Añadido
    hiddenBatteryIds, // Añadido
    loading,
    error
  } = useBattery();

  const [battery, setBattery] = useState(null);

  const [alerts, setAlerts] = useState([]);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [maintenanceRecords, setMaintenanceRecords] = useState([]);
  const [historicalData, setHistoricalData] = useState(null);

  const [loadingAdditionalData, setLoadingAdditionalData] = useState(false);
  const [additionalDataError, setAdditionalDataError] = useState(null);

  const isBatteryHidden = battery ? hiddenBatteryIds.has(battery.id) : false;

  useEffect(() => {
    const fetchBatteryDetails = async () => {
      if (id) {
        const result = await getBatteryById(parseInt(id));
        if (result && result.success) {
          setBattery(result.data);
        } else {
          console.error("Error fetching battery details:", result ? result.error : "Unknown error");
          setBattery(null);
        }
      }
    };
    fetchBatteryDetails();
  }, [id, getBatteryById]);

  useEffect(() => {
    const fetchAdditionalData = async () => {
      if (battery?.id) {
        setLoadingAdditionalData(true);
        setAdditionalDataError(null);

        const batteryId = battery.id;

        // Cargar Alertas
        const alertsResult = await getAlertsByBatteryId(batteryId);
        if (alertsResult.success) {
          setAlerts(alertsResult.data);
        } else {
          console.error("Error fetching alerts:", alertsResult.error);
          setAdditionalDataError(prev => ({ ...prev, alerts: alertsResult.error || 'Error al cargar alertas' }));
        }

        // Cargar Resultados de Análisis
        const analysisResult = await getAnalysisResultsByBatteryId(batteryId);
        if (analysisResult.success) {
          setAnalysisResults(analysisResult.data);
        } else {
          console.error("Error fetching analysis results:", analysisResult.error);
          setAdditionalDataError(prev => ({ ...prev, analysis: analysisResult.error || 'Error al cargar resultados de análisis' }));
        }

        // Cargar Registros de Mantenimiento
        const maintenanceResult = await getMaintenanceRecordsByBatteryId(batteryId);
        if (maintenanceResult.success) {
          setMaintenanceRecords(maintenanceResult.data);
        } else {
          console.error("Error fetching maintenance records:", maintenanceResult.error);
          setAdditionalDataError(prev => ({ ...prev, maintenance: maintenanceResult.error || 'Error al cargar registros de mantenimiento' }));
        }

        // Cargar Datos Históricos (para gráficos)
        const historicalDataResult = await getBatteryHistoricalData(batteryId, { time_range: 'last_30_days', interval: 'daily' });
        if (historicalDataResult.success) {
          setHistoricalData(historicalDataResult.data);
        } else {
          console.error("Error fetching historical data:", historicalDataResult.error);
          setAdditionalDataError(prev => ({ ...prev, historical: historicalDataResult.error || 'Error al cargar datos históricos' }));
        }

        setLoadingAdditionalData(false);
      }
    };

    fetchAdditionalData();
  }, [battery?.id, getAlertsByBatteryId, getAnalysisResultsByBatteryId, getMaintenanceRecordsByBatteryId, getBatteryHistoricalData]);

  // Funciones para eliminar y ocultar
  const handleDeleteBattery = async () => {
    if (battery?.id) {
      const result = await deleteBattery(battery.id);
      if (result.success) {
        toast({
          title: "Batería Eliminada",
          description: `La batería "${battery.name}" ha sido eliminada exitosamente.`,
        });
        navigate('/batteries'); // Redirigir a la lista de baterías
      } else {
        toast({
          title: "Error al Eliminar",
          description: result.error || "No se pudo eliminar la batería.",
          variant: "destructive",
        });
      }
    }
  };

  const handleToggleVisibility = async () => {
    if (battery?.id) {
      const result = await toggleBatteryVisibility(battery.id);
      if (result.success) {
        toast({
          title: isBatteryHidden ? "Batería Visible" : "Batería Oculta",
          description: isBatteryHidden ? `La batería "${battery.name}" ahora es visible.` : `La batería "${battery.name}" ha sido oculta.`,
        });
      } else {
        toast({
          title: "Error al cambiar visibilidad",
          description: result.error || "No se pudo cambiar la visibilidad de la batería.",
          variant: "destructive",
        });
      }
    }
  };


  // Derivar la última o única medición de historicalData para las métricas clave
  const latestBatteryMetrics = Array.isArray(historicalData) && historicalData.length > 0
    ? historicalData[historicalData.length - 1]
    : (historicalData && typeof historicalData === 'object' ? historicalData : null);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p>Cargando detalles de la batería...</p>
      </div>
    );
  }

  if (error || !battery) {
    return (
      <div className="space-y-6">
        <Button variant="outline" onClick={() => navigate('/batteries')} className="mb-4">
          <ChevronLeft className="h-4 w-4 mr-2" /> Volver a Baterías
        </Button>
        <Card>
          <CardHeader>
            <CardTitle>Error al cargar la batería</CardTitle>
          </CardHeader>
          <CardContent className="text-center py-12">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-muted-foreground">
              {error || "No se pudo encontrar la batería o hubo un error inesperado."}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'excellent':
      case 'good':
      case 'active':
        return <Activity className="h-6 w-6 text-green-500" />;
      case 'fair':
        return <Clock className="h-6 w-6 text-yellow-500" />;
      case 'poor':
      case 'critical':
        return <AlertTriangle className="h-6 w-6 text-red-500" />;
      default:
        return <Battery className="h-6 w-6 text-gray-500" />;
    }
  };

  const formattedInstallationDate = battery.installation_date
    ? new Date(battery.installation_date).toLocaleDateString('es-CO', { year: 'numeric', month: 'long', day: 'numeric' })
    : 'N/A';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-foreground">Detalle de {battery.name || 'Batería Desconocida'}</h1>
        <div className="flex space-x-2"> {/* Contenedor para los botones */}
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" className="text-red-600 hover:bg-red-50 hover:text-red-700">
                <Trash2 className="h-4 w-4 mr-2" /> Eliminar Batería
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>¿Estás absolutamente seguro?</AlertDialogTitle>
                <AlertDialogDescription>
                  Esta acción no se puede deshacer. Esto eliminará permanentemente la batería y todos sus datos asociados de nuestros servidores.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancelar</AlertDialogCancel>
                <AlertDialogAction onClick={handleDeleteBattery} className="bg-red-500 hover:bg-red-600 text-white">
                  Sí, eliminar batería
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>

          <Button variant="outline" onClick={handleToggleVisibility}>
            {isBatteryHidden ? (
              <>
                <Eye className="h-4 w-4 mr-2" /> Mostrar Batería
              </>
            ) : (
              <>
                <EyeOff className="h-4 w-4 mr-2" /> Ocultar Batería
              </>
            )}
          </Button>

          <Button variant="outline" onClick={() => navigate('/batteries')}>
            <ChevronLeft className="h-4 w-4 mr-2" /> Volver a Baterías
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center space-x-4">
            {getStatusIcon(battery.status)}
            <div>
              <CardTitle className="text-2xl font-bold">{battery.name || 'N/A'}</CardTitle>
              <CardDescription>
                {battery.type || 'N/A'} {battery.model && `(${battery.model})`} - SN: {battery.serial_number || 'N/A'}
              </CardDescription>
            </div>
            <Badge variant="outline" className={cn("ml-auto", getBatteryStatusColor(battery.status))}>
              {battery.status || 'unknown'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Fabricante</span>
            <span className="text-base font-semibold">{battery.manufacturer || 'N/A'}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Capacidad Nominal</span>
            <span className="text-base font-semibold">{battery.capacity ? `${battery.capacity} Ah` : 'N/A'}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Voltaje Nominal</span>
            <span className="text-base font-semibold">{battery.voltage_nominal ? `${battery.voltage_nominal} V` : 'N/A'}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Fecha de Instalación</span>
            <span className="text-base font-semibold">{formattedInstallationDate}</span>
          </div>
        </CardContent>
      </Card>

      {/* Tarjeta de Métricas Clave */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center"><Activity className="h-5 w-5 mr-2" /> Métricas Clave</CardTitle>
          <CardDescription>Estado actual y salud operativa de la batería.</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* SOC - State of Charge */}
          <div className="flex flex-col items-center">
            <Battery className="h-10 w-10 text-blue-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Estado de Carga (SOC)</span>
            <span className="text-2xl font-bold text-blue-600">
              {latestBatteryMetrics?.soc ? formatPercentage(latestBatteryMetrics.soc, 0) : '--'}
            </span>
            <Progress value={latestBatteryMetrics?.soc || 0} className="w-full mt-2" />
          </div>

          {/* SOH - State of Health */}
          <div className="flex flex-col items-center">
            <TrendingUp className="h-10 w-10 text-green-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Salud de la Batería (SOH)</span>
            <span className="text-2xl font-bold text-green-600">
              {latestBatteryMetrics?.soh ? formatPercentage(latestBatteryMetrics.soh, 0) : '--'}
            </span>
            <Progress value={latestBatteryMetrics?.soh || 0} className="w-full mt-2" />
          </div>

          {/* Temperatura */}
          <div className="flex flex-col items-center">
            <Thermometer className="h-10 w-10 text-orange-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Temperatura</span>
            <span className="text-2xl font-bold text-orange-600">
              {latestBatteryMetrics?.temperature ? `${formatNumber(latestBatteryMetrics.temperature, 1)}°C` : '--°C'}
            </span>
          </div>

          {/* RUL - Remaining Useful Life (Nota: Este dato no estaba en el objeto de datos históricos proporcionado, se mantiene de la batería principal si existe) */}
          <div className="flex flex-col items-center">
            <Clock className="h-10 w-10 text-purple-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Vida Útil Restante (RUL)</span>
            <span className="text-2xl font-bold text-purple-600">
              {battery.rul ? `${formatNumber(battery.rul, 1)} años` : '--'}
            </span>
          </div>

          {/* Current (Corriente) */}
          <div className="flex flex-col items-center">
            <Zap className="h-10 w-10 text-red-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Corriente</span>
            <span className="text-2xl font-bold text-red-600">
              {latestBatteryMetrics?.current ? `${formatNumber(latestBatteryMetrics.current, 2)} A` : '--'}
            </span>
          </div>

          {/* Cycles (Ciclos de carga/descarga) */}
          <div className="flex flex-col items-center">
            <Activity className="h-10 w-10 text-cyan-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Ciclos</span>
            <span className="text-2xl font-bold text-cyan-600">
              {latestBatteryMetrics?.cycles ? formatNumber(latestBatteryMetrics.cycles, 0) : '--'}
            </span>
          </div>

          {/* NUEVAS MÉTRICAS */}

          {/* Voltage */}
          <div className="flex flex-col items-center">
            <Bolt className="h-10 w-10 text-yellow-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Voltaje</span>
            <span className="text-2xl font-bold text-yellow-600">
              {latestBatteryMetrics?.voltage ? `${formatNumber(latestBatteryMetrics.voltage, 2)} V` : '--'}
            </span>
          </div>

          {/* Efficiency */}
          <div className="flex flex-col items-center">
            <Gauge className="h-10 w-10 text-teal-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Eficiencia</span>
            <span className="text-2xl font-bold text-teal-600">
              {latestBatteryMetrics?.efficiency ? formatPercentage(latestBatteryMetrics.efficiency * 100, 1) : '--'}
            </span>
          </div>

          {/* Internal Resistance */}
          <div className="flex flex-col items-center">
            <SlidersHorizontal className="h-10 w-10 text-gray-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Resistencia Interna</span>
            <span className="text-2xl font-bold text-gray-600">
              {latestBatteryMetrics?.internal_resistance ? `${formatNumber(latestBatteryMetrics.internal_resistance, 3)} Ω` : '--'}
            </span>
          </div>

          {/* Power */}
          <div className="flex flex-col items-center">
            <Power className="h-10 w-10 text-indigo-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Potencia</span>
            <span className="text-2xl font-bold text-indigo-600">
              {latestBatteryMetrics?.power ? `${formatNumber(latestBatteryMetrics.power, 2)} W` : '--'}
            </span>
          </div>

          {/* Last Measurement Timestamp */}
          <div className="flex flex-col items-center">
            <Calendar className="h-10 w-10 text-pink-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Última Medición</span>
            <span className="text-base font-bold text-pink-600 text-center">
              {latestBatteryMetrics?.timestamp ? formatDateTime(latestBatteryMetrics.timestamp) : '--'}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Sección de Alertas Activas */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" /> Alertas Activas
          </CardTitle>
          <CardDescription>Advertencias y notificaciones recientes de la batería.</CardDescription>
        </CardHeader>
        <CardContent>
          {loadingAdditionalData && !additionalDataError?.alerts && <p>Cargando alertas...</p>}
          {additionalDataError?.alerts && <p className="text-red-500">Error al cargar alertas: {additionalDataError.alerts}</p>}
          {!loadingAdditionalData && alerts.length === 0 && !additionalDataError?.alerts && (
            <p className="text-muted-foreground">No hay alertas activas para esta batería.</p>
          )}
          {alerts.length > 0 && (
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-start space-x-3 p-3 border border-border rounded-lg">
                  <AlertTriangle className={cn("h-4 w-4 mt-0.5", alert.severity === 'high' ? 'text-red-500' : (alert.severity === 'medium' ? 'text-orange-500' : 'text-yellow-500'))} />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-foreground">{alert.message}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {alert.timestamp ? formatDateTime(alert.timestamp) : 'Fecha desconocida'}
                    </p>
                  </div>
                  <Badge variant="outline" className="text-xs">{alert.severity}</Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sección de Resultados de Análisis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <BarChart2 className="h-5 w-5 mr-2" /> Resultados de Análisis
          </CardTitle>
          <CardDescription>Informes de diagnóstico y análisis predictivo.</CardDescription>
        </CardHeader>
        <CardContent>
          {loadingAdditionalData && !additionalDataError?.analysis && <p>Cargando resultados de análisis...</p>}
          {additionalDataError?.analysis && <p className="text-red-500">Error al cargar resultados de análisis: {additionalDataError.analysis}</p>}
          {!loadingAdditionalData && analysisResults.length === 0 && !additionalDataError?.analysis && (
            <p className="text-muted-foreground">No hay resultados de análisis disponibles para esta batería.</p>
          )}
          {analysisResults.length > 0 && (
            <div className="space-y-3">
              {analysisResults.map((result) => (
                <div key={result.id} className="flex flex-col p-3 border border-border rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <p className="font-medium">{result.type || 'Análisis'}</p>
                    <Badge variant="outline" className={cn(
                      result.status === 'completed' && 'bg-green-100 text-green-800',
                      result.status === 'in_progress' && 'bg-yellow-100 text-yellow-800',
                      result.status === 'failed' && 'bg-red-100 text-red-800',
                    )}>
                      {result.status || 'desconocido'}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{result.summary || 'Sin resumen.'}</p>
                  <p className="text-xs text-muted-foreground">
                    Fecha: {result.timestamp ? formatDateTime(result.timestamp) : 'N/A'}
                  </p>
                  {result.recommendations && result.recommendations.length > 0 && (
                    <div className="mt-2 text-sm">
                      <span className="font-semibold">Recomendaciones:</span>
                      <ul className="list-disc list-inside text-muted-foreground">
                        {result.recommendations.map((rec, idx) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sección de Registros de Mantenimiento */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Wrench className="h-5 w-5 mr-2" /> Registros de Mantenimiento
          </CardTitle>
          <CardDescription>Historial de servicios y mantenimientos realizados.</CardDescription>
        </CardHeader>
        <CardContent>
          {loadingAdditionalData && !additionalDataError?.maintenance && <p>Cargando registros de mantenimiento...</p>}
          {additionalDataError?.maintenance && <p className="text-red-500">Error al cargar registros de mantenimiento: {additionalDataError.maintenance}</p>}
          {!loadingAdditionalData && maintenanceRecords.length === 0 && !additionalDataError?.maintenance && (
            <p className="text-muted-foreground">No hay registros de mantenimiento para esta batería.</p>
          )}
          {maintenanceRecords.length > 0 && (
            <div className="space-y-3">
              {maintenanceRecords.map((record) => (
                <div key={record.id} className="flex flex-col p-3 border border-border rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <p className="font-medium">{record.type || 'Mantenimiento'}</p>
                    <Badge variant="outline">{record.status || 'completado'}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{record.description || 'Sin descripción.'}</p>
                  <p className="text-xs text-muted-foreground">
                    Fecha: {record.date ? formatDate(record.date) : 'N/A'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Realizado por: {record.performed_by || 'Desconocido'}
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Placeholder para Gráficos de Datos Históricos (se usará 'historicalData' aquí) */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <BarChart2 className="h-5 w-5 mr-2" /> Gráficos de Datos Históricos
          </CardTitle>
          <CardDescription>Visualización de tendencias de rendimiento a lo largo del tiempo.</CardDescription>
        </CardHeader>
        <CardContent>
          {loadingAdditionalData && !additionalDataError?.historical && <p>Cargando datos históricos...</p>}
          {additionalDataError?.historical && <p className="text-red-500">Error al cargar datos históricos: {additionalDataError.historical}</p>}
          {!loadingAdditionalData && !historicalData && !additionalDataError?.historical && (
            <p className="text-muted-foreground">No hay datos históricos disponibles para mostrar gráficos.</p>
          )}
          {historicalData && (Array.isArray(historicalData) ? historicalData.length > 0 : typeof historicalData === 'object') && (
            <div className="text-muted-foreground">
              <p>Datos históricos cargados (
                {Array.isArray(historicalData) ? `${historicalData.length} puntos` : '1 punto'}
                ). Ahora puedes integrar aquí los componentes de gráficos para visualizar estos datos.</p>
              {/* Para depuración, puedes descomentar la siguiente línea para ver los datos crudos: */}
              {/* <pre className="text-xs overflow-auto h-40">{JSON.stringify(historicalData, null, 2)}</pre> */}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
