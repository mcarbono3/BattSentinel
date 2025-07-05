// src/pages/BatteryDetailPage.jsx
import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
// Se han añadido Bolt, Gauge, SlidersHorizontal, Power, Calendar, EyeOff, Trash2, Eye
import { Activity, Battery, Thermometer, Zap, Clock, TrendingUp, AlertTriangle, ChevronLeft, Wrench, BarChart2, Bolt, Gauge, SlidersHorizontal, Power, Calendar, EyeOff, Trash2, Eye, BatteryCharging } from 'lucide-react';
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
// Importar Popover y CalendarComponent para el selector de fechas
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";
import { format, addDays } from "date-fns"; // Importar format y addDays

// Importar DropdownMenu para el selector de gráficos
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu";


export default function BatteryDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast(); // Inicializar toast

  const [settingsVersion, setSettingsVersion] = useState(0); //

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

  // --- Nuevos estados para el manejo de gráficos (manteniendo la lógica de la mejora anterior) ---
  const initialChartMetrics = [
    'soc', 'soh', 'temperature', 'current', 'cycles',
    'voltage', 'efficiency', 'internal_resistance', 'power' // Usar internal_resistance para que coincida con el backend
  ];

// Función para obtener el estado inicial de dateRange desde localStorage
const getInitialDateRange = () => {
      const storedDateRange = localStorage.getItem('battSentinelDateRange');
    try {
        const storedRange = localStorage.getItem('battSentinelDateRange');
        if (storedRange) {
            const parsedRange = JSON.parse(storedRange);
            // Asegúrate de que las fechas sean objetos Date
            return {
                from: parsedRange.from ? new Date(parsedRange.from) : undefined,
                to: parsedRange.to ? new Date(parsedRange.to) : undefined,
            };
        }
    } catch (e) {
        console.error("Error parsing date range from localStorage", e);
    }
    return { from: undefined, to: undefined };
};

// Función para obtener el estado inicial de los gráficos desde localStorage
const getInitialVisibleCharts = () => {
    try {
        const storedCharts = localStorage.getItem('battSentinelVisibleCharts');
        if (storedCharts) {
            const parsedCharts = JSON.parse(storedCharts);
            // Asegúrate de que los gráficos guardados son un array y válidos
            if (Array.isArray(parsedCharts) && parsedCharts.every(chart => initialChartMetrics.includes(chart))) {
                return parsedCharts;
            }
        }
    } catch (e) {
        console.error("Error parsing visible charts from localStorage", e);
    }
    return initialChartMetrics; // Por defecto, todos los gráficos
};

  const [pendingVisibleCharts, setPendingVisibleCharts] = useState(getInitialVisibleCharts); // Nuevo estado
  const [visibleCharts, setVisibleCharts] = useState(getInitialVisibleCharts);
  const [dateRange, setDateRange] = useState(getInitialDateRange);

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
    localStorage.setItem('battSentinelDateRange', JSON.stringify(dateRange));
}, [dateRange]);

useEffect(() => {
    localStorage.setItem('battSentinelVisibleCharts', JSON.stringify(pendingVisibleCharts));
}, [pendingVisibleCharts]);

// Opcional: También guardar visibleCharts si difiere de pendingVisibleCharts al aplicar
useEffect(() => {
    // Asegura que visibleCharts también se guarda después de aplicar los filtros
    localStorage.setItem('battSentinelVisibleCharts', JSON.stringify(visibleCharts));
}, [visibleCharts]);

  // --- Función para cargar datos históricos con filtros ---
  const fetchHistoricalData = useCallback(async (start, end) => {
    if (!battery?.id) return;
    setLoadingAdditionalData(true);
    setAdditionalDataError(prev => ({ ...prev, historical: null }));
    try {
      const params = {};
      if (start) {
        params.startDate = format(start, 'yyyy-MM-dd');
      }
      if (end) {
        const adjustedEndDate = addDays(end, 1);
	params.endDate = format(adjustedEndDate, 'yyyy-MM-dd');
      }
      const historicalDataResult = await getBatteryHistoricalData(battery.id, params);
      if (historicalDataResult.success) {
        setHistoricalData(historicalDataResult.data);
      } else {
        console.error("Error fetching historical data:", historicalDataResult.error);
        setAdditionalDataError(prev => ({ ...prev, historical: historicalDataResult.error || 'Error al cargar datos históricos' }));
      }
    } catch (err) {
      console.error("Failed to fetch historical data:", err);
      setAdditionalDataError(prev => ({ ...prev, historical: "Error inesperado al cargar datos históricos." }));
    } finally {
      setLoadingAdditionalData(false);
    }
  }, [battery?.id, getBatteryHistoricalData]);


  // Uso de useCallback para optimizar las funciones de fetching
  const fetchCurrentBatteryDetails = useCallback(async () => {
    if (id) {
      const result = await getBatteryById(parseInt(id));
      if (result && result.success) {
        setBattery(result.data);
      } else {
        console.error("Error fetching battery details for polling:", result ? result.error : "Unknown error");
      }
    }
  }, [id, getBatteryById]);

  const fetchAllRelatedData = useCallback(async () => {
    if (battery?.id) {
      const batteryId = battery.id;

      // Cargar Alertas
      const alertsResult = await getAlertsByBatteryId(batteryId);
      if (alertsResult.success) {
        setAlerts(alertsResult.data);
      } else {
        console.error("Error fetching alerts for polling:", alertsResult.error);
      }

      // Cargar Resultados de Análisis
      const analysisResult = await getAnalysisResultsByBatteryId(batteryId);
      if (analysisResult.success) {
        setAnalysisResults(analysisResult.data);
      } else {
        console.error("Error fetching analysis results for polling:", analysisResult.error);
      }

      // Cargar Registros de Mantenimiento
      const maintenanceResult = await getMaintenanceRecordsByBatteryId(batteryId);
      if (maintenanceResult.success) {
        setMaintenanceRecords(maintenanceResult.data);
      } else {
        console.error("Error fetching maintenance records for polling:", maintenanceResult.error);
      }
    }
  }, [battery?.id, getAlertsByBatteryId, getAnalysisResultsByBatteryId, getMaintenanceRecordsByBatteryId]);


  // *******************************************************************
  // ** INICIO DE LA SECCIÓN A MODIFICAR PARA APLICAR LAS CONFIGURACIONES DE REFRESH **
  // *******************************************************************

  useEffect(() => {
    let intervalId;
    const handleSettingsChanged = () => { // <-- AÑADIR ESTA FUNCIÓN
      console.log("Evento 'battSentinelSettingsChanged' recibido. Re-evaluando polling.");
      setSettingsVersion(prev => prev + 1); // Incrementa para forzar re-ejecución del useEffect
    };

    // Añadir el event listener
    window.addEventListener('battSentinelSettingsChanged', handleSettingsChanged);

    let autoRefreshEnabled = true; // Por defecto si no se encuentra en localStorage
    let refreshInterval = 30000; // Valor por defecto: 30 segundos (30000 ms)

    try {
      const storedEnabled = localStorage.getItem('battSentinelAutoRefreshEnabled');
      if (storedEnabled !== null) {
        autoRefreshEnabled = JSON.parse(storedEnabled);
      }

      const storedInterval = localStorage.getItem('battSentinelRefreshIntervalMs');
      if (storedInterval) {
        const parsedInterval = parseInt(storedInterval, 10);
        if (!isNaN(parsedInterval) && parsedInterval > 0) {
          refreshInterval = parsedInterval;
        } else {
          console.warn("Valor inválido para el intervalo de refresco en localStorage, usando el valor por defecto.");
        }
      }
    } catch (e) {
      console.error("Error al leer la configuración de refresco de localStorage, usando valores por defecto:", e);
    }

    if (autoRefreshEnabled) {
      // Ejecutar las funciones inmediatamente al montar y luego en el intervalo
      fetchCurrentBatteryDetails();
      fetchAllRelatedData();
      if (!dateRange.from && !dateRange.to) {
        fetchHistoricalData(undefined, undefined);
      }

      intervalId = setInterval(() => {
        fetchCurrentBatteryDetails();
        fetchAllRelatedData();
        // Refresca datos históricos solo si no hay filtros de fecha aplicados
        if (!dateRange.from && !dateRange.to) {
          fetchHistoricalData(undefined, undefined);
        }
      }, refreshInterval);
    } else {
      console.log("Refresco automático deshabilitado en la configuración.");
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId); // Limpieza: detener el intervalo al desmontar o al cambiar las dependencias
      }
      // Limpieza: remover el event listener
      window.removeEventListener('battSentinelSettingsChanged', handleSettingsChanged); // <-- AÑADIR ESTA LÍNEA
    };
  }, [fetchCurrentBatteryDetails, fetchAllRelatedData, fetchHistoricalData, settingsVersion]); 


  // *******************************************************************
  // ** FIN DE LA SECCIÓN A MODIFICAR PARA APLICAR LAS CONFIGURACIONES DE REFRESH **
  // *******************************************************************


  // Nuevo useEffect para cargar datos históricos cuando cambie la batería o el rango de fechas
  useEffect(() => {
    if (battery?.id) {
      fetchHistoricalData(dateRange.from, dateRange.to);
    }
  }, [battery?.id, fetchHistoricalData]);


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

  // Funciones para los filtros de gráficos
  const handleToggleChart = (chartName) => {
    setPendingVisibleCharts(prev =>
      prev.includes(chartName)
        ? prev.filter(name => name !== chartName)
        : [...prev, chartName]
    );
  };

  const handleApplyFilters = () => {
     setVisibleCharts(pendingVisibleCharts); 
    // La re-llamada a fetchHistoricalData ya se maneja en el useEffect
    fetchHistoricalData(dateRange.from, dateRange.to); // Forzar recarga con las fechas y gráficos seleccionados
  };

  const handleClearFilters = () => {
    setVisibleCharts(initialChartMetrics); // Restablecer a todos los gráficos
    setPendingVisibleCharts(initialChartMetrics);
    const newDateRange = { from: undefined, to: undefined };
    setDateRange(newDateRange); // Limpiar rango de fechas
    // El useEffect para historicalData se encargará de volver a cargar sin filtros.
    localStorage.removeItem('battSentinelDateRange');
    localStorage.removeItem('battSentinelVisibleCharts');

    fetchHistoricalData(newDateRange.from, newDateRange.to);
  };

  const today = new Date();
  const last15Days = addDays(today, -15);
  const lastMonth = addDays(today, -30);


  // Derivar la última o única medición de historicalData para las métricas clave
  // Se ha adaptado para usar el primer elemento si no hay un `latestBatteryMetrics` directo
  const latestBatteryMetrics = Array.isArray(historicalData) && historicalData.length > 0
    ? historicalData[historicalData.length - 1] // Última medición
    : (battery || null); // Si no hay datos históricos, usar los datos de la batería principal


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
                {battery.chemistry || 'N/A'} {battery.model && `(${battery.model})`} - SN: {battery.serial_number || 'N/A'}
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
            <span className="text-base font-semibold">{battery.full_charge_capacity ? `${battery.full_charge_capacity} Ah` : 'N/A'}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Voltaje Nominal</span>
            <span className="text-base font-semibold">{battery.designvoltage ? `${battery.designvoltage} V` : 'N/A'}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Fecha de Instalación</span>
            <span className="text-base font-semibold">{formattedInstallationDate}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-muted-foreground">Tipo de batería</span>
            <span className="text-base font-semibold">{battery.chemistry || 'N/A'}</span>
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
              {latestBatteryMetrics?.rul_days ? `${formatNumber(latestBatteryMetrics?.rul_days, 1)} dias` : '--'}
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

          {/* Estado */}
          <div className="flex flex-col items-center">
            {latestBatteryMetrics ? (
              <>
                {latestBatteryMetrics.is_plugged ? (
                  <BatteryCharging className="h-10 w-10 text-green-500 mb-2" />
                ) : (
                  <Battery className="h-10 w-10 text-red-500 mb-2" />
                )}
                <span className="text-sm font-medium text-muted-foreground">Estado</span>
                <span
                  className={`text-2xl font-bold ${
                    latestBatteryMetrics.is_plugged ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {latestBatteryMetrics.is_plugged ? 'Carga' : 'Descarga'}
                </span>
              </>
            ) : (
              <>
                <Battery className="h-10 w-10 text-gray-400 mb-2" />
                <span className="text-sm font-medium text-muted-foreground">Estado</span>
                <span className="text-2xl font-bold text-gray-500">--</span>
              </>
            )}
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

      {/* Gráficos de Datos Históricos */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <BarChart2 className="h-5 w-5 mr-2" /> Gráficos de Datos Históricos
          </CardTitle>
          <CardDescription>Visualización de tendencias de rendimiento a lo largo del tiempo.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4 items-center mb-6"> {/* flex-wrap para responsividad */}
            {/* Selector de Rango de Fechas */}
            <div className="flex items-center space-x-2">
              <Calendar className="h-4 w-4 text-gray-500" />
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    id="date"
                    variant={"outline"}
                    className={cn(
                      "w-[260px] justify-start text-left font-normal",
                      !dateRange?.from && "text-muted-foreground"
                    )}
                  >
                    {dateRange?.from ? (
                      dateRange?.to ? (
                        <>
                          {format(dateRange.from, "LLL dd, y")} -{" "}
                          {format(dateRange.to, "LLL dd, y")}
                        </>
                      ) : (
                        format(dateRange.from, "LLL dd, y")
                      )
                    ) : (
                      <span>Seleccionar rango de fechas</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <CalendarComponent
                    initialFocus
                    mode="range"
                    defaultMonth={dateRange?.from}
                    selected={dateRange}
                    onSelect={setDateRange}
                    numberOfMonths={2}
                  />
                  <div className="p-4 flex flex-col space-y-2">
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => setDateRange({ from: last15Days, to: today })}
                    >
                      Últimos 15 días
                    </Button>
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => setDateRange({ from: lastMonth, to: today })}
                    >
                      Último mes
                    </Button>
                  </div>
                </PopoverContent>
              </Popover>
            </div>

            {/* Selector de Gráficos */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="min-w-[180px]">
                  <BarChart2 className="h-4 w-4 mr-2" />
                  Visualizar Gráficos
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>Seleccionar Gráficos</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {initialChartMetrics.map((metric) => (
                  <DropdownMenuCheckboxItem
                    key={metric}
                    checked={pendingVisibleCharts.includes(metric)}
                    onCheckedChange={() => handleToggleChart(metric)}
                  >
                    {metric.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())} {/* Formatear camelCase a texto legible */}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Botones de Acción */}
            <div className="flex space-x-2">
              <Button onClick={handleApplyFilters}>Aplicar Filtros</Button>
              <Button variant="outline" onClick={handleClearFilters}>
                Limpiar Filtros
              </Button>
            </div>
          </div>


          {loadingAdditionalData && !additionalDataError?.historical && <p>Cargando datos históricos...</p>}
          {additionalDataError?.historical && <p className="text-red-500">Error al cargar datos históricos: {additionalDataError.historical}</p>}
          {!loadingAdditionalData && (!historicalData || historicalData.length === 0) && !additionalDataError?.historical && (
            <p className="text-muted-foreground">No hay datos históricos disponibles para mostrar gráficos.</p>
          )}

          {/* Gráfico de SOC (Barras) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('soc') && (
            <div className="mb-8"> {/* Añadimos margen inferior para separar gráficos */}
              <h3 className="text-lg font-semibold mb-2">Estado de Carga (SOC)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${value}%`}
                    label={{ value: 'SOC (%)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}%`, 'SOC']}
                  />
                  <Legend />
                  <Bar dataKey="soc" fill="#8884d8" name="Estado de Carga (SOC)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de SOH (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('soh') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Salud de la Batería (SOH)</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${value}%`}
                    label={{ value: 'SOH (%)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}%`, 'SOH']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="soh" stroke="#82ca9d" name="Salud de la Batería (SOH)" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Temperatura (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('temperature') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Temperatura</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${formatNumber(value)}°C`}
                    label={{ value: 'Temperatura (°C)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}°C`, 'Temperatura']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="temperature" stroke="#ffc658" name="Temperatura" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Corriente (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('current') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Corriente</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${formatNumber(value)}A`}
                    label={{ value: 'Corriente (A)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}A`, 'Corriente']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="current" stroke="#ff7300" name="Corriente" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Ciclos (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('cycles') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Ciclos</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    label={{ value: 'Ciclos', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [formatNumber(value), 'Ciclos']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="cycles" stroke="#ff65a3" name="Ciclos" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Voltaje (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('voltage') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Voltaje</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${formatNumber(value)}V`}
                    label={{ value: 'Voltaje (V)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}V`, 'Voltaje']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="voltage" stroke="#8dd1e1" name="Voltaje" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Eficiencia (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('efficiency') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Eficiencia</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${formatPercentage(value)}`}
                    label={{ value: 'Eficiencia (%)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatPercentage(value)}`, 'Eficiencia']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="efficiency" stroke="#a4de6c" name="Eficiencia" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Resistencia Interna (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('internal_resistance') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Resistencia Interna</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${formatNumber(value)}Ω`}
                    label={{ value: 'Resistencia Interna (Ω)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}Ω`, 'Resistencia Interna']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="internal_resistance" stroke="#d0ed57" name="Resistencia Interna" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Potencia (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('power') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Potencia</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={historicalData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    label={{ value: 'Fecha', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickFormatter={(value) => `${formatNumber(value)}W`}
                    label={{ value: 'Potencia (W)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}W`, 'Potencia']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="power" stroke="#f08700" name="Potencia" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

        </CardContent>
      </Card>
    </div>
  );
}
