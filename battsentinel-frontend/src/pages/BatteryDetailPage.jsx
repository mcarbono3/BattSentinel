// src/pages/BatteryDetailPage.jsx
import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Se han añadido Bolt, Gauge, SlidersHorizontal, Power, Calendar, EyeOff, Trash2, Eye, Loader2
import { Activity, Battery, Thermometer, Zap, Clock, TrendingUp, AlertTriangle, ChevronLeft, Wrench, BarChart2, Bolt, Gauge, SlidersHorizontal, Power, Calendar, EyeOff, Trash2, Eye, BatteryCharging, Loader2, CheckCircle, XCircle, Pencil } from 'lucide-react';
import { useBattery } from '@/contexts/BatteryContext';
import { cn, formatNumber, formatPercentage, getBatteryStatusColor, formatDate, formatDateTime } from '@/lib/utils';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';

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
import { useToast } from '@/hooks/use-toast';
// Importar Popover y CalendarComponent para el selector de fechas
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";
import { format, addDays } from "date-fns";

// NUEVA IMPORTACIÓN: Importa aiAPI desde tu archivo api.js
import { aiAPI } from '@/lib/api';

// Importar DropdownMenu para el selector de gráficos
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu";


// CORRECCIÓN: Eliminar la prop manualBatteryData si ya no se usa, y eliminar la declaración duplicada.
export default function BatteryDetailPage() { // <-- Una sola declaración
  const { id } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [settingsVersion, setSettingsVersion] = useState(0);

  const {
    getBatteryById,
    getAlertsByBatteryId,
    getAnalysisResultsByBatteryId,
    getMaintenanceRecordsByBatteryId,
    getBatteryHistoricalData,
    deleteBattery,
    toggleBatteryVisibility,
    hiddenBatteryIds,
    loading,
    error,
    selectedBattery, // Usaremos este para actualizar el estado local de la batería
  } = useBattery();

  const [battery, setBattery] = useState(null);

  const [alerts, setAlerts] = useState([]);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [maintenanceRecords, setMaintenanceRecords] = useState([]);
  const [historicalData, setHistoricalData] = useState(null);

  const [loadingAdditionalData, setLoadingAdditionalData] = useState(false);
  const [additionalDataError, setAdditionalDataError] = useState(null);

  // Nuevos estados para el análisis de IA
  const [aiAnalysisResults, setAiAnalysisResults] = useState(null);
  const [loadingAIAnalysis, setLoadingAIAnalysis] = useState(false);
  const [aiAnalysisError, setAiAnalysisError] = useState(null);

  const isBatteryHidden = battery ? hiddenBatteryIds.has(battery.id) : false;

  // useEffect para cargar la batería cuando el ID o selectedBattery cambie
  useEffect(() => {
    if (id) {
      // Intenta primero usar selectedBattery si ya está cargada y coincide
      if (selectedBattery && selectedBattery.id === parseInt(id)) {
        setBattery(selectedBattery);
      } else {
        // Si no, carga la batería específica usando la función del contexto
        const fetchBattery = async () => {
          const result = await getBatteryById(parseInt(id));
          if (result.success) {
            setBattery(result.data);
          } else {
            // Manejar error, por ejemplo, mostrar un toast
            // toast({
            //   title: "Error al cargar la batería",
            //   description: result.error || "No se pudo obtener la información de la batería.",
            //   variant: "destructive",
            // });
          }
        };
        fetchBattery();
      }
    }
  }, [id, selectedBattery, getBatteryById]); // Dependencias correctas

  // Función auxiliar para obtener valores anidados de un objeto
  const getNestedValue = (obj, path) => {
    return path.split('.').reduce((o, i) => (o ? o[i] : undefined), obj);
  };

const getDisplayValue = useCallback((path, defaultValue = 'N/A') => {
  // ... (código existente)

  if (battery) {
    let value = getNestedValue(battery, path);

    // Lógica específica para la ciudad: priorizar cityOther si existe y city está vacío o "Otra"
    if (path === 'location.city') {
        const actualCity = getNestedValue(battery, 'location.city');
        const otherCity = getNestedValue(battery, 'location.cityOther');
        if (otherCity && otherCity !== '') {
            value = otherCity;
        } else if (actualCity) {
            value = actualCity;
        } else {
            value = undefined; // o un valor por defecto si ambos están vacíos
        }
    }
    // ... (resto de la función)
    if (value !== undefined && value !== null && value !== '') {
      // ... (manejo de fechas)
      return value;
    }
  }
  return defaultValue;
}, [battery]);


  // NUEVAS VARIABLES PARA LOS VALORES A MOSTRAR (derivados)
  const displayId = getDisplayValue('id');
  const displayName = getDisplayValue('name');
  const displayModel = getDisplayValue('model');
  const displaySerialNumber = getDisplayValue('serial_number');
  const displayManufacturer = getDisplayValue('manufacturer');
  const displayCountry = getDisplayValue('location.country');
  const displayCity = getDisplayValue('location.city');
  const displayCountryOther = getDisplayValue('location.countryOther');
  const displayStatus = getDisplayValue('status');

  const displayFullChargeCapacity = getDisplayValue('full_charge_capacity');
  const displayCapacityUnit = getDisplayValue('capacity_unit');
  const displayDesignVoltage = getDisplayValue('designvoltage');

  // La fecha de instalación ya se maneja dentro de getDisplayValue si el path es 'installation_date'
  const displayInstallationDate = getDisplayValue('installation_date');

  const displayChemistry = getDisplayValue('chemistry');

  // --- Nuevos estados para el manejo de gráficos (manteniendo la lógica de la mejora anterior) ---
  const initialChartMetrics = [
    'soc', 'soh', 'temperature', 'current', 'cycles',
    'voltage', 'efficiency', 'internal_resistance', 'power'
  ];

  // Función para obtener el estado inicial de dateRange desde localStorage
  const getInitialDateRange = () => {
    // ... (código existente para getInitialDateRange)
    const storedDateRange = localStorage.getItem('battSentinelDateRange');
    try {
        const storedRange = localStorage.getItem('battSentinelDateRange');
        if (storedRange) {
            const parsedRange = JSON.parse(storedRange);
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

      // Cargar Resultados del Análisis
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
    let refreshInterval = 420000; // Valor por defecto: 7 min (420000 ms)

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


  // Nueva función para ejecutar el análisis de IA
const runAIAnalysis = async () => {
  if (!battery?.id) {
    toast({
      title: "Error",
      description: "No se pudo iniciar el análisis: ID de batería no disponible.",
      variant: "destructive",
    });
    return;
  }

  setLoadingAIAnalysis(true);
  setAiAnalysisError(null);
  setAiAnalysisResults(null); // Limpiar resultados anteriores
  try {
    const response = await aiAPI.analyzeBattery(battery.id, {
      time_window_hours: 24
    });

    // ¡ASÍ ES COMO DEBES MANEJAR LA RESPUESTA DE api.js!
    if (response.success) { // Accede directamente a 'success'
      setAiAnalysisResults(response.data); // Accede directamente a 'results'
      toast({
        title: "Análisis de IA Completado",
        description: "Se han generado nuevos resultados del análisis de inteligencia artificial.",
      });
    } else {
      // Accede directamente a 'error'
      setAiAnalysisError(response.error || "Error desconocido al ejecutar el análisis de IA.");
      toast({
        title: "Error en Análisis de IA",
        description: response.error || "No se pudo completar el análisis de IA.",
        variant: "destructive",
      });
    }
  } catch (err) {
    // Este catch solo se activaría para errores MUY inesperados no manejados por api.js.
    // Para errores de red o HTTP, api.js ya devuelve { success: false, error: ... }
    console.error("Error inesperado en AI analysis:", err);
    setAiAnalysisError(err.message || "Error inesperado al ejecutar el análisis de IA.");
    toast({
      title: "Error Inesperado",
      description: "Ocurrió un error inesperado al procesar la solicitud.",
      variant: "destructive",
    });
  } finally {
    setLoadingAIAnalysis(false);
  }
};


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
        <Button variant="outline" onClick={() => navigate('/dashboard')} className="mb-4">
          <ChevronLeft className="h-4 w-4 mr-2" /> Volver al Dashboard
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
      {/* 1. Volver al Dashboard */}
      <Button variant="outline" onClick={() => navigate('/dashboard')}>
        <ChevronLeft className="h-4 w-4 mr-2" /> Volver al Dashboard
      </Button>

      {/* 2. Ocultar Batería / Mostrar Batería */}
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

      {/* 3. Eliminar Batería */}
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
    </div>
  </div>

{/* Tarjeta de Información General */}
<Card>
  {/* MODIFICACIÓN AQUÍ: Añade flexbox para alinear título y botón */}
  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
    <CardTitle className="flex items-center">
      {/* Usa displayStatus para el icono */}
      {getStatusIcon(displayStatus)}
      <span className="ml-2">Información General</span>
    </CardTitle>
    {/* AÑADE ESTE BOTÓN DE EDICIÓN */}
    <Button
      variant="ghost"
      size="icon"
      onClick={() => navigate(`/batteries?editId=${id}`)} // Redirige a la URL especificada
      className="hover:bg-gray-100 dark:hover:bg-gray-800"
      title="Editar Información General"
    >
      <Pencil className="h-5 w-5 text-gray-500" />
    </Button>
    {/* FIN DEL BOTÓN DE EDICIÓN */}
  </CardHeader>
  {/* CardDescription se mueve fuera de CardHeader pero dentro de Card */}
  <CardDescription className="px-6">Detalles básicos y estado de la batería.</CardDescription>
  <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">ID de Batería</span>
      {/* Usa displayId */}
      <span className="text-base font-semibold">{displayId || 'N/A'}</span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Nombre</span>
      {/* Usa displayName */}
      <span className="text-base font-semibold">{displayName || 'N/A'}</span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Modelo</span>
      {/* Usa displayModel */}
      <span className="text-base font-semibold">{displayModel || 'N/A'}</span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Número de Serie</span>
      {/* Usa displaySerialNumber */}
      <span className="text-base font-semibold">{displaySerialNumber || 'N/A'}</span>
    </div>
<div className="flex flex-col">
  <span className="text-sm font-medium text-muted-foreground">Ciudad de Ubicación</span>
  <span className="text-base font-semibold">{displayCity}</span> {/* Usamos displayCity */}
</div>
<div className="flex flex-col mt-2"> {/* Agregado un margen para separar visualmente */}
  <span className="text-sm font-medium text-muted-foreground">País de Ubicación</span>
  <span className="text-base font-semibold">{displayCountry}</span> {/* Usamos displayCountry */}
</div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Estado</span>
      <Badge
        variant="outline"
        className={cn(
          // Usa displayStatus para el color
          getBatteryStatusColor(displayStatus),
          displayStatus === 'Active'
            ? 'bg-green-500 text-white'
            : 'bg-red-500 text-white'
        )}
      >
        {/* Usa displayStatus */}
        {displayStatus || 'unknown'}
      </Badge>
    </div>
  </CardContent>
</Card>

{/* Tarjeta de Especificaciones Técnicas */}
<Card>
  {/* MODIFICACIÓN AQUÍ: Añade flexbox para alinear título y botón */}
  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
    <CardTitle className="flex items-center"><Wrench className="h-5 w-5 mr-2" /> Especificaciones Técnicas</CardTitle>
    {/* AÑADE ESTE BOTÓN DE EDICIÓN */}
    <Button
      variant="ghost"
      size="icon"
      onClick={() => navigate(`/batteries?editId=${id}`)} // Redirige a la URL especificada
      className="hover:bg-gray-100 dark:hover:bg-gray-800"
      title="Editar Especificaciones Técnicas"
    >
      <Pencil className="h-5 w-5 text-gray-500" />
    </Button>
    {/* FIN DEL BOTÓN DE EDICIÓN */}
  </CardHeader>
  {/* CardDescription se mueve fuera de CardHeader pero dentro de Card */}
  <CardDescription className="px-6">Detalles técnicos y de fabricación.</CardDescription>
  <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Fabricante</span>
      {/* Usa displayManufacturer */}
      <span className="text-base font-semibold">{displayManufacturer || 'N/A'}</span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Capacidad Nominal</span>
      <span className="text-base font-semibold">
        {/* Usa displayFullChargeCapacity */}
        {(displayFullChargeCapacity !== undefined && displayFullChargeCapacity !== null && displayFullChargeCapacity !== '')
          ? `${displayFullChargeCapacity} Ah`
          : 'N/A'}
      </span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Voltaje Nominal</span>
      <span className="text-base font-semibold">
        {/* Usa displayDesignVoltage */}
        {(displayDesignVoltage !== undefined && displayDesignVoltage !== null && displayDesignVoltage !== '')
          ? `${displayDesignVoltage} V`
          : 'N/A'}
      </span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Fecha de Instalación</span>
      {/* Usa displayInstallationDate */}
      <span className="text-base font-semibold">{displayInstallationDate}</span>
    </div>
    <div className="flex flex-col">
      <span className="text-sm font-medium text-muted-foreground">Tipo de batería</span>
      {/* Usa displayChemistry */}
      <span className="text-base font-semibold">{displayChemistry || 'N/A'}</span>
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

          {/* Temperatura */}
          <div className="flex flex-col items-center">
            <Thermometer className="h-10 w-10 text-orange-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Temperatura</span>
            <span className="text-2xl font-bold text-orange-600">
              {latestBatteryMetrics?.temperature ? `${formatNumber(latestBatteryMetrics.temperature, 1)}°C` : '--°C'}
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

          {/* Voltage */}
          <div className="flex flex-col items-center">
            <Bolt className="h-10 w-10 text-yellow-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Voltaje</span>
            <span className="text-2xl font-bold text-yellow-600">
              {latestBatteryMetrics?.voltage ? `${formatNumber(latestBatteryMetrics.voltage, 2)} V` : '--'}
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

          {/* Efficiency */}
          <div className="flex flex-col items-center">
            <Gauge className="h-10 w-10 text-teal-500 mb-2" />
            <span className="text-sm font-medium text-muted-foreground">Eficiencia</span>
            <span className="text-2xl font-bold text-teal-600">
              {latestBatteryMetrics?.efficiency ? formatPercentage(latestBatteryMetrics.efficiency * 100, 1) : '--'}
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

      {/* BOTÓN EJECUTAR ANÁLISIS DE IA - ¡NUEVA UBICACIÓN AQUÍ! */}
      <div className="flex justify-end mb-6 mt-6"> {/* Añadido mt-6 para más espacio */}
        <Button
          onClick={runAIAnalysis}
          disabled={loadingAIAnalysis}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-200 text-lg"
        >
          {loadingAIAnalysis ? (
            <>
              <Loader2 className="mr-3 h-5 w-5 animate-spin" />
              Analizando Datos...
            </>
          ) : (
            <>
              <Zap className="mr-3 h-5 w-5" />
              Ejecutar Análisis de IA
            </>
          )}
        </Button>
      </div>
      {/* FIN DEL BOTÓN */}

{/* Nueva Tarjeta de Resultados de Análisis de IA - MEJORADA UI/UX */}
{aiAnalysisResults && (
  <Card className="mb-6 shadow-lg border-t-4 border-blue-500"> {/* Añadido sombra y borde superior para destacar */}
    <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-t-lg"> {/* Fondo degradado */}
      <CardTitle className="flex items-center text-2xl font-bold text-blue-800">
        <Zap className="h-7 w-7 mr-3 text-blue-600" /> Resultados del Análisis de IA
      </CardTitle>
      <CardDescription className="text-blue-700 text-sm mt-1">
        Último análisis realizado el: <span className="font-semibold">{formatDateTime(aiAnalysisResults.analysis_timestamp)}</span> (Puntos de datos analizados: <span className="font-semibold">{aiAnalysisResults.data_points_analyzed}</span>)
      </CardDescription>
    </CardHeader>
    <CardContent className="p-6">
      {aiAnalysisError && (
        <p className="text-red-600 bg-red-50 p-3 rounded-md mb-4 border border-red-200">
          <AlertTriangle className="inline h-4 w-4 mr-2" /> Error al obtener resultados: {aiAnalysisError}
        </p>
      )}

      {/* Sección de Detección de Fallas */}
      {aiAnalysisResults.results?.fault_detection && (
        <div className="mb-8 p-4 border border-orange-200 rounded-lg bg-orange-50">
          <h3 className="text-xl font-semibold mb-3 flex items-center text-orange-700">
            <AlertTriangle className="h-5 w-5 mr-2 text-orange-500" /> Detección de Fallas
            {/* Estado general de falla */}
            {aiAnalysisResults.results.fault_detection.fault_detected ? (
              <Badge variant="destructive" className="ml-3 px-3 py-1 text-sm">
                <XCircle className="h-4 w-4 mr-1" /> Falla Detectada
              </Badge>
            ) : (
              <Badge className="bg-green-500 text-white ml-3 px-3 py-1 text-sm">
                <CheckCircle className="h-4 w-4 mr-1" /> Normal
              </Badge>
            )}
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            <div className="bg-white p-3 rounded-md shadow-sm border border-orange-100">
              <p className="text-sm text-gray-500">Tipo de Falla</p>
              <Badge
                className={cn(
                  'text-base font-bold mt-1',
                  aiAnalysisResults.results.fault_detection.fault_type === 'normal' && 'bg-green-100 text-green-800 hover:bg-green-100',
                  aiAnalysisResults.results.fault_detection.severity === 'low' && 'bg-blue-100 text-blue-800 hover:bg-blue-100',
                  aiAnalysisResults.results.fault_detection.severity === 'medium' && 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100',
                  aiAnalysisResults.results.fault_detection.severity === 'high' && 'bg-orange-100 text-orange-800 hover:bg-orange-100',
                  aiAnalysisResults.results.fault_detection.severity === 'critical' && 'bg-red-100 text-red-800 hover:bg-red-100',
                )}
              >
                {aiAnalysisResults.results.fault_detection.fault_type || 'N/A'}
              </Badge>
            </div>
            <div className="bg-white p-3 rounded-md shadow-sm border border-orange-100">
              <p className="text-sm text-gray-500">Severidad</p>
              <p className="text-lg font-bold mt-1">
                <Badge
                  className={cn(
                    aiAnalysisResults.results.fault_detection.severity === 'low' && 'bg-blue-100 text-blue-800',
                    aiAnalysisResults.results.fault_detection.severity === 'medium' && 'bg-yellow-100 text-yellow-800',
                    aiAnalysisResults.results.fault_detection.severity === 'high' && 'bg-orange-100 text-orange-800',
                    aiAnalysisResults.results.fault_detection.severity === 'critical' && 'bg-red-100 text-red-800',
                  )}
                >
                  {aiAnalysisResults.results.fault_detection.severity || 'N/A'}
                </Badge>
              </p>
            </div>
            <div className="bg-white p-3 rounded-md shadow-sm border border-orange-100">
              <p className="text-sm text-gray-500">Confianza</p>
              <p className="text-lg font-bold mt-1 text-orange-600">
                {formatPercentage(aiAnalysisResults.results.fault_detection.confidence * 100, 1)}
              </p>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mt-3">
            <span className="font-medium text-gray-700">Explicación:</span>{' '}
            {aiAnalysisResults.results.fault_detection.explanation?.method || 'No hay explicación disponible.'}
            {aiAnalysisResults.results.fault_detection.explanation?.parameters_analyzed?.length > 0 &&
              ` - Parámetros Analizados: ${aiAnalysisResults.results.fault_detection.explanation.parameters_analyzed.join(', ')}`}
          </p>

          {aiAnalysisResults.results.fault_detection.predictions && aiAnalysisResults.results.fault_detection.predictions.length > 0 && (
            <div className="mt-5 p-4 bg-white rounded-md shadow-inner border border-gray-100">
              <p className="font-medium text-gray-700 mb-2">Detalle de Fallas Específicas:</p>
              <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                {aiAnalysisResults.results.fault_detection.predictions.map((fault, index) => (
                  <li key={index}>
                    <span className="font-semibold">{fault.type || 'Falla Desconocida'}</span>
                    {' - '}Severidad: <Badge variant="outline" className={cn(
                        fault.severity === 'low' && 'bg-blue-50 text-blue-700',
                        fault.severity === 'medium' && 'bg-yellow-50 text-yellow-700',
                        fault.severity === 'high' && 'bg-orange-50 text-orange-700',
                        fault.severity === 'critical' && 'bg-red-50 text-red-700',
                      )}>{fault.severity || 'N/A'}</Badge>
                    {fault.value && `, Valor: ${formatNumber(fault.value, 2)}`}
                    {fault.threshold && `, Umbral: ${formatNumber(fault.threshold, 2)}`}
                    {fault.variability && `, Variabilidad: ${formatNumber(fault.variability, 2)}`}
                    {fault.soh && `, SOH: ${formatNumber(fault.soh, 2)}%`}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Secciones de Anomalías y Top Features (se mostrarán solo si hay datos de modelos avanzados) */}
          {aiAnalysisResults.results.fault_detection.anomalies && aiAnalysisResults.results.fault_detection.anomalies.length > 0 && (
            <div className="mt-5 p-4 bg-white rounded-md shadow-inner border border-gray-100">
              <p className="font-medium text-gray-700 mb-2">Anomalías Detectadas (Adicional):</p>
              <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                {aiAnalysisResults.results.fault_detection.anomalies.map((anomaly, index) => (
                  <li key={index}>
                    Parámetro: <span className="font-semibold">{anomaly.parameter}</span>, Valor: {formatNumber(anomaly.value, 2)},
                    Timestamp: {formatDateTime(anomaly.timestamp)}, Severidad: {anomaly.severity}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {aiAnalysisResults.results.fault_detection.top_features && Object.keys(aiAnalysisResults.results.fault_detection.top_features).length > 0 && (
            <div className="mt-5 p-4 bg-white rounded-md shadow-inner border border-gray-100">
              <p className="font-medium text-gray-700 mb-2">Características más influyentes (Adicional):</p>
              <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
                {Object.entries(aiAnalysisResults.results.fault_detection.top_features).map(([feature, importance]) => (
                  <li key={feature}>
                    <span className="font-semibold">{feature}</span>: {formatNumber(importance * 100, 2)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <Separator className="my-8 bg-gray-200" /> {/* Separador más prominente */}

      {/* Sección de Predicción de Salud */}
      {aiAnalysisResults.results?.health_prediction && (
        <div className="p-4 border border-green-200 rounded-lg bg-green-50">
          <h3 className="text-xl font-semibold mb-3 flex items-center text-green-700">
            <BatteryCharging className="h-5 w-5 mr-2 text-green-500" /> Predicción de Salud
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            {/* SOH Actual */}
            <div className="bg-white p-3 rounded-md shadow-sm border border-green-100">
              <p className="text-sm text-gray-500">Estado de Salud (SOH)</p>
              <div className="flex items-center mt-1">
                <p className="text-2xl font-bold mr-2">
                  {formatPercentage(aiAnalysisResults.results.health_prediction.predictions.current_soh, 1)}
                </p>
                <Progress
                  value={aiAnalysisResults.results.health_prediction.predictions.current_soh}
                  className={cn(
                    "w-full h-2",
                    aiAnalysisResults.results.health_prediction.predictions.current_soh >= 90 && 'bg-green-500',
                    aiAnalysisResults.results.health_prediction.predictions.current_soh < 90 && aiAnalysisResults.results.health_prediction.predictions.current_soh >= 80 && 'bg-yellow-500',
                    aiAnalysisResults.results.health_prediction.predictions.current_soh < 80 && 'bg-red-500',
                  )}
                />
              </div>
              <Badge
                variant="outline"
                className={cn(
                  aiAnalysisResults.results.health_prediction.predictions.current_soh >= 90 && 'bg-green-100 text-green-800',
                  aiAnalysisResults.results.health_prediction.predictions.current_soh < 90 && aiAnalysisResults.results.health_prediction.predictions.current_soh >= 80 && 'bg-yellow-100 text-yellow-800',
                  aiAnalysisResults.results.health_prediction.predictions.current_soh < 80 && 'bg-red-100 text-red-800',
                )}
              >
                {/* Puedes derivar el estado de salud aquí si no viene del backend */}
                {aiAnalysisResults.results.health_prediction.predictions.current_soh >= 90 ? 'Excelente' :
                 aiAnalysisResults.results.health_prediction.predictions.current_soh >= 80 ? 'Bueno' :
                 'Bajo'}
              </Badge>
            </div>
            
            {/* RUL Estimado */}
            <div className="bg-white p-3 rounded-md shadow-sm border border-green-100 flex flex-col justify-between">
              <p className="text-sm text-gray-500">Vida Útil Restante (RUL)</p>
              <p className="text-2xl font-bold text-green-600 mt-1">
                {aiAnalysisResults.results.health_prediction.predictions.rul_estimate} <span className="text-base font-normal">días</span>
              </p>
              <Badge variant="secondary" className="mt-2 text-sm bg-blue-100 text-blue-800">
                <Clock className="h-4 w-4 mr-1"/> Estimación
              </Badge>
            </div>

            {/* Confianza */}
            <div className="bg-white p-3 rounded-md shadow-sm border border-green-100 flex flex-col justify-between">
              <p className="text-sm text-gray-500">Confianza de Predicción</p>
              <p className="text-2xl font-bold text-blue-600 mt-1">
                {formatPercentage(aiAnalysisResults.results.health_prediction.confidence * 100, 1)}
              </p>
              <Progress
                value={aiAnalysisResults.results.health_prediction.confidence * 100} // Multiplicar por 100 para la barra de progreso
                className="w-full h-2 bg-blue-200"
                indicatorClassName="bg-blue-500"
              />
            </div>

            {/* SOH Predicho a 30 Días */}
            {aiAnalysisResults.results.health_prediction.predictions.predicted_soh_30_days !== undefined && (
              <div className="bg-white p-3 rounded-md shadow-sm border border-green-100">
                <p className="text-sm text-gray-500">SOH Predicho (30 días)</p>
                <p className="text-xl font-bold mt-1">
                  {formatPercentage(aiAnalysisResults.results.health_prediction.predictions.predicted_soh_30_days)}
                </p>
              </div>
            )}
            {/* SOH Predicho a 90 Días */}
            {aiAnalysisResults.results.health_prediction.predictions.predicted_soh_90_days !== undefined && (
              <div className="bg-white p-3 rounded-md shadow-sm border border-green-100">
                <p className="text-sm text-gray-500">SOH Predicho (90 días)</p>
                <p className="text-xl font-bold mt-1">
                  {formatPercentage(aiAnalysisResults.results.health_prediction.predictions.predicted_soh_90_days)}
                </p>
              </div>
            )}
            {/* Tasa de Degradación */}
            <div className="bg-white p-3 rounded-md shadow-sm border border-green-100">
              <p className="text-sm text-gray-500">Tasa de Degradación</p>
              <p className="text-xl font-bold mt-1 text-gray-700">
                {aiAnalysisResults.results.health_prediction.degradation_rate?.toExponential(2) || 'N/A'}
              </p>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mt-3">
            <span className="font-medium text-gray-700">Explicación:</span>{' '}
            {aiAnalysisResults.results.health_prediction.explanation?.method || 'No hay explicación disponible.'}
            {aiAnalysisResults.results.health_prediction.explanation?.factors?.length > 0 &&
              ` - Factores: ${aiAnalysisResults.results.health_prediction.explanation.factors.join(', ')}`}
          </p>
        </div>
      )}

      {(!aiAnalysisResults.results || (!aiAnalysisResults.results.fault_detection && !aiAnalysisResults.results.health_prediction)) && (
        <p className="text-muted-foreground text-center p-8 text-lg">
          No hay resultados de análisis de IA disponibles para mostrar.
          <br/>Por favor, asegúrate de que haya datos de batería y que el backend de análisis esté funcionando.
        </p>
      )}
    </CardContent>
  </Card>
)}

      {/* Sección de Registros de Mantenimiento */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center"><Wrench className="h-5 w-5 mr-2" /> Registros de Mantenimiento</CardTitle>
          <CardDescription>Historial de mantenimientos realizados en la batería.</CardDescription>
        </CardHeader>
        <CardContent>
          {loadingAdditionalData && !additionalDataError?.maintenance && <p>Cargando registros de mantenimiento...</p>}
          {additionalDataError?.maintenance && <p className="text-red-500">Error al cargar registros de mantenimiento: {additionalDataError.maintenance}</p>}
          {!loadingAdditionalData && maintenanceRecords.length === 0 && !additionalDataError?.maintenance && (
            <p className="text-muted-foreground">No hay registros de mantenimiento disponibles para esta batería.</p>
          )}
          {maintenanceRecords.length > 0 && (
            <div className="space-y-3">
              {maintenanceRecords.map((record) => (
                <div key={record.id} className="flex flex-col p-3 border border-border rounded-lg">
                  <p className="font-medium">{record.type || 'Mantenimiento'}</p>
                  <p className="text-sm text-muted-foreground mb-2">{record.description || 'Sin descripción.'}</p>
                  <p className="text-xs text-muted-foreground">
                    Fecha: {record.date ? formatDate(record.date) : 'N/A'} |
                    Realizado por: {record.performed_by || 'Desconocido'}
                  </p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Tarjeta de Datos Históricos y Gráficos */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center"><TrendingUp className="h-5 w-5 mr-2" /> Datos Históricos y Gráficos</CardTitle>
          <CardDescription>Visualización de métricas de la batería a lo largo del tiempo.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4 mb-6">
            {/* Selector de Fechas */}
            <div className="flex items-center space-x-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
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
                    <Button variant="outline" className="w-full" onClick={() => setDateRange({ from: last15Days, to: today })} >
                      Últimos 15 días
                    </Button>
                    <Button variant="outline" className="w-full" onClick={() => setDateRange({ from: lastMonth, to: today })} >
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
                  <BarChart2 className="h-4 w-4 mr-2" /> Visualizar Gráficos
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
                    {metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' ')}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            <Button onClick={handleApplyFilters}>Aplicar Filtros</Button>
            <Button variant="outline" onClick={handleClearFilters}>Limpiar Filtros</Button>
          </div>

          <Separator className="my-6" />

          {loadingAdditionalData && !additionalDataError?.historical && <p>Cargando datos históricos...</p>}
          {additionalDataError?.historical && <p className="text-red-500">Error al cargar datos históricos: {additionalDataError.historical}</p>}
          {!loadingAdditionalData && (!historicalData || historicalData.length === 0) && !additionalDataError?.historical && (
            <p className="text-muted-foreground text-center py-8">
              No hay datos históricos disponibles para el período seleccionado.
            </p>
          )}

          {/* Gráfico de SOC (Líneas) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('soc') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Estado de Carga (SOC)</h3>
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
                    label={{ value: 'SOC (%)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value)}%`, 'SOC']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="soc" stroke="#8884d8" name="Estado de Carga (SOC)" strokeWidth={2} />
                </LineChart>
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
                  <Line type="monotone" dataKey="current" stroke="#d0ed57" name="Corriente" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Gráfico de Ciclos (Barras) */}
          {historicalData && Array.isArray(historicalData) && historicalData.length > 0 && visibleCharts.includes('cycles') && (
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-2">Ciclos</h3>
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
                    label={{ value: 'Ciclos', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatNumber(value, 0)}`, 'Ciclos']}
                  />
                  <Legend />
                  <Bar dataKey="cycles" fill="#82c" name="Ciclos" />
                </BarChart>
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
                  <Line type="monotone" dataKey="voltage" stroke="#a4de6c" name="Voltaje" strokeWidth={2} />
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
                    tickFormatter={(value) => `${formatPercentage(value * 100)}`}
                    label={{ value: 'Eficiencia (%)', angle: -90, position: 'insideLeft', offset: 10 }}
                  />
                  <Tooltip
                    labelFormatter={(label) => `Fecha: ${new Date(label).toLocaleString()}`}
                    formatter={(value) => [`${formatPercentage(value * 100)}`, 'Eficiencia']}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="efficiency" stroke="#8dd1e1" name="Eficiencia" strokeWidth={2} />
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
                  <Line type="monotone" dataKey="internal_resistance" stroke="#c06c84" name="Resistencia Interna" strokeWidth={2} />
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
