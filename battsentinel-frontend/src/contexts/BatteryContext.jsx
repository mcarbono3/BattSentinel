// src/contexts/BatteryContext.jsx
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { batteryAPI } from '@/lib/api';
import { useAuth } from './AuthContext';

const BatteryContext = createContext({});

// Clave para localStorage
const HIDDEN_BATTERIES_KEY = 'hiddenBatteryIds';
const LOCAL_BATTERIES_KEY = 'localBatteries'; // Nueva clave para las baterías editadas
const REFRESH_INTERVAL_KEY = 'battSentinelRefreshIntervalMs';
const AUTO_REFRESH_ENABLED_KEY = 'battSentinelAutoRefreshEnabled';

export function BatteryProvider({ children }) {
  // Inicializar baterías desde localStorage
  const [batteries, setBatteries] = useState(() => {
    try {
      const storedBatteries = localStorage.getItem(LOCAL_BATTERIES_KEY);
      return storedBatteries ? JSON.parse(storedBatteries) : [];
    } catch (e) {
      console.error("Failed to parse batteries from localStorage", e);
      return [];
    }
  });

  const [selectedBattery, setSelectedBattery] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { isAuthenticated } = useAuth();
  // Nuevo estado para las IDs de baterías ocultas
  const [hiddenBatteryIds, setHiddenBatteryIds] = useState(() => {
    try {
      const storedIds = localStorage.getItem(HIDDEN_BATTERIES_KEY);
      return storedIds ? new Set(JSON.parse(storedIds)) : new Set();
    } catch (e) {
      console.error("Failed to parse hidden battery IDs from localStorage", e);
      return new Set();
    }
  });

  // Nuevo estado para el intervalo de refresco, inicializado desde localStorage
  const [refreshInterval, setRefreshInterval] = useState(() => {
    try {
      const storedInterval = localStorage.getItem(REFRESH_INTERVAL_KEY);
      const parsedInterval = storedInterval ? parseInt(storedInterval, 10) : 420000; // Valor por defecto: 7 minutos
      return (!isNaN(parsedInterval) && parsedInterval > 0) ? parsedInterval : 420000;
    } catch (e) {
      console.error("Error reading refresh interval from localStorage, using default:", e);
      return 420000; // Valor por defecto en caso de error
    }
  });

  // <-- NUEVO ESTADO para autoRefreshEnabled, inicializado desde localStorage
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(() => {
    try {
      const storedEnabled = localStorage.getItem(AUTO_REFRESH_ENABLED_KEY);
      return storedEnabled !== null ? JSON.parse(storedEnabled) : true; // Por defecto: habilitado
    } catch (e) {
      console.error("Error reading autoRefreshEnabled from localStorage, using default:", e);
      return true;
    }
  });

  // Guardar hiddenBatteryIds en localStorage cada vez que cambie
  useEffect(() => {
    localStorage.setItem(HIDDEN_BATTERIES_KEY, JSON.stringify(Array.from(hiddenBatteryIds)));
  }, [hiddenBatteryIds]);

  // Guardar batteries en localStorage cada vez que cambie
  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_BATTERIES_KEY, JSON.stringify(batteries));
    } catch (e) {
      console.error("Failed to save batteries to localStorage", e);
    }
  }, [batteries]);

  // Guardar refreshInterval en localStorage cada vez que cambie
  useEffect(() => {
    try {
      localStorage.setItem(REFRESH_INTERVAL_KEY, String(refreshInterval));
    } catch (e) {
      console.error("Failed to save refresh interval to localStorage", e);
    }
  }, [refreshInterval]);

  // <-- NUEVO useEffect para guardar autoRefreshEnabled en localStorage
  useEffect(() => {
    try {
      localStorage.setItem(AUTO_REFRESH_ENABLED_KEY, JSON.stringify(autoRefreshEnabled));
    } catch (e) {
      console.error("Failed to save autoRefreshEnabled to localStorage", e);
    }
  }, [autoRefreshEnabled]);

  // Cargar baterías desde la API y fusionar con datos locales
  // Se llama cada vez que el estado de autenticación cambia, para asegurar que los datos
  // estén actualizados y fusionados correctamente con las ediciones locales.
  useEffect(() => {
    if (isAuthenticated) {
      loadBatteriesFromAPI();
    }
  }, [isAuthenticated]); // Removido 'batteries.length' de las dependencias

  const loadBatteriesFromAPI = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await batteryAPI.getBatteries();
      if (response.success) {
        const apiBatteries = response.data;

        // Crear un mapa de las baterías actuales (que incluyen ediciones locales) por ID
        // Esto usa el estado 'batteries' actual, que ya ha sido inicializado desde localStorage
        const currentLocalBatteriesMap = new Map(batteries.map(b => [b.id, b]));

        const finalBatteriesMap = new Map();

        // Primero, añadir todas las baterías de la API al mapa
        apiBatteries.forEach(apiBattery => {
          finalBatteriesMap.set(apiBattery.id, apiBattery);
        });

        // Luego, fusionar con las baterías locales. Si hay una ID coincidente,
        // la versión local (con sus ediciones) sobrescribe la versión de la API.
        // Si es una batería nueva que solo existe localmente, también se añade.
        currentLocalBatteriesMap.forEach(localBattery => {
          finalBatteriesMap.set(localBattery.id, localBattery);
        });

        setBatteries(Array.from(finalBatteriesMap.values()));
      } else {
        setError(response.error);
        // Si hay un error al cargar de la API, se asegura de usar lo que esté en localStorage
        const storedBatteries = localStorage.getItem(LOCAL_BATTERIES_KEY);
        if (storedBatteries) {
          setBatteries(JSON.parse(storedBatteries));
        }
      }
    } catch (error) {
      setError('Error al cargar las baterías');
      console.error('Error loading batteries:', error);
      // Si la llamada a la API falla completamente, recurre a localStorage
      const storedBatteries = localStorage.getItem(LOCAL_BATTERIES_KEY);
      if (storedBatteries) {
        setBatteries(JSON.parse(storedBatteries));
      }
    } finally {
      setLoading(false);
    }
  };

  const createBattery = async (batteryData) => {
    try {
      const response = await batteryAPI.createBattery(batteryData);
      if (response.success) {
        setBatteries(prev => [...prev, response.data]);
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      return { success: false, error: 'Error al crear la batería' };
    }
  };

  // Función updateBattery modificada para persistencia solo en frontend
  const updateBattery = async (id, batteryData) => {
    try {
      const updatedBattery = { ...batteryData, id }; // Asegurar que el ID esté presente
      setBatteries(prev => prev.map(b => b.id === id ? updatedBattery : b));
      setSelectedBattery(updatedBattery); // Actualizar si es la batería seleccionada
      return { success: true, data: updatedBattery };
    } catch (error) {
      console.error('Error al actualizar la batería localmente:', error);
      return { success: false, error: 'Error al actualizar la batería localmente' };
    }
  };

  const deleteBattery = async (id) => {
    try {
      const response = await batteryAPI.deleteBattery(id);
      if (response.success) {
        setBatteries(prev => prev.filter(b => b.id !== id));
        if (selectedBattery?.id === id) {
          setSelectedBattery(null); // Si se elimina la batería seleccionada, deseleccionar
        }
        // Asegurarse de quitarla también de la lista de ocultas si lo estaba
        setHiddenBatteryIds(prev => {
          const newSet = new Set(prev);
          newSet.delete(id);
          return newSet;
        });
        return { success: true };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      return { success: false, error: 'Error al eliminar la batería' };
    }
  };

  const getBatteryById = useCallback(async (batteryId) => {
    try {
      setLoading(true);
      setError(null);
      // Intentar obtener de las baterías ya cargadas localmente
      const foundBattery = batteries.find(b => b.id === batteryId);
      if (foundBattery) {
        setSelectedBattery(foundBattery);
        return { success: true, data: foundBattery };
      }

      // Si no se encuentra localmente, intentar del API (para datos iniciales)
      const response = await batteryAPI.getBattery(batteryId);
      if (response.success) {
        setSelectedBattery(response.data);
        return { success: true, data: response.data };
      } else {
        setError(response.error);
        return { success: false, error: response.error };
      }
    } catch (error) {
      setError('Error al obtener la batería');
      return { success: false, error: 'Error al obtener la batería' };
    } finally {
      setLoading(false);
    }
  }, [batteries]);

  const getBatteryData = useCallback(async (batteryId) => {
    try {
      const response = await batteryAPI.getBatteryData(batteryId);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching battery data:', error);
      return { success: false, error: 'Error al obtener los datos de la batería' };
    }
  }, []);

  const getBatterySummary = useCallback(async () => {
    try {
      const response = await batteryAPI.getBatterySummary();
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching battery summary:', error);
      return { success: false, error: 'Error al obtener el resumen de baterías' };
    }
  }, []);

  const uploadBatteryData = useCallback(async (batteryId, data) => {
    try {
      const response = await batteryAPI.uploadBatteryData(batteryId, data);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error uploading battery data:', error);
      return { success: false, error: 'Error al subir datos de la batería' };
    }
  }, []);

  const uploadThermalImage = useCallback(async (batteryId, imageFile) => {
    try {
      const response = await batteryAPI.uploadThermalImage(batteryId, imageFile);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error uploading thermal image:', error);
      return { success: false, error: 'Error al subir imagen térmica' };
    }
  }, []);

  const getThermalImages = useCallback(async (batteryId) => {
    try {
      const response = await batteryAPI.getThermalImages(batteryId);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching thermal images:', error);
      return { success: false, error: 'Error al obtener imágenes térmicas' };
    }
  }, []);

  const addBatteryData = useCallback(async (batteryId, data) => {
    try {
      const response = await batteryAPI.addBatteryData(batteryId, data);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error adding battery data:', error);
      return { success: false, error: 'Error al añadir datos de la batería' };
    }
  }, []);

  // --- Funciones corregidas para datos detallados de batería ---
  const getAlertsByBatteryId = useCallback(async (batteryId) => {
    try {
      const response = await batteryAPI.getAlertsByBatteryId(batteryId);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching battery alerts:', error);
      return { success: false, error: 'Error al obtener alertas de la batería' };
    }
  }, []);

  const getAnalysisResultsByBatteryId = useCallback(async (batteryId) => {
    try {
      const response = await batteryAPI.getAnalysisResultsByBatteryId(batteryId);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching battery analysis results:', error);
      return { success: false, error: 'Error al obtener resultados de análisis' };
    }
  }, []);

  const getMaintenanceRecordsByBatteryId = useCallback(async (batteryId) => {
    try {
      const response = await batteryAPI.getMaintenanceRecordsByBatteryId(batteryId);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching maintenance records:', error);
      return { success: false, error: 'Error al obtener registros de mantenimiento' };
    }
  }, []);

  const addMaintenanceRecord = useCallback(async (batteryId, recordData) => {
    try {
      const response = await batteryAPI.addMaintenanceRecord(batteryId, recordData);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error adding maintenance record:', error);
      return { success: false, error: 'Error al añadir registro de mantenimiento' };
    }
  }, []);

  const getBatteryHistoricalData = useCallback(async (batteryId, params) => {
    try {
      const response = await batteryAPI.getHistoricalData(batteryId, params);
      if (response.success) {
        return { success: true, data: response.data };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Error fetching historical battery data:', error);
      return { success: false, error: 'Error al obtener datos históricos de la batería' };
    }
  }, []);

  // Nueva función para ocultar/mostrar baterías
  const toggleBatteryVisibility = useCallback((batteryId) => {
    setHiddenBatteryIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(batteryId)) {
        newSet.delete(batteryId);
      } else {
        newSet.add(batteryId);
      }
      return newSet;
    });
  }, []);

  // Función para obtener las baterías visibles (no ocultas)
  const getVisibleBatteries = useCallback(() => {
    return batteries.filter(battery => !hiddenBatteryIds.has(battery.id));
  }, [batteries, hiddenBatteryIds]);

  const value = {
    batteries,
    selectedBattery,
    loading,
    error,
    setSelectedBattery,
    loadBatteries: loadBatteriesFromAPI,
    createBattery,
    updateBattery,
    deleteBattery,
    getBatteryById,
    getBatteryData,
    getBatterySummary,
    uploadBatteryData,
    uploadThermalImage,
    getThermalImages,
    addBatteryData,
    batteryCount: batteries.length,
    activeBatteries: batteries.filter(b => b.active !== false),
    criticalBatteries: batteries.filter(b => {
      // Lógica para determinar baterías críticas
      return false;
    }),
    getAlertsByBatteryId,
    getAnalysisResultsByBatteryId,
    getMaintenanceRecordsByBatteryId,
    addMaintenanceRecord,
    getBatteryHistoricalData,
    // Nuevas propiedades y funciones para ocultar/mostrar
    hiddenBatteryIds,
    toggleBatteryVisibility,
    getVisibleBatteries,
    refreshInterval,
    setRefreshInterval,
    autoRefreshEnabled, // <-- EXPUERTO autoRefreshEnabled en el contexto
    setAutoRefreshEnabled, // 
  };

  return (
    <BatteryContext.Provider value={value}>
      {children}
    </BatteryContext.Provider>
  );
}

export function useBattery() {
  const context = useContext(BatteryContext);
  if (context === undefined) {
    throw new Error('useBattery must be used within a BatteryProvider');
  }
  return context;
}
