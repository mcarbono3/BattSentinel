// src/contexts/BatteryContext.jsx
import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { batteryAPI } from '@/lib/api'
import { useAuth } from './AuthContext'

const BatteryContext = createContext({})

// Clave para localStorage
const HIDDEN_BATTERIES_KEY = 'hiddenBatteryIds';

export function BatteryProvider({ children }) {
  const [batteries, setBatteries] = useState([])
  const [selectedBattery, setSelectedBattery] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const { isAuthenticated } = useAuth()
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

  // Guardar hiddenBatteryIds en localStorage cada vez que cambie
  useEffect(() => {
    localStorage.setItem(HIDDEN_BATTERIES_KEY, JSON.stringify(Array.from(hiddenBatteryIds)));
  }, [hiddenBatteryIds]);

  // Cargar baterías cuando el usuario esté autenticado
  useEffect(() => {
    if (isAuthenticated) {
      loadBatteries()
    }
  }, [isAuthenticated])

  const loadBatteries = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await batteryAPI.getBatteries()
      if (response.success) {
        setBatteries(response.data)
      } else {
        setError(response.error)
      }
    } catch (error) {
      setError('Error al cargar las baterías')
      console.error('Error loading batteries:', error)
    } finally {
      setLoading(false)
    }
  }

  const createBattery = async (batteryData) => {
    try {
      const response = await batteryAPI.createBattery(batteryData)
      if (response.success) {
        setBatteries(prev => [...prev, response.data])
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al crear la batería' }
    }
  }

  const updateBattery = async (id, batteryData) => {
    try {
      const response = await batteryAPI.updateBattery(id, batteryData)
      if (response.success) {
        setBatteries(prev => prev.map(b => b.id === id ? response.data : b))
        setSelectedBattery(response.data); // Actualizar si es la batería seleccionada
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al actualizar la batería' }
    }
  }

  const deleteBattery = async (id) => {
    try {
      const response = await batteryAPI.deleteBattery(id)
      if (response.success) {
        setBatteries(prev => prev.filter(b => b.id !== id))
        if (selectedBattery?.id === id) {
          setSelectedBattery(null); // Si se elimina la batería seleccionada, deseleccionar
        }
        // Asegurarse de quitarla también de la lista de ocultas si lo estaba
        setHiddenBatteryIds(prev => {
          const newSet = new Set(prev);
          newSet.delete(id);
          return newSet;
        });
        return { success: true }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al eliminar la batería' }
    }
  }

  const getBatteryById = useCallback(async (batteryId) => {
    try {
      setLoading(true); // Se agrega para mostrar carga al obtener una batería
      setError(null);
      const response = await batteryAPI.getBattery(batteryId)
      if (response.success) {
        setSelectedBattery(response.data); // Actualiza la batería seleccionada
        return { success: true, data: response.data }
      } else {
        setError(response.error);
        return { success: false, error: response.error }
      }
    } catch (error) {
      setError('Error al obtener la batería');
      return { success: false, error: 'Error al obtener la batería' }
    } finally {
        setLoading(false); // Finaliza la carga
    }
  }, []);


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
    loadBatteries,
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
      return false
    }),
    getAlertsByBatteryId,
    getAnalysisResultsByBatteryId,
    getMaintenanceRecordsByBatteryId,
    addMaintenanceRecord,
    getBatteryHistoricalData,
    // Nuevas propiedades y funciones para ocultar/mostrar
    hiddenBatteryIds,
    toggleBatteryVisibility,
    getVisibleBatteries, // Exportar para usar en Dashboard
  }

  return (
    <BatteryContext.Provider value={value}>
      {children}
    </BatteryContext.Provider>
  )
}

export function useBattery() {
  const context = useContext(BatteryContext)
  if (context === undefined) {
    throw new Error('useBattery must be used within a BatteryProvider')
  }
  return context
}
