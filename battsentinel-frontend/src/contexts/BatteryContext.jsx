import { createContext, useContext, useState, useEffect } from 'react'
import { batteryAPI } from '@/lib/api'
import { useAuth } from './AuthContext'

const BatteryContext = createContext({})

export function BatteryProvider({ children }) {
  const [batteries, setBatteries] = useState([])
  const [selectedBattery, setSelectedBattery] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const { isAuthenticated } = useAuth()

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

  const updateBattery = async (batteryId, batteryData) => {
    try {
      const response = await batteryAPI.updateBattery(batteryId, batteryData)
      if (response.success) {
        setBatteries(prev => 
          prev.map(battery => 
            battery.id === batteryId ? response.data : battery
          )
        )
        if (selectedBattery?.id === batteryId) {
          setSelectedBattery(response.data)
        }
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al actualizar la batería' }
    }
  }

  const deleteBattery = async (batteryId) => {
    try {
      const response = await batteryAPI.deleteBattery(batteryId)
      if (response.success) {
        setBatteries(prev => prev.filter(battery => battery.id !== batteryId))
        if (selectedBattery?.id === batteryId) {
          setSelectedBattery(null)
        }
        return { success: true }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al eliminar la batería' }
    }
  }

  const getBatteryById = async (batteryId) => {
    try {
      const response = await batteryAPI.getBattery(batteryId)
      if (response.success) {
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al obtener la batería' }
    }
  }

  const getBatteryData = async (batteryId, params = {}) => {
    try {
      const response = await batteryAPI.getBatteryData(batteryId, params)
      if (response.success) {
        return { success: true, data: response.data, pagination: response.pagination }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al obtener los datos de la batería' }
    }
  }

  const getBatterySummary = async (batteryId) => {
    try {
      const response = await batteryAPI.getBatterySummary(batteryId)
      if (response.success) {
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al obtener el resumen de la batería' }
    }
  }

  const uploadBatteryData = async (batteryId, file) => {
    try {
      const response = await batteryAPI.uploadData(batteryId, file)
      if (response.success) {
        // Recargar datos de la batería después de la carga
        await loadBatteries()
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al cargar los datos' }
    }
  }

  const uploadThermalImage = async (batteryId, file) => {
    try {
      const response = await batteryAPI.uploadThermalImage(batteryId, file)
      if (response.success) {
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al cargar la imagen térmica' }
    }
  }

  const getThermalImages = async (batteryId) => {
    try {
      const response = await batteryAPI.getThermalImages(batteryId)
      if (response.success) {
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al obtener las imágenes térmicas' }
    }
  }

  const addBatteryData = async (batteryId, dataPoint) => {
    try {
      const response = await batteryAPI.addData(batteryId, dataPoint)
      if (response.success) {
        return { success: true, data: response.data }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error al agregar los datos' }
    }
  }

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
    // Computed values
    batteryCount: batteries.length,
    activeBatteries: batteries.filter(b => b.active !== false),
    criticalBatteries: batteries.filter(b => {
      // Lógica para determinar baterías críticas
      return false // Placeholder
    })
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

