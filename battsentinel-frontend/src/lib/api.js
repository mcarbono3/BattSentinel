// API Configuration - Sin autenticación
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

// Helper function to get headers - Sin autenticación
const getHeaders = () => {
  return {
    'Content-Type': 'application/json'
  }
}

// Helper function to handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
  }
  return response.json()
}

// Helper function to make API requests - Sin autenticación
const apiRequest = async (endpoint, options = {}) => {
  try {
    const url = `${API_BASE_URL}${endpoint}`
    const config = {
      headers: getHeaders(),
      ...options
    }

    const response = await fetch(url, config)
    return await handleResponse(response)
  } catch (error) {
    console.error(`API request failed: ${endpoint}`, error)
    throw error
  }
}

// Helper function for file uploads - Sin autenticación
const uploadFile = async (endpoint, file, additionalData = {}) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    
    Object.keys(additionalData).forEach(key => {
      formData.append(key, additionalData[key])
    })

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData
    })

    return await handleResponse(response)
  } catch (error) {
    console.error(`File upload failed: ${endpoint}`, error)
    throw error
  }
}

// Battery API - Rutas actualizadas sin autenticación
export const batteryAPI = {
  getBatteries: () => 
    apiRequest('/api/batteries'),

  getBattery: (batteryId) => 
    apiRequest(`/api/batteries/${batteryId}`),

  createBattery: (batteryData) => 
    apiRequest('/api/batteries', {
      method: 'POST',
      body: JSON.stringify(batteryData)
    }),

  updateBattery: (batteryId, batteryData) => 
    apiRequest(`/api/batteries/${batteryId}`, {
      method: 'PUT',
      body: JSON.stringify(batteryData)
    }),

  deleteBattery: (batteryId) => 
    apiRequest(`/api/batteries/${batteryId}`, { method: 'DELETE' }),

  getBatteryData: (batteryId, params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/api/batteries/${batteryId}/data${queryString ? `?${queryString}` : ''}`)
  },

  getBatterySummary: (batteryId) => 
    apiRequest(`/api/batteries/${batteryId}/summary`),

  uploadData: (batteryId, file) => 
    uploadFile(`/api/batteries/${batteryId}/upload-data`, file),

  uploadThermalImage: (batteryId, file) => 
    uploadFile(`/api/batteries/${batteryId}/upload-thermal`, file),

  getThermalImages: (batteryId) => 
    apiRequest(`/api/batteries/${batteryId}/thermal-images`),

  addData: (batteryId, dataPoint) => 
    apiRequest(`/api/batteries/${batteryId}/add-data`, {
      method: 'POST',
      body: JSON.stringify(dataPoint)
    }),

  // Nuevos endpoints para datos en tiempo real
  getRealTimeData: () => 
    apiRequest('/api/battery/real-time'),

  updateWithRealTimeData: (batteryId) => 
    apiRequest(`/api/batteries/${batteryId}/update-real-time`, {
      method: 'POST'
    })
}

// AI Analysis API - Rutas actualizadas sin autenticación
export const aiAPI = {
  analyzeBattery: (batteryId, analysisParams = {}) => 
    apiRequest(`/api/analyze/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(analysisParams)
    }),

  detectFaults: (batteryId) => 
    apiRequest(`/api/fault-detection/${batteryId}`, { method: 'POST' }),

  predictHealth: (batteryId) => 
    apiRequest(`/api/health-prediction/${batteryId}`, { method: 'POST' }),

  detectAnomalies: (batteryId) => 
    apiRequest(`/api/anomaly-detection/${batteryId}`, { method: 'POST' }),

  getAnalysesHistory: (batteryId, params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/api/analyses/${batteryId}${queryString ? `?${queryString}` : ''}`)
  },

  getModelInfo: () => 
    apiRequest('/api/model-info')
}

// Digital Twin API - Rutas actualizadas sin autenticación
export const digitalTwinAPI = {
  createTwin: (batteryId) => 
    apiRequest(`/api/twin/create/${batteryId}`, { method: 'POST' }),

  simulateResponse: (batteryId, simulationParams) => 
    apiRequest(`/api/twin/simulate/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(simulationParams)
    }),

  getTwinState: (batteryId) => 
    apiRequest(`/api/twin/state/${batteryId}`),

  getTwinParameters: (batteryId) => 
    apiRequest(`/api/twin/parameters/${batteryId}`),

  predictFutureBehavior: (batteryId, predictionParams) => 
    apiRequest(`/api/twin/predict/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(predictionParams)
    }),

  optimizeUsage: (batteryId, optimizationParams) => 
    apiRequest(`/api/twin/optimize/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(optimizationParams)
    }),

  calibrateModel: (batteryId) => 
    apiRequest(`/api/twin/calibrate/${batteryId}`, { method: 'POST' }),

  compareScenarios: (batteryId, scenarios) => 
    apiRequest(`/api/twin/compare/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify({ scenarios })
    })
}

// Notifications API - Rutas actualizadas sin autenticación
export const notificationsAPI = {
  sendAlert: (alertData) => 
    apiRequest('/api/notifications/send-alert', {
      method: 'POST',
      body: JSON.stringify(alertData)
    }),

  getBatteryAlerts: (batteryId, params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/api/notifications/alerts/${batteryId}${queryString ? `?${queryString}` : ''}`)
  },

  acknowledgeAlert: (alertId) => 
    apiRequest(`/api/notifications/alerts/${alertId}/acknowledge`, { method: 'POST' }),

  resolveAlert: (alertId) => 
    apiRequest(`/api/notifications/alerts/${alertId}/resolve`, { method: 'POST' }),

  testNotification: (notificationData) => 
    apiRequest('/api/notifications/test-notification', {
      method: 'POST',
      body: JSON.stringify(notificationData)
    }),

  getNotificationSettings: (userId) => 
    apiRequest(`/api/notifications/settings/${userId}`),

  updateNotificationSettings: (userId, settings) => 
    apiRequest(`/api/notifications/settings/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(settings)
    })
}

// System API - Nuevos endpoints
export const systemAPI = {
  // Endpoint de salud del sistema
  getHealth: () => 
    apiRequest('/api/health'),

  // Obtener datos de batería en tiempo real del sistema
  getRealTimeBatteryData: () => 
    apiRequest('/api/battery/real-time'),

  // Obtener información del sistema
  getSystemInfo: () => 
    apiRequest('/api/system/info')
}

// Utility functions
export const utils = {
  // Format date for API
  formatDateForAPI: (date) => {
    if (!date) return null
    return date instanceof Date ? date.toISOString() : date
  },

  // Parse API date
  parseAPIDate: (dateString) => {
    if (!dateString) return null
    return new Date(dateString)
  },

  // Check if API is available
  checkAPIHealth: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`)
      return response.ok
    } catch (error) {
      return false
    }
  },

  // Get API base URL
  getAPIBaseURL: () => API_BASE_URL,

  // Simulate real-time data for demo purposes
  generateMockBatteryData: () => {
    const now = new Date()
    return {
      timestamp: now.toISOString(),
      voltage: 12.0 + (Math.random() - 0.5) * 0.5,
      current: 2.5 + (Math.random() - 0.5) * 1.0,
      temperature: 25 + (Math.random() - 0.5) * 10,
      soc: Math.max(20, Math.min(100, 75 + (Math.random() - 0.5) * 30)),
      soh: Math.max(70, Math.min(100, 85 + (Math.random() - 0.5) * 20)),
      cycles: 150 + Math.floor(Math.random() * 100)
    }
  }
}

// Export all APIs - Sin auth
export default {
  battery: batteryAPI,
  ai: aiAPI,
  digitalTwin: digitalTwinAPI,
  notifications: notificationsAPI,
  system: systemAPI,
  utils
}

