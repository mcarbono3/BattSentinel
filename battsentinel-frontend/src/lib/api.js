// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api'

// Helper function to get auth headers
const getAuthHeaders = () => {
  const token = localStorage.getItem('battsentinel-token')
  return {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` })
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

// Helper function to make API requests
const apiRequest = async (endpoint, options = {}) => {
  try {
    const url = `${API_BASE_URL}${endpoint}`
    const config = {
      headers: getAuthHeaders(),
      ...options
    }

    const response = await fetch(url, config)
    return await handleResponse(response)
  } catch (error) {
    console.error(`API request failed: ${endpoint}`, error)
    throw error
  }
}

// Helper function for file uploads
const uploadFile = async (endpoint, file, additionalData = {}) => {
  try {
    const formData = new FormData()
    formData.append('file', file)
    
    Object.keys(additionalData).forEach(key => {
      formData.append(key, additionalData[key])
    })

    const token = localStorage.getItem('battsentinel-token')
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        ...(token && { Authorization: `Bearer ${token}` })
      },
      body: formData
    })

    return await handleResponse(response)
  } catch (error) {
    console.error(`File upload failed: ${endpoint}`, error)
    throw error
  }
}

// Authentication API
export const authAPI = {
  login: (credentials) => 
    apiRequest('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials)
    }),

  register: (userData) => 
    apiRequest('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData)
    }),

  logout: () => 
    apiRequest('/auth/logout', { method: 'POST' }),

  verifyToken: (token) => 
    apiRequest('/auth/verify-token', {
      method: 'POST',
      body: JSON.stringify({ token })
    }),

  refreshToken: (token) => 
    apiRequest('/auth/refresh-token', {
      method: 'POST',
      body: JSON.stringify({ token })
    }),

  changePassword: (userId, passwordData) => 
    apiRequest('/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, ...passwordData })
    }),

  updateProfile: (userId, profileData) => 
    apiRequest(`/auth/profile/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(profileData)
    }),

  getUsers: (params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/auth/users${queryString ? `?${queryString}` : ''}`)
  },

  activateUser: (userId) => 
    apiRequest(`/auth/users/${userId}/activate`, { method: 'POST' }),

  deactivateUser: (userId) => 
    apiRequest(`/auth/users/${userId}/deactivate`, { method: 'POST' })
}

// Battery API
export const batteryAPI = {
  getBatteries: () => 
    apiRequest('/battery/batteries'),

  getBattery: (batteryId) => 
    apiRequest(`/battery/batteries/${batteryId}`),

  createBattery: (batteryData) => 
    apiRequest('/battery/batteries', {
      method: 'POST',
      body: JSON.stringify(batteryData)
    }),

  updateBattery: (batteryId, batteryData) => 
    apiRequest(`/battery/batteries/${batteryId}`, {
      method: 'PUT',
      body: JSON.stringify(batteryData)
    }),

  deleteBattery: (batteryId) => 
    apiRequest(`/battery/batteries/${batteryId}`, { method: 'DELETE' }),

  getBatteryData: (batteryId, params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/battery/batteries/${batteryId}/data${queryString ? `?${queryString}` : ''}`)
  },

  getBatterySummary: (batteryId) => 
    apiRequest(`/battery/batteries/${batteryId}/summary`),

  uploadData: (batteryId, file) => 
    uploadFile(`/battery/batteries/${batteryId}/upload-data`, file),

  uploadThermalImage: (batteryId, file) => 
    uploadFile(`/battery/batteries/${batteryId}/upload-thermal`, file),

  getThermalImages: (batteryId) => 
    apiRequest(`/battery/batteries/${batteryId}/thermal-images`),

  addData: (batteryId, dataPoint) => 
    apiRequest(`/battery/batteries/${batteryId}/add-data`, {
      method: 'POST',
      body: JSON.stringify(dataPoint)
    })
}

// AI Analysis API
export const aiAPI = {
  analyzeBattery: (batteryId, analysisParams = {}) => 
    apiRequest(`/ai/analyze/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(analysisParams)
    }),

  detectFaults: (batteryId) => 
    apiRequest(`/ai/fault-detection/${batteryId}`, { method: 'POST' }),

  predictHealth: (batteryId) => 
    apiRequest(`/ai/health-prediction/${batteryId}`, { method: 'POST' }),

  detectAnomalies: (batteryId) => 
    apiRequest(`/ai/anomaly-detection/${batteryId}`, { method: 'POST' }),

  getAnalysesHistory: (batteryId, params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/ai/analyses/${batteryId}${queryString ? `?${queryString}` : ''}`)
  },

  getModelInfo: () => 
    apiRequest('/ai/model-info')
}

// Digital Twin API
export const digitalTwinAPI = {
  createTwin: (batteryId) => 
    apiRequest(`/twin/create/${batteryId}`, { method: 'POST' }),

  simulateResponse: (batteryId, simulationParams) => 
    apiRequest(`/twin/simulate/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(simulationParams)
    }),

  getTwinState: (batteryId) => 
    apiRequest(`/twin/state/${batteryId}`),

  getTwinParameters: (batteryId) => 
    apiRequest(`/twin/parameters/${batteryId}`),

  predictFutureBehavior: (batteryId, predictionParams) => 
    apiRequest(`/twin/predict/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(predictionParams)
    }),

  optimizeUsage: (batteryId, optimizationParams) => 
    apiRequest(`/twin/optimize/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify(optimizationParams)
    }),

  calibrateModel: (batteryId) => 
    apiRequest(`/twin/calibrate/${batteryId}`, { method: 'POST' }),

  compareScenarios: (batteryId, scenarios) => 
    apiRequest(`/twin/compare/${batteryId}`, {
      method: 'POST',
      body: JSON.stringify({ scenarios })
    })
}

// Notifications API
export const notificationsAPI = {
  sendAlert: (alertData) => 
    apiRequest('/notifications/send-alert', {
      method: 'POST',
      body: JSON.stringify(alertData)
    }),

  getBatteryAlerts: (batteryId, params = {}) => {
    const queryString = new URLSearchParams(params).toString()
    return apiRequest(`/notifications/alerts/${batteryId}${queryString ? `?${queryString}` : ''}`)
  },

  acknowledgeAlert: (alertId) => 
    apiRequest(`/notifications/alerts/${alertId}/acknowledge`, { method: 'POST' }),

  resolveAlert: (alertId) => 
    apiRequest(`/notifications/alerts/${alertId}/resolve`, { method: 'POST' }),

  testNotification: (notificationData) => 
    apiRequest('/notifications/test-notification', {
      method: 'POST',
      body: JSON.stringify(notificationData)
    }),

  getNotificationSettings: (userId) => 
    apiRequest(`/notifications/settings/${userId}`),

  updateNotificationSettings: (userId, settings) => 
    apiRequest(`/notifications/settings/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(settings)
    })
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
      const response = await fetch(`${API_BASE_URL.replace('/api', '')}/`)
      return response.ok
    } catch (error) {
      return false
    }
  },

  // Get API base URL
  getAPIBaseURL: () => API_BASE_URL
}

// Export all APIs
export default {
  auth: authAPI,
  battery: batteryAPI,
  ai: aiAPI,
  digitalTwin: digitalTwinAPI,
  notifications: notificationsAPI,
  utils
}
