import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

// Utility function to merge Tailwind classes
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

// Date formatting utilities
export const formatDate = (date, options = {}) => {
  if (!date) return ''
  
  const dateObj = date instanceof Date ? date : new Date(date)
  
  const defaultOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...options
  }
  
  return dateObj.toLocaleDateString('es-ES', defaultOptions)
}

export const formatDateTime = (date, options = {}) => {
  if (!date) return ''
  
  const dateObj = date instanceof Date ? date : new Date(date)
  
  const defaultOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...options
  }
  
  return dateObj.toLocaleDateString('es-ES', defaultOptions)
}

export const formatTime = (date) => {
  if (!date) return ''
  
  const dateObj = date instanceof Date ? date : new Date(date)
  return dateObj.toLocaleTimeString('es-ES', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

export const getRelativeTime = (date) => {
  if (!date) return ''
  
  const dateObj = date instanceof Date ? date : new Date(date)
  const now = new Date()
  const diffInSeconds = Math.floor((now - dateObj) / 1000)
  
  if (diffInSeconds < 60) return 'Hace unos segundos'
  if (diffInSeconds < 3600) return `Hace ${Math.floor(diffInSeconds / 60)} minutos`
  if (diffInSeconds < 86400) return `Hace ${Math.floor(diffInSeconds / 3600)} horas`
  if (diffInSeconds < 2592000) return `Hace ${Math.floor(diffInSeconds / 86400)} días`
  
  return formatDate(dateObj)
}

// Number formatting utilities
export const formatNumber = (number, decimals = 2) => {
  if (number === null || number === undefined) return '--'
  return Number(number).toFixed(decimals)
}

export const formatPercentage = (value, decimals = 1) => {
  if (value === null || value === undefined) return '--'
  return `${Number(value).toFixed(decimals)}%`
}

export const formatCurrency = (amount, currency = 'EUR') => {
  if (amount === null || amount === undefined) return '--'
  return new Intl.NumberFormat('es-ES', {
    style: 'currency',
    currency: currency
  }).format(amount)
}

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// Battery-specific utilities
export const getBatteryStatusColor = (status) => {
  const statusColors = {
    excellent: 'text-green-600 bg-green-50 border-green-200',
    good: 'text-blue-600 bg-blue-50 border-blue-200',
    fair: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    poor: 'text-orange-600 bg-orange-50 border-orange-200',
    critical: 'text-red-600 bg-red-50 border-red-200'
  }
  return statusColors[status] || 'text-gray-600 bg-gray-50 border-gray-200'
}

export const getAlertSeverityColor = (severity) => {
  const severityColors = {
    low: 'text-green-600 bg-green-50 border-green-200',
    medium: 'text-yellow-600 bg-yellow-50 border-yellow-200',
    high: 'text-orange-600 bg-orange-50 border-orange-200',
    critical: 'text-red-600 bg-red-50 border-red-200'
  }
  return severityColors[severity] || 'text-gray-600 bg-gray-50 border-gray-200'
}

export const getBatteryHealthPercentage = (soh) => {
  if (soh === null || soh === undefined) return 0
  return Math.max(0, Math.min(100, Number(soh)))
}

export const getBatteryChargeLevel = (soc) => {
  if (soc === null || soc === undefined) return 0
  return Math.max(0, Math.min(100, Number(soc)))
}

export const calculateRemainingTime = (soc, current, capacity) => {
  if (!soc || !current || !capacity || current <= 0) return null
  
  const remainingCapacity = (soc / 100) * capacity
  const hoursRemaining = remainingCapacity / current
  
  if (hoursRemaining < 1) {
    return `${Math.round(hoursRemaining * 60)} min`
  } else if (hoursRemaining < 24) {
    return `${Math.round(hoursRemaining)} h`
  } else {
    return `${Math.round(hoursRemaining / 24)} días`
  }
}

// Validation utilities
export const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

export const validatePhoneNumber = (phone) => {
  const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/
  return phoneRegex.test(phone.replace(/\s/g, ''))
}

export const validatePassword = (password) => {
  return password && password.length >= 6
}

// File utilities
export const getFileExtension = (filename) => {
  return filename.split('.').pop().toLowerCase()
}

export const isValidFileType = (filename, allowedTypes) => {
  const extension = getFileExtension(filename)
  return allowedTypes.includes(extension)
}

export const isImageFile = (filename) => {
  const imageTypes = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'tif']
  return isValidFileType(filename, imageTypes)
}

export const isDataFile = (filename) => {
  const dataTypes = ['csv', 'txt', 'xlsx', 'xls']
  return isValidFileType(filename, dataTypes)
}

// Chart utilities
export const generateChartColors = (count) => {
  const baseColors = [
    '#10b981', // green
    '#3b82f6', // blue
    '#8b5cf6', // purple
    '#f59e0b', // amber
    '#ef4444', // red
    '#06b6d4', // cyan
    '#84cc16', // lime
    '#f97316', // orange
  ]
  
  const colors = []
  for (let i = 0; i < count; i++) {
    colors.push(baseColors[i % baseColors.length])
  }
  
  return colors
}

export const formatChartData = (data, xKey, yKey) => {
  return data.map(item => ({
    x: item[xKey],
    y: item[yKey]
  }))
}

// Local storage utilities
export const setLocalStorage = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch (error) {
    console.error('Error saving to localStorage:', error)
  }
}

export const getLocalStorage = (key, defaultValue = null) => {
  try {
    const item = localStorage.getItem(key)
    return item ? JSON.parse(item) : defaultValue
  } catch (error) {
    console.error('Error reading from localStorage:', error)
    return defaultValue
  }
}

export const removeLocalStorage = (key) => {
  try {
    localStorage.removeItem(key)
  } catch (error) {
    console.error('Error removing from localStorage:', error)
  }
}

// URL utilities
export const buildQueryString = (params) => {
  const searchParams = new URLSearchParams()
  
  Object.keys(params).forEach(key => {
    const value = params[key]
    if (value !== null && value !== undefined && value !== '') {
      searchParams.append(key, value)
    }
  })
  
  return searchParams.toString()
}

export const parseQueryString = (queryString) => {
  const params = new URLSearchParams(queryString)
  const result = {}
  
  for (const [key, value] of params.entries()) {
    result[key] = value
  }
  
  return result
}

// Debounce utility
export const debounce = (func, wait) => {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

// Throttle utility
export const throttle = (func, limit) => {
  let inThrottle
  return function() {
    const args = arguments
    const context = this
    if (!inThrottle) {
      func.apply(context, args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}

// Deep clone utility
export const deepClone = (obj) => {
  if (obj === null || typeof obj !== 'object') return obj
  if (obj instanceof Date) return new Date(obj.getTime())
  if (obj instanceof Array) return obj.map(item => deepClone(item))
  if (typeof obj === 'object') {
    const clonedObj = {}
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key])
      }
    }
    return clonedObj
  }
}

// Array utilities
export const groupBy = (array, key) => {
  return array.reduce((groups, item) => {
    const group = item[key]
    groups[group] = groups[group] || []
    groups[group].push(item)
    return groups
  }, {})
}

export const sortBy = (array, key, direction = 'asc') => {
  return [...array].sort((a, b) => {
    const aVal = a[key]
    const bVal = b[key]
    
    if (direction === 'asc') {
      return aVal > bVal ? 1 : -1
    } else {
      return aVal < bVal ? 1 : -1
    }
  })
}

export const uniqueBy = (array, key) => {
  const seen = new Set()
  return array.filter(item => {
    const value = item[key]
    if (seen.has(value)) {
      return false
    }
    seen.add(value)
    return true
  })
}

// Error handling utilities
export const getErrorMessage = (error) => {
  if (typeof error === 'string') return error
  if (error?.message) return error.message
  if (error?.error) return error.error
  return 'Ha ocurrido un error inesperado'
}

export const isNetworkError = (error) => {
  return error?.message?.includes('fetch') || 
         error?.message?.includes('network') ||
         error?.code === 'NETWORK_ERROR'
}

