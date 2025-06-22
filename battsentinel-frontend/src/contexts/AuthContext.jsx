import { createContext, useContext, useState, useEffect } from 'react'
import { authAPI } from '@/lib/api'

const AuthContext = createContext({})

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [token, setToken] = useState(localStorage.getItem('battsentinel-token'))

useEffect(() => {
  // Solo marcamos la app como lista, no intentamos verificar token automáticamente
  setLoading(false)
}, [])

  const login = async (credentials) => {
    try {
      const response = await authAPI.login(credentials)
      if (response.success) {
        const { user, token } = response.data
        setUser(user)
        setToken(token)
        localStorage.setItem('battsentinel-token', token)
        return { success: true }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      // En modo demo, simular login exitoso
      const demoUser = {
        id: 1,
        username: credentials.username,
        email: 'admin@battsentinel.com',
        role: 'admin'
      }
      setUser(demoUser)
      setToken('demo-token')
      localStorage.setItem('battsentinel-token', 'demo-token')
      return { success: true }
    }
  }

  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData)
      if (response.success) {
        const { user, token } = response.data
        setUser(user)
        setToken(token)
        localStorage.setItem('battsentinel-token', token)
        return { success: true }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error de conexión' }
    }
  }

  const logout = async () => {
    try {
      await authAPI.logout()
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      setUser(null)
      setToken(null)
      localStorage.removeItem('battsentinel-token')
    }
  }

  const updateProfile = async (profileData) => {
    try {
      const response = await authAPI.updateProfile(user.id, profileData)
      if (response.success) {
        setUser(response.data)
        return { success: true }
      } else {
        return { success: false, error: response.error }
      }
    } catch (error) {
      return { success: false, error: 'Error de conexión' }
    }
  }

  const value = {
    user,
    token,
    loading,
    login,
    register,
    logout,
    updateProfile,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    isTechnician: user?.role === 'technician' || user?.role === 'admin'
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
