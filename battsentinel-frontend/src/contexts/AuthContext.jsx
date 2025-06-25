import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext({})

export function AuthProvider({ children }) {
  // Usuario simulado para mantener compatibilidad sin autenticación
  const [user, setUser] = useState({
    id: 1,
    username: 'demo',
    email: 'demo@battsentinel.com',
    role: 'admin',
    first_name: 'Usuario',
    last_name: 'Demo'
  })
  const [loading, setLoading] = useState(false)
  const [token, setToken] = useState('demo-token')

  // Funciones simuladas para mantener compatibilidad
  const login = async (credentials) => {
    return { success: true, user: user }
  }

  const logout = () => {
    // No hacer nada en esta versión sin autenticación
    return true
  }

  const register = async (userData) => {
    return { success: true, user: user }
  }

  const updateProfile = async (profileData) => {
    setUser(prev => ({ ...prev, ...profileData }))
    return { success: true, user: user }
  }

  const isAuthenticated = () => true // Siempre autenticado en esta versión

  const hasRole = (role) => true // Siempre tiene permisos en esta versión

  const value = {
    user,
    token,
    loading,
    login,
    logout,
    register,
    updateProfile,
    isAuthenticated,
    hasRole,
    setUser,
    setToken: () => {}, // No hacer nada
    setLoading
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext

