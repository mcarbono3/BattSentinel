import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import { ThemeProvider } from '@/components/theme-provider'
import { AuthProvider, useAuth } from '@/contexts/AuthContext'
import { BatteryProvider } from '@/contexts/BatteryContext'

// Layout Components
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'

// Page Components
import LandingPage from '@/pages/LandingPage'
import LoginPage from '@/pages/LoginPage'
import Dashboard from '@/pages/Dashboard'
import BatteriesPage from '@/pages/BatteriesPage'
import BatteryDetailPage from '@/pages/BatteryDetailPage'
import DigitalTwinPage from '@/pages/DigitalTwinPage'
import AnalyticsPage from '@/pages/AnalyticsPage'
import AlertsPage from '@/pages/AlertsPage'
import SettingsPage from '@/pages/SettingsPage'
import LoadingScreen from '@/components/ui/LoadingScreen'

import './App.css'

// Protected Route Component
function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  
  if (loading) {
    return <LoadingScreen />
  }
  
  if (!user) {
    return <Navigate to="/login" replace />
  }
  
  return children
}

// Main App Layout
function AppLayout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  
  return (
    <div className="flex h-screen bg-background">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
        <main className="flex-1 overflow-auto p-6 bg-background">
          {children}
        </main>
      </div>
    </div>
  )
}

// App Content Component
function AppContent() {
  const { user, loading } = useAuth()
  
  if (loading) {
    return <LoadingScreen />
  }
  
  return (
    <Routes>
      <Route 
        path="/" 
        element={user ? <Navigate to="/dashboard" replace /> : <LandingPage />} 
      />
      <Route 
        path="/login" 
        element={user ? <Navigate to="/dashboard" replace /> : <LoginPage />} 
      />
      <Route 
        path="/dashboard" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <Dashboard />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/batteries" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <BatteriesPage />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/batteries/:id" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <BatteryDetailPage />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/digital-twin/:id" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <DigitalTwinPage />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/analytics" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <AnalyticsPage />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/alerts" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <AlertsPage />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/settings" 
        element={
          <ProtectedRoute>
            <AppLayout>
              <SettingsPage />
            </AppLayout>
          </ProtectedRoute>
        } 
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

// Main App Component
function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="battsentinel-theme">
      <Router>
        <AuthProvider>
          <BatteryProvider>
            <div className="min-h-screen bg-background font-sans antialiased">
              <AppContent />
              <Toaster />
            </div>
          </BatteryProvider>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  )
}

export default App
