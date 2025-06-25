import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import { ThemeProvider } from '@/components/theme-provider'
import { BatteryProvider } from '@/contexts/BatteryContext'

// Layout Components
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'

// Page Components
import LandingPage from '@/pages/LandingPage'
import Dashboard from '@/pages/Dashboard'
import BatteriesPage from '@/pages/BatteriesPage'
import BatteryDetailPage from '@/pages/BatteryDetailPage'
import DigitalTwinPage from '@/pages/DigitalTwinPage'
import AnalyticsPage from '@/pages/AnalyticsPage'
import AlertsPage from '@/pages/AlertsPage'
import SettingsPage from '@/pages/SettingsPage'

import './App.css'

// Main App Layout - Sin restricciones de autenticación
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

// App Content Component - Sin autenticación
function AppContent() {
  return (
    <Routes>
      {/* Ruta principal va a LandingPage */}
      <Route path="/" element={<LandingPage />} />
      
      {/* Todas las rutas del dashboard son accesibles sin autenticación */}
      <Route 
        path="/dashboard" 
        element={
          <AppLayout>
            <Dashboard />
          </AppLayout>
        } 
      />
      <Route 
        path="/batteries" 
        element={
          <AppLayout>
            <BatteriesPage />
          </AppLayout>
        } 
      />
      <Route 
        path="/batteries/:id" 
        element={
          <AppLayout>
            <BatteryDetailPage />
          </AppLayout>
        } 
      />
      <Route
        path="/digital-twin"
        element={
          <AppLayout>
            <DigitalTwinPage />
          </AppLayout>
        }
      />
      <Route
        path="/digital-twin/:id"
        element={
          <AppLayout>
            <DigitalTwinPage />
          </AppLayout>
        }
      />
      <Route
        path="/analytics"
        element={
          <AppLayout>
            <AnalyticsPage />
          </AppLayout>
        }
      />
      <Route
        path="/alerts"
        element={
          <AppLayout>
            <AlertsPage />
          </AppLayout>
        }
      />
      <Route
        path="/settings"
        element={
          <AppLayout>
            <SettingsPage />
          </AppLayout>
        }
      />
      {/* Cualquier ruta no encontrada redirige a landing page */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

// Main App Component - Sin AuthProvider
function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="battsentinel-theme">
      <Router>
        <BatteryProvider>
          <div className="min-h-screen bg-background font-sans antialiased">
            <AppContent />
            <Toaster />
          </div>
        </BatteryProvider>
      </Router>
    </ThemeProvider>
  )
}

export default App

