// AnalyticsPage.jsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { BarChart3, Bell, Settings } from 'lucide-react'

export default function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-foreground">Análisis con IA</h1>
      <Card>
        <CardHeader>
          <CardTitle>Análisis Avanzado</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">Análisis con IA en desarrollo</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// AlertsPage component
export function AlertsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-foreground">Alertas y Notificaciones</h1>
      <Card>
        <CardHeader>
          <CardTitle>Centro de Alertas</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Bell className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">Sistema de alertas en desarrollo</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// SettingsPage component
export function SettingsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-foreground">Configuración</h1>
      <Card>
        <CardHeader>
          <CardTitle>Configuración del Sistema</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Settings className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">Configuración en desarrollo</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

