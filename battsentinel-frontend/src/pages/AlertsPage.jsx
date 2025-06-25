import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Bell } from 'lucide-react'

export default function AlertsPage() {
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

