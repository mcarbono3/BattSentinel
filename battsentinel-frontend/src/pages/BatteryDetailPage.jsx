// BatteryDetailPage.jsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Activity } from 'lucide-react'

export default function BatteryDetailPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-foreground">Detalle de Batería</h1>
      <Card>
        <CardHeader>
          <CardTitle>Información Detallada</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">Funcionalidad en desarrollo</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

