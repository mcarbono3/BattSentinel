import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Battery, Plus, Upload, Search } from 'lucide-react'

export default function BatteriesPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Gestión de Baterías</h1>
          <p className="text-muted-foreground">
            Administra y monitorea todas las baterías del sistema
          </p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Nueva Batería
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Lista de Baterías</CardTitle>
          <CardDescription>
            Funcionalidad completa en desarrollo
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <Battery className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">
              Gestión de Baterías
            </h3>
            <p className="text-muted-foreground mb-4">
              Esta funcionalidad estará disponible próximamente
            </p>
            <div className="flex justify-center space-x-2">
              <Button variant="outline">
                <Upload className="h-4 w-4 mr-2" />
                Cargar Datos
              </Button>
              <Button variant="outline">
                <Search className="h-4 w-4 mr-2" />
                Buscar
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

