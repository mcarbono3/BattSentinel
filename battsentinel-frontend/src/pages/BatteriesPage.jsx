import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Battery, Plus, Upload, Search, Eye, EyeOff, Trash2, ArrowUpDown } from 'lucide-react';
import { useBattery } from '@/contexts/BatteryContext';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Switch } from '@/components/ui/switch'; // Asumiendo que tienes un componente Switch
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"; // Asumiendo que tienes componentes de Table
import { useToast } from '@/hooks/use-toast'; // Para notificaciones

export default function BatteriesPage() {
  const navigate = useNavigate();
  const { toast } = useToast();

  const {
    batteries,
    loading,
    error,
    loadBatteries, // Asegurar que loadBatteries esté disponible para reintentar
    hiddenBatteryIds,
    toggleBatteryVisibility,
    deleteBattery,
  } = useBattery();

  // useEffect(() => {
  //   // La carga inicial de baterías ya la maneja BatteryProvider al autenticarse.
  //   // Este useEffect solo sería necesario si quieres forzar una recarga aquí.
  //   // loadBatteries();
  // }, []); // Eliminar dependencia de loadBatteries si no se usa para evitar ciclos


  const handleToggleVisibility = async (batteryId, batteryName, isCurrentlyHidden) => {
    try {
      toggleBatteryVisibility(batteryId);
      toast({
        title: "Visibilidad Actualizada",
        description: `Batería "${batteryName}" ha sido ${isCurrentlyHidden ? 'mostrada' : 'ocultada'}.`,
        variant: "default",
      });
    } catch (err) {
      toast({
        title: "Error",
        description: `No se pudo actualizar la visibilidad de la batería "${batteryName}".`,
        variant: "destructive",
      });
    }
  };

  const handleDeleteBattery = async (batteryId, batteryName) => {
    try {
      const result = await deleteBattery(batteryId);
      if (result.success) {
        toast({
          title: "Batería Eliminada",
          description: `La batería "${batteryName}" ha sido eliminada permanentemente.`,
          variant: "default",
        });
        // No es necesario actualizar el estado aquí, BatteryContext.jsx se encarga de setBatteries
        // Si quieres redirigir después de eliminar desde esta página:
        // navigate('/batteries'); 
      } else {
        toast({
          title: "Error al eliminar",
          description: result.error || `No se pudo eliminar la batería "${batteryName}".`,
          variant: "destructive",
        });
      }
    } catch (err) {
      toast({
        title: "Error al eliminar",
        description: `Ocurrió un error inesperado al intentar eliminar la batería "${batteryName}".`,
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Gestión de Baterías</h1>
          <p className="text-muted-foreground">
            Administra y monitorea todas las baterías del sistema, incluyendo su visibilidad.
          </p>
        </div>
        <Button onClick={() => navigate('/batteries/new')}> {/* Asumiendo /batteries/new para formulario de nueva batería */}
          <Plus className="h-4 w-4 mr-2" />
          Nueva Batería
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Lista de Baterías</CardTitle>
          <CardDescription>
            Visualiza y gestiona el estado de todas tus baterías.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8">
              <p>Cargando baterías...</p>
            </div>
          ) : error ? (
            <div className="text-center py-8 text-red-500">
              <p>Error al cargar las baterías: {error}</p>
              <Button onClick={loadBatteries} className="mt-4">Reintentar Carga</Button>
            </div>
          ) : batteries.length === 0 ? (
            <div className="text-center py-12">
              <Battery className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">
                No hay baterías registradas
              </h3>
              <p className="text-muted-foreground mb-4">
                Puedes añadir una nueva batería o cargar datos.
              </p>
              <div className="flex justify-center space-x-2">
                <Button variant="outline" onClick={() => console.log('Navegar a Cargar Datos')}> {/* Placeholder para navegar a página de carga */}
                  <Upload className="h-4 w-4 mr-2" />
                  Cargar Datos
                </Button>
                <Button onClick={() => navigate('/batteries/new')}>
                  <Plus className="h-4 w-4 mr-2" />
                  Añadir Nueva Batería
                </Button>
              </div>
            </div>
          ) : (
            <>
              {/* Sección de "Buscar" y "Cargar Datos" se mantiene si es parte del diseño existente */}
              <div className="mb-4 flex justify-between items-center">
                <div className="relative flex-1 max-w-sm">
                  {/* Aquí podrías integrar un <Input /> para la búsqueda si el diseño lo requiere,
                      pero el botón de "Buscar" individual se mantiene si es así tu UI actual. */}
                  {/* <Input
                    placeholder="Buscar baterías..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-8"
                  />
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" /> */}
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline" onClick={() => console.log('Cargar Datos Clicked')}>
                    <Upload className="h-4 w-4 mr-2" />
                    Cargar Datos
                  </Button>
                  {/* Si el botón "Buscar" es independiente del Input de búsqueda: */}
                  {/* <Button variant="outline">
                    <Search className="h-4 w-4 mr-2" />
                    Buscar
                  </Button> */}
                </div>
              </div>

              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Nombre</TableHead>
                      <TableHead>Tipo</TableHead>
                      <TableHead>Estado</TableHead>
                      <TableHead className="text-right">Acciones</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {batteries.map((battery) => {
                      const isHidden = hiddenBatteryIds.has(battery.id);
                      return (
                        <TableRow key={battery.id}>
                          <TableCell className="font-medium">
                            <Link to={`/batteries/${battery.id}`} className="hover:underline">
                              {battery.name}
                            </Link>
                          </TableCell>
                          <TableCell>{battery.type}</TableCell>
                          <TableCell>
                            <Badge variant={isHidden ? "secondary" : "default"} className={isHidden ? "bg-gray-500 hover:bg-gray-600 text-white" : "bg-green-500 hover:bg-green-600 text-white"}>
                              {isHidden ? 'Oculta' : 'Activa'}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right flex justify-end items-center space-x-2">
                            {/* Botón para ver detalles */}
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => navigate(`/batteries/${battery.id}`)}
                              title="Ver Detalles"
                            >
                              <Eye className="h-4 w-4" />
                            </Button>

                            {/* Interruptor para ocultar/mostrar */}
                            <Switch
                              checked={!isHidden} // true si visible, false si oculta
                              onCheckedChange={() => handleToggleVisibility(battery.id, battery.name, isHidden)}
                              aria-label={isHidden ? `Mostrar batería ${battery.name}` : `Ocultar batería ${battery.name}`}
                              title={isHidden ? `Mostrar ${battery.name}` : `Ocultar ${battery.name}`}
                            />

                            {/* Botón de eliminar con confirmación */}
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button variant="ghost" size="sm" className="text-red-500 hover:text-red-600" title="Eliminar">
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>¿Estás absolutamente seguro?</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    Esta acción no se puede deshacer. Esto eliminará permanentemente la batería <span className="font-bold">"{battery.name}"</span> de nuestros servidores.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancelar</AlertDialogCancel>
                                  <AlertDialogAction onClick={() => handleDeleteBattery(battery.id, battery.name)}>
                                    Eliminar
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
