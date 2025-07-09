import { useEffect, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom'; // Added useSearchParams
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Battery, Plus, Upload, Search, Eye, EyeOff, Trash2, ArrowUpDown, Pencil, CalendarIcon } from 'lucide-react'; // Added CalendarIcon
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
import { Switch } from '@/components/ui/switch';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useToast } from '@/hooks/use-toast';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Calendar } from '@/components/ui/calendar';
import { format } from 'date-fns';
import { cn } from '@/lib/utils';

export default function BatteriesPage() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [searchParams, setSearchParams] = useSearchParams(); // Initialize useSearchParams

  const {
    batteries,
    loading,
    error,
    loadBatteries,
    hiddenBatteryIds,
    toggleBatteryVisibility,
    deleteBattery,
    updateBattery,
    getBatteryById, // Added to fetch battery by ID for URL editing
  } = useBattery();

  const [editingBattery, setEditingBattery] = useState(null);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editFormData, setEditFormData] = useState({});

  // [Punto 2] Datos actualizados para los selects de País y Ciudad.
  const countriesAndCities = [
    { country: 'Colombia', cities: ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Santa Marta', 'Bucaramanga', 'Otra...'] },
    { country: 'Estados Unidos', cities: ['Nueva York', 'Los Ángeles', 'Chicago', 'Houston', 'Otra...'] },
    { country: 'México', cities: ['Ciudad de México', 'Guadalajara', 'Monterrey', 'Otra...'] },
    { country: 'Chile', cities: ['Santiago', 'Valparaíso', 'Concepción', 'Otra...'] },
    { country: 'Otro', cities: [] }
  ];

  const manufacturers = [
    'Tesla', 'LG Energy Solution', 'Panasonic', 'CATL', 'Samsung SDI', 'BYD', 'Enphase', 'Sonnen', 'Otro'
  ];

  const batteryTypes = [
    'Litio-ion (Li-ion)', 'Plomo-Ácido', 'Níquel-Cadmio (NiCd)', 'Níquel-Metal Hidruro (NiMH)', 'Flujo (Flow Battery)', 'Estado Sólido', 'Otro'
  ];

  const handleEditBattery = (battery) => {
    setEditingBattery(battery);
    setEditFormData({
      ...battery,
      installationDate: battery.installationDate ? new Date(battery.installationDate) : null,
      soh: battery.soh || '', // SOH should retain its value if present
      temperature: battery.temperature || '', // Temperature should retain its value if present
      name: battery.name || '',
      model: battery.model || '',
      serial_number: battery.serial_number || '',
      status: battery.status || '',
      location: {
        country: battery.location?.country || '',
        city: battery.location?.city || '',
        countryOther: battery.location?.countryOther || '',
        cityOther: battery.location?.cityOther || '', // Initialize cityOther
      },
      manufacturer: battery.manufacturer || '',
      manufacturerOther: battery.manufacturerOther || '',
      type: battery.type || '',
      typeOther: battery.typeOther || '',
      full_charge_capacity: battery.full_charge_capacity || '',
      capacity_unit: battery.capacity_unit || 'Ah', // Initialize capacity_unit
      nominalVoltageV: battery.nominalVoltageV || '',
      cycles: battery.cycles || '',
      initialCapacityWh: battery.initialCapacityWh || '',
      currentCapacityWh: battery.currentCapacityWh || '',
      dod: battery.dod || '',
    });
    setIsEditModalOpen(true);
  };

  const handleFormChange = (e) => {
    const { id, value, type } = e.target;
    let newValue = value;

    // Handle comma to dot conversion for number inputs and parse to Number
    if (type === 'number') {
      const formattedValue = value.replace(/,/g, '.'); // Replace commas with dots
      newValue = formattedValue !== '' ? Number(formattedValue) : '';
    }

    setEditFormData(prevData => ({
      ...prevData,
      [id]: newValue
    }));
  };

  const handleSelectChange = (id, value) => {
    setEditFormData(prevData => {
      if (id === 'locationCountry') {
        return {
          ...prevData,
          location: { ...prevData.location, country: value, city: '', cityOther: '' } // Reset city and cityOther when country changes
        };
      }
      if (id === 'locationCity') {
        return {
          ...prevData,
          location: { ...prevData.location, city: value, cityOther: value === 'Otra...' ? (prevData.location?.cityOther || '') : '' }
        };
      }
      return {
        ...prevData,
        [id]: value
      };
    });
  };

  const handleDateSelect = (date) => {
    setEditFormData(prevData => ({
      ...prevData,
      installationDate: date
    }));
  };

  // [Punto 4] Función handleSaveEdit ahora usa la 'updateBattery' del contexto con 'localOnly = true'
  const handleSaveEdit = async () => {
    if (!editingBattery) return;

    try {
      // Build location object based on selections
      let locationToSave = {};
      if (editFormData.location?.country === 'Otro') {
        locationToSave = {
          country: editFormData.location.country,
          countryOther: editFormData.location.countryOther,
          city: '', // Ensure city is empty if country is 'Otro'
          cityOther: '', // Ensure cityOther is empty if country is 'Otro'
        };
      } else if (editFormData.location?.city === 'Otra...') {
        locationToSave = {
          country: editFormData.location?.country,
          city: editFormData.location.city,
          cityOther: editFormData.location.cityOther,
        };
      } else {
        locationToSave = {
          country: editFormData.location?.country,
          city: editFormData.location?.city,
          cityOther: '', // Ensure cityOther is empty if a specific city is selected
        };
      }

      const dataToSave = {
        ...editFormData,
        installationDate: editFormData.installationDate ? editFormData.installationDate.toISOString() : null,
        location: locationToSave, // Use the dynamically created location object
        capacity_unit: editFormData.capacity_unit, // Include capacity_unit
      };

      // Remove specific fields if "Otro" is not selected to avoid sending unnecessary data
      if (dataToSave.manufacturer !== 'Otro') {
        delete dataToSave.manufacturerOther;
      }
      if (dataToSave.type !== 'Otro') {
        delete dataToSave.typeOther;
      }

      // If country is not 'Otro', clear countryOther
      if (dataToSave.location.country !== 'Otro') {
        dataToSave.location.countryOther = '';
      }
      // If city is not 'Otra...', clear cityOther
      if (dataToSave.location.city !== 'Otra...') {
        dataToSave.location.cityOther = '';
      }


      // Llama a updateBattery del contexto, indicando que es una actualización local
      const result = await updateBattery(editingBattery.id, dataToSave, true); // true para localOnly

      if (result.success) {
        toast({
          title: "Batería Actualizada",
          description: `La batería "${result.data.name}" ha sido actualizada exitosamente en el frontend.`,
          variant: "default",
        });
        setIsEditModalOpen(false);
        setEditingBattery(null);
        setEditFormData({});
      } else {
        toast({
          title: "Error al actualizar",
          description: result.error || `No se pudo actualizar la batería "${editingBattery.name}".`,
          variant: "destructive",
        });
      }
    } catch (err) {
      console.error("Error saving battery:", err); // Log the error for debugging
      toast({
        title: "Error al actualizar",
        description: `Ocurrió un error inesperado al intentar actualizar la batería "${editingBattery.name}".`,
        variant: "destructive",
      });
    }
  };

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

  // Handle URL parameter for editing
  useEffect(() => {
    const editId = searchParams.get('editId');
    if (editId && batteries.length > 0) {
      const batteryToEdit = batteries.find(battery => battery.id === editId);
      if (batteryToEdit) {
        handleEditBattery(batteryToEdit);
        // Clear the editId from the URL to prevent re-opening on refresh
        setSearchParams(prev => {
          prev.delete('editId');
          return prev;
        }, { replace: true });
      }
    }
  }, [searchParams, batteries, setSearchParams]); // Dependency on batteries to ensure they are loaded

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Gestión de Baterías</h1>
          <p className="text-muted-foreground">
            Administra y monitorea todas las baterías del sistema, incluyendo su visibilidad.
          </p>
        </div>
        <Button onClick={() => navigate('/batteries/new')}>
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
                <Button variant="outline" onClick={() => console.log('Navegar a Cargar Datos')}>
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
              <div className="mb-4 flex justify-between items-center">
                <div className="relative flex-1 max-w-sm">
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline" onClick={() => console.log('Cargar Datos Clicked')}>
                    <Upload className="h-4 w-4 mr-2" />
                    Cargar Datos
                  </Button>
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
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => navigate(`/batteries/${battery.id}`)}
                              title="Ver Detalles"
                            >
                              <Eye className="h-4 w-4" />
                            </Button>

                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleEditBattery(battery)}
                              title="Editar Batería"
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>

                            <Switch
                              checked={!isHidden}
                              onCheckedChange={() => handleToggleVisibility(battery.id, battery.name, isHidden)}
                              aria-label={isHidden ? `Mostrar batería ${battery.name}` : `Ocultar batería ${battery.name}`}
                              title={isHidden ? `Mostrar ${battery.name}` : `Ocultar ${battery.name}`}
                            />

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

      {/* Edit Battery AlertDialog */}
      {editingBattery && (
        <AlertDialog open={isEditModalOpen} onOpenChange={setIsEditModalOpen}>
          {/* [Punto 1] Clases para responsividad y scroll en la ventana emergente */}
          <AlertDialogContent className="sm:max-w-[90vw] md:max-w-[700px] lg:max-w-[800px] max-h-[85vh] overflow-y-auto">
            <AlertDialogHeader>
              <AlertDialogTitle>Editar Batería: {editingBattery.name}</AlertDialogTitle>
              <AlertDialogDescription>
                Realiza cambios en la información de la batería.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="grid gap-4 py-4">
              <h3 className="text-lg font-semibold border-b pb-2 mb-4">Información General</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Nombre</Label>
                  <Input id="name" value={editFormData.name} onChange={handleFormChange} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="model">Modelo</Label>
                  <Input id="model" value={editFormData.model} onChange={handleFormChange} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="serial_number">Número de Serie</Label>
                  <Input id="serial_number" value={editFormData.serial_number} onChange={handleFormChange} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="status">Estado</Label>
                  <Select
                    id="status"
                    value={editFormData.status}
                    onValueChange={(value) => handleSelectChange('status', value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona un estado" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="active">Activa</SelectItem>
                      <SelectItem value="inactive">Inactiva</SelectItem>
                      <SelectItem value="maintenance">Mantenimiento</SelectItem>
                      <SelectItem value="critical">Crítica</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="locationCountry">País de Ubicación</Label>
                  <Select
                    id="locationCountry"
                    value={editFormData.location?.country || ''}
                    onValueChange={(value) => {
                      setEditFormData(prevData => ({
                        ...prevData,
                        location: { ...prevData.location, country: value, city: '', cityOther: '' }
                      }));
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona un país" />
                    </SelectTrigger>
                    <SelectContent>
                      {countriesAndCities.map((loc) => (
                        <SelectItem key={loc.country} value={loc.country}>
                          {loc.country}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {editFormData.location?.country === 'Otro' ? (
                  <div className="space-y-2">
                    <Label htmlFor="locationCountryOther">Otro País</Label>
                    <Input
                      id="locationCountryOther"
                      value={editFormData.location.countryOther || ''}
                      onChange={(e) => setEditFormData(prevData => ({
                        ...prevData,
                        location: { ...prevData.location, countryOther: e.target.value }
                      }))}
                    />
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Label htmlFor="locationCity">Ciudad de Ubicación</Label>
                    <Select
                      id="locationCity"
                      value={editFormData.location?.city || ''}
                      onValueChange={(value) => {
                        setEditFormData(prevData => ({
                          ...prevData,
                          location: {
                            ...prevData.location,
                            city: value,
                            cityOther: value === 'Otra...' ? (prevData.location?.cityOther || '') : '' // Keep or clear cityOther
                          }
                        }));
                      }}
                      disabled={!editFormData.location?.country}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Selecciona una ciudad" />
                      </SelectTrigger>
                      <SelectContent>
                        {countriesAndCities.find(c => c.country === editFormData.location?.country)?.cities.map((city) => (
                          <SelectItem key={city} value={city}>
                            {city}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {editFormData.location?.city === 'Otra...' && (
                      <Input
                        id="locationCityOther"
                        placeholder="Ingresa otra ciudad"
                        value={editFormData.location.cityOther || ''}
                        onChange={(e) => setEditFormData(prevData => ({
                          ...prevData,
                          location: { ...prevData.location, cityOther: e.target.value }
                        }))}
                      />
                    )}
                  </div>
                )}
              </div>

              <h3 className="text-lg font-semibold border-b pb-2 mb-4 mt-6">Especificaciones Técnicas</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="manufacturer">Fabricante</Label>
                  <Select
                    id="manufacturer"
                    value={editFormData.manufacturer}
                    onValueChange={(value) => handleSelectChange('manufacturer', value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona un fabricante" />
                    </SelectTrigger>
                    <SelectContent>
                      {manufacturers.map((m) => (
                        <SelectItem key={m} value={m}>
                          {m}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {editFormData.manufacturer === 'Otro' && (
                  <div className="space-y-2">
                    <Label htmlFor="manufacturerOther">Otro Fabricante</Label>
                    <Input
                      id="manufacturerOther"
                      value={editFormData.manufacturerOther || ''}
                      onChange={handleFormChange}
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="type">Tipo de Batería (Electroquímica)</Label>
                  <Select
                    id="type"
                    value={editFormData.type}
                    onValueChange={(value) => handleSelectChange('type', value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Selecciona un tipo" />
                    </SelectTrigger>
                    <SelectContent>
                      {batteryTypes.map((t) => (
                        <SelectItem key={t} value={t}>
                          {t}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {editFormData.type === 'Otro' && (
                  <div className="space-y-2">
                    <Label htmlFor="typeOther">Otro Tipo de Batería</Label>
                    <Input
                      id="typeOther"
                      value={editFormData.typeOther || ''}
                      onChange={handleFormChange}
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="full_charge_capacity">Capacidad Nominal</Label>
                  <div className="flex space-x-2">
                    <Input
                      id="full_charge_capacity"
                      type="number"
                      value={editFormData.full_charge_capacity}
                      onChange={handleFormChange}
                      className="flex-1"
                    />
                    <Select
                      id="capacity_unit"
                      value={editFormData.capacity_unit}
                      onValueChange={(value) => handleSelectChange('capacity_unit', value)}
                    >
                      <SelectTrigger className="w-[100px]">
                        <SelectValue placeholder="Unidad" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Ah">Ah</SelectItem>
                        <SelectItem value="Wh">Wh</SelectItem>
                        <SelectItem value="mAh">mAh</SelectItem>
                        <SelectItem value="mWh">mWh</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="nominalVoltageV">Voltaje Nominal (V)</Label>
                  <Input
                    id="nominalVoltageV"
                    type="number"
                    value={editFormData.nominalVoltageV}
                    onChange={handleFormChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="cycles">Ciclos</Label>
                  <Input
                    id="cycles"
                    type="number"
                    value={editFormData.cycles}
                    onChange={handleFormChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="initialCapacityWh">Capacidad Inicial (Wh)</Label>
                  <Input
                    id="initialCapacityWh"
                    type="number"
                    value={editFormData.initialCapacityWh}
                    onChange={handleFormChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="currentCapacityWh">Capacidad Actual (Wh)</Label>
                  <Input
                    id="currentCapacityWh"
                    type="number"
                    value={editFormData.currentCapacityWh}
                    onChange={handleFormChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="dod">DOD (Depth of Discharge) (%)</Label>
                  <Input
                    id="dod"
                    type="number"
                    value={editFormData.dod}
                    onChange={handleFormChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="soh">SOH (State of Health) (%)</Label>
                  <Input
                    id="soh"
                    type="number"
                    value={editFormData.soh}
                    onChange={handleFormChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="temperature">Temperatura (°C)</Label>
                  <Input
                    id="temperature"
                    type="number"
                    value={editFormData.temperature}
                    onChange={handleFormChange}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="installationDate">Fecha de Instalación</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant={"outline"}
                        className={cn(
                          "w-full justify-start text-left font-normal",
                          !editFormData.installationDate && "text-muted-foreground"
                        )}
                      >
                        <CalendarIcon className="mr-2 h-4 w-4" /> {/* Replaced Calendar with CalendarIcon */}
                        {editFormData.installationDate ? (
                          format(editFormData.installationDate, "PPP")
                        ) : (
                          <span>Selecciona una fecha</span>
                        )}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <Calendar
                        mode="single"
                        selected={editFormData.installationDate}
                        onSelect={handleDateSelect}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                </div>
              </div>
            </div>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={() => setIsEditModalOpen(false)}>Cancelar</AlertDialogCancel>
              <AlertDialogAction onClick={handleSaveEdit}>Guardar Cambios</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
    </div>
  );
}
