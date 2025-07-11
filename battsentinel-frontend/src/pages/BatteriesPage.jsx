// src/pages/BatteriesPage.jsx
import { useEffect, useState, useCallback } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Battery, Plus, Upload, Search, Eye, Trash2, ArrowUpDown, Pencil, CalendarIcon, Activity, Thermometer, Zap, Clock } from 'lucide-react';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog";


// Opciones de países y ciudades
const COUNTRY_OPTIONS = [
  "Colombia",
  "Mexico",
  "Estados Unidos",
  "Chile",
  "Otro"
];

const CITY_OPTIONS_MAP = {
  "Colombia": ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena", "Bucaramanga", "Santa Marta", "Otro"],
  "Mexico": ["Ciudad de México", "Guadalajara", "Monterrey", "Puebla", "Otro"],
  "Estados Unidos": ["Nueva York", "Los Ángeles", "Chicago", "Houston", "Otro"],
  "Chile": ["Santiago", "Valparaíso", "Concepción", "Otro"],
  "Otro": ["Otro"] // Para el caso de "Otro" país, solo "Otro" ciudad
};

const CAPACITY_UNITS = ["Wh", "Ah", "mAh", "mWh"]; // Default 'Wh'

// Helper function for battery status colors
const getBatteryStatusClass = (status, isHidden) => {
  if (isHidden) {
    return 'bg-gray-400 text-gray-800'; // Grey out if hidden
  }
  switch (status) {
    case 'active':
      return 'bg-green-500 text-white';
    case 'inactive':
      return 'bg-red-500 text-white';
    case 'maintenance':
      return 'bg-yellow-500 text-black';
    case 'retired':
      return 'bg-amber-800 text-white'; // Changed to amber-800 for brown-like color
    default:
      return 'bg-gray-200 text-gray-800';
  }
};


export default function BatteriesPage() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [searchParams, setSearchParams] = useSearchParams();

  const {
    batteries,
    loading,
    error,
    loadBatteries, // This now maps to loadBatteriesFromAPI which handles local storage
    createBattery,
    updateBattery, // This now updates local storage only
    deleteBattery,
    uploadBatteryData,
    toggleBatteryVisibility,
    hiddenBatteryIds,
    // Removed getVisibleBatteries here as per new requirement
  } = useBattery();

  const [filteredBatteries, setFilteredBatteries] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState(null);
  const [sortDirection, setSortDirection] = useState('asc');

  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editFormData, setEditFormData] = useState(null); // Estado para los datos del formulario de edición

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newBatteryData, setNewBatteryData] = useState({
    name: '',
    model: '',
    manufacturer: '',
    serial_number: '',
    full_charge_capacity: '',
    designvoltage: '',
    chemistry: 'Li-ion',
    installation_date: new Date(),
    location: '',
    status: 'active',
    monitoring_source: 'Simulado',
  });

  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [selectedBatteryForUpload, setSelectedBatteryForUpload] = useState('');
  const [fileToUpload, setFileToUpload] = useState(null);

  // Cargar baterías solo una vez al montar el componente
  // Este useEffect llama a loadBatteries que está definido en BatteryContext.jsx
  // y que ahora gestiona la carga desde localStorage y la API.
  useEffect(() => {
    console.log('BatteriesPage: useEffect calling loadBatteries...');
    loadBatteries();
  }, []); // Dependencia loadBatteries para asegurar que se llama si cambia (aunque es estable)


  // Aplicar filtros y ordenamiento - NOW APPLIES TO ALL BATTERIES, NOT JUST VISIBLE
  useEffect(() => {
    console.log('BatteriesPage: useEffect filtering/sorting batteries...');
    // Use all batteries, regardless of their hidden status for display on this page
    let currentBatteries = [...batteries];

    if (searchTerm) {
      currentBatteries = currentBatteries.filter(battery =>
        battery.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (battery.serial_number && battery.serial_number.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (battery.location && battery.location.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (battery.status && battery.status.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    if (sortBy) {
      currentBatteries = [...currentBatteries].sort((a, b) => {
        const aValue = a[sortBy];
        const bValue = b[sortBy];

        if (aValue === null || aValue === undefined) return sortDirection === 'asc' ? 1 : -1;
        if (bValue === null || bValue === undefined) return sortDirection === 'asc' ? -1 : 1;

        if (typeof aValue === 'string') {
          return sortDirection === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
        }
        return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
      });
    }
    setFilteredBatteries(currentBatteries);
  }, [batteries, searchTerm, sortBy, sortDirection]); // hiddenBatteryIds removed from dependencies as it no longer filters the list


  // Handler para cambios en el formulario de edición
  const handleEditChange = (e) => {
    const { id, value } = e.target;
    setEditFormData(prev => ({ ...prev, [id]: value }));
  };

  const handleEditDateSelect = (date, fieldName) => {
    setEditFormData(prev => ({ ...prev, [fieldName]: date }));
  };

  const handleEditSelectChange = (value, fieldName) => {
    setEditFormData(prev => ({ ...prev, [fieldName]: value }));
  };

  const handleSaveEdit = async () => {
    if (!editFormData.name) {
      toast({
        title: "Error de Edición",
        description: "El nombre de la batería no puede estar vacío.",
        variant: "destructive",
      });
      return;
    }

    const dataToUpdate = { ...editFormData };

    // Reconstruct 'location' from 'country' and 'city' for the backend
    if (dataToUpdate.country || dataToUpdate.city) {
      let fullLocation = '';
      if (dataToUpdate.city && dataToUpdate.city !== 'Otro') {
        fullLocation += dataToUpdate.city;
      } else if (dataToUpdate.otherCity) { // Use 'otherCity' if 'Otro' was selected
        fullLocation += dataToUpdate.otherCity;
      }

      if (dataToUpdate.country && dataToUpdate.country !== 'Otro') {
        fullLocation = fullLocation ? `${fullLocation}, ${dataToUpdate.country}` : dataToUpdate.country;
      } else if (dataToUpdate.otherCountry) { // Use 'otherCountry' if 'Otro' was selected
        fullLocation = fullLocation ? `${fullLocation}, ${dataToUpdate.otherCountry}` : dataToUpdate.otherCountry;
      }
      dataToUpdate.location = fullLocation;
    } else {
      dataToUpdate.location = ''; // No location selected
    }

    // Convert dates to ISO string before sending
    if (dataToUpdate.installation_date) {
      dataToUpdate.installation_date = dataToUpdate.installation_date.toISOString();
    }
    if (dataToUpdate.last_maintenance_date) {
      dataToUpdate.last_maintenance_date = dataToUpdate.last_maintenance_date.toISOString();
    }
    if (dataToUpdate.warranty_expiry_date) {
      dataToUpdate.warranty_expiry_date = dataToUpdate.warranty_expiry_date.toISOString();
    }

    // Remove temporary UI fields not for backend
    delete dataToUpdate.country;
    delete dataToUpdate.city;
    delete dataToUpdate.otherCountry;
    delete dataToUpdate.otherCity;
    // For now, units and cycles are frontend-only, assume backend will ignore if not in model
    // If backend model needs these, `battery.py` should be updated
    // For example, you might want to send `full_charge_capacity_unit` as a separate field
    // or concatenate it with the capacity value. For now, sending as separate.

    const result = await updateBattery(editFormData.id, dataToUpdate);
    if (result.success) {
      toast({
        title: "Batería Actualizada",
        description: `La batería "${result.data.name}" ha sido actualizada exitosamente.`,
        variant: "success",
      });
      setIsEditModalOpen(false);
    } else {
      toast({
        title: "Error al Actualizar",
        description: result.error || "Ocurrió un error inesperado al actualizar la batería.",
        variant: "destructive",
      });
    }
  };

  const handleDeleteBattery = async (id) => {
    const result = await deleteBattery(id);
    if (result.success) {
      toast({
        title: "Batería Eliminada",
        description: "La batería ha sido eliminada exitosamente.",
        variant: "success",
      });
    } else {
      toast({
        title: "Error al Eliminar",
        description: result.error || "Ocurrió un error inesperado al eliminar la batería.",
        variant: "destructive",
      });
    }
  };

  const handleNewBatteryChange = (e) => {
    const { id, value } = e.target;
    setNewBatteryData(prev => ({ ...prev, [id]: value }));
  };

  const handleNewBatterySelectChange = (id, value) => {
    setNewBatteryData(prev => ({ ...prev, [id]: value }));
  };

  const handleNewBatteryDateSelect = (date) => {
    setNewBatteryData(prev => ({ ...prev, installation_date: date }));
  };

  const handleCreateBattery = async () => {
    if (!newBatteryData.name) {
      toast({
        title: "Error",
        description: "El nombre de la batería es obligatorio.",
        variant: "destructive",
      });
      return;
    }

    const batteryDataToSend = { ...newBatteryData };
    delete batteryDataToSend.monitoring_source;

    if (batteryDataToSend.installation_date) {
      batteryDataToSend.installation_date = new Date(batteryDataToSend.installation_date).toISOString();
    }

    const result = await createBattery(batteryDataToSend);

    if (result.success) {
      toast({
        title: "Batería creada",
        description: `La batería "${result.data.name}" ha sido creada exitosamente.`,
        variant: "success",
      });
      setNewBatteryData({
        name: '', model: '', manufacturer: '', serial_number: '', full_charge_capacity: '',
        designvoltage: '', chemistry: 'Li-ion', installation_date: new Date(), location: '',
        status: 'active', monitoring_source: 'Simulado',
      });
      setIsCreateModalOpen(false);
    } else {
      toast({
        title: "Error al crear batería",
        description: result.error || "Ocurrió un error inesperado.",
        variant: "destructive",
      });
    }
  };

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setFileToUpload(event.target.files[0]);
    } else {
      setFileToUpload(null);
    }
  };

  const handleUploadData = async () => {
    if (!selectedBatteryForUpload) {
      toast({
        title: "Error de Carga",
        description: "Debe seleccionar una batería para cargar datos.",
        variant: "destructive",
      });
      return;
    }
    if (!fileToUpload) {
      toast({
        title: "Error de Carga",
        description: "Debe seleccionar un archivo para cargar.",
        variant: "destructive",
      });
      return;
    }

    const batteryToUpload = batteries.find(b => b.id === selectedBatteryForUpload);
    if (batteryToUpload && (batteryToUpload.monitoring_source === 'Real (Directo)' || batteryToUpload.monitoring_source === 'Simulado')) {
      toast({
        title: "Carga no permitida",
        description: `No se permite la carga manual de datos para baterías con fuente "${batteryToUpload.monitoring_source}".`,
        variant: "destructive",
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', fileToUpload);

    const result = await uploadBatteryData(selectedBatteryForUpload, formData);

    if (result.success) {
      toast({
        title: "Datos Cargados",
        description: "Los datos de la batería se han cargado exitosamente.",
        variant: "success",
      });
      setFileToUpload(null);
      setSelectedBatteryForUpload('');
      setIsUploadModalOpen(false);
    } else {
      toast({
        title: "Error al Cargar Datos",
        description: result.error || "Ocurrió un error al cargar los datos.",
        variant: "destructive",
      });
    }
  };

  const handleSort = useCallback((column) => {
    if (sortBy === column) {
      setSortDirection(prevDirection => prevDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortDirection('asc');
    }
  }, [sortBy]);

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Gestión de Baterías</h1>
          <p className="text-muted-foreground">
            Administra y monitorea todas las baterías del sistema, incluyendo su visibilidad.
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {/* Botón "Cargar Datos" */}
          <Dialog open={isUploadModalOpen} onOpenChange={setIsUploadModalOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm">
                <Upload className="h-4 w-4 mr-2" />
                Cargar Datos
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[425px]">
              <DialogHeader>
                <DialogTitle>Cargar Datos de Batería</DialogTitle>
                <DialogDescription>
                  Selecciona una batería y sube un archivo de datos.
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="batterySelect" className="text-right">
                    Batería
                  </Label>
                  <Select
                    value={selectedBatteryForUpload}
                    onValueChange={setSelectedBatteryForUpload}
                  >
                    <SelectTrigger className="col-span-3">
                      <SelectValue placeholder="Selecciona una batería" />
                    </SelectTrigger>
                    <SelectContent>
                      {batteries.map((battery) => (
                        <SelectItem key={battery.id} value={battery.id}>
                          {battery.name} (ID: {battery.id})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="fileInput" className="text-right">
                    Archivo
                  </Label>
                  <Input
                    id="fileInput"
                    type="file"
                    className="col-span-3"
                    onChange={handleFileChange}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsUploadModalOpen(false)}>Cancelar</Button>
                <Button
                  onClick={handleUploadData}
                  disabled={!selectedBatteryForUpload || !fileToUpload || (
                    batteries.find(b => b.id === selectedBatteryForUpload && (b.monitoring_source === 'Real (Directo)' || b.monitoring_source === 'Simulado'))
                  )}
                >
                  Cargar
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Botón "Nueva Batería" */}
          <Dialog open={isCreateModalOpen} onOpenChange={setIsCreateModalOpen}>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="h-4 w-4 mr-2" />
                Nueva Batería
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[500px]">
              <DialogHeader>
                <DialogTitle>Crear Nueva Batería</DialogTitle>
                <DialogDescription>
                  Completa los campos para añadir una nueva batería al sistema. El campo "Nombre" es obligatorio.
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="name" className="text-right">Nombre <span className="text-red-500">*</span></Label>
                  <Input id="name" value={newBatteryData.name} onChange={handleNewBatteryChange} className="col-span-3" required />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="model" className="text-right">Modelo</Label>
                  <Input id="model" value={newBatteryData.model} onChange={handleNewBatteryChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="manufacturer" className="text-right">Fabricante</Label>
                  <Input id="manufacturer" value={newBatteryData.manufacturer} onChange={handleNewBatteryChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="serial_number" className="text-right">Número de Serie</Label>
                  <Input id="serial_number" value={newBatteryData.serial_number} onChange={handleNewBatteryChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="full_charge_capacity" className="text-right">Capacidad (mAh)</Label>
                  <Input type="number" id="full_charge_capacity" value={newBatteryData.full_charge_capacity} onChange={handleNewBatteryChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="designvoltage" className="text-right">Voltaje Diseño (V)</Label>
                  <Input type="number" id="designvoltage" value={newBatteryData.designvoltage} onChange={handleNewBatteryChange} className="col-span-3" />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="chemistry" className="text-right">Química</Label>
                  <Select value={newBatteryData.chemistry} onValueChange={(value) => handleNewBatterySelectChange('chemistry', value)}>
                    <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona la química" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Li-ion">Li-ion</SelectItem>
                      <SelectItem value="NiMH">NiMH</SelectItem>
                      <SelectItem value="LiPo">LiPo</SelectItem>
                      <SelectItem value="PbA">Plomo-Ácido</SelectItem>
                      <SelectItem value="other">Otro</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="status" className="text-right">Estado Inicial</Label>
                  <Select value={newBatteryData.status} onValueChange={(value) => handleNewBatterySelectChange('status', value)}>
                    <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona el estado" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="active">Activa</SelectItem>
                      <SelectItem value="inactive">Inactiva</SelectItem>
                      <SelectItem value="maintenance">Mantenimiento</SelectItem>
                      <SelectItem value="retired">Retirada</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="monitoring_source" className="text-right">Fuente de Monitoreo</Label>
                  <Select value={newBatteryData.monitoring_source} onValueChange={(value) => handleNewBatterySelectChange('monitoring_source', value)}>
                    <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona la fuente" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Real (Directo)">Real (Directo)</SelectItem>
                      <SelectItem value="Simulado">Simulado</SelectItem>
                      <SelectItem value="Histórico (Archivo/BD)">Histórico (Archivo/BD)</SelectItem>
                      <SelectItem value="Prueba">Prueba</SelectItem>
                      <SelectItem value="otro">Otro</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="installation_date" className="text-right">Fecha Instalación</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant={"outline"} className={cn("col-span-3 justify-start text-left font-normal", !newBatteryData.installation_date && "text-muted-foreground")}>
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {newBatteryData.installation_date ? format(newBatteryData.installation_date, "PPP") : <span>Selecciona una fecha</span>}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <Calendar mode="single" selected={newBatteryData.installation_date} onSelect={handleNewBatteryDateSelect} initialFocus />
                    </PopoverContent>
                  </Popover>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="location" className="text-right">Ubicación</Label>
                  <Input id="location" value={newBatteryData.location} onChange={handleNewBatteryChange} className="col-span-3" />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsCreateModalOpen(false)}>Cancelar</Button>
                <Button onClick={handleCreateBattery}>Crear Batería</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Search Input */}
      <div className="flex items-center space-x-2">
        <Input
          placeholder="Buscar baterías por nombre o SN..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="max-w-sm"
        />
        <Button variant="outline" size="icon" onClick={() => setSearchTerm('')}>
          <Search className="h-4 w-4" />
        </Button>
      </div>

      {/* Battery List Table */}
      <Card>
        <CardHeader>
          <CardTitle>Lista de Baterías</CardTitle>
          <CardDescription>
            Visualiza y gestiona el estado de todas tus baterías.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8">Cargando baterías...</div>
          ) : error ? (
            <div className="text-center py-8 text-red-500">Error: {error}</div>
          ) : filteredBatteries.length === 0 ? (
            <div className="text-center py-8">
              <Battery className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-lg text-muted-foreground">No se encontraron baterías.</p>
              <p className="text-sm text-muted-foreground mt-2">
                Intenta ajustar tu búsqueda o crea una nueva batería.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[150px]">
                      <Button variant="ghost" onClick={() => handleSort('name')}>
                        Nombre
                        <ArrowUpDown className="ml-2 h-4 w-4" />
                      </Button>
                    </TableHead>
                    <TableHead>
                      <Button variant="ghost" onClick={() => handleSort('chemistry')}>
                        Tipo
                        <ArrowUpDown className="ml-2 h-4 w-4" />
                      </Button>
                    </TableHead>
                    <TableHead>
                      <Button variant="ghost" onClick={() => handleSort('status')}>
                        Estado
                        <ArrowUpDown className="ml-2 h-4 w-4" />
                      </Button>
                    </TableHead>
                    <TableHead>Visibilidad</TableHead>
                    <TableHead className="text-right">Acciones</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredBatteries.map((battery) => (
                    <TableRow key={battery.id}>
                      <TableCell className="font-medium">{battery.name}</TableCell>
                      <TableCell>{battery.chemistry}</TableCell>
                      <TableCell>
                        <Badge className={getBatteryStatusClass(battery.status, hiddenBatteryIds.has(battery.id))}>
                          {battery.status}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Switch
                          checked={!hiddenBatteryIds.has(battery.id)}
                          onCheckedChange={() => toggleBatteryVisibility(battery.id)}
                          aria-label={hiddenBatteryIds.has(battery.id) ? "Mostrar batería" : "Ocultar batería"}
                          title={hiddenBatteryIds.has(battery.id) ? "Mostrar batería" : "Ocultar batería"}
                        />
                      </TableCell>
                      <TableCell className="text-right flex space-x-2 justify-end">
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
                          onClick={() => {
                            // Parse location into country and city
                            let country = '';
                            let city = '';
                            let otherCountry = '';
                            let otherCity = '';
                            if (battery.location) {
                              const parts = battery.location.split(', ').map(p => p.trim());
                              if (parts.length === 2) {
                                city = parts[0];
                                country = parts[1];
                              } else if (parts.length === 1) {
                                city = parts[0]; // Assume it's a city or generic location
                              }

                              // Check if parsed country/city are in predefined lists
                              if (!COUNTRY_OPTIONS.includes(country)) {
                                otherCountry = country;
                                country = 'Otro';
                              }
                              if (country !== 'Otro' && (!CITY_OPTIONS_MAP[country] || !CITY_OPTIONS_MAP[country].includes(city))) {
                                otherCity = city;
                                city = 'Otro';
                              } else if (country === 'Otro' && city && city !== 'Otro') {
                                // If country is 'Otro' but a city was parsed, keep it as 'otherCity'
                                otherCity = city;
                                city = 'Otro';
                              }
                            }

                            setEditFormData({
                              id: battery.id,
                              name: battery.name,
                              model: battery.model || '',
                              manufacturer: battery.manufacturer || '',
                              serial_number: battery.serial_number || '',
                              // Renamed: full_charge_capacity to design_capacity
                              design_capacity: battery.full_charge_capacity || '',
                              design_capacity_unit: battery.design_capacity_unit || 'Wh', // Default to Wh
                              // New: nominal_capacity to current_capacity
                              current_capacity: battery.nominal_capacity || '', // Assuming nominal_capacity from backend
                              current_capacity_unit: battery.current_capacity_unit || (battery.design_capacity_unit || 'Wh'), // Default to design unit
                              designvoltage: battery.designvoltage || '',
                              chemistry: battery.chemistry || 'Li-ion',
                              installation_date: battery.installation_date ? new Date(battery.installation_date) : null,
                              // Location split into country and city
                              country: country || 'Otro',
                              city: city || 'Otro',
                              otherCountry: otherCountry,
                              otherCity: otherCity,
                              status: battery.status || 'active',
                              description: battery.description || '',
                              last_maintenance_date: battery.last_maintenance_date ? new Date(battery.last_maintenance_date) : null,
                              warranty_expiry_date: battery.warranty_expiry_date ? new Date(battery.warranty_expiry_date) : null,
                              monitoring_source: battery.monitoring_source || 'Simulado',
                              cycles: battery.cycles || 0, // New field
                            });
                            setIsEditModalOpen(true);
                          }}
                          title="Editar Batería"
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-800" title="Eliminar Batería">
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>¿Estás absolutamente seguro?</AlertDialogTitle>
                              <AlertDialogDescription>
                                Esta acción no se puede deshacer. Esto eliminará permanentemente
                                la batería de nuestros servidores.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancelar</AlertDialogCancel>
                              <AlertDialogAction onClick={() => handleDeleteBattery(battery.id)}>
                                Eliminar
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Edit Battery Dialog (AlertDialog para edición) */}
      {isEditModalOpen && editFormData && (
        <AlertDialog open={isEditModalOpen} onOpenChange={setIsEditModalOpen}>
          <AlertDialogContent className="sm:max-w-[500px]">
            <AlertDialogHeader>
              <AlertDialogTitle>Editar Batería</AlertDialogTitle>
              <AlertDialogDescription>
                Modifica los detalles de la batería.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <div className="grid gap-4 py-4 max-h-[70vh] overflow-y-auto pr-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">Nombre</Label>
                <Input id="name" value={editFormData.name} onChange={handleEditChange} className="col-span-3" />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="model" className="text-right">Modelo</Label>
                <Input id="model" value={editFormData.model} onChange={handleEditChange} className="col-span-3" />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="manufacturer" className="text-right">Fabricante</Label>
                <Input id="manufacturer" value={editFormData.manufacturer} onChange={handleEditChange} className="col-span-3" />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="serial_number" className="text-right">Número de Serie</Label>
                <Input id="serial_number" value={editFormData.serial_number} onChange={handleEditChange} className="col-span-3" />
              </div>

              {/* Capacidad de diseño (antes Capacidad de Carga Completa) */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="design_capacity" className="text-right">Capacidad de Diseño</Label>
                <Input
                  type="number"
                  id="design_capacity"
                  value={editFormData.design_capacity}
                  onChange={handleEditChange}
                  className="col-span-2"
                />
                <Select
                  value={editFormData.design_capacity_unit}
                  onValueChange={(value) => handleEditSelectChange(value, 'design_capacity_unit')}
                >
                  <SelectTrigger className="col-span-1">
                    <SelectValue placeholder="Unidad" />
                  </SelectTrigger>
                  <SelectContent>
                    {CAPACITY_UNITS.map(unit => (
                      <SelectItem key={unit} value={unit}>{unit}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Capacidad Actual (antes Capacidad Nominal) */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="current_capacity" className="text-right">Capacidad Actual</Label>
                <Input
                  type="number"
                  id="current_capacity"
                  value={editFormData.current_capacity}
                  onChange={handleEditChange}
                  className="col-span-2"
                />
                <Select
                  value={editFormData.current_capacity_unit || editFormData.design_capacity_unit || 'Wh'}
                  onValueChange={(value) => handleEditSelectChange(value, 'current_capacity_unit')}
                >
                  <SelectTrigger className="col-span-1">
                    <SelectValue placeholder="Unidad" />
                  </SelectTrigger>
                  <SelectContent>
                    {CAPACITY_UNITS.map(unit => (
                      <SelectItem key={unit} value={unit}>{unit}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Ciclos */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="cycles" className="text-right">Ciclos</Label>
                <Input
                  type="number"
                  id="cycles"
                  value={editFormData.cycles}
                  onChange={handleEditChange}
                  className="col-span-3"
                />
              </div>

              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="designvoltage" className="text-right">Voltaje Diseño (V)</Label>
                <Input type="number" id="designvoltage" value={editFormData.designvoltage} onChange={handleEditChange} className="col-span-3" />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="chemistry" className="text-right">Química</Label>
                <Select value={editFormData.chemistry} onValueChange={(value) => handleEditSelectChange(value, 'chemistry')}>
                  <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona la química" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Li-ion">Li-ion</SelectItem>
                    <SelectItem value="NiMH">NiMH</SelectItem>
                    <SelectItem value="LiPo">LiPo</SelectItem>
                    <SelectItem value="PbA">Plomo-Ácido</SelectItem>
                    <SelectItem value="other">Otro</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="installation_date" className="text-right">Fecha Instalación</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant={"outline"} className={cn("col-span-3 justify-start text-left font-normal", !editFormData.installation_date && "text-muted-foreground")}>
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {editFormData.installation_date ? format(editFormData.installation_date, "PPP") : <span>Selecciona una fecha</span>}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar mode="single" selected={editFormData.installation_date} onSelect={(date) => handleEditDateSelect(date, 'installation_date')} initialFocus />
                  </PopoverContent>
                </Popover>
              </div>

              {/* País de Ubicación */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="country" className="text-right">País de Ubicación</Label>
                <Select value={editFormData.country} onValueChange={(value) => {
                  setEditFormData(prev => ({ ...prev, country: value, city: 'Otro', otherCity: '' })); // Reset city when country changes
                }}>
                  <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona el país" /></SelectTrigger>
                  <SelectContent>
                    {COUNTRY_OPTIONS.map(option => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {editFormData.country === 'Otro' && (
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="otherCountry" className="text-right">Especifica País</Label>
                  <Input id="otherCountry" value={editFormData.otherCountry} onChange={handleEditChange} className="col-span-3" />
                </div>
              )}

              {/* Ciudad de Ubicación */}
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="city" className="text-right">Ciudad de Ubicación</Label>
                <Select value={editFormData.city} onValueChange={(value) => handleEditSelectChange(value, 'city')}>
                  <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona la ciudad" /></SelectTrigger>
                  <SelectContent>
                    {CITY_OPTIONS_MAP[editFormData.country]?.map(option => (
                      <SelectItem key={option} value={option}>{option}</SelectItem>
                    )) || <SelectItem value="Otro">Otro</SelectItem>}
                  </SelectContent>
                </Select>
              </div>
              {editFormData.city === 'Otro' && (
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="otherCity" className="text-right">Especifica Ciudad</Label>
                  <Input id="otherCity" value={editFormData.otherCity} onChange={handleEditChange} className="col-span-3" />
                </div>
              )}


              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="status" className="text-right">Estado</Label>
                <Select value={editFormData.status} onValueChange={(value) => handleEditSelectChange(value, 'status')}>
                  <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona el estado" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="active">Activa</SelectItem>
                    <SelectItem value="inactive">Inactiva</SelectItem>
                    <SelectItem value="maintenance">Mantenimiento</SelectItem>
                    <SelectItem value="retired">Retirada</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="monitoring_source" className="text-right">Fuente de Monitoreo</Label>
                <Select value={editFormData.monitoring_source} onValueChange={(value) => handleEditSelectChange(value, 'monitoring_source')}>
                  <SelectTrigger className="col-span-3"><SelectValue placeholder="Selecciona la fuente" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Real (Directo)">Real (Directo)</SelectItem>
                    <SelectItem value="Simulado">Simulado</SelectItem>
                    <SelectItem value="Histórico (Archivo/BD)">Histórico (Archivo/BD)</SelectItem>
                    <SelectItem value="Prueba">Prueba</SelectItem>
                    <SelectItem value="otro">Otro</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="description" className="text-right">Descripción</Label>
                <Input id="description" value={editFormData.description} onChange={handleEditChange} className="col-span-3" />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="last_maintenance_date" className="text-right">Último Mantenimiento</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant={"outline"} className={cn("col-span-3 justify-start text-left font-normal", !editFormData.last_maintenance_date && "text-muted-foreground")}>
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {editFormData.last_maintenance_date ? format(editFormData.last_maintenance_date, "PPP") : <span>Selecciona una fecha</span>}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar mode="single" selected={editFormData.last_maintenance_date} onSelect={(date) => handleEditDateSelect(date, 'last_maintenance_date')} initialFocus />
                  </PopoverContent>
                </Popover>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="warranty_expiry_date" className="text-right">Fin Garantía</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant={"outline"} className={cn("col-span-3 justify-start text-left font-normal", !editFormData.warranty_expiry_date && "text-muted-foreground")}>
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {editFormData.warranty_expiry_date ? format(editFormData.warranty_expiry_date, "PPP") : <span>Selecciona una fecha</span>}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar mode="single" selected={editFormData.warranty_expiry_date} onSelect={(date) => handleEditDateSelect(date, 'warranty_expiry_date')} initialFocus />
                  </PopoverContent>
                </Popover>
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
