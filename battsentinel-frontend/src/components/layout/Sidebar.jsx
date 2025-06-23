import { useState } from 'react' // <--- No es necesario si isOpen y onToggle vienen de props
import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import { useAuth } from '@/contexts/AuthContext'
import { useBattery } from '@/contexts/BatteryContext'
import battSentinelLogo from '@/assets/BattSentinel_Logo.png'

import {
  LayoutDashboard,
  Battery,
  Bot,
  BarChart3,
  Bell,
  Settings,
  ChevronLeft,
  ChevronRight,
  // Zap, // No utilizada, puedes eliminarla si no la necesitas
  Activity,
  AlertTriangle
} from 'lucide-react'

const navigationItems = [
  {
    title: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    description: 'Vista general del sistema'
  },
  {
    title: 'Baterías',
    href: '/batteries',
    icon: Battery,
    description: 'Gestión de baterías'
  },
  {
    title: 'Gemelo Digital',
    href: '/digital-twin',
    icon: Bot,
    description: 'Simulación y modelado',
    requiresBattery: true
  },
  {
    title: 'Análisis',
    href: '/analytics',
    icon: BarChart3,
    description: 'Análisis avanzado con IA'
  },
  {
    title: 'Alertas',
    href: '/alerts',
    icon: Bell,
    description: 'Notificaciones y alertas'
  },
  {
    title: 'Configuración',
    href: '/settings',
    icon: Settings,
    description: 'Configuración del sistema'
  }
]

// Recibe isOpen y onToggle como props desde el componente padre
export default function Sidebar({ isOpen, onToggle }) {
  const location = useLocation()
  const { user } = useAuth()
  const { batteries, alerts } = useBattery()

  // Calcula batteryCount si no viene del contexto
  // const batteryCount = batteries?.length || 0; // Línea original 
  const batteryCount = batteries?.length > 0 ? batteries.length : 1; // Fuerza a 1 si no hay
  // Calcula criticalBatteries aquí si no viene del contexto, o verifica si tu contexto ya lo proporciona
  const criticalBatteries = batteries?.filter(battery =>
    battery.soh < 70 || battery.soc < 20 || battery.temperature > 45
  ) || [];

  const activeAlerts = alerts?.filter(alert => alert.status === 'active') || []
  
  const isActive = (href) => {
    if (href === '/dashboard') {
      return location.pathname === '/' || location.pathname === '/dashboard'
    }
    return location.pathname.startsWith(href)
  }

  const getItemBadge = (item) => {
    switch (item.href) {
      case '/batteries':
        return batteryCount > 0 ? (
          <Badge variant="secondary" className="ml-auto">
            {batteryCount}
          </Badge>
        ) : null
      case '/alerts':
        return activeAlerts.length > 0 ? (
          <Badge variant="destructive" className="ml-auto">
            {activeAlerts.length}
          </Badge>
        ) : null
      default:
        return null
    }
  }

  return (
    <div className={cn(
      "relative flex flex-col bg-card border-r border-border transition-all duration-300 h-screen", // Añadir h-screen para altura completa
      isOpen ? "w-64" : "w-16",
      // Ajustes responsivos: el sidebar se ocultará en pantallas pequeñas y aparecerá con un overlay si decides implementarlo
      "hidden md:flex" // Ocultar por defecto en móvil, mostrar en pantallas medianas y grandes
    )}>
      {/* Header con el botón de toggle siempre visible y ajustado */}
      <div className="flex items-center justify-between p-4 border-b border-border relative"> {/* relative para posicionar el botón de toggle */}
        <div className={cn(
          "flex items-center space-x-3 transition-opacity duration-200",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none" // pointer-events-none para que el botón toggle sea clicable
        )}>
          <img 
            src={battSentinelLogo} 
            alt="BattSentinel" 
            className="h-8 w-8"
          />
          <div>
            <h1 className="text-lg font-bold text-foreground">BattSentinel</h1>
            <p className="text-xs text-muted-foreground">Monitoreo Inteligente</p>
          </div>
        </div>
        
        {/* Botón de toggle siempre en la esquina superior derecha del header del sidebar */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className={cn(
            "h-8 w-8 p-0",
            isOpen ? "" : "absolute top-4 right-4" // Si está colapsado, posiciona en la esquina superior derecha
          )}
        >
          {isOpen ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* User Info */}
      {isOpen && user && (
        <div className="p-4 border-b border-border">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
              <span className="text-sm font-medium text-primary-foreground">
                {user.name?.charAt(0).toUpperCase() || 'U'}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                {user.name || 'Usuario'}
              </p>
              <p className="text-xs text-muted-foreground capitalize">
                {user.role}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <ScrollArea className="flex-1 px-3 py-4">
        <nav className="space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            const badge = getItemBadge(item)
            
            const isDisabled = item.requiresBattery && batteryCount === 0;

            if (item.requiresBattery && batteryCount === 0) {
              return null
            }

            return (
              <Link
                key={item.href}
                to={isDisabled ? '#' : item.href}
                className={cn(
                  "flex items-center space-x-3 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200",
                  "hover:bg-accent hover:text-accent-foreground",
                  active 
                    ? "bg-primary text-primary-foreground" 
                    : "text-muted-foreground",
                  isDisabled && "opacity-50 cursor-not-allowed",
                  !isOpen && "justify-center" // Centrar icono cuando está colapsado
                )}
                title={!isOpen ? item.title : undefined}
                onClick={(e) => {
                  if (isDisabled) {
                    e.preventDefault();
                  }
                }}
              >
                <Icon className="h-4 w-4 flex-shrink-0" />
                {isOpen && (
                  <>
                    <span className="flex-1">{item.title}</span>
                    {badge}
                  </>
                )}
              </Link>
            )
          })}
        </nav>
      </ScrollArea>

      {/* System Status */}
      {isOpen && (
        <div className="p-4 border-t border-border">
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Estado del Sistema</span>
              <div className="flex items-center space-x-1">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse-green"></div>
                <span className="text-green-600 font-medium">Activo</span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex items-center space-x-2">
                <Battery className="h-3 w-3 text-blue-500" />
                <span className="text-muted-foreground">{batteryCount}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Activity className="h-3 w-3 text-green-500" />
                <span className="text-muted-foreground">Online</span>
              </div>
            </div>
            
            {criticalBatteries.length > 0 && (
              <div className="flex items-center space-x-2 p-2 bg-red-50 dark:bg-red-900/20 rounded-md">
                <AlertTriangle className="h-3 w-3 text-red-500" />
                <span className="text-xs text-red-600 dark:text-red-400">
                  {criticalBatteries.length} alerta{criticalBatteries.length !== 1 ? 's' : ''} crítica{criticalBatteries.length !== 1 ? 's' : ''}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Collapsed state indicators */}
      {!isOpen && (
        <div className="p-2 border-t border-border">
          <div className="flex flex-col items-center space-y-2">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse-green" title="Sistema Activo"></div>
            {criticalBatteries.length > 0 && (
              <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse-red" title="Alertas Críticas"></div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
