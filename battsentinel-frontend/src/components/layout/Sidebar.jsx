import { useState } from 'react'
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
  Zap,
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

export default function Sidebar({ isOpen, onToggle }) {
  const location = useLocation()
  const { user } = useAuth()
  const { batteries, batteryCount, criticalBatteries } = useBattery()
  
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
        return criticalBatteries.length > 0 ? (
          <Badge variant="destructive" className="ml-auto">
            {criticalBatteries.length}
          </Badge>
        ) : null
      default:
        return null
    }
  }

  return (
    <div className={cn(
      "relative flex flex-col bg-card border-r border-border transition-all duration-300",
      isOpen ? "w-64" : "w-16"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className={cn(
          "flex items-center space-x-3 transition-opacity duration-200",
          isOpen ? "opacity-100" : "opacity-0"
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
        
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="h-8 w-8 p-0"
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
                {user.username?.charAt(0).toUpperCase()}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                {user.username}
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
            
            // Skip items that require a battery if none are available
            if (item.requiresBattery && batteryCount === 0) {
              return null
            }

            return (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  "flex items-center space-x-3 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200",
                  "hover:bg-accent hover:text-accent-foreground",
                  active 
                    ? "bg-primary text-primary-foreground" 
                    : "text-muted-foreground"
                )}
                title={!isOpen ? item.title : undefined}
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

