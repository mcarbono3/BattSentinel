import { Loader2 } from 'lucide-react'
import battSentinelLogo from '@/assets/BattSentinel_Logo.png'

export default function LoadingScreen({ message = 'Cargando...' }) {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <img 
            src={battSentinelLogo} 
            alt="BattSentinel" 
            className="h-16 w-auto"
          />
        </div>
        
        <div className="flex items-center justify-center space-x-3">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <span className="text-lg font-medium text-foreground">{message}</span>
        </div>
        
        <div className="w-64 h-1 bg-muted rounded-full overflow-hidden">
          <div className="h-full bg-primary rounded-full animate-pulse"></div>
        </div>
      </div>
    </div>
  )
}

