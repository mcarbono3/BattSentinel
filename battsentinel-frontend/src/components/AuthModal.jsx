import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { useAuth } from '@/contexts/AuthContext'
import { Eye, EyeOff, Loader2, AlertCircle, X } from 'lucide-react'

export default function AuthModal({ isOpen, onClose }) {
  const navigate = useNavigate()
  const { login, register } = useAuth()
  
  const [activeTab, setActiveTab] = useState('login')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  const [loginForm, setLoginForm] = useState({
    username: '',
    password: ''
  })
  
  const [registerForm, setRegisterForm] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'user'
  })

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const result = await login(loginForm)
      if (result.success) {
        onClose()
        navigate('/dashboard')
      } else {
        setError(result.error || 'Error al iniciar sesión')
      }
    } catch (error) {
      setError('Error de conexión. Verifique su conexión a internet.')
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    // Validaciones
    if (registerForm.password !== registerForm.confirmPassword) {
      setError('Las contraseñas no coinciden')
      setLoading(false)
      return
    }

    if (registerForm.password.length < 6) {
      setError('La contraseña debe tener al menos 6 caracteres')
      setLoading(false)
      return
    }

    try {
      const { confirmPassword, ...userData } = registerForm
      const result = await register(userData)
      if (result.success) {
        onClose()
        navigate('/dashboard')
      } else {
        setError(result.error || 'Error al registrar usuario')
      }
    } catch (error) {
      setError('Error de conexión. Verifique su conexión a internet.')
    } finally {
      setLoading(false)
    }
  }

  const updateLoginForm = (field, value) => {
    setLoginForm(prev => ({ ...prev, [field]: value }))
    setError('')
  }

  const updateRegisterForm = (field, value) => {
    setRegisterForm(prev => ({ ...prev, [field]: value }))
    setError('')
  }

  const handleClose = () => {
    setError('')
    setLoginForm({ username: '', password: '' })
    setRegisterForm({ username: '', email: '', password: '', confirmPassword: '', role: 'user' })
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-center">
            {activeTab === 'login' ? 'Iniciar Sesión' : 'Crear Cuenta'}
          </DialogTitle>
          <DialogDescription className="text-center">
            {activeTab === 'login' 
              ? 'Accede a tu cuenta de BattSentinel' 
              : 'Regístrate para comenzar a monitorear tus baterías'
            }
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">Iniciar Sesión</TabsTrigger>
              <TabsTrigger value="register">Registrarse</TabsTrigger>
            </TabsList>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <TabsContent value="login" className="space-y-4">
              <form onSubmit={handleLogin} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="login-username">Usuario o Email</Label>
                  <Input
                    id="login-username"
                    type="text"
                    placeholder="Ingrese su usuario o email"
                    value={loginForm.username}
                    onChange={(e) => updateLoginForm('username', e.target.value)}
                    required
                    disabled={loading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="login-password">Contraseña</Label>
                  <div className="relative">
                    <Input
                      id="login-password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="Ingrese su contraseña"
                      value={loginForm.password}
                      onChange={(e) => updateLoginForm('password', e.target.value)}
                      required
                      disabled={loading}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowPassword(!showPassword)}
                      disabled={loading}
                    >
                      {showPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>

                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Iniciando sesión...
                    </>
                  ) : (
                    'Iniciar Sesión'
                  )}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="register" className="space-y-4">
              <form onSubmit={handleRegister} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="register-username">Nombre de Usuario</Label>
                  <Input
                    id="register-username"
                    type="text"
                    placeholder="Elija un nombre de usuario"
                    value={registerForm.username}
                    onChange={(e) => updateRegisterForm('username', e.target.value)}
                    required
                    disabled={loading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="register-email">Email</Label>
                  <Input
                    id="register-email"
                    type="email"
                    placeholder="Ingrese su email"
                    value={registerForm.email}
                    onChange={(e) => updateRegisterForm('email', e.target.value)}
                    required
                    disabled={loading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="register-password">Contraseña</Label>
                  <div className="relative">
                    <Input
                      id="register-password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="Cree una contraseña segura"
                      value={registerForm.password}
                      onChange={(e) => updateRegisterForm('password', e.target.value)}
                      required
                      disabled={loading}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowPassword(!showPassword)}
                      disabled={loading}
                    >
                      {showPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="register-confirm-password">Confirmar Contraseña</Label>
                  <Input
                    id="register-confirm-password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Confirme su contraseña"
                    value={registerForm.confirmPassword}
                    onChange={(e) => updateRegisterForm('confirmPassword', e.target.value)}
                    required
                    disabled={loading}
                  />
                </div>

                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creando cuenta...
                    </>
                  ) : (
                    'Crear Cuenta'
                  )}
                </Button>
              </form>
            </TabsContent>
          </Tabs>

          {/* Demo credentials */}
          <Card className="bg-muted/50">
            <CardContent className="pt-4">
              <p className="text-sm font-medium text-foreground mb-2">Credenciales de demostración:</p>
              <div className="text-xs text-muted-foreground space-y-1">
                <p><strong>Usuario:</strong> admin</p>
                <p><strong>Contraseña:</strong> admin123</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </DialogContent>
    </Dialog>
  )
}

