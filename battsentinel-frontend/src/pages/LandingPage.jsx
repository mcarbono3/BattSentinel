import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import AuthModal from '@/components/AuthModal'
import { 
  Zap, 
  Shield, 
  BarChart3, 
  Brain, 
  Eye, 
  AlertTriangle,
  Users,
  TrendingUp,
  Cpu,
  Activity,
  ChevronRight,
  Play,
  ArrowRight,
  CheckCircle,
  Star,
  Gauge,
  Thermometer,
  Battery,
  Lightbulb,
  Target,
  Globe
} from 'lucide-react'
import battSentinelLogo from '@/assets/BattSentinel_Logo.png'

export default function LandingPage() {
  const navigate = useNavigate()
  const [currentFeature, setCurrentFeature] = useState(0)
  const [isVisible, setIsVisible] = useState({})
  const [showAuthModal, setShowAuthModal] = useState(false)

  const features = [
    {
      icon: <Zap className="h-8 w-8" />,
      title: "Gemelo Digital Interactivo",
      description: "Simulación en tiempo real del comportamiento de las baterías con capacidad de predicción avanzada.",
      details: "Visualización dinámica de parámetros, control de simulación y predicciones de vida útil."
    },
    {
      icon: <Brain className="h-8 w-8" />,
      title: "IA y AutoML",
      description: "Detección inteligente de fallas utilizando Machine Learning y Deep Learning avanzado.",
      details: "Random Forest, SVM, Neural Networks e Isolation Forest para análisis predictivo."
    },
    {
      icon: <Eye className="h-8 w-8" />,
      title: "XAI - Explicabilidad",
      description: "Implementación de SHAP y LIME para explicar las decisiones del sistema de IA.",
      details: "Transparencia total en las predicciones y recomendaciones del sistema."
    },
    {
      icon: <Thermometer className="h-8 w-8" />,
      title: "Análisis Térmico",
      description: "Procesamiento de imágenes térmicas para detección de hotspots y anomalías.",
      details: "OpenCV para análisis de distribución térmica y alertas preventivas."
    },
    {
      icon: <AlertTriangle className="h-8 w-8" />,
      title: "Sistema de Alertas",
      description: "Notificaciones inteligentes y preventivas basadas en análisis predictivo.",
      details: "Alertas críticas, advertencias y notificaciones informativas en tiempo real."
    },
    {
      icon: <Users className="h-8 w-8" />,
      title: "Gestión Multiusuario",
      description: "Roles diferenciados para administradores, técnicos y usuarios finales.",
      details: "Control de acceso granular y perfiles personalizables por rol."
    }
  ]

  const analysisTypes = [
    {
      title: "Clasificación de Fallas",
      description: "Detección automática de degradación acelerada, cortocircuitos y sobrecarga",
      icon: <Shield className="h-6 w-6" />
    },
    {
      title: "Predicción RUL",
      description: "Estimación de vida útil restante con precisión avanzada",
      icon: <TrendingUp className="h-6 w-6" />
    },
    {
      title: "Análisis de Anomalías",
      description: "Identificación de patrones inusuales y comportamientos atípicos",
      icon: <Target className="h-6 w-6" />
    },
    {
      title: "Optimización de Rendimiento",
      description: "Recomendaciones para maximizar eficiencia y durabilidad",
      icon: <Gauge className="h-6 w-6" />
    }
  ]

  const benefits = [
    {
      title: "Reducción de Costos",
      description: "Hasta 40% menos gastos por prevención de fallas",
      icon: <TrendingUp className="h-6 w-6" />
    },
    {
      title: "Seguridad Mejorada",
      description: "Detección anticipada de condiciones peligrosas",
      icon: <Shield className="h-6 w-6" />
    },
    {
      title: "Optimización Energética",
      description: "Maximización de la vida útil de las baterías",
      icon: <Battery className="h-6 w-6" />
    },
    {
      title: "Decisiones Informadas",
      description: "Análisis basado en datos y predicciones precisas",
      icon: <Brain className="h-6 w-6" />
    }
  ]

  const technologies = [
    { name: "React 19.1.0", category: "Frontend" },
    { name: "Flask 3.1.0", category: "Backend" },
    { name: "TensorFlow 2.18.0", category: "Deep Learning" },
    { name: "scikit-learn 1.6.0", category: "Machine Learning" },
    { name: "SHAP 0.46.0", category: "XAI" },
    { name: "OpenCV 4.10.0", category: "Visión Computacional" }
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length)
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setIsVisible(prev => ({
            ...prev,
            [entry.target.id]: entry.isIntersecting
          }))
        })
      },
      { threshold: 0.1 }
    )

    document.querySelectorAll('[id]').forEach((el) => {
      observer.observe(el)
    })

    return () => observer.disconnect()
  }, [])

  const handleGetStarted = () => {
    setShowAuthModal(true)
  }

  const handleDemo = () => {
    // Aquí se podría implementar una demo interactiva
    setShowAuthModal(true)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center space-x-4">
            <img src={battSentinelLogo} alt="BattSentinel" className="h-10 w-10" />
            <div>
              <h1 className="text-xl font-bold text-foreground">BattSentinel</h1>
              <p className="text-xs text-muted-foreground">Monitoreo Inteligente</p>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center space-x-6">
            <a href="#features" className="text-sm font-medium hover:text-primary transition-colors">
              Características
            </a>
            <a href="#technology" className="text-sm font-medium hover:text-primary transition-colors">
              Tecnología
            </a>
            <a href="#benefits" className="text-sm font-medium hover:text-primary transition-colors">
              Beneficios
            </a>
            <Button onClick={handleGetStarted} size="sm">
              Acceder
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-background via-background to-primary/5 py-20 md:py-32">
        <div className="container relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="space-y-4">
                <Badge variant="secondary" className="w-fit">
                  <Lightbulb className="mr-2 h-3 w-3" />
                  Industria 4.0 + IA
                </Badge>
                <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
                  Monitoreo{' '}
                  <span className="text-primary">Inteligente</span>{' '}
                  de Baterías
                </h1>
                <p className="text-xl text-muted-foreground leading-relaxed">
                  Sistema avanzado para el monitoreo, análisis y diagnóstico de baterías de ion de litio 
                  basado en metodologías de la Industria 4.0 e Inteligencia Artificial.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Button size="lg" onClick={handleGetStarted} className="group">
                  Comenzar Ahora
                  <ChevronRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Button>
                <Button size="lg" variant="outline" onClick={handleDemo} className="group">
                  <Play className="mr-2 h-4 w-4" />
                  Ver Demo
                </Button>
              </div>

              <div className="flex items-center space-x-6 text-sm text-muted-foreground">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Gemelo Digital</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>IA Explicable</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Análisis Predictivo</span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="relative bg-gradient-to-br from-primary/10 to-primary/5 rounded-2xl p-8">
                <div className="grid grid-cols-2 gap-4">
                  <Card className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Battery className="h-5 w-5 text-green-500" />
                      <span className="text-sm font-medium">SOC</span>
                    </div>
                    <div className="text-2xl font-bold">85%</div>
                    <div className="text-xs text-muted-foreground">Estado de Carga</div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Activity className="h-5 w-5 text-blue-500" />
                      <span className="text-sm font-medium">SOH</span>
                    </div>
                    <div className="text-2xl font-bold">92%</div>
                    <div className="text-xs text-muted-foreground">Estado de Salud</div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Thermometer className="h-5 w-5 text-orange-500" />
                      <span className="text-sm font-medium">Temp</span>
                    </div>
                    <div className="text-2xl font-bold">25°C</div>
                    <div className="text-xs text-muted-foreground">Temperatura</div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Gauge className="h-5 w-5 text-purple-500" />
                      <span className="text-sm font-medium">RUL</span>
                    </div>
                    <div className="text-2xl font-bold">2.3y</div>
                    <div className="text-xs text-muted-foreground">Vida Útil</div>
                  </Card>
                </div>
                
                <div className="mt-6 p-4 bg-background/50 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Brain className="h-5 w-5 text-primary" />
                    <span className="text-sm font-medium">Análisis IA</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Estado: <span className="text-green-500 font-medium">Óptimo</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Próxima revisión recomendada en 30 días
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-muted/30">
        <div className="container">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              <Star className="mr-2 h-3 w-3" />
              Características Principales
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Tecnología de Vanguardia
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Combinamos inteligencia artificial, gemelos digitales y análisis predictivo 
              para ofrecer la solución más avanzada de monitoreo de baterías.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              {features.map((feature, index) => (
                <Card 
                  key={index}
                  className={`p-6 cursor-pointer transition-all duration-300 ${
                    currentFeature === index 
                      ? 'border-primary shadow-lg scale-105' 
                      : 'hover:shadow-md'
                  }`}
                  onClick={() => setCurrentFeature(index)}
                >
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-lg ${
                      currentFeature === index 
                        ? 'bg-primary text-primary-foreground' 
                        : 'bg-primary/10 text-primary'
                    }`}>
                      {feature.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                      <p className="text-muted-foreground mb-2">{feature.description}</p>
                      {currentFeature === index && (
                        <p className="text-sm text-primary font-medium">{feature.details}</p>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>

            <div className="relative">
              <div className="bg-gradient-to-br from-primary/5 to-primary/10 rounded-2xl p-8 h-96 flex items-center justify-center">
                <div className="text-center">
                  <div className="mb-6">
                    {features[currentFeature].icon}
                  </div>
                  <h3 className="text-2xl font-bold mb-4">
                    {features[currentFeature].title}
                  </h3>
                  <p className="text-muted-foreground">
                    {features[currentFeature].details}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* AI Analysis Types */}
      <section className="py-20">
        <div className="container">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              <Brain className="mr-2 h-3 w-3" />
              Análisis con IA
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Tipos de Análisis Implementados
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Nuestros modelos de inteligencia artificial proporcionan análisis 
              profundos y predicciones precisas para optimizar el rendimiento de las baterías.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {analysisTypes.map((analysis, index) => (
              <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="mb-4 p-3 bg-primary/10 rounded-lg w-fit mx-auto">
                  {analysis.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2">{analysis.title}</h3>
                <p className="text-sm text-muted-foreground">{analysis.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section id="technology" className="py-20 bg-muted/30">
        <div className="container">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              <Cpu className="mr-2 h-3 w-3" />
              Stack Tecnológico
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Tecnologías de Última Generación
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Construido con las mejores herramientas y frameworks para garantizar 
              rendimiento, escalabilidad y confiabilidad.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {technologies.map((tech, index) => (
              <Card key={index} className="p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{tech.name}</div>
                    <div className="text-sm text-muted-foreground">{tech.category}</div>
                  </div>
                  <Badge variant="outline">{tech.category}</Badge>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits */}
      <section id="benefits" className="py-20">
        <div className="container">
          <div className="text-center mb-16">
            <Badge variant="secondary" className="mb-4">
              <TrendingUp className="mr-2 h-3 w-3" />
              Beneficios
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Impacto Medible en su Operación
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              BattSentinel no solo monitorea, sino que transforma la gestión 
              de baterías con resultados tangibles y medibles.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {benefits.map((benefit, index) => (
              <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="mb-4 p-3 bg-primary/10 rounded-lg w-fit mx-auto">
                  {benefit.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2">{benefit.title}</h3>
                <p className="text-sm text-muted-foreground">{benefit.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary text-primary-foreground">
        <div className="container text-center">
          <div className="max-w-3xl mx-auto space-y-8">
            <h2 className="text-3xl md:text-4xl font-bold">
              ¿Listo para Revolucionar su Monitoreo de Baterías?
            </h2>
            <p className="text-xl opacity-90">
              Únase a la nueva era del monitoreo inteligente con BattSentinel. 
              Comience hoy mismo y experimente el poder de la IA aplicada a la gestión de baterías.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                variant="secondary" 
                onClick={handleGetStarted}
                className="group"
              >
                Acceder al Sistema
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                onClick={handleDemo}
                className="border-primary-foreground text-primary-foreground hover:bg-primary-foreground hover:text-primary"
              >
                <Play className="mr-2 h-4 w-4" />
                Ver Demostración
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t bg-background">
        <div className="container">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <img src={battSentinelLogo} alt="BattSentinel" className="h-8 w-8" />
                <span className="font-bold">BattSentinel</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Sistema inteligente de monitoreo de baterías basado en IA y metodologías de la Industria 4.0.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Producto</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#features" className="hover:text-foreground transition-colors">Características</a></li>
                <li><a href="#technology" className="hover:text-foreground transition-colors">Tecnología</a></li>
                <li><a href="#benefits" className="hover:text-foreground transition-colors">Beneficios</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Soporte</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#" className="hover:text-foreground transition-colors">Documentación</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">API</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Contacto</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Empresa</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#" className="hover:text-foreground transition-colors">Acerca de</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Privacidad</a></li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 pt-8 border-t text-center text-sm text-muted-foreground">
            <p>&copy; 2025 BattSentinel. Todos los derechos reservados.</p>
          </div>
        </div>
      </footer>

      {/* Auth Modal */}
      <AuthModal 
        isOpen={showAuthModal} 
        onClose={() => setShowAuthModal(false)} 
      />
    </div>
  )
}

