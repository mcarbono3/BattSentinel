import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
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
  Globe,
  Menu,
  X,
  ChevronDown
} from 'lucide-react'

export default function LandingPage() {
  const navigate = useNavigate()
  const [currentFeature, setCurrentFeature] = useState(0)
  const [isVisible, setIsVisible] = useState({})
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [activeSection, setActiveSection] = useState('hero')
  
  // Referencias para smooth scroll
  const heroRef = useRef(null)
  const featuresRef = useRef(null)
  const technologyRef = useRef(null)
  const benefitsRef = useRef(null)

  const features = [
    {
      icon: <Zap className="h-8 w-8" />,
      title: "Gemelo Digital Interactivo",
      description: "Simulación en tiempo real del comportamiento de las baterías con capacidad de predicción avanzada.",
      details: "Visualización dinámica de parámetros, control de simulación y predicciones de vida útil.",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Brain className="h-8 w-8" />,
      title: "IA y AutoML",
      description: "Detección inteligente de fallas utilizando Machine Learning y Deep Learning avanzado.",
      details: "Random Forest, SVM, Neural Networks e Isolation Forest para análisis predictivo.",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      icon: <Eye className="h-8 w-8" />,
      title: "XAI - Explicabilidad",
      description: "Implementación de SHAP y LIME para explicar las decisiones del sistema de IA.",
      details: "Transparencia total en las predicciones y recomendaciones del sistema.",
      gradient: "from-green-500 to-emerald-500"
    },
    {
      icon: <Thermometer className="h-8 w-8" />,
      title: "Análisis Térmico",
      description: "Procesamiento de imágenes térmicas para detección de hotspots y anomalías.",
      details: "OpenCV para análisis de distribución térmica y alertas preventivas.",
      gradient: "from-orange-500 to-red-500"
    },
    {
      icon: <AlertTriangle className="h-8 w-8" />,
      title: "Sistema de Alertas",
      description: "Notificaciones inteligentes y preventivas basadas en análisis predictivo.",
      details: "Alertas críticas, advertencias y notificaciones informativas en tiempo real.",
      gradient: "from-yellow-500 to-orange-500"
    },
    {
      icon: <Users className="h-8 w-8" />,
      title: "Gestión Multiusuario",
      description: "Roles diferenciados para administradores, técnicos y usuarios finales.",
      details: "Control de acceso granular y perfiles personalizables por rol.",
      gradient: "from-indigo-500 to-purple-500"
    }
  ]

  const analysisTypes = [
    {
      title: "Clasificación de Fallas",
      description: "Detección automática de degradación acelerada, cortocircuitos y sobrecarga",
      icon: <Shield className="h-6 w-6" />,
      color: "text-red-500"
    },
    {
      title: "Predicción RUL",
      description: "Estimación de vida útil restante con precisión avanzada",
      icon: <TrendingUp className="h-6 w-6" />,
      color: "text-blue-500"
    },
    {
      title: "Análisis de Anomalías",
      description: "Identificación de patrones inusuales y comportamientos atípicos",
      icon: <Target className="h-6 w-6" />,
      color: "text-purple-500"
    },
    {
      title: "Optimización de Rendimiento",
      description: "Recomendaciones para maximizar eficiencia y durabilidad",
      icon: <Gauge className="h-6 w-6" />,
      color: "text-green-500"
    }
  ]

  const benefits = [
    {
      title: "Reducción de Costos",
      description: "Hasta 40% menos gastos por prevención de fallas",
      icon: <TrendingUp className="h-6 w-6" />,
      metric: "40%",
      color: "text-green-500"
    },
    {
      title: "Seguridad Mejorada",
      description: "Detección anticipada de condiciones peligrosas",
      icon: <Shield className="h-6 w-6" />,
      metric: "99.9%",
      color: "text-blue-500"
    },
    {
      title: "Optimización Energética",
      description: "Maximización de la vida útil de las baterías",
      icon: <Battery className="h-6 w-6" />,
      metric: "+25%",
      color: "text-purple-500"
    },
    {
      title: "Decisiones Informadas",
      description: "Análisis basado en datos y predicciones precisas",
      icon: <Brain className="h-6 w-6" />,
      metric: "100%",
      color: "text-orange-500"
    }
  ]

  const technologies = [
    { name: "React 19.1.0", category: "Frontend", color: "bg-blue-100 text-blue-800" },
    { name: "Flask 3.1.0", category: "Backend", color: "bg-green-100 text-green-800" },
    { name: "TensorFlow 2.18.0", category: "Deep Learning", color: "bg-orange-100 text-orange-800" },
    { name: "scikit-learn 1.6.0", category: "Machine Learning", color: "bg-purple-100 text-purple-800" },
    { name: "SHAP 0.46.0", category: "XAI", color: "bg-pink-100 text-pink-800" },
    { name: "OpenCV 4.10.0", category: "Visión Computacional", color: "bg-indigo-100 text-indigo-800" }
  ]

  // Smooth scroll function
  const scrollToSection = (sectionRef, sectionName) => {
    sectionRef.current?.scrollIntoView({ 
      behavior: 'smooth',
      block: 'start'
    })
    setActiveSection(sectionName)
    setMobileMenuOpen(false)
  }

  // Auto-rotate features
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length)
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  // Intersection Observer for animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(prev => ({
              ...prev,
              [entry.target.id]: true
            }))
            setActiveSection(entry.target.id)
          }
        })
      },
      { threshold: 0.3 }
    )

    const sections = document.querySelectorAll('[data-section]')
    sections.forEach(section => observer.observe(section))

    return () => observer.disconnect()
  }, [])

  // Handle direct access to dashboard - Sin autenticación
  const handleAccessSystem = () => {
    navigate('/dashboard')
  }

  const handleDemoAccess = () => {
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Modern Header */}
      <header className="fixed top-0 w-full bg-white/80 backdrop-blur-md border-b border-gray-200/50 z-50 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  BattSentinel
                </h1>
                <p className="text-xs text-gray-500">Monitoreo Inteligente</p>
              </div>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-8">
              <button
                onClick={() => scrollToSection(featuresRef, 'features')}
                className={`text-sm font-medium transition-colors hover:text-blue-600 ${
                  activeSection === 'features' ? 'text-blue-600' : 'text-gray-700'
                }`}
              >
                Características
              </button>
              <button
                onClick={() => scrollToSection(technologyRef, 'technology')}
                className={`text-sm font-medium transition-colors hover:text-blue-600 ${
                  activeSection === 'technology' ? 'text-blue-600' : 'text-gray-700'
                }`}
              >
                Tecnología
              </button>
              <button
                onClick={() => scrollToSection(benefitsRef, 'benefits')}
                className={`text-sm font-medium transition-colors hover:text-blue-600 ${
                  activeSection === 'benefits' ? 'text-blue-600' : 'text-gray-700'
                }`}
              >
                Beneficios
              </button>
              <Button
                onClick={handleAccessSystem}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-2 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                Acceder al Sistema
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </nav>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-white border-t border-gray-200">
            <div className="px-4 py-4 space-y-3">
              <button
                onClick={() => scrollToSection(featuresRef, 'features')}
                className="block w-full text-left px-3 py-2 text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              >
                Características
              </button>
              <button
                onClick={() => scrollToSection(technologyRef, 'technology')}
                className="block w-full text-left px-3 py-2 text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              >
                Tecnología
              </button>
              <button
                onClick={() => scrollToSection(benefitsRef, 'benefits')}
                className="block w-full text-left px-3 py-2 text-gray-700 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              >
                Beneficios
              </button>
              <Button
                onClick={handleAccessSystem}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl"
              >
                Acceder al Sistema
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </header>

      {/* Hero Section */}
      <section 
        ref={heroRef}
        id="hero" 
        data-section
        className="pt-32 pb-20 px-4 sm:px-6 lg:px-8 relative overflow-hidden"
      >
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-indigo-400/20 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-green-400/20 to-emerald-400/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        </div>

        <div className="max-w-7xl mx-auto relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column - Content */}
            <div className={`space-y-8 ${isVisible.hero ? 'animate-fade-in-up' : 'opacity-0'}`}>
              <div className="space-y-4">
                <Badge className="bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-800 border-blue-200 px-4 py-2 text-sm font-medium">
                  <Cpu className="h-4 w-4 mr-2" />
                  Industria 4.0 + IA
                </Badge>
                
                <h1 className="text-5xl lg:text-6xl font-bold leading-tight">
                  <span className="bg-gradient-to-r from-gray-900 via-blue-900 to-indigo-900 bg-clip-text text-transparent">
                    Monitoreo
                  </span>
                  <br />
                  <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Inteligente
                  </span>
                  <br />
                  <span className="bg-gradient-to-r from-gray-900 via-blue-900 to-indigo-900 bg-clip-text text-transparent">
                    de Baterías
                  </span>
                </h1>
                
                <p className="text-xl text-gray-600 leading-relaxed max-w-2xl">
                  Sistema avanzado para el monitoreo, análisis y diagnóstico de baterías de ion de litio basado en metodologías de la Industria 4.0 e Inteligencia Artificial.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Button
                  onClick={handleAccessSystem}
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                >
                  <ArrowRight className="mr-2 h-5 w-5" />
                  Acceder al Sistema
                </Button>
                
                <Button
                  onClick={handleDemoAccess}
                  variant="outline"
                  size="lg"
                  className="border-2 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300"
                >
                  <Play className="mr-2 h-5 w-5" />
                  Ver Demo Interactivo
                </Button>
              </div>

              {/* Feature highlights */}
              <div className="flex flex-wrap gap-6 pt-4">
                <div className="flex items-center space-x-2 text-green-600">
                  <CheckCircle className="h-5 w-5" />
                  <span className="font-medium">Gemelo Digital</span>
                </div>
                <div className="flex items-center space-x-2 text-blue-600">
                  <CheckCircle className="h-5 w-5" />
                  <span className="font-medium">IA Explicable</span>
                </div>
                <div className="flex items-center space-x-2 text-purple-600">
                  <CheckCircle className="h-5 w-5" />
                  <span className="font-medium">Análisis Predictivo</span>
                </div>
              </div>
            </div>

            {/* Right Column - Interactive Dashboard Preview */}
            <div className={`relative ${isVisible.hero ? 'animate-fade-in-left' : 'opacity-0'}`}>
              <div className="relative bg-white rounded-2xl shadow-2xl p-6 border border-gray-200/50">
                <div className="flex items-center space-x-2 mb-4">
                  <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                  <div className="flex-1 bg-gray-100 rounded-full h-6 flex items-center px-3">
                    <span className="text-xs text-gray-500">battsentinel.com/dashboard</span>
                  </div>
                </div>
                
                {/* Simulated Dashboard Content */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-gray-900">Estado de Baterías</h3>
                    <Badge className="bg-green-100 text-green-800">En línea</Badge>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <Card className="p-4">
                      <div className="flex items-center space-x-2">
                        <Battery className="h-5 w-5 text-blue-600" />
                        <div>
                          <p className="text-sm text-gray-600">SOC</p>
                          <p className="text-lg font-semibold">85%</p>
                        </div>
                      </div>
                    </Card>
                    
                    <Card className="p-4">
                      <div className="flex items-center space-x-2">
                        <Thermometer className="h-5 w-5 text-orange-600" />
                        <div>
                          <p className="text-sm text-gray-600">Temp</p>
                          <p className="text-lg font-semibold">32°C</p>
                        </div>
                      </div>
                    </Card>
                  </div>
                  
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Brain className="h-5 w-5 text-purple-600" />
                      <span className="font-medium text-gray-900">Análisis IA</span>
                    </div>
                    <p className="text-sm text-gray-600">Estado: Normal</p>
                    <p className="text-sm text-gray-600">RUL: 2.3 años</p>
                  </div>
                </div>
              </div>
              
              {/* Floating elements */}
              <div className="absolute -top-4 -right-4 w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-full flex items-center justify-center shadow-lg animate-bounce">
                <Zap className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Rest of the component remains the same... */}
      {/* Features Section, Technology Section, Benefits Section, etc. */}
      
      {/* Features Section */}
      <section 
        ref={featuresRef}
        id="features" 
        data-section
        className="py-20 px-4 sm:px-6 lg:px-8 bg-white"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 ${isVisible.features ? 'animate-fade-in-up' : 'opacity-0'}`}>
            <Badge className="bg-blue-100 text-blue-800 mb-4">
              Características Principales
            </Badge>
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Tecnología de Vanguardia
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Combinamos las últimas innovaciones en IA, IoT y análisis predictivo para ofrecer 
              una solución integral de monitoreo de baterías.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card 
                key={index}
                className={`group hover:shadow-xl transition-all duration-300 border-0 bg-gradient-to-br from-white to-gray-50 ${
                  isVisible.features ? 'animate-fade-in-up' : 'opacity-0'
                }`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <CardHeader className="pb-4">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <CardTitle className="text-xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-600 mb-4 leading-relaxed">
                    {feature.description}
                  </CardDescription>
                  <p className="text-sm text-gray-500">
                    {feature.details}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-600 to-indigo-600">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-white mb-6">
            ¿Listo para Revolucionar tu Monitoreo de Baterías?
          </h2>
          <p className="text-xl text-blue-100 mb-8 leading-relaxed">
            Únete a la nueva era del monitoreo inteligente y descubre cómo la IA 
            puede transformar la gestión de tus sistemas de energía.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              onClick={handleAccessSystem}
              size="lg"
              className="bg-white text-blue-600 hover:bg-gray-100 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              <ArrowRight className="mr-2 h-5 w-5" />
              Comenzar Ahora
            </Button>
            <Button
              onClick={() => scrollToSection(featuresRef, 'features')}
              variant="outline"
              size="lg"
              className="border-2 border-white text-white hover:bg-white hover:text-blue-600 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300"
            >
              Conocer Más
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="col-span-2">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                  <Zap className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">BattSentinel</h3>
                  <p className="text-sm text-gray-400">Monitoreo Inteligente</p>
                </div>
              </div>
              <p className="text-gray-400 mb-4 max-w-md">
                Sistema avanzado de monitoreo de baterías basado en Industria 4.0 
                e Inteligencia Artificial para optimizar el rendimiento y seguridad.
              </p>
              <div className="flex space-x-4">
                <Badge variant="outline" className="border-gray-600 text-gray-300">
                  Industria 4.0
                </Badge>
                <Badge variant="outline" className="border-gray-600 text-gray-300">
                  IA Explicable
                </Badge>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Tecnologías</h4>
              <ul className="space-y-2 text-gray-400">
                <li>Machine Learning</li>
                <li>Deep Learning</li>
                <li>Computer Vision</li>
                <li>IoT Integration</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Contacto</h4>
              <ul className="space-y-2 text-gray-400">
                <li>info@battsentinel.com</li>
                <li>+1 (555) 123-4567</li>
                <li>Soporte 24/7</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 BattSentinel. Todos los derechos reservados.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

