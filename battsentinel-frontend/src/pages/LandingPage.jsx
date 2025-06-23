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
import battSentinelLogo from '@/assets/BattSentinel_Logo.png'

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

  // Handle demo access - direct navigation without auth modal
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
                onClick={handleDemoAccess}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-2 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                Acceder al Demo
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
                onClick={handleDemoAccess}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl"
              >
                Acceder al Demo
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
                  onClick={handleDemoAccess}
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                >
                  <Play className="mr-2 h-5 w-5" />
                  Ver Demo Interactivo
                </Button>
                
                <Button
                  onClick={() => scrollToSection(featuresRef, 'features')}
                  variant="outline"
                  size="lg"
                  className="border-2 border-gray-300 hover:border-blue-600 hover:text-blue-600 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300"
                >
                  Explorar Características
                  <ChevronDown className="ml-2 h-5 w-5" />
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
              <div className="relative">
                {/* Main dashboard card */}
                <div className="bg-white/70 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20">
                  <div className="grid grid-cols-2 gap-6">
                    {/* SOC Card */}
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-6 border border-green-100">
                      <div className="flex items-center justify-between mb-4">
                        <Battery className="h-6 w-6 text-green-600" />
                        <span className="text-sm font-medium text-green-700">SOC</span>
                      </div>
                      <div className="text-3xl font-bold text-green-800 mb-2">85%</div>
                      <div className="text-sm text-green-600">Estado de Carga</div>
                    </div>

                    {/* SOH Card */}
                    <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border border-blue-100">
                      <div className="flex items-center justify-between mb-4">
                        <Activity className="h-6 w-6 text-blue-600" />
                        <span className="text-sm font-medium text-blue-700">SOH</span>
                      </div>
                      <div className="text-3xl font-bold text-blue-800 mb-2">92%</div>
                      <div className="text-sm text-blue-600">Estado de Salud</div>
                    </div>

                    {/* Temperature Card */}
                    <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-2xl p-6 border border-orange-100">
                      <div className="flex items-center justify-between mb-4">
                        <Thermometer className="h-6 w-6 text-orange-600" />
                        <span className="text-sm font-medium text-orange-700">Temp</span>
                      </div>
                      <div className="text-3xl font-bold text-orange-800 mb-2">25°C</div>
                      <div className="text-sm text-orange-600">Temperatura</div>
                    </div>

                    {/* RUL Card */}
                    <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-2xl p-6 border border-purple-100">
                      <div className="flex items-center justify-between mb-4">
                        <TrendingUp className="h-6 w-6 text-purple-600" />
                        <span className="text-sm font-medium text-purple-700">RUL</span>
                      </div>
                      <div className="text-3xl font-bold text-purple-800 mb-2">2.3y</div>
                      <div className="text-sm text-purple-600">Vida Útil</div>
                    </div>
                  </div>

                  {/* AI Analysis Section */}
                  <div className="mt-6 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-2xl border border-indigo-100">
                    <div className="flex items-center space-x-3 mb-3">
                      <Brain className="h-5 w-5 text-indigo-600" />
                      <span className="font-semibold text-indigo-800">Análisis IA</span>
                    </div>
                    <div className="text-lg font-bold text-green-700 mb-1">Estado: Óptimo</div>
                    <div className="text-sm text-gray-600">Próxima revisión recomendada en 30 días</div>
                  </div>
                </div>

                {/* Floating elements */}
                <div className="absolute -top-4 -right-4 w-8 h-8 bg-gradient-to-br from-yellow-400 to-orange-400 rounded-full animate-bounce delay-300"></div>
                <div className="absolute -bottom-4 -left-4 w-6 h-6 bg-gradient-to-br from-green-400 to-emerald-400 rounded-full animate-bounce delay-700"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section 
        ref={featuresRef}
        id="features" 
        data-section
        className="py-20 px-4 sm:px-6 lg:px-8 bg-white"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 ${isVisible.features ? 'animate-fade-in-up' : 'opacity-0'}`}>
            <Badge className="bg-blue-100 text-blue-800 px-4 py-2 mb-4">
              <Star className="h-4 w-4 mr-2" />
              Características Principales
            </Badge>
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-gray-900 to-blue-900 bg-clip-text text-transparent">
                Tecnología de Vanguardia
              </span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Combinamos inteligencia artificial, gemelos digitales y análisis predictivo para ofrecer la solución más avanzada de monitoreo de baterías.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card 
                key={index}
                className={`group hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-2 border-0 bg-gradient-to-br from-white to-gray-50 ${
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
                  <p className="text-sm text-blue-600 font-medium">
                    {feature.details}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section 
        ref={technologyRef}
        id="technology" 
        data-section
        className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-gray-50 to-blue-50"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 ${isVisible.technology ? 'animate-fade-in-up' : 'opacity-0'}`}>
            <Badge className="bg-purple-100 text-purple-800 px-4 py-2 mb-4">
              <Cpu className="h-4 w-4 mr-2" />
              Stack Tecnológico
            </Badge>
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-gray-900 to-purple-900 bg-clip-text text-transparent">
                Tecnologías de Última Generación
              </span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Construido con las mejores herramientas y frameworks para garantizar rendimiento, escalabilidad y confiabilidad.
            </p>
          </div>

          {/* Analysis Types */}
          <div className="mb-16">
            <h3 className="text-2xl font-bold text-center mb-8 text-gray-900">Tipos de Análisis Implementados</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {analysisTypes.map((type, index) => (
                <Card 
                  key={index}
                  className={`text-center hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 ${
                    isVisible.technology ? 'animate-fade-in-up' : 'opacity-0'
                  }`}
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <CardContent className="pt-6">
                    <div className={`w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center mx-auto mb-4 ${type.color}`}>
                      {type.icon}
                    </div>
                    <h4 className="font-bold text-gray-900 mb-2">{type.title}</h4>
                    <p className="text-sm text-gray-600">{type.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Tech Stack */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {technologies.map((tech, index) => (
              <Card 
                key={index}
                className={`hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 ${
                  isVisible.technology ? 'animate-fade-in-up' : 'opacity-0'
                }`}
                style={{ animationDelay: `${(index + 4) * 100}ms` }}
              >
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-bold text-gray-900 mb-1">{tech.name}</h4>
                      <Badge className={`${tech.color} text-xs`}>
                        {tech.category}
                      </Badge>
                    </div>
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center">
                      <Cpu className="h-6 w-6 text-white" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section 
        ref={benefitsRef}
        id="benefits" 
        data-section
        className="py-20 px-4 sm:px-6 lg:px-8 bg-white"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 ${isVisible.benefits ? 'animate-fade-in-up' : 'opacity-0'}`}>
            <Badge className="bg-green-100 text-green-800 px-4 py-2 mb-4">
              <TrendingUp className="h-4 w-4 mr-2" />
              Beneficios
            </Badge>
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              <span className="bg-gradient-to-r from-gray-900 to-green-900 bg-clip-text text-transparent">
                Impacto Medible en su Operación
              </span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              BattSentinel no solo monitorea, sino que transforma la gestión de baterías con resultados tangibles y medibles.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {benefits.map((benefit, index) => (
              <Card 
                key={index}
                className={`text-center hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-2 border-0 bg-gradient-to-br from-white to-gray-50 ${
                  isVisible.benefits ? 'animate-fade-in-up' : 'opacity-0'
                }`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <CardContent className="pt-8 pb-6">
                  <div className={`w-16 h-16 rounded-2xl bg-gray-100 flex items-center justify-center mx-auto mb-4 ${benefit.color}`}>
                    {benefit.icon}
                  </div>
                  <div className={`text-3xl font-bold mb-2 ${benefit.color}`}>
                    {benefit.metric}
                  </div>
                  <h4 className="font-bold text-gray-900 mb-3">{benefit.title}</h4>
                  <p className="text-sm text-gray-600 leading-relaxed">{benefit.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 relative overflow-hidden">
        {/* Background elements */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.05"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-20"></div>
        
        <div className="max-w-4xl mx-auto text-center relative">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
            ¿Listo para Revolucionar su Monitoreo de Baterías?
          </h2>
          <p className="text-xl text-blue-100 mb-8 leading-relaxed">
            Únase a la nueva era del monitoreo inteligente con BattSentinel. Comience hoy mismo y experimente el poder de la IA aplicada a la gestión de baterías.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              onClick={handleDemoAccess}
              size="lg"
              className="bg-white text-blue-900 hover:bg-gray-100 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
            >
              Acceder al Sistema
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
            
            <Button
              onClick={handleDemoAccess}
              variant="outline"
              size="lg"
              className="border-2 border-white text-white hover:bg-white hover:text-blue-900 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-300"
            >
              <Play className="mr-2 h-5 w-5" />
              Ver Demostración
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
              <Zap className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold">BattSentinel</h3>
              <p className="text-sm text-gray-400">Monitoreo Inteligente de Baterías</p>
            </div>
          </div>
          <p className="text-gray-400 mb-4">
            Sistema avanzado basado en Industria 4.0 e Inteligencia Artificial
          </p>
          <p className="text-sm text-gray-500">
            © 2025 BattSentinel. Tecnología de vanguardia para el futuro energético.
          </p>
        </div>
      </footer>

      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes fade-in-left {
          from {
            opacity: 0;
            transform: translateX(30px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out forwards;
        }
        
        .animate-fade-in-left {
          animation: fade-in-left 0.8s ease-out forwards;
        }
      `}</style>
    </div>
  )
}
