import cv2
import numpy as np
from PIL import Image
import json
import os

class ThermalAnalyzer:
    """Analizador de imágenes térmicas para detección de hotspots y análisis térmico"""
    
    def __init__(self):
        self.min_hotspot_area = 50  # Área mínima para considerar un hotspot
        self.temperature_threshold_percentile = 95  # Percentil para detectar hotspots
        
    def analyze_image(self, image_path):
        """Analizar imagen térmica y extraer información de temperatura"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convertir a escala de grises para análisis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Análisis básico de temperatura (simulado basado en intensidad)
            # En una implementación real, esto dependería del formato específico de la imagen térmica
            temperature_data = self._simulate_temperature_from_intensity(gray)
            
            # Calcular estadísticas de temperatura
            max_temp = float(np.max(temperature_data))
            min_temp = float(np.min(temperature_data))
            avg_temp = float(np.mean(temperature_data))
            
            # Detectar hotspots
            hotspots = self._detect_hotspots(temperature_data, image.shape)
            
            # Análisis de distribución térmica
            thermal_distribution = self._analyze_thermal_distribution(temperature_data)
            
            result = {
                'max_temperature': max_temp,
                'min_temperature': min_temp,
                'avg_temperature': avg_temp,
                'temperature_std': float(np.std(temperature_data)),
                'hotspot_detected': len(hotspots) > 0,
                'hotspot_count': len(hotspots),
                'hotspot_coordinates': hotspots,
                'thermal_distribution': thermal_distribution,
                'analysis_metadata': {
                    'image_size': image.shape[:2],
                    'analysis_method': 'intensity_based_simulation',
                    'threshold_percentile': self.temperature_threshold_percentile
                }
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Error analyzing thermal image: {str(e)}")
    
    def _simulate_temperature_from_intensity(self, gray_image):
        """Simular datos de temperatura basados en intensidad de píxeles"""
        # Esta es una simulación. En una implementación real, se usarían
        # bibliotecas específicas para imágenes térmicas como FLIR o similar
        
        # Normalizar intensidades a rango de temperatura (20-80°C)
        normalized = gray_image.astype(np.float32) / 255.0
        temperature_data = 20 + (normalized * 60)  # Rango 20-80°C
        
        # Agregar algo de ruido realista
        noise = np.random.normal(0, 0.5, temperature_data.shape)
        temperature_data += noise
        
        return temperature_data
    
    def _detect_hotspots(self, temperature_data, image_shape):
        """Detectar hotspots en la imagen térmica"""
        try:
            # Calcular umbral para hotspots
            threshold = np.percentile(temperature_data, self.temperature_threshold_percentile)
            
            # Crear máscara binaria para hotspots
            hotspot_mask = (temperature_data > threshold).astype(np.uint8)
            
            # Encontrar contornos de hotspots
            contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hotspots = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_hotspot_area:
                    # Calcular centroide
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calcular bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Temperatura máxima en el hotspot
                        hotspot_region = temperature_data[y:y+h, x:x+w]
                        max_temp_in_hotspot = float(np.max(hotspot_region))
                        avg_temp_in_hotspot = float(np.mean(hotspot_region))
                        
                        hotspot_info = {
                            'centroid': {'x': cx, 'y': cy},
                            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h},
                            'area': float(area),
                            'max_temperature': max_temp_in_hotspot,
                            'avg_temperature': avg_temp_in_hotspot,
                            'severity': self._classify_hotspot_severity(max_temp_in_hotspot)
                        }
                        
                        hotspots.append(hotspot_info)
            
            # Ordenar hotspots por temperatura máxima (descendente)
            hotspots.sort(key=lambda x: x['max_temperature'], reverse=True)
            
            return hotspots
            
        except Exception as e:
            print(f"Error detecting hotspots: {e}")
            return []
    
    def _classify_hotspot_severity(self, temperature):
        """Clasificar severidad del hotspot basado en temperatura"""
        if temperature > 70:
            return 'critical'
        elif temperature > 60:
            return 'high'
        elif temperature > 50:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_thermal_distribution(self, temperature_data):
        """Analizar distribución térmica de la imagen"""
        try:
            # Calcular histograma de temperaturas
            hist, bin_edges = np.histogram(temperature_data.flatten(), bins=20)
            
            # Encontrar zonas de temperatura
            temp_zones = {
                'cold': float(np.sum(temperature_data < 30)),
                'normal': float(np.sum((temperature_data >= 30) & (temperature_data < 50))),
                'warm': float(np.sum((temperature_data >= 50) & (temperature_data < 70))),
                'hot': float(np.sum(temperature_data >= 70))
            }
            
            # Calcular porcentajes
            total_pixels = temperature_data.size
            temp_zones_percent = {k: (v / total_pixels) * 100 for k, v in temp_zones.items()}
            
            # Análisis de gradientes térmicos
            grad_x = np.gradient(temperature_data, axis=1)
            grad_y = np.gradient(temperature_data, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            distribution_info = {
                'temperature_zones': temp_zones_percent,
                'histogram': {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist()
                },
                'thermal_gradients': {
                    'max_gradient': float(np.max(gradient_magnitude)),
                    'avg_gradient': float(np.mean(gradient_magnitude)),
                    'gradient_std': float(np.std(gradient_magnitude))
                },
                'uniformity_index': float(1.0 / (1.0 + np.std(temperature_data)))  # 0-1, 1 = uniforme
            }
            
            return distribution_info
            
        except Exception as e:
            print(f"Error analyzing thermal distribution: {e}")
            return {}
    
    def generate_thermal_overlay(self, image_path, output_path, hotspots=None):
        """Generar imagen con overlay térmico y hotspots marcados"""
        try:
            # Cargar imagen original
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Crear overlay térmico (colormap)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thermal_overlay = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            
            # Combinar imagen original con overlay
            result = cv2.addWeighted(image, 0.6, thermal_overlay, 0.4, 0)
            
            # Marcar hotspots si se proporcionan
            if hotspots:
                for hotspot in hotspots:
                    bbox = hotspot['bounding_box']
                    centroid = hotspot['centroid']
                    severity = hotspot['severity']
                    
                    # Color según severidad
                    color_map = {
                        'critical': (0, 0, 255),    # Rojo
                        'high': (0, 165, 255),      # Naranja
                        'medium': (0, 255, 255),   # Amarillo
                        'low': (0, 255, 0)         # Verde
                    }
                    color = color_map.get(severity, (255, 255, 255))
                    
                    # Dibujar bounding box
                    cv2.rectangle(result, 
                                (bbox['x'], bbox['y']), 
                                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                                color, 2)
                    
                    # Marcar centroide
                    cv2.circle(result, (centroid['x'], centroid['y']), 5, color, -1)
                    
                    # Agregar etiqueta de temperatura
                    temp_text = f"{hotspot['max_temperature']:.1f}°C"
                    cv2.putText(result, temp_text, 
                              (bbox['x'], bbox['y'] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Guardar imagen resultado
            cv2.imwrite(output_path, result)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating thermal overlay: {str(e)}")
    
    def extract_temperature_profile(self, image_path, line_coords):
        """Extraer perfil de temperatura a lo largo de una línea"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Could not load image")
            
            # Simular datos de temperatura
            temperature_data = self._simulate_temperature_from_intensity(image)
            
            # Extraer perfil a lo largo de la línea
            x1, y1, x2, y2 = line_coords
            
            # Crear línea de puntos
            num_points = max(abs(x2 - x1), abs(y2 - y1))
            x_points = np.linspace(x1, x2, num_points, dtype=int)
            y_points = np.linspace(y1, y2, num_points, dtype=int)
            
            # Extraer temperaturas
            temperatures = []
            distances = []
            
            for i, (x, y) in enumerate(zip(x_points, y_points)):
                if 0 <= x < temperature_data.shape[1] and 0 <= y < temperature_data.shape[0]:
                    temp = temperature_data[y, x]
                    temperatures.append(float(temp))
                    distances.append(float(i))
            
            profile = {
                'distances': distances,
                'temperatures': temperatures,
                'line_coordinates': line_coords,
                'statistics': {
                    'max_temp': float(np.max(temperatures)) if temperatures else 0,
                    'min_temp': float(np.min(temperatures)) if temperatures else 0,
                    'avg_temp': float(np.mean(temperatures)) if temperatures else 0,
                    'temp_range': float(np.max(temperatures) - np.min(temperatures)) if temperatures else 0
                }
            }
            
            return profile
            
        except Exception as e:
            raise Exception(f"Error extracting temperature profile: {str(e)}")
    
    def validate_thermal_image(self, image_path):
        """Validar si la imagen es adecuada para análisis térmico"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(image_path):
                return False, "File does not exist"
            
            # Intentar cargar la imagen
            image = cv2.imread(image_path)
            if image is None:
                return False, "Could not load image"
            
            # Verificar dimensiones mínimas
            height, width = image.shape[:2]
            if width < 100 or height < 100:
                return False, "Image too small for analysis"
            
            # Verificar que no sea completamente uniforme
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if np.std(gray) < 5:
                return False, "Image appears to be uniform (no thermal variation)"
            
            return True, "Valid thermal image"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
