import subprocess
import json
import psutil
import platform
from datetime import datetime, timezone
import re

class WindowsBatteryService:
    """Servicio para obtener datos reales de batería en Windows"""
    
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        self.last_data = {}
    
    def get_battery_info(self):
        """Obtener información completa de la batería"""
        try:
            if self.is_windows:
                return self._get_windows_battery_info()
            else:
                # Simulación para desarrollo en Linux/Mac
                return self._get_simulated_battery_info()
        except Exception as e:
            print(f"Error obteniendo información de batería: {e}")
            return self._get_fallback_battery_info()
    
    def _get_windows_battery_info(self):
        """Obtener información real de batería en Windows usando WMI"""
        try:
            battery_data = {}
            
            # Obtener información básica con psutil
            battery = psutil.sensors_battery()
            if battery:
                battery_data.update({
                    'soc': round(battery.percent, 2),
                    'is_plugged': battery.power_plugged,
                    'time_left': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                })
            
            # Obtener información detallada con PowerShell/WMI
            powershell_commands = [
                # Información básica de batería
                "Get-WmiObject -Class Win32_Battery | Select-Object EstimatedChargeRemaining, BatteryStatus, EstimatedRunTime",
                
                # Información de diseño de batería
                "Get-WmiObject -Class Win32_Battery | Select-Object DesignCapacity, FullChargeCapacity, DesignVoltage",
                
                # Información del sistema de energía
                "powercfg /batteryreport /output temp_battery_report.html /duration 1",
                
                # Información térmica (si está disponible)
                "Get-WmiObject -Namespace root/WMI -Class MSAcpi_ThermalZoneTemperature | Select-Object CurrentTemperature"
            ]
            
            # Ejecutar comandos de PowerShell
            for cmd in powershell_commands[:3]:  # Evitar el reporte por ahora
                try:
                    result = subprocess.run(
                        ["powershell", "-Command", cmd],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0 and result.stdout:
                        self._parse_powershell_output(result.stdout, battery_data)
                except Exception as e:
                    print(f"Error ejecutando comando PowerShell: {e}")
            
            # Obtener información adicional con WMIC
            wmic_commands = [
                "wmic path Win32_Battery get EstimatedChargeRemaining,BatteryStatus,DesignCapacity /format:list",
                "wmic path Win32_PortableBattery get DesignVoltage,DesignCapacity,FullChargeCapacity /format:list"
            ]
            
            for cmd in wmic_commands:
                try:
                    result = subprocess.run(
                        cmd.split(),
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0 and result.stdout:
                        self._parse_wmic_output(result.stdout, battery_data)
                except Exception as e:
                    print(f"Error ejecutando comando WMIC: {e}")
            
            # Calcular valores derivados
            self._calculate_derived_values(battery_data)
            
            # Agregar timestamp
            battery_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            battery_data['source'] = 'windows_wmi'
            
            self.last_data = battery_data
            return battery_data
            
        except Exception as e:
            print(f"Error en _get_windows_battery_info: {e}")
            return self._get_fallback_battery_info()
    
    def _get_simulated_battery_info(self):
        """Obtener información simulada de batería para desarrollo"""
        import random
        import time
        
        # Simular datos realistas que cambien gradualmente
        base_time = time.time()
        
        # Usar psutil si está disponible
        try:
            battery = psutil.sensors_battery()
            if battery:
                base_soc = battery.percent
                is_plugged = battery.power_plugged
            else:
                base_soc = 75.0
                is_plugged = False
        except:
            base_soc = 75.0
            is_plugged = False
        
        # Simular variaciones realistas
        soc = max(20, min(100, base_soc + random.uniform(-2, 2)))
        voltage = 11.1 + (soc / 100) * 1.5 + random.uniform(-0.1, 0.1)
        current = 2.5 + random.uniform(-0.5, 0.5) if not is_plugged else -1.5 + random.uniform(-0.3, 0.3)
        temperature = 25 + random.uniform(-3, 8)
        
        # Simular degradación gradual
        cycles = 150 + int((base_time % 10000) / 100)
        soh = max(70, 100 - (cycles * 0.02) + random.uniform(-2, 2))
        
        battery_data = {
            'voltage': round(voltage, 2),
            'current': round(current, 2),
            'temperature': round(temperature, 1),
            'soc': round(soc, 1),
            'soh': round(soh, 1),
            'cycles': cycles,
            'is_plugged': is_plugged,
            'design_capacity': 50000,  # mWh
            'full_charge_capacity': int(50000 * (soh / 100)),
            'design_voltage': 11100,  # mV
            'internal_resistance': round(0.05 + (100 - soh) * 0.001, 4),
            'power': round(voltage * abs(current), 2),
            'energy_rate': round(voltage * current, 2),  # Positivo = descarga, Negativo = carga
            'time_remaining': int((soc / 100) * 8 * 3600) if not is_plugged else None,  # segundos
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'simulated'
        }
        
        # Calcular valores derivados
        self._calculate_derived_values(battery_data)
        
        self.last_data = battery_data
        return battery_data
    
    def _get_fallback_battery_info(self):
        """Información de respaldo si fallan otros métodos"""
        return {
            'voltage': 12.0,
            'current': 2.5,
            'temperature': 25.0,
            'soc': 75.0,
            'soh': 85.0,
            'cycles': 150,
            'is_plugged': False,
            'design_capacity': 50000,
            'full_charge_capacity': 42500,
            'design_voltage': 11100,
            'internal_resistance': 0.075,
            'power': 30.0,
            'energy_rate': 30.0,
            'time_remaining': 21600,  # 6 horas
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'fallback'
        }
    
    def _parse_powershell_output(self, output, battery_data):
        """Parsear salida de PowerShell"""
        try:
            lines = output.strip().split('\n')
            for line in lines:
                line = line.strip()
                if 'EstimatedChargeRemaining' in line and ':' in line:
                    value = line.split(':')[1].strip()
                    if value and value.isdigit():
                        battery_data['soc'] = float(value)
                elif 'BatteryStatus' in line and ':' in line:
                    value = line.split(':')[1].strip()
                    if value and value.isdigit():
                        battery_data['battery_status'] = int(value)
                elif 'EstimatedRunTime' in line and ':' in line:
                    value = line.split(':')[1].strip()
                    if value and value.isdigit():
                        battery_data['estimated_runtime'] = int(value)
        except Exception as e:
            print(f"Error parseando salida PowerShell: {e}")
    
    def _parse_wmic_output(self, output, battery_data):
        """Parsear salida de WMIC"""
        try:
            lines = output.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'DesignCapacity' and value and value.isdigit():
                        battery_data['design_capacity'] = int(value)
                    elif key == 'FullChargeCapacity' and value and value.isdigit():
                        battery_data['full_charge_capacity'] = int(value)
                    elif key == 'DesignVoltage' and value and value.isdigit():
                        battery_data['design_voltage'] = int(value)
                    elif key == 'EstimatedChargeRemaining' and value and value.isdigit():
                        battery_data['soc'] = float(value)
        except Exception as e:
            print(f"Error parseando salida WMIC: {e}")
    
    def _calculate_derived_values(self, battery_data):
        """Calcular valores derivados"""
        try:
            # Calcular SOH si no está disponible
            if 'soh' not in battery_data:
                if 'full_charge_capacity' in battery_data and 'design_capacity' in battery_data:
                    if battery_data['design_capacity'] > 0:
                        battery_data['soh'] = round(
                            (battery_data['full_charge_capacity'] / battery_data['design_capacity']) * 100, 2
                        )
                else:
                    battery_data['soh'] = 85.0  # Valor por defecto
            
            # Calcular voltaje si no está disponible
            if 'voltage' not in battery_data:
                if 'design_voltage' in battery_data:
                    # Estimar voltaje basado en SOC
                    soc = battery_data.get('soc', 75)
                    design_v = battery_data['design_voltage'] / 1000  # Convertir mV a V
                    battery_data['voltage'] = round(design_v * 0.85 + (soc / 100) * design_v * 0.15, 2)
                else:
                    battery_data['voltage'] = 12.0
            
            # Calcular corriente estimada si no está disponible
            if 'current' not in battery_data:
                if 'power' in battery_data and battery_data['voltage'] > 0:
                    battery_data['current'] = round(battery_data['power'] / battery_data['voltage'], 2)
                else:
                    battery_data['current'] = 2.5
            
            # Calcular potencia si no está disponible
            if 'power' not in battery_data:
                voltage = battery_data.get('voltage', 12.0)
                current = battery_data.get('current', 2.5)
                battery_data['power'] = round(voltage * abs(current), 2)
            
            # Calcular resistencia interna estimada
            if 'internal_resistance' not in battery_data:
                soh = battery_data.get('soh', 85)
                battery_data['internal_resistance'] = round(0.05 + (100 - soh) * 0.001, 4)
            
            # Estimar temperatura si no está disponible
            if 'temperature' not in battery_data:
                battery_data['temperature'] = 25.0
            
            # Estimar ciclos si no están disponibles
            if 'cycles' not in battery_data:
                soh = battery_data.get('soh', 85)
                battery_data['cycles'] = int((100 - soh) * 50)  # Estimación muy aproximada
            
            # Calcular RUL (Remaining Useful Life) estimado
            soh = battery_data.get('soh', 85)
            cycles = battery_data.get('cycles', 150)
            
            # RUL en días (estimación simple)
            if soh > 80:
                rul_days = (soh - 70) * 30  # Días estimados hasta SOH crítico
            else:
                rul_days = max(0, (soh - 70) * 10)
            
            battery_data['rul_days'] = int(rul_days)
            
            # Calcular eficiencia estimada
            soh = battery_data.get('soh', 85)
            battery_data['efficiency'] = round(min(0.98, 0.85 + (soh / 100) * 0.13), 3)
            
        except Exception as e:
            print(f"Error calculando valores derivados: {e}")
    
    def get_battery_health_analysis(self):
        """Obtener análisis de salud de la batería"""
        try:
            battery_info = self.get_battery_info()
            
            analysis = {
                'overall_health': self._assess_overall_health(battery_info),
                'degradation_analysis': self._analyze_degradation(battery_info),
                'thermal_analysis': self._analyze_thermal_status(battery_info),
                'performance_analysis': self._analyze_performance(battery_info),
                'recommendations': self._generate_recommendations(battery_info),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error en análisis de salud: {e}")
            return {
                'overall_health': 'unknown',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _assess_overall_health(self, battery_info):
        """Evaluar salud general de la batería"""
        soh = battery_info.get('soh', 85)
        temperature = battery_info.get('temperature', 25)
        voltage = battery_info.get('voltage', 12)
        
        if soh >= 90 and temperature < 35 and voltage > 11.5:
            return 'excellent'
        elif soh >= 80 and temperature < 40 and voltage > 11.0:
            return 'good'
        elif soh >= 70 and temperature < 45 and voltage > 10.5:
            return 'fair'
        elif soh >= 60:
            return 'poor'
        else:
            return 'critical'
    
    def _analyze_degradation(self, battery_info):
        """Analizar degradación de la batería"""
        soh = battery_info.get('soh', 85)
        cycles = battery_info.get('cycles', 150)
        
        if cycles > 0:
            degradation_rate = (100 - soh) / cycles
        else:
            degradation_rate = 0.1
        
        return {
            'current_soh': soh,
            'cycles_completed': cycles,
            'degradation_rate_per_cycle': round(degradation_rate, 4),
            'estimated_total_cycles': int(100 / max(degradation_rate, 0.01)) if degradation_rate > 0 else 1000,
            'remaining_cycles': max(0, int((soh - 70) / max(degradation_rate, 0.01)))
        }
    
    def _analyze_thermal_status(self, battery_info):
        """Analizar estado térmico"""
        temperature = battery_info.get('temperature', 25)
        
        if temperature < 30:
            status = 'optimal'
            risk = 'low'
        elif temperature < 40:
            status = 'normal'
            risk = 'low'
        elif temperature < 50:
            status = 'elevated'
            risk = 'medium'
        elif temperature < 60:
            status = 'high'
            risk = 'high'
        else:
            status = 'critical'
            risk = 'critical'
        
        return {
            'current_temperature': temperature,
            'thermal_status': status,
            'risk_level': risk,
            'optimal_range': '20-30°C',
            'warning_threshold': 45,
            'critical_threshold': 60
        }
    
    def _analyze_performance(self, battery_info):
        """Analizar rendimiento de la batería"""
        soc = battery_info.get('soc', 75)
        voltage = battery_info.get('voltage', 12)
        current = battery_info.get('current', 2.5)
        power = battery_info.get('power', 30)
        efficiency = battery_info.get('efficiency', 0.9)
        
        return {
            'current_performance': {
                'soc': soc,
                'voltage': voltage,
                'current': current,
                'power': power,
                'efficiency': efficiency
            },
            'performance_rating': self._rate_performance(soc, voltage, power, efficiency),
            'capacity_utilization': round((soc / 100) * 100, 1),
            'power_delivery': 'normal' if 20 <= power <= 100 else 'abnormal'
        }
    
    def _rate_performance(self, soc, voltage, power, efficiency):
        """Calificar rendimiento general"""
        score = 0
        
        # SOC score (0-25 points)
        if soc >= 80:
            score += 25
        elif soc >= 60:
            score += 20
        elif soc >= 40:
            score += 15
        elif soc >= 20:
            score += 10
        else:
            score += 5
        
        # Voltage score (0-25 points)
        if voltage >= 12.0:
            score += 25
        elif voltage >= 11.5:
            score += 20
        elif voltage >= 11.0:
            score += 15
        elif voltage >= 10.5:
            score += 10
        else:
            score += 5
        
        # Power score (0-25 points)
        if 20 <= power <= 50:
            score += 25
        elif 15 <= power <= 60:
            score += 20
        elif 10 <= power <= 70:
            score += 15
        else:
            score += 10
        
        # Efficiency score (0-25 points)
        if efficiency >= 0.95:
            score += 25
        elif efficiency >= 0.90:
            score += 20
        elif efficiency >= 0.85:
            score += 15
        elif efficiency >= 0.80:
            score += 10
        else:
            score += 5
        
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 60:
            return 'fair'
        elif score >= 45:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_recommendations(self, battery_info):
        """Generar recomendaciones basadas en el estado de la batería"""
        recommendations = []
        
        soh = battery_info.get('soh', 85)
        temperature = battery_info.get('temperature', 25)
        soc = battery_info.get('soc', 75)
        voltage = battery_info.get('voltage', 12)
        is_plugged = battery_info.get('is_plugged', False)
        
        # Recomendaciones basadas en SOH
        if soh < 70:
            recommendations.append({
                'type': 'critical',
                'category': 'replacement',
                'message': 'La batería requiere reemplazo inmediato (SOH < 70%)',
                'priority': 'high'
            })
        elif soh < 80:
            recommendations.append({
                'type': 'warning',
                'category': 'maintenance',
                'message': 'Considere planificar el reemplazo de la batería pronto',
                'priority': 'medium'
            })
        
        # Recomendaciones basadas en temperatura
        if temperature > 45:
            recommendations.append({
                'type': 'warning',
                'category': 'thermal',
                'message': 'Temperatura elevada - Mejorar ventilación o reducir carga',
                'priority': 'high'
            })
        elif temperature > 35:
            recommendations.append({
                'type': 'info',
                'category': 'thermal',
                'message': 'Monitorear temperatura - Mantener en ambiente fresco',
                'priority': 'low'
            })
        
        # Recomendaciones basadas en SOC
        if soc < 20:
            recommendations.append({
                'type': 'warning',
                'category': 'charging',
                'message': 'Nivel de carga crítico - Cargar inmediatamente',
                'priority': 'high'
            })
        elif soc > 95 and is_plugged:
            recommendations.append({
                'type': 'info',
                'category': 'charging',
                'message': 'Desconectar cargador para evitar sobrecarga',
                'priority': 'low'
            })
        
        # Recomendaciones basadas en voltaje
        if voltage < 10.5:
            recommendations.append({
                'type': 'critical',
                'category': 'voltage',
                'message': 'Voltaje crítico - Verificar conexiones y estado de la batería',
                'priority': 'high'
            })
        
        # Recomendaciones generales
        if not recommendations:
            recommendations.append({
                'type': 'info',
                'category': 'general',
                'message': 'La batería está funcionando dentro de parámetros normales',
                'priority': 'low'
            })
        
        return recommendations

# Instancia global del servicio
windows_battery_service = WindowsBatteryService()

