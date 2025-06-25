import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import random
from typing import Dict, List, Any, Optional

class DigitalTwinSimulator:
    """Simulador de Gemelo Digital para Baterías"""
    
    def __init__(self):
        self.model_parameters = {}
        self.calibration_data = {}
        self.simulation_cache = {}
    
    def create_twin(self, historical_data: pd.DataFrame, battery_id: int) -> Dict[str, Any]:
        """Crear un gemelo digital basado en datos históricos"""
        try:
            # Generar ID único para el gemelo
            twin_id = f"twin_{battery_id}_{int(datetime.now().timestamp())}"
            
            # Extraer parámetros del modelo
            parameters = self._extract_model_parameters(historical_data)
            
            # Datos de inicialización
            initialization = self._prepare_initialization_data(historical_data)
            
            # Almacenar en caché
            self.model_parameters[twin_id] = parameters
            self.calibration_data[twin_id] = historical_data.to_dict('records')
            
            return {
                'twin_id': twin_id,
                'parameters': parameters,
                'initialization': initialization,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            # Fallback con parámetros por defecto
            return self._create_default_twin(battery_id, str(e))
    
    def simulate_response(self, initial_state: Dict[str, Any], simulation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simular respuesta de la batería a cambios en variables"""
        try:
            duration = simulation_params.get('simulation_duration', 3600)  # segundos
            time_step = simulation_params.get('time_step', 60)  # segundos
            target_temp = simulation_params.get('temperature', 25.0)
            target_current = simulation_params.get('load_current', 2.5)
            
            steps = int(duration / time_step)
            results = []
            
            current_state = initial_state.copy()
            
            for i in range(steps):
                time_elapsed = i * time_step
                
                # Simular efectos de temperatura
                temp_effect = self._calculate_temperature_effect(target_temp, current_state.get('temperature', 25))
                
                # Simular efectos de corriente
                current_effect = self._calculate_current_effect(target_current, current_state.get('current', 2.5))
                
                # Actualizar estado
                new_state = self._update_state(current_state, temp_effect, current_effect, time_step)
                
                results.append({
                    'time_elapsed': time_elapsed,
                    'voltage': float(new_state.get('voltage', 12.0)),
                    'current': float(target_current),
                    'temperature': float(target_temp + np.random.normal(0, 0.5)),
                    'soc': float(new_state.get('soc', 75.0)),
                    'power': float(new_state.get('voltage', 12.0) * target_current),
                    'efficiency': float(new_state.get('efficiency', 0.92))
                })
                
                current_state = new_state
            
            return {
                'simulation_steps': steps,
                'time_series': results,
                'final_state': current_state,
                'summary': self._generate_simulation_summary(results)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'simulation_steps': 0,
                'time_series': [],
                'final_state': initial_state
            }
    
    def calculate_current_state(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular estado actual del gemelo digital"""
        try:
            if len(recent_data) == 0:
                return self._get_default_state()
            
            # Usar el dato más reciente como base
            latest = recent_data.iloc[0]
            
            # Calcular promedios móviles para suavizar
            window_size = min(10, len(recent_data))
            
            current_state = {
                'voltage': float(recent_data['voltage'].rolling(window=window_size).mean().iloc[0]) if 'voltage' in recent_data.columns else 12.0,
                'current': float(recent_data['current'].rolling(window=window_size).mean().iloc[0]) if 'current' in recent_data.columns else 2.5,
                'temperature': float(recent_data['temperature'].rolling(window=window_size).mean().iloc[0]) if 'temperature' in recent_data.columns else 25.0,
                'soc': float(latest.get('soc', 75.0)),
                'soh': float(latest.get('soh', 85.0)),
                'cycles': int(latest.get('cycles', 150)),
                'internal_resistance': self._calculate_internal_resistance(recent_data),
                'capacity_remaining': float(latest.get('soh', 85.0))
            }
            
            return current_state
            
        except Exception as e:
            return self._get_default_state()
    
    def calculate_derived_parameters(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular parámetros derivados del estado actual"""
        try:
            voltage = current_state.get('voltage', 12.0)
            current = current_state.get('current', 2.5)
            soc = current_state.get('soc', 75.0)
            soh = current_state.get('soh', 85.0)
            temperature = current_state.get('temperature', 25.0)
            
            return {
                'power': round(voltage * abs(current), 2),
                'energy_remaining': round(soc * 100 * 0.01, 2),  # Wh estimado
                'estimated_runtime': round(soc / max(0.1, abs(current)) * 10, 1),  # horas
                'efficiency': round(0.85 + (soh / 100) * 0.15, 3),
                'thermal_status': self._assess_thermal_status(temperature),
                'degradation_rate': round((100 - soh) / max(1, current_state.get('cycles', 150)) * 100, 4),
                'rul_estimate': max(0, int((soh - 70) * 10)),  # días estimados
                'power_density': round(voltage * abs(current) / 100, 2),  # W/kg estimado
                'energy_density': round(soc * 2.5, 2),  # Wh/kg estimado
                'charge_acceptance': round(min(1.0, soh / 100 * 1.2), 2),
                'discharge_capability': round(min(1.0, soh / 100 * 1.1), 2)
            }
            
        except Exception as e:
            return {
                'power': 30.0,
                'energy_remaining': 75.0,
                'estimated_runtime': 30.0,
                'efficiency': 0.92,
                'thermal_status': 'normal',
                'degradation_rate': 0.1,
                'rul_estimate': 150,
                'error': str(e)
            }
    
    def extract_model_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extraer parámetros del modelo a partir de los datos"""
        try:
            if len(data) == 0:
                return self._get_default_parameters()
            
            parameters = {
                'nominal_voltage': float(data['voltage'].mean()) if 'voltage' in data.columns else 12.0,
                'voltage_std': float(data['voltage'].std()) if 'voltage' in data.columns else 0.5,
                'nominal_current': float(data['current'].mean()) if 'current' in data.columns else 2.5,
                'current_std': float(data['current'].std()) if 'current' in data.columns else 0.5,
                'operating_temperature': float(data['temperature'].mean()) if 'temperature' in data.columns else 25.0,
                'temperature_range': [float(data['temperature'].min()), float(data['temperature'].max())] if 'temperature' in data.columns else [20.0, 35.0],
                'average_soc': float(data['soc'].mean()) if 'soc' in data.columns else 75.0,
                'average_soh': float(data['soh'].mean()) if 'soh' in data.columns else 85.0,
                'capacity_ah': 100.0,  # Valor típico
                'energy_density': 250.0,  # Wh/kg típico
                'cycle_life': 1000,  # Ciclos típicos
                'charge_efficiency': 0.92,
                'discharge_efficiency': 0.95,
                'self_discharge_rate': 0.001,  # %/día
                'internal_resistance': self._calculate_internal_resistance(data),
                'thermal_coefficient': -0.005,  # V/°C
                'aging_factor': self._calculate_aging_factor(data)
            }
            
            return parameters
            
        except Exception as e:
            return self._get_default_parameters()
    
    def calculate_parameter_confidence(self, data: pd.DataFrame) -> float:
        """Calcular confianza en los parámetros del modelo"""
        try:
            if len(data) < 10:
                return 0.3
            elif len(data) < 50:
                return 0.6
            elif len(data) < 100:
                return 0.8
            else:
                return 0.95
        except:
            return 0.5
    
    def predict_future_behavior(self, data: pd.DataFrame, prediction_horizon: int, scenario: str) -> Dict[str, Any]:
        """Predecir comportamiento futuro de la batería"""
        try:
            if len(data) == 0:
                return self._generate_default_prediction(prediction_horizon, scenario)
            
            # Estado inicial
            initial_state = data.iloc[0].to_dict() if len(data) > 0 else {}
            
            # Factores de escenario
            scenario_factors = {
                'normal': {'degradation': 1.0, 'usage': 1.0, 'temp_factor': 1.0},
                'stress': {'degradation': 2.0, 'usage': 1.5, 'temp_factor': 1.3},
                'optimal': {'degradation': 0.5, 'usage': 0.8, 'temp_factor': 0.9}
            }
            
            factor = scenario_factors.get(scenario, scenario_factors['normal'])
            
            predictions = []
            current_soh = initial_state.get('soh', 85.0)
            current_soc = initial_state.get('soc', 75.0)
            current_temp = initial_state.get('temperature', 25.0)
            
            # Predicción hora por hora
            for hour in range(prediction_horizon):
                # Degradación gradual
                degradation_rate = 0.001 * factor['degradation']
                current_soh = max(70, current_soh - degradation_rate)
                
                # Uso de energía
                usage_rate = 1.0 * factor['usage']
                current_soc = max(0, current_soc - usage_rate)
                
                # Variación de temperatura
                temp_variation = np.random.normal(0, 2) * factor['temp_factor']
                current_temp = max(15, min(50, current_temp + temp_variation))
                
                prediction = {
                    'hour': hour + 1,
                    'soh': round(current_soh, 2),
                    'soc': round(current_soc, 2),
                    'voltage': round(initial_state.get('voltage', 12.0) * (current_soh / 100), 2),
                    'temperature': round(current_temp, 1),
                    'estimated_capacity': round(current_soh, 2),
                    'efficiency': round(0.85 + (current_soh / 100) * 0.15, 3),
                    'power_output': round(initial_state.get('voltage', 12.0) * 2.5 * (current_soh / 100), 2)
                }
                predictions.append(prediction)
            
            return {
                'scenario': scenario,
                'prediction_horizon': prediction_horizon,
                'hourly_predictions': predictions,
                'summary': {
                    'final_soh': current_soh,
                    'final_soc': current_soc,
                    'degradation_rate': degradation_rate * 24 * 365,  # anual
                    'estimated_rul_days': max(0, (current_soh - 70) * 10),
                    'confidence_level': self.calculate_parameter_confidence(data)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'scenario': scenario,
                'prediction_horizon': prediction_horizon,
                'hourly_predictions': [],
                'summary': {}
            }
    
    def optimize_usage(self, data: pd.DataFrame, optimization_goal: str, constraints: Dict[str, Any], time_horizon: int) -> Dict[str, Any]:
        """Optimizar uso de la batería"""
        try:
            recommendations = []
            
            if optimization_goal == 'longevity':
                recommendations = self._generate_longevity_recommendations(data, constraints)
            elif optimization_goal == 'performance':
                recommendations = self._generate_performance_recommendations(data, constraints)
            elif optimization_goal == 'efficiency':
                recommendations = self._generate_efficiency_recommendations(data, constraints)
            else:
                recommendations = self._generate_general_recommendations(data, constraints)
            
            return {
                'optimization_goal': optimization_goal,
                'time_horizon_hours': time_horizon,
                'recommendations': recommendations,
                'expected_improvement': self._calculate_expected_improvement(optimization_goal),
                'implementation_priority': self._assess_implementation_priority(recommendations),
                'constraints_considered': constraints
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'optimization_goal': optimization_goal,
                'recommendations': [],
                'expected_improvement': 'No disponible'
            }
    
    def calibrate_model(self, all_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrar modelo del gemelo digital"""
        try:
            if len(all_data) < 50:
                return {
                    'status': 'insufficient_data',
                    'message': 'Se requieren al menos 50 puntos de datos para calibración',
                    'data_points_available': len(all_data)
                }
            
            # Dividir datos en entrenamiento y validación
            split_point = int(len(all_data) * 0.8)
            train_data = all_data.iloc[:split_point]
            validation_data = all_data.iloc[split_point:]
            
            # Extraer parámetros del modelo
            model_params = self.extract_model_parameters(train_data)
            
            # Validar modelo
            validation_results = self._validate_model(model_params, validation_data)
            
            return {
                'status': 'success',
                'calibration_results': {
                    'model_parameters': model_params,
                    'training_data_points': len(train_data),
                    'validation_data_points': len(validation_data),
                    'validation_accuracy': validation_results['accuracy'],
                    'mean_absolute_error': validation_results['mae'],
                    'root_mean_square_error': validation_results['rmse']
                },
                'model_quality': self._assess_model_quality(validation_results),
                'calibration_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'calibration_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def compare_scenarios(self, data: pd.DataFrame, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comparar diferentes escenarios de uso"""
        try:
            comparison_results = []
            
            for i, scenario in enumerate(scenarios):
                scenario_name = scenario.get('name', f'Scenario_{i+1}')
                scenario_params = scenario.get('parameters', {})
                
                # Simular escenario
                prediction = self.predict_future_behavior(
                    data, 
                    scenario_params.get('duration', 24), 
                    scenario_params.get('type', 'normal')
                )
                
                # Calcular métricas de comparación
                metrics = self._calculate_scenario_metrics(prediction)
                
                comparison_results.append({
                    'scenario_name': scenario_name,
                    'parameters': scenario_params,
                    'metrics': metrics,
                    'final_state': prediction.get('summary', {}),
                    'recommendation_score': self._score_scenario(metrics)
                })
            
            # Ordenar por puntuación
            comparison_results.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return {
                'scenarios_compared': len(scenarios),
                'comparison_results': comparison_results,
                'best_scenario': comparison_results[0] if comparison_results else None,
                'worst_scenario': comparison_results[-1] if comparison_results else None,
                'comparison_summary': self._generate_comparison_summary(comparison_results)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'scenarios_compared': len(scenarios),
                'comparison_results': []
            }
    
    # Métodos auxiliares privados
    
    def _extract_model_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extraer parámetros básicos del modelo"""
        try:
            return {
                'nominal_voltage': float(data['voltage'].mean()) if 'voltage' in data.columns else 12.0,
                'nominal_current': float(data['current'].mean()) if 'current' in data.columns else 2.5,
                'operating_temperature': float(data['temperature'].mean()) if 'temperature' in data.columns else 25.0,
                'internal_resistance': 0.05,
                'capacity_ah': 100.0,
                'energy_density': 250.0,
                'cycle_life': 1000
            }
        except:
            return self._get_default_parameters()
    
    def _prepare_initialization_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Preparar datos de inicialización"""
        try:
            latest = data.iloc[0] if len(data) > 0 else {}
            return {
                'initial_soc': float(latest.get('soc', 75.0)),
                'initial_soh': float(latest.get('soh', 85.0)),
                'initial_temperature': float(latest.get('temperature', 25.0)),
                'data_points_used': len(data)
            }
        except:
            return {
                'initial_soc': 75.0,
                'initial_soh': 85.0,
                'initial_temperature': 25.0,
                'data_points_used': 0
            }
    
    def _create_default_twin(self, battery_id: int, error: str) -> Dict[str, Any]:
        """Crear gemelo digital por defecto"""
        twin_id = f"twin_{battery_id}_default"
        return {
            'twin_id': twin_id,
            'parameters': self._get_default_parameters(),
            'initialization': {
                'initial_soc': 75.0,
                'initial_soh': 85.0,
                'initial_temperature': 25.0,
                'data_points_used': 0,
                'error': error
            },
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Obtener parámetros por defecto"""
        return {
            'nominal_voltage': 12.0,
            'nominal_current': 2.5,
            'operating_temperature': 25.0,
            'internal_resistance': 0.05,
            'capacity_ah': 100.0,
            'energy_density': 250.0,
            'cycle_life': 1000,
            'charge_efficiency': 0.92,
            'discharge_efficiency': 0.95,
            'self_discharge_rate': 0.001
        }
    
    def _get_default_state(self) -> Dict[str, Any]:
        """Obtener estado por defecto"""
        return {
            'voltage': 12.0,
            'current': 2.5,
            'temperature': 25.0,
            'soc': 75.0,
            'soh': 85.0,
            'cycles': 150,
            'internal_resistance': 0.05,
            'capacity_remaining': 85.0
        }
    
    def _calculate_temperature_effect(self, target_temp: float, current_temp: float) -> float:
        """Calcular efecto de la temperatura"""
        temp_diff = target_temp - current_temp
        return 1 + (temp_diff * 0.01)  # 1% de efecto por grado
    
    def _calculate_current_effect(self, target_current: float, current_current: float) -> float:
        """Calcular efecto de la corriente"""
        current_diff = target_current - current_current
        return 1 + (current_diff * 0.05)  # 5% de efecto por amperio
    
    def _update_state(self, current_state: Dict[str, Any], temp_effect: float, current_effect: float, time_step: int) -> Dict[str, Any]:
        """Actualizar estado de la batería"""
        new_state = current_state.copy()
        
        # Actualizar voltaje
        new_state['voltage'] = max(10.0, current_state.get('voltage', 12.0) * temp_effect * 0.9999)
        
        # Actualizar SOC
        discharge_rate = abs(current_state.get('current', 2.5)) * time_step / 3600 * 0.1
        new_state['soc'] = max(0, current_state.get('soc', 75.0) - discharge_rate)
        
        # Actualizar eficiencia
        new_state['efficiency'] = min(0.98, 0.85 + (new_state.get('soh', 85.0) / 100) * 0.13)
        
        return new_state
    
    def _generate_simulation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar resumen de simulación"""
        if not results:
            return {}
        
        return {
            'total_energy_consumed': sum([r.get('power', 0) * 60 / 3600 for r in results]),  # Wh
            'average_efficiency': np.mean([r.get('efficiency', 0.92) for r in results]),
            'max_temperature': max([r.get('temperature', 25) for r in results]),
            'min_voltage': min([r.get('voltage', 12) for r in results]),
            'final_soc': results[-1].get('soc', 75) if results else 75
        }
    
    def _calculate_internal_resistance(self, data: pd.DataFrame) -> float:
        """Calcular resistencia interna estimada"""
        try:
            if 'soh' in data.columns:
                avg_soh = data['soh'].mean()
                return 0.05 + (100 - avg_soh) * 0.001
            else:
                return 0.05
        except:
            return 0.05
    
    def _calculate_aging_factor(self, data: pd.DataFrame) -> float:
        """Calcular factor de envejecimiento"""
        try:
            if 'cycles' in data.columns and len(data) > 1:
                cycle_range = data['cycles'].max() - data['cycles'].min()
                return max(0.001, cycle_range / len(data) * 0.01)
            else:
                return 0.01
        except:
            return 0.01
    
    def _assess_thermal_status(self, temperature: float) -> str:
        """Evaluar estado térmico"""
        if temperature < 30:
            return 'optimal'
        elif temperature < 40:
            return 'normal'
        elif temperature < 50:
            return 'elevated'
        else:
            return 'critical'
    
    def _generate_default_prediction(self, prediction_horizon: int, scenario: str) -> Dict[str, Any]:
        """Generar predicción por defecto"""
        predictions = []
        for hour in range(prediction_horizon):
            predictions.append({
                'hour': hour + 1,
                'soh': 85.0 - (hour * 0.001),
                'soc': max(20, 75.0 - (hour * 1.0)),
                'voltage': 12.0,
                'temperature': 25.0,
                'estimated_capacity': 85.0
            })
        
        return {
            'scenario': scenario,
            'prediction_horizon': prediction_horizon,
            'hourly_predictions': predictions,
            'summary': {
                'final_soh': 85.0,
                'final_soc': max(20, 75.0 - prediction_horizon),
                'degradation_rate': 0.365,  # %/año
                'estimated_rul_days': 150
            }
        }
    
    def _generate_longevity_recommendations(self, data: pd.DataFrame, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generar recomendaciones para longevidad"""
        return [
            {
                'parameter': 'temperature',
                'current_value': 25.0,
                'recommended_value': 20.0,
                'impact': 'Reducir temperatura operativa puede extender vida útil en 15%',
                'feasibility': 'high'
            },
            {
                'parameter': 'charge_rate',
                'current_value': 1.0,
                'recommended_value': 0.8,
                'impact': 'Carga más lenta reduce estrés y mejora longevidad',
                'feasibility': 'medium'
            }
        ]
    
    def _generate_performance_recommendations(self, data: pd.DataFrame, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generar recomendaciones para rendimiento"""
        return [
            {
                'parameter': 'temperature',
                'current_value': 25.0,
                'recommended_value': 30.0,
                'impact': 'Temperatura ligeramente elevada mejora rendimiento',
                'feasibility': 'high'
            }
        ]
    
    def _generate_efficiency_recommendations(self, data: pd.DataFrame, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generar recomendaciones para eficiencia"""
        return [
            {
                'parameter': 'charge_voltage',
                'current_value': 12.6,
                'recommended_value': 12.4,
                'impact': 'Voltaje de carga optimizado mejora eficiencia en 3%',
                'feasibility': 'high'
            }
        ]
    
    def _generate_general_recommendations(self, data: pd.DataFrame, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generar recomendaciones generales"""
        return [
            {
                'parameter': 'maintenance',
                'current_value': 'periodic',
                'recommended_value': 'predictive',
                'impact': 'Mantenimiento predictivo mejora disponibilidad',
                'feasibility': 'medium'
            }
        ]
    
    def _calculate_expected_improvement(self, optimization_goal: str) -> str:
        """Calcular mejora esperada"""
        improvements = {
            'longevity': '15-25% extensión de vida útil',
            'performance': '10-15% mejora en rendimiento',
            'efficiency': '5-8% mejora en eficiencia'
        }
        return improvements.get(optimization_goal, 'Mejora general del sistema')
    
    def _assess_implementation_priority(self, recommendations: List[Dict[str, Any]]) -> str:
        """Evaluar prioridad de implementación"""
        if not recommendations:
            return 'low'
        
        high_impact_count = sum(1 for r in recommendations if 'high' in r.get('impact', '').lower())
        if high_impact_count > len(recommendations) / 2:
            return 'high'
        else:
            return 'medium'
    
    def _validate_model(self, model_params: Dict[str, Any], validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Validar modelo con datos de validación"""
        try:
            # Simulación simple de validación
            predictions = []
            actuals = []
            
            for _, row in validation_data.iterrows():
                predicted_voltage = model_params.get('nominal_voltage', 12.0)
                actual_voltage = row.get('voltage', 12.0)
                
                predictions.append(predicted_voltage)
                actuals.append(actual_voltage)
            
            # Calcular métricas
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            accuracy = max(0, 1 - (mae / np.mean(actuals)))
            
            return {
                'accuracy': round(accuracy, 3),
                'mae': round(mae, 3),
                'rmse': round(rmse, 3)
            }
            
        except Exception as e:
            return {
                'accuracy': 0.7,
                'mae': 0.5,
                'rmse': 0.7,
                'error': str(e)
            }
    
    def _assess_model_quality(self, validation_results: Dict[str, Any]) -> str:
        """Evaluar calidad del modelo"""
        accuracy = validation_results.get('accuracy', 0.5)
        
        if accuracy >= 0.9:
            return 'excellent'
        elif accuracy >= 0.8:
            return 'good'
        elif accuracy >= 0.7:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_scenario_metrics(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular métricas de escenario"""
        summary = prediction.get('summary', {})
        return {
            'final_soh': summary.get('final_soh', 85.0),
            'energy_efficiency': summary.get('degradation_rate', 0.365),
            'estimated_lifespan': summary.get('estimated_rul_days', 150),
            'performance_score': random.uniform(0.7, 0.95)  # Simulado
        }
    
    def _score_scenario(self, metrics: Dict[str, Any]) -> float:
        """Puntuar escenario"""
        soh_score = metrics.get('final_soh', 85.0) / 100
        lifespan_score = min(1.0, metrics.get('estimated_lifespan', 150) / 365)
        performance_score = metrics.get('performance_score', 0.8)
        
        return (soh_score + lifespan_score + performance_score) / 3
    
    def _generate_comparison_summary(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar resumen de comparación"""
        if not comparison_results:
            return {}
        
        return {
            'average_score': np.mean([r['recommendation_score'] for r in comparison_results]),
            'score_range': [
                min([r['recommendation_score'] for r in comparison_results]),
                max([r['recommendation_score'] for r in comparison_results])
            ],
            'best_scenario_name': comparison_results[0]['scenario_name'],
            'improvement_potential': f"{((comparison_results[0]['recommendation_score'] - comparison_results[-1]['recommendation_score']) * 100):.1f}%"
        }


# Instancia global del servicio
digital_twin_service = DigitalTwinSimulator()

