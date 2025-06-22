import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import joblib
import shap
import lime
import lime.tabular
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FaultDetectionModel:
    """Modelo de detección de fallas en baterías"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['voltage', 'current', 'temperature', 'soc', 'soh', 'internal_resistance']
        self.fault_types = {
            0: 'normal',
            1: 'degradation',
            2: 'short_circuit', 
            3: 'overcharge',
            4: 'overheat',
            5: 'thermal_runaway'
        }
        self.severity_mapping = {
            'normal': 'none',
            'degradation': 'medium',
            'short_circuit': 'critical',
            'overcharge': 'high',
            'overheat': 'high',
            'thermal_runaway': 'critical'
        }
        
    def _prepare_features(self, df):
        """Preparar características para el modelo"""
        # Seleccionar columnas disponibles
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError("No valid features found in data")
        
        # Extraer características
        X = df[available_features].copy()
        
        # Rellenar valores faltantes con la mediana
        X = X.fillna(X.median())
        
        # Agregar características derivadas
        if 'voltage' in X.columns and 'current' in X.columns:
            X['power_calculated'] = X['voltage'] * X['current']
        
        if 'temperature' in X.columns:
            X['temp_gradient'] = X['temperature'].diff().fillna(0)
        
        if 'soc' in X.columns:
            X['soc_change_rate'] = X['soc'].diff().fillna(0)
        
        if 'voltage' in X.columns:
            X['voltage_stability'] = X['voltage'].rolling(window=5, min_periods=1).std().fillna(0)
        
        return X
    
    def _simulate_training_data(self, df):
        """Simular datos de entrenamiento con etiquetas de fallas"""
        X = self._prepare_features(df)
        
        # Simular etiquetas basadas en reglas heurísticas
        y = np.zeros(len(X))
        
        for i, row in X.iterrows():
            # Reglas para detectar fallas
            if 'temperature' in row and row['temperature'] > 80:
                y[i] = 5 if row['temperature'] > 100 else 4  # thermal_runaway o overheat
            elif 'voltage' in row and row['voltage'] < 2.5:
                y[i] = 2  # short_circuit
            elif 'voltage' in row and row['voltage'] > 4.5:
                y[i] = 3  # overcharge
            elif 'soh' in row and row['soh'] < 70:
                y[i] = 1  # degradation
            else:
                y[i] = 0  # normal
        
        return X, y
    
    def train(self, df):
        """Entrenar el modelo de detección de fallas"""
        try:
            X, y = self._simulate_training_data(df)
            
            # Escalar características
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar modelo
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_scaled, y)
            
            return True
            
        except Exception as e:
            print(f"Error training fault detection model: {e}")
            return False
    
    def analyze(self, df):
        """Analizar datos para detectar fallas"""
        try:
            # Preparar características
            X = self._prepare_features(df)
            
            # Entrenar modelo si no existe
            if self.model is None:
                self.train(df)
            
            # Escalar características
            X_scaled = self.scaler.transform(X)
            
            # Realizar predicciones
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Analizar resultados
            unique_faults, counts = np.unique(predictions, return_counts=True)
            fault_distribution = dict(zip(unique_faults, counts))
            
            # Determinar falla principal
            main_fault_idx = np.argmax(counts)
            main_fault = unique_faults[main_fault_idx]
            main_fault_name = self.fault_types[main_fault]
            
            # Calcular confianza
            confidence = np.max(probabilities, axis=1).mean()
            
            # Detectar si hay falla
            fault_detected = main_fault != 0 or (counts[main_fault_idx] / len(predictions)) > 0.3
            
            result = {
                'fault_detected': fault_detected,
                'fault_type': main_fault_name if fault_detected else None,
                'severity': self.severity_mapping[main_fault_name] if fault_detected else 'none',
                'confidence': float(confidence),
                'predictions': {
                    'fault_distribution': {self.fault_types[k]: int(v) for k, v in fault_distribution.items()},
                    'main_fault': main_fault_name,
                    'fault_probability': float(1 - (fault_distribution.get(0, 0) / len(predictions)))
                },
                'analysis_details': {
                    'total_samples': len(predictions),
                    'normal_samples': int(fault_distribution.get(0, 0)),
                    'fault_samples': int(len(predictions) - fault_distribution.get(0, 0)),
                    'feature_importance': self._get_feature_importance(X.columns)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'fault_detected': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _get_feature_importance(self, feature_names):
        """Obtener importancia de características"""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(feature_names, [float(imp) for imp in importance]))

class HealthPredictionModel:
    """Modelo de predicción de salud de batería"""
    
    def __init__(self):
        self.soh_model = None
        self.rul_model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['voltage', 'current', 'temperature', 'cycles', 'capacity', 'internal_resistance']
        
    def _prepare_features(self, df):
        """Preparar características para predicción de salud"""
        # Ordenar por timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Seleccionar columnas disponibles
        available_features = [col for col in self.feature_columns if col in df_sorted.columns]
        
        if not available_features:
            raise ValueError("No valid features found in data")
        
        X = df_sorted[available_features].copy()
        
        # Rellenar valores faltantes
        X = X.fillna(X.median())
        
        # Características derivadas
        if 'cycles' in X.columns:
            X['cycle_rate'] = X['cycles'].diff().fillna(0)
        
        if 'capacity' in X.columns:
            X['capacity_fade'] = X['capacity'].pct_change().fillna(0)
        
        if 'internal_resistance' in X.columns:
            X['resistance_growth'] = X['internal_resistance'].pct_change().fillna(0)
        
        # Características temporales
        X['days_since_start'] = (pd.to_datetime(df_sorted['timestamp']) - pd.to_datetime(df_sorted['timestamp']).min()).dt.days
        
        return X
    
    def _simulate_health_targets(self, df):
        """Simular objetivos de salud basados en datos"""
        X = self._prepare_features(df)
        
        # Simular SOH basado en ciclos y capacidad
        if 'cycles' in df.columns and 'capacity' in df.columns:
            # SOH decrece con los ciclos
            max_cycles = df['cycles'].max() if df['cycles'].max() > 0 else 1000
            soh_target = 100 - (df['cycles'] / max_cycles) * 30  # Degradación simulada
            
            # Ajustar por capacidad si está disponible
            if df['capacity'].notna().any():
                initial_capacity = df['capacity'].dropna().iloc[0]
                current_capacity = df['capacity'].dropna().iloc[-1]
                capacity_ratio = current_capacity / initial_capacity if initial_capacity > 0 else 1
                soh_target = soh_target * capacity_ratio
        else:
            # SOH basado en tiempo si no hay ciclos
            days_elapsed = (pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()).days
            soh_target = 100 - (days_elapsed / 365) * 10  # 10% degradación por año
        
        # Simular RUL (días restantes hasta SOH < 80%)
        current_soh = soh_target.iloc[-1] if len(soh_target) > 0 else 90
        if current_soh > 80:
            # Estimar tasa de degradación
            if len(soh_target) > 10:
                degradation_rate = (soh_target.iloc[0] - soh_target.iloc[-1]) / len(soh_target)
                rul_days = (current_soh - 80) / degradation_rate if degradation_rate > 0 else 365
            else:
                rul_days = 365  # Valor por defecto
        else:
            rul_days = 0
        
        return soh_target, np.full(len(X), max(0, rul_days))
    
    def train(self, df):
        """Entrenar modelos de predicción de salud"""
        try:
            X = self._prepare_features(df)
            soh_target, rul_target = self._simulate_health_targets(df)
            
            # Escalar características
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar modelo SOH
            self.soh_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.soh_model.fit(X_scaled, soh_target)
            
            # Entrenar modelo RUL
            self.rul_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rul_model.fit(X_scaled, rul_target)
            
            return True
            
        except Exception as e:
            print(f"Error training health prediction model: {e}")
            return False
    
    def analyze(self, df):
        """Analizar salud de la batería y predecir RUL"""
        try:
            # Preparar características
            X = self._prepare_features(df)
            
            # Entrenar modelos si no existen
            if self.soh_model is None or self.rul_model is None:
                self.train(df)
            
            # Escalar características
            X_scaled = self.scaler.transform(X)
            
            # Realizar predicciones
            soh_predictions = self.soh_model.predict(X_scaled)
            rul_predictions = self.rul_model.predict(X_scaled)
            
            # Calcular métricas
            current_soh = float(soh_predictions[-1])
            current_rul = float(rul_predictions[-1])
            
            # Calcular tendencia de degradación
            if len(soh_predictions) > 10:
                recent_soh = soh_predictions[-10:]
                degradation_trend = np.polyfit(range(len(recent_soh)), recent_soh, 1)[0]
            else:
                degradation_trend = 0
            
            # Calcular confianza basada en estabilidad de predicciones
            soh_std = np.std(soh_predictions[-10:]) if len(soh_predictions) > 10 else 0
            confidence = max(0.5, 1.0 - (soh_std / 100))
            
            # Clasificar estado de salud
            if current_soh > 90:
                health_status = 'excellent'
            elif current_soh > 80:
                health_status = 'good'
            elif current_soh > 70:
                health_status = 'fair'
            elif current_soh > 60:
                health_status = 'poor'
            else:
                health_status = 'critical'
            
            result = {
                'current_soh': current_soh,
                'rul_days': max(0, current_rul),
                'health_status': health_status,
                'degradation_trend': float(degradation_trend),
                'confidence': float(confidence),
                'predictions': {
                    'soh_history': [float(x) for x in soh_predictions[-50:]],  # Últimas 50 predicciones
                    'rul_history': [float(x) for x in rul_predictions[-50:]],
                    'timestamps': df['timestamp'].tail(min(50, len(df))).tolist()
                },
                'analysis_details': {
                    'total_samples': len(soh_predictions),
                    'prediction_stability': float(1.0 - soh_std / 100) if soh_std > 0 else 1.0,
                    'feature_importance': self._get_feature_importance(X.columns)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'current_soh': 0,
                'rul_days': 0,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _get_feature_importance(self, feature_names):
        """Obtener importancia de características"""
        if self.soh_model is None:
            return {}
        
        importance = self.soh_model.feature_importances_
        return dict(zip(feature_names, [float(imp) for imp in importance]))

class XAIExplainer:
    """Explicador de IA usando SHAP y LIME"""
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
    
    def explain_fault_detection(self, df, prediction_result):
        """Explicar predicciones de detección de fallas"""
        try:
            # Preparar datos para explicación
            fault_model = FaultDetectionModel()
            X = fault_model._prepare_features(df)
            
            if fault_model.model is None:
                fault_model.train(df)
            
            X_scaled = fault_model.scaler.transform(X)
            
            # Explicación SHAP (simplificada)
            # En una implementación real, se usaría SHAP con el modelo entrenado
            feature_importance = fault_model._get_feature_importance(X.columns)
            
            # Crear explicación basada en importancia de características
            explanation = {
                'method': 'feature_importance',
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                'feature_contributions': feature_importance,
                'explanation_text': self._generate_fault_explanation(prediction_result, feature_importance)
            }
            
            return explanation
            
        except Exception as e:
            return {'error': str(e), 'method': 'failed'}
    
    def explain_health_prediction(self, df, prediction_result):
        """Explicar predicciones de salud"""
        try:
            health_model = HealthPredictionModel()
            X = health_model._prepare_features(df)
            
            if health_model.soh_model is None:
                health_model.train(df)
            
            feature_importance = health_model._get_feature_importance(X.columns)
            
            explanation = {
                'method': 'feature_importance',
                'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                'feature_contributions': feature_importance,
                'explanation_text': self._generate_health_explanation(prediction_result, feature_importance)
            }
            
            return explanation
            
        except Exception as e:
            return {'error': str(e), 'method': 'failed'}
    
    def _generate_fault_explanation(self, prediction_result, feature_importance):
        """Generar explicación textual para detección de fallas"""
        if not prediction_result.get('fault_detected'):
            return "La batería muestra un comportamiento normal. Todos los parámetros están dentro de rangos aceptables."
        
        fault_type = prediction_result.get('fault_type', 'unknown')
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanations = {
            'degradation': f"Se detectó degradación de la batería. Los factores más influyentes son: {', '.join([f[0] for f in top_features])}.",
            'short_circuit': f"Posible cortocircuito interno detectado. Principales indicadores: {', '.join([f[0] for f in top_features])}.",
            'overcharge': f"Condición de sobrecarga detectada. Parámetros críticos: {', '.join([f[0] for f in top_features])}.",
            'overheat': f"Sobrecalentamiento detectado. Factores de riesgo: {', '.join([f[0] for f in top_features])}.",
            'thermal_runaway': f"¡ALERTA CRÍTICA! Posible fuga térmica. Factores determinantes: {', '.join([f[0] for f in top_features])}."
        }
        
        return explanations.get(fault_type, f"Falla de tipo {fault_type} detectada.")
    
    def _generate_health_explanation(self, prediction_result, feature_importance):
        """Generar explicación textual para predicción de salud"""
        current_soh = prediction_result.get('current_soh', 0)
        rul_days = prediction_result.get('rul_days', 0)
        health_status = prediction_result.get('health_status', 'unknown')
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"Estado de salud actual: {current_soh:.1f}% ({health_status}). "
        explanation += f"Vida útil restante estimada: {rul_days:.0f} días. "
        explanation += f"Los factores más influyentes en esta predicción son: {', '.join([f[0] for f in top_features])}."
        
        if current_soh < 80:
            explanation += " Se recomienda considerar el reemplazo de la batería."
        elif current_soh < 90:
            explanation += " Se recomienda monitoreo frecuente."
        
        return explanation

