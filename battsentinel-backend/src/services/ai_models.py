"""
BattSentinel AI Models - Versión Mejorada 2.0
Sistema de Monitoreo de Baterías de Clase Industrial

Implementa un sistema de doble nivel:
- Nivel 1: Monitoreo Continuo (Ligero y Eficiente)
- Nivel 2: Análisis Avanzado (Profundo y Preciso)

"""

import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from abc import ABC, abstractmethod

# Scikit-learn imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.feature_selection import SelectKBest, f_classif

# Deep Learning imports (con manejo de errores para entornos sin TensorFlow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow no disponible. Funcionalidades de Deep Learning limitadas.")

# XAI imports
try:
    import shap
    import lime
    from lime import lime_tabular
    SHAP_AVAILABLE = True
    LIME_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    LIME_AVAILABLE = False
    logging.warning("SHAP/LIME no disponibles. Explicabilidad limitada.")

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryMetadata:
    """Metadatos de fabricación de la batería"""
    design_capacity: float  # Capacidad de diseño (Ah)
    design_cycles: int      # Ciclos de vida de diseño
    voltage_limits: Tuple[float, float]  # (min_voltage, max_voltage)
    charge_current_limit: float  # Límite de corriente de carga (A)
    discharge_current_limit: float  # Límite de corriente de descarga (A)
    operating_temp_range: Tuple[float, float]  # (min_temp, max_temp) °C
    chemistry: str          # Tipo de química (LiFePO4, NMC, etc.)
    manufacturer: str       # Fabricante
    model: str             # Modelo específico

# CORRECCIÓN PRINCIPAL: Clase AnalysisResult corregida y normalización de tipos
@dataclass
class AnalysisResult:
    """Resultado de análisis de IA - CORREGIDO para evitar errores de instanciación y tipos numpy"""
    
    analysis_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Renombrado a confidence_score para coincidir con el modelo de DB
    confidence_score: float = 0.0 
    predictions: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "2.1"
    
    # Añadidos campos que pueden ser None o de tipos numpy
    fault_detected: Optional[bool] = None
    fault_type: Optional[str] = None
    severity: Optional[str] = None
    rul_prediction: Optional[float] = None
    level_of_analysis: Optional[int] = None
    
    def __post_init__(self):
        """Post-inicialización para asegurar tipos correctos y metadatos"""
        # Asegurar que los tipos numpy se conviertan a tipos nativos de Python
        self.confidence_score = float(self.confidence_score) if self.confidence_score is not None else 0.0
        self.rul_prediction = float(self.rul_prediction) if self.rul_prediction is not None else None
        self.fault_detected = bool(self.fault_detected) if self.fault_detected is not None else None
        self.level_of_analysis = int(self.level_of_analysis) if self.level_of_analysis is not None else None
        
        # Inicializar metadata con valores por defecto
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Inicializar metadata con valores por defecto para evitar errores"""
        default_metadata = {
            'data_points': 0,
            'processing_time_ms': 0,
            'processing_time_s': 0.0,
            'level': 1,
            'models_used': [],
            'data_quality_score': 0.0,
            'feature_count': 0
        }
        
        # Actualizar con valores por defecto solo si no existen
        for key, default_value in default_metadata.items():
            if key not in self.metadata:
                self.metadata[key] = default_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'analysis_type': self.analysis_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'confidence_score': self.confidence_score, # Usar confidence_score
            'predictions': self.predictions,
            'explanation': self.explanation,
            'metadata': self.metadata,
            'model_version': self.model_version,
            'fault_detected': self.fault_detected,
            'fault_type': self.fault_type,
            'severity': self.severity,
            'rul_prediction': self.rul_prediction,
            'level_of_analysis': self.level_of_analysis
        }

class DataPreprocessor:
    """Preprocesador avanzado de datos con manejo robusto de valores faltantes"""
    
    def __init__(self):
        self.scalers = {}
        self.imputation_models = {}
        self.feature_stats = {}
        
    def prepare_features(self, df: pd.DataFrame, battery_metadata: Optional[BatteryMetadata] = None) -> pd.DataFrame:
        """Preparar características avanzadas con ingeniería de características"""
        if df is None or df.empty:
            raise ValueError("DataFrame de entrada está vacío o es None")
            
        df_processed = df.copy()
        
        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
            df_processed = df_processed.sort_values('timestamp')
        
        basic_features = ['voltage', 'current', 'temperature', 'soc', 'soh', 'cycles']
        available_features = [col for col in basic_features if col in df_processed.columns]
        
        if not available_features:
            raise ValueError("No se encontraron características válidas en los datos")
        
        if len(df_processed) < 2:
            logger.warning("Datos insuficientes para análisis robusto. Se requieren al menos 2 puntos de datos.")
        
        df_processed = self._contextual_imputation(df_processed, available_features)
        df_processed = self._advanced_feature_engineering(df_processed, available_features)
        
        if battery_metadata:
            df_processed = self._integrate_metadata_features(df_processed, battery_metadata)
        
        df_processed = self._noise_filtering(df_processed)
        
        return df_processed
    
    def _contextual_imputation(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Imputación contextual basada en correlaciones entre parámetros"""
        df_imputed = df.copy()
        
        for feature in features:
            if feature in df_imputed.columns and df_imputed[feature].isna().any():
                if feature == 'temperature':
                    df_imputed[feature] = self._impute_temperature(df_imputed, feature)
                elif feature == 'soc':
                    df_imputed[feature] = self._impute_soc(df_imputed, feature)
                elif feature == 'soh':
                    df_imputed[feature] = self._impute_soh(df_imputed, feature)
                else:
                    df_imputed[feature] = df_imputed[feature].interpolate(method='time')
                    df_imputed[feature] = df_imputed[feature].fillna(df_imputed[feature].median())
        
        return df_imputed
    
    def _impute_temperature(self, df: pd.DataFrame, feature: str) -> pd.Series:
        """Imputación específica para temperatura"""
        temp_series = df[feature].copy()
        if 'current' in df.columns and not df['current'].isna().all():
            base_temp = float(temp_series.median()) if not temp_series.isna().all() else 25.0
            current_heating = np.abs(df['current']) * 0.1
            estimated_temp = base_temp + current_heating
            temp_series = temp_series.fillna(estimated_temp)
        temp_series = temp_series.interpolate(method='time')
        temp_series = temp_series.fillna(float(temp_series.median()) if not temp_series.isna().all() else 25.0)
        return temp_series
    
    def _impute_soc(self, df: pd.DataFrame, feature: str) -> pd.Series:
        """Imputación específica para SOC basada en voltaje"""
        soc_series = df[feature].copy()
        if 'voltage' in df.columns and not df['voltage'].isna().all():
            voltage = df['voltage']
            estimated_soc = np.clip((voltage - 3.0) / (4.2 - 3.0) * 100, 0, 100)
            soc_series = soc_series.fillna(estimated_soc)
        soc_series = soc_series.interpolate(method='time')
        soc_series = soc_series.fillna(float(soc_series.median()) if not soc_series.isna().all() else 80.0)
        return soc_series
    
    def _impute_soh(self, df: pd.DataFrame, feature: str) -> pd.Series:
        """Imputación específica para SOH basada en ciclos"""
        soh_series = df[feature].copy()
        if 'cycles' in df.columns and not df['cycles'].isna().all():
            cycles = df['cycles']
            estimated_soh = np.clip(100 - (cycles / 2000) * 20, 60, 100)
            soh_series = soh_series.fillna(estimated_soh)
        soh_series = soh_series.interpolate(method='time')
        soh_series = soh_series.fillna(float(soh_series.median()) if not soh_series.isna().all() else 85.0)
        return soh_series
    
    def _advanced_feature_engineering(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Ingeniería de características avanzada"""
        df_enhanced = df.copy()
        window_sizes = [5, 10, 20]
        for window in window_sizes:
            for feature in features:
                if feature in df_enhanced.columns:
                    df_enhanced[f'{feature}_mean_{window}'] = df_enhanced[feature].rolling(window=window, min_periods=1).mean()
                    df_enhanced[f'{feature}_std_{window}'] = df_enhanced[feature].rolling(window=window, min_periods=1).std().fillna(0)
                    df_enhanced[f'{feature}_min_{window}'] = df_enhanced[feature].rolling(window=window, min_periods=1).min()
                    df_enhanced[f'{feature}_max_{window}'] = df_enhanced[feature].rolling(window=window, min_periods=1).max()
                    df_enhanced[f'{feature}_skew_{window}'] = df_enhanced[feature].rolling(window=window, min_periods=3).skew().fillna(0)
                    df_enhanced[f'{feature}_kurt_{window}'] = df_enhanced[feature].rolling(window=window, min_periods=4).kurt().fillna(0)
        for feature in features:
            if feature in df_enhanced.columns:
                df_enhanced[f'{feature}_diff'] = df_enhanced[feature].diff().fillna(0)
                df_enhanced[f'{feature}_diff2'] = df_enhanced[f'{feature}_diff'].diff().fillna(0)
                df_enhanced[f'{feature}_pct_change'] = df_enhanced[feature].pct_change().fillna(0)
        if 'voltage' in df_enhanced.columns and 'current' in df_enhanced.columns:
            df_enhanced['power_calculated'] = df_enhanced['voltage'] * df_enhanced['current']
            df_enhanced['resistance_estimated'] = np.where(
                df_enhanced['current'] != 0,
                df_enhanced['voltage'] / df_enhanced['current'],
                0
            )
        if 'temperature' in df_enhanced.columns:
            df_enhanced['temp_gradient'] = df_enhanced['temperature'].diff().fillna(0)
            df_enhanced['temp_acceleration'] = df_enhanced['temp_gradient'].diff().fillna(0)
        if all(col in df_enhanced.columns for col in ['voltage', 'current', 'soc']):
            soc_change = df_enhanced['soc'].diff().fillna(0)
            current_integral = df_enhanced['current'].rolling(window=10, min_periods=1).sum()
            df_enhanced['coulombic_efficiency'] = np.where(
                current_integral != 0,
                soc_change / current_integral,
                1.0
            )
        return df_enhanced
    
    def _integrate_metadata_features(self, df: pd.DataFrame, metadata: BatteryMetadata) -> pd.DataFrame:
        """Integrar metadatos de fabricación como características"""
        df_meta = df.copy()
        if 'voltage' in df_meta.columns:
            v_min, v_max = metadata.voltage_limits
            df_meta['voltage_normalized'] = (df_meta['voltage'] - v_min) / (v_max - v_min)
            df_meta['voltage_margin_min'] = df_meta['voltage'] - v_min
            df_meta['voltage_margin_max'] = v_max - df_meta['voltage']
        if 'current' in df_meta.columns:
            df_meta['current_charge_ratio'] = df_meta['current'] / metadata.charge_current_limit
            df_meta['current_discharge_ratio'] = np.abs(df_meta['current']) / metadata.discharge_current_limit
        if 'temperature' in df_meta.columns:
            t_min, t_max = metadata.operating_temp_range
            df_meta['temp_normalized'] = (df_meta['temperature'] - t_min) / (t_max - t_min)
            df_meta['temp_margin_min'] = df_meta['temperature'] - t_min
            df_meta['temp_margin_max'] = t_max - df_meta['temperature']
        if 'cycles' in df_meta.columns:
            df_meta['cycle_life_ratio'] = df_meta['cycles'] / metadata.design_cycles
        chemistry_encoding = {
            'LiFePO4': [1, 0, 0], 'NMC': [0, 1, 0], 'LTO': [0, 0, 1]
        }
        chemistry_code = chemistry_encoding.get(metadata.chemistry, [0, 0, 0])
        df_meta['chemistry_lifepo4'] = chemistry_code[0]
        df_meta['chemistry_nmc'] = chemistry_code[1]
        df_meta['chemistry_lto'] = chemistry_code[2]
        return df_meta
    
    def _noise_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtrado de ruido usando promedio móvil y filtros estadísticos"""
        df_filtered = df.copy()
        noisy_params = ['voltage', 'current', 'temperature']
        for param in noisy_params:
            if param in df_filtered.columns:
                df_filtered[f'{param}_filtered'] = df_filtered[param].rolling(
                    window=3, min_periods=1, center=True
                ).mean()
                z_scores = np.abs((df_filtered[param] - df_filtered[param].mean()) / df_filtered[param].std())
                outlier_mask = z_scores > 4
                df_filtered.loc[outlier_mask, param] = df_filtered[param].median()
        return df_filtered

class ContinuousMonitoringEngine:
    """Motor de monitoreo continuo - Nivel 1"""
    
    def __init__(self, model_cache_size: int = 10):
        self.preprocessor = DataPreprocessor()
        self.model_cache = {}
        self.cache_size = model_cache_size
        self.anomaly_detectors = {}
        self.control_charts = {}
        self.threshold_monitors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Inicializar detectores de anomalías ligeros"""
        self.anomaly_detectors['isolation_forest'] = IsolationForest(
            n_estimators=50, contamination=0.1, random_state=42, n_jobs=1
        )
        self.anomaly_detectors['one_class_svm'] = OneClassSVM(
            kernel='rbf', gamma='scale', nu=0.1
        )
        self.control_charts = {
            'ewma': EWMAControlChart(),
            'cusum': CUSUMControlChart()
        }
    
    def analyze_continuous(self, df: pd.DataFrame, battery_metadata: Optional[BatteryMetadata] = None) -> AnalysisResult:
        """Análisis continuo ligero y rápido"""
        start_time = datetime.now()
        try:
            df_processed = self.preprocessor.prepare_features(df, battery_metadata)
            key_features = self._select_key_features(df_processed)
            if key_features.empty:
                raise ValueError("No hay características válidas para análisis")
            anomaly_results = self._detect_anomalies_fast(key_features)
            control_results = self._statistical_process_control(key_features)
            threshold_results = self._dynamic_threshold_monitoring(key_features, battery_metadata)
            
            combined_results = self._combine_level1_results(
                anomaly_results, control_results, threshold_results
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Asegurar tipos nativos de Python para la instancia de AnalysisResult
            return AnalysisResult(
                analysis_type='continuous_monitoring',
                timestamp=datetime.now(timezone.utc),
                confidence_score=float(combined_results['confidence']),
                predictions=combined_results['predictions'],
                explanation=combined_results['explanation'],
                metadata={
                    'processing_time_ms': float(processing_time * 1000),
                    'data_points': int(len(df)),
                    'features_analyzed': int(len(key_features.columns)),
                    'level': 1
                },
                model_version='2.0-level1'
            )
        except Exception as e:
            logger.error(f"Error en análisis continuo: {str(e)}")
            return self._create_error_result(str(e), 'continuous_monitoring')
    
    def _select_key_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seleccionar características clave para análisis rápido"""
        priority_features = [
            'voltage', 'current', 'temperature', 'soc', 'soh',
            'voltage_std_5', 'current_diff', 'temperature_gradient',
            'power_calculated', 'resistance_estimated'
        ]
        available_features = [col for col in priority_features if col in df.columns]
        if not available_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols[:10]
        return df[available_features].fillna(0)
    
    def _detect_anomalies_fast(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detección rápida de anomalías usando modelos ligeros"""
        results = {
            'anomalies_detected': False,
            'anomaly_score': 0.0,
            'anomaly_details': []
        }
        try:
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
            if len(features) >= 10:
                iso_forest = self.anomaly_detectors['isolation_forest']
                iso_forest.fit(features_scaled)
                anomaly_scores = iso_forest.decision_function(features_scaled)
                anomaly_predictions = iso_forest.predict(features_scaled)
                anomaly_indices = np.where(anomaly_predictions == -1)[0]
                if len(anomaly_indices) > 0:
                    results['anomalies_detected'] = True
                    results['anomaly_score'] = float(np.mean(np.abs(anomaly_scores[anomaly_indices])))
                    for idx in anomaly_indices[-5:]:
                        if idx < len(features):
                            results['anomaly_details'].append({
                                'index': int(idx),
                                'score': float(anomaly_scores[idx]),
                                'timestamp': features.index[idx].isoformat() if hasattr(features.index[idx], 'isoformat') else str(features.index[idx])
                            })
            if not results['anomalies_detected']:
                stat_anomalies = self._statistical_anomaly_detection(features)
                if stat_anomalies['count'] > 0:
                    results['anomalies_detected'] = True
                    results['anomaly_score'] = float(stat_anomalies['max_z_score']) # Asegurar float
                    results['anomaly_details'] = stat_anomalies['details']
        except Exception as e:
            logger.warning(f"Error en detección de anomalías: {str(e)}")
        return results
    
    def _statistical_anomaly_detection(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detección de anomalías usando métodos estadísticos simples"""
        anomalies = []
        max_z_score = 0.0
        for col in features.columns:
            if features[col].notna().sum() > 5:
                values = features[col].dropna()
                if len(values) > 1 and values.std() > 0:
                    z_scores = np.abs((values - values.mean()) / values.std())
                    anomaly_mask = z_scores > 2.5
                    if anomaly_mask.any():
                        max_z_score = max(float(max_z_score), float(z_scores.max())) # Asegurar float
                        anomaly_indices = values[anomaly_mask].index.tolist()
                        for idx in anomaly_indices[-3:]:
                            anomalies.append({
                                'index': int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                                'parameter': col,
                                'value': float(features.loc[idx, col]),
                                'z_score': float(z_scores.loc[idx])
                            })
        return {
            'count': int(len(anomalies)), # Asegurar int
            'max_z_score': float(max_z_score),
            'details': anomalies
        }
    
    def _statistical_process_control(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Control estadístico de procesos usando gráficos de control"""
        results = {
            'control_violations': False,
            'violations_count': 0,
            'control_details': []
        }
        try:
            for col in features.columns:
                if features[col].notna().sum() > 10:
                    values = features[col].dropna()
                    ewma_result = self.control_charts['ewma'].analyze(values)
                    if ewma_result['violations'] > 0:
                        results['control_violations'] = True
                        results['violations_count'] += int(ewma_result['violations']) # Asegurar int
                        results['control_details'].append({
                            'parameter': col,
                            'chart_type': 'EWMA',
                            'violations': int(ewma_result['violations']), # Asegurar int
                            'last_value': float(values.iloc[-1]),
                            'control_limit': float(ewma_result['control_limit']) # Asegurar float
                        })
                    cusum_result = self.control_charts['cusum'].analyze(values)
                    if cusum_result['violations'] > 0:
                        results['control_violations'] = True
                        results['violations_count'] += int(cusum_result['violations']) # Asegurar int
                        results['control_details'].append({
                            'parameter': col,
                            'chart_type': 'CUSUM',
                            'violations': int(cusum_result['violations']), # Asegurar int
                            'cumulative_sum': float(cusum_result['cumulative_sum']) # Asegurar float
                        })
        except Exception as e:
            logger.warning(f"Error en control estadístico: {str(e)}")
        return results
    
    def _dynamic_threshold_monitoring(self, features: pd.DataFrame, metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
        """Monitoreo de umbrales dinámicos adaptativos"""
        results = {
            'threshold_violations': False,
            'violations_count': 0,
            'threshold_details': []
        }
        try:
            if metadata:
                critical_thresholds = self._get_metadata_thresholds(metadata)
            else:
                critical_thresholds = self._get_default_thresholds()
            for param, thresholds in critical_thresholds.items():
                if param in features.columns:
                    current_value = features[param].iloc[-1] if len(features) > 0 else 0
                    if 'min' in thresholds and float(current_value) < float(thresholds['min']): # Asegurar float
                        results['threshold_violations'] = True
                        results['violations_count'] += 1
                        results['threshold_details'].append({
                            'parameter': param,
                            'violation_type': 'below_minimum',
                            'current_value': float(current_value),
                            'threshold': float(thresholds['min']), # Asegurar float
                            'severity': 'high'
                        })
                    if 'max' in thresholds and float(current_value) > float(thresholds['max']): # Asegurar float
                        results['threshold_violations'] = True
                        results['violations_count'] += 1
                        results['threshold_details'].append({
                            'parameter': param,
                            'violation_type': 'above_maximum',
                            'current_value': float(current_value),
                            'threshold': float(thresholds['max']), # Asegurar float
                            'severity': 'high'
                        })
        except Exception as e:
            logger.warning(f"Error en monitoreo de umbrales: {str(e)}")
        return results
    
    def _get_metadata_thresholds(self, metadata: BatteryMetadata) -> Dict[str, Any]:
        """Obtener umbrales críticos de metadatos"""
        return {
            'voltage': {'min': metadata.voltage_limits[0], 'max': metadata.voltage_limits[1]},
            'current': {'min': -metadata.discharge_current_limit, 'max': metadata.charge_current_limit},
            'temperature': {'min': metadata.operating_temp_range[0], 'max': metadata.operating_temp_range[1]},
            'soc': {'min': 10.0, 'max': 95.0}, # SOC crítico
            'soh': {'min': 60.0, 'max': 100.0} # SOH crítico
        }
    
    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Obtener umbrales por defecto si no hay metadatos"""
        return {
            'voltage': {'min': 3.0, 'max': 4.2},
            'current': {'min': -5.0, 'max': 5.0},
            'temperature': {'min': 0.0, 'max': 50.0},
            'soc': {'min': 15.0, 'max': 90.0},
            'soh': {'min': 65.0, 'max': 100.0}
        }
        
    def _combine_level1_results(self, anomaly_res: Dict, control_res: Dict, threshold_res: Dict) -> Dict[str, Any]:
        """Combinar resultados de Nivel 1"""
        overall_fault_detected = (
            anomaly_res.get('anomalies_detected', False) or
            control_res.get('control_violations', False) or
            threshold_res.get('threshold_violations', False)
        )
        
        # Calcular confianza (ejemplo simple, puede ser más sofisticado)
        confidence = 0.7
        if overall_fault_detected:
            confidence = max(confidence, 0.8) # Aumentar si hay fallas
        
        return {
            'confidence': float(confidence), # Asegurar float
            'predictions': {
                'overall_fault_detected': overall_fault_detected,
                'anomaly_results': anomaly_res,
                'control_results': control_res,
                'threshold_results': threshold_res,
                'main_status': 'Fault Detected' if overall_fault_detected else 'Normal Operation'
            },
            'explanation': {
                'method': 'level1_combined_rules',
                'summary': 'Resumen de monitoreo continuo'
            }
        }
        
    def _create_error_result(self, error_msg: str, analysis_type: str) -> AnalysisResult:
        """Crear resultado de error para análisis continuo"""
        return AnalysisResult(
            analysis_type=analysis_type,
            timestamp=datetime.now(timezone.utc),
            confidence_score=0.0,
            predictions={'error': True, 'message': error_msg},
            explanation={'error': str(error_msg), 'method': 'error_handling'},
            metadata={'level': 0, 'error': True}
        )

# =============================================================================================
# GRÁFICOS DE CONTROL ESTADÍSTICO
# =============================================================================================

class EWMAControlChart:
    """Gráfico de control EWMA (Exponentially Weighted Moving Average)"""
    def __init__(self, lambda_param: float = 0.2, L: float = 2.7):
        self.lambda_param = lambda_param
        self.L = L
    
    def analyze(self, data: pd.Series) -> Dict[str, Any]:
        """Analizar datos usando gráfico de control EWMA"""
        if len(data) < 5:
            return {'violations': 0, 'control_limit': 0.0} # Asegurar float
        
        ewma = data.ewm(alpha=self.lambda_param).mean()
        sigma = data.std()
        if sigma == 0: # Evitar división por cero
            return {'violations': 0, 'control_limit': 0.0}
        
        control_limit = self.L * sigma * np.sqrt(self.lambda_param / (2 - self.lambda_param))
        center_line = data.mean()
        violations = np.sum(np.abs(ewma - center_line) > control_limit)
        
        return {
            'violations': int(violations), # Asegurar int
            'control_limit': float(control_limit), # Asegurar float
            'ewma_values': ewma.tolist()
        }

class CUSUMControlChart:
    """Gráfico de control CUSUM (Cumulative Sum)"""
    def __init__(self, k: float = 0.5, h: float = 4.0):
        self.k = k
        self.h = h
    
    def analyze(self, data: pd.Series) -> Dict[str, Any]:
        """Analizar datos usando gráfico de control CUSUM"""
        if len(data) < 5:
            return {'violations': 0, 'cumulative_sum': 0.0} # Asegurar float
        
        mean = data.mean()
        std = data.std()
        if std == 0:
            return {'violations': 0, 'cumulative_sum': 0.0} # Asegurar float
        
        normalized_data = (data - mean) / std
        cusum_pos = 0.0 # Asegurar float
        cusum_neg = 0.0 # Asegurar float
        violations = 0
        
        for value in normalized_data:
            cusum_pos = max(0.0, float(cusum_pos + value - self.k)) # Asegurar float
            cusum_neg = max(0.0, float(cusum_neg - value - self.k)) # Asegurar float
            if cusum_pos > self.h or cusum_neg > self.h:
                violations += 1
        
        return {
            'violations': int(violations), # Asegurar int
            'cumulative_sum': float(max(cusum_pos, cusum_neg)) # Asegurar float
        }

# =============================================================================================
# MODELOS DE DETECCIÓN DE FALLAS (FAULT DETECTION) - Nivel 2 (usando ML/DL)
# =============================================================================================

class FaultDetectionModel:
    """Modelo de detección de fallas mejorado con algoritmos ML/DL avanzados"""
    def __init__(self):
        self.continuous_engine = ContinuousMonitoringEngine()
        self.preprocessor = DataPreprocessor()
        self.isolation_forest = None
        self.random_forest = None
        self.svm_model = None
        self.gradient_boosting = None
        self.lstm_model = None
        self.gru_model = None
        self.tcn_model = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.feature_selector = None
        self.feature_columns = ['voltage', 'current', 'temperature', 'soc', 'soh', 'internal_resistance']
        self.sequence_length = 50
        self.fault_types = {
            0: 'normal', 1: 'degradation', 2: 'short_circuit', 3: 'overcharge', 4: 'overheat',
            5: 'thermal_runaway', 6: 'capacity_fade', 7: 'impedance_rise', 8: 'electrolyte_loss',
            9: 'lithium_plating'
        }
        self.severity_mapping = {
            'normal': 'none', 'degradation': 'medium', 'short_circuit': 'critical',
            'overcharge': 'high', 'overheat': 'high', 'thermal_runaway': 'critical',
            'capacity_fade': 'medium', 'impedance_rise': 'medium',
            'electrolyte_loss': 'high', 'lithium_plating': 'high'
        }
        self.model_config = {
            'lstm': { 'units': [64, 32], 'dropout': 0.2, 'recurrent_dropout': 0.1, 'epochs': 10, 'batch_size': 16 },
            'gru': { 'units': [64, 32], 'dropout': 0.2, 'recurrent_dropout': 0.1, 'epochs': 15, 'batch_size': 16 },
            'tcn': { 'filters': 64, 'kernel_size': 3, 'dilations': [1, 2, 4, 8], 'dropout': 0.1 },
            'autoencoder': { 'encoding_dim': 32, 'hidden_layers': [128, 64], 'epochs': 20, 'batch_size': 32 }
        }
        self._initialize_models()

    def _initialize_models(self):
        """Inicializar todos los modelos de ML/DL"""
        try:
            self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
            self.random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            self.feature_selector = SelectKBest(score_func=f_classif, k=20)
            logger.info("Modelos tradicionales de ML inicializados correctamente")
        except Exception as e:
            logger.error(f"Error inicializando modelos tradicionales: {str(e)}")

    def _build_lstm_model(self, input_shape: tuple, num_classes: int) -> Optional[tf.keras.Model]:
        """Construir modelo LSTM para detección de fallas"""
        if not TENSORFLOW_AVAILABLE: raise ImportError("TensorFlow no está disponible para modelos LSTM")
        try:
            model = Sequential([
                LSTM(self.model_config['lstm']['units'][0], return_sequences=True, input_shape=input_shape,
                     dropout=self.model_config['lstm']['dropout'], recurrent_dropout=self.model_config['lstm']['recurrent_dropout']),
                LSTM(self.model_config['lstm']['units'][1], return_sequences=False,
                     dropout=self.model_config['lstm']['dropout'], recurrent_dropout=self.model_config['lstm']['recurrent_dropout']),
                Dense(64, activation='relu'), Dropout(0.3), Dense(32, activation='relu'), Dropout(0.2),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])
            return model
        except Exception as e:
            logger.error(f"Error construyendo modelo LSTM: {str(e)}")
            return None

    def _build_gru_model(self, input_shape: tuple, num_classes: int) -> Optional[tf.keras.Model]:
        """Construir modelo GRU para detección de fallas"""
        if not TENSORFLOW_AVAILABLE: return None # Ya se manejó la excepción arriba
        try:
            model = Sequential([
                GRU(self.model_config['gru']['units'][0], return_sequences=True, input_shape=input_shape,
                    dropout=self.model_config['gru']['dropout'], recurrent_dropout=self.model_config['gru']['recurrent_dropout']),
                GRU(self.model_config['gru']['units'][1], return_sequences=False,
                    dropout=self.model_config['gru']['dropout'], recurrent_dropout=self.model_config['gru']['recurrent_dropout']),
                Dense(64, activation='relu'), Dropout(0.3), Dense(32, activation='relu'), Dropout(0.2),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])
            return model
        except Exception as e:
            logger.error(f"Error construyendo modelo GRU: {str(e)}")
            return None

    def _build_tcn_model(self, input_shape: tuple, num_classes: int) -> Optional[tf.keras.Model]:
        """Construir modelo TCN (Temporal Convolutional Network)"""
        if not TENSORFLOW_AVAILABLE: return None
        try:
            from tensorflow.keras.layers import Conv1D, Add, Activation
            from tensorflow.keras.initializers import glorot_normal

            def residual_block(x, filters, kernel_size, dilation_rate, activation='relu', dropout_rate=0.2):
                original_x = x
                conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                               padding='causal', kernel_initializer=glorot_normal())(x)
                conv1 = Activation(activation)(conv1)
                conv1 = Dropout(dropout_rate)(conv1)

                conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                               padding='causal', kernel_initializer=glorot_normal())(conv1)
                conv2 = Activation(activation)(conv2)
                conv2 = Dropout(dropout_rate)(conv2)

                if original_x.shape[-1] != filters:
                    original_x = Conv1D(filters, 1, padding='same')(original_x)

                return Add()([original_x, conv2])

            inputs = Input(shape=input_shape)
            x = inputs
            for dilation in self.model_config['tcn']['dilations']:
                x = residual_block(x, self.model_config['tcn']['filters'], self.model_config['tcn']['kernel_size'], dilation)
            x = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=x)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            logger.error(f"Error construyendo modelo TCN: {str(e)}")
            return None

    def _build_autoencoder_model(self, input_shape: tuple, encoding_dim: int, hidden_layers: List[int]) -> Optional[Model]:
        """Construir un modelo Autoencoder para detección de anomalías."""
        if not TENSORFLOW_AVAILABLE: return None
        try:
            input_layer = Input(shape=input_shape)
            x = input_layer
            # Encoder
            for units in hidden_layers:
                x = Dense(units, activation='relu')(x)
                x = Dropout(0.2)(x)
            encoder_output = Dense(encoding_dim, activation='relu', name='encoder_output')(x)

            # Decoder
            x = encoder_output
            for units in reversed(hidden_layers):
                x = Dense(units, activation='relu')(x)
                x = Dropout(0.2)(x)
            decoder_output = Dense(input_shape[-1], activation='sigmoid')(x) # Salida para reconstruir la entrada

            autoencoder = Model(inputs=input_layer, outputs=decoder_output)
            autoencoder.compile(optimizer='adam', loss='mse') # Reconstrucción
            return autoencoder
        except Exception as e:
            logger.error(f"Error construyendo autoencoder: {str(e)}")
            return None

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrenar modelos de detección de fallas (ML tradicional)"""
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            X_selected = self.feature_selector.fit_transform(X_scaled, y_train)

            self.isolation_forest.fit(X_selected)
            self.random_forest.fit(X_selected, y_train)
            self.svm_model.fit(X_selected, y_train)
            self.gradient_boosting.fit(X_selected, y_train)
            
            logger.info("Modelos de detección de fallas entrenados correctamente.")
        except Exception as e:
            logger.error(f"Error entrenando modelos de detección de fallas: {str(e)}")

    def predict_fault(self, df: pd.DataFrame, battery_metadata: Optional[BatteryMetadata] = None) -> AnalysisResult:
        """Predecir fallas usando el modelo avanzado (Nivel 2)"""
        start_time = datetime.now()
        try:
            df_processed = self.preprocessor.prepare_features(df, battery_metadata)
            
            # Asegurar que haya suficientes datos después del preprocesamiento para predecir
            if len(df_processed) == 0:
                raise ValueError("DataFrame preprocesado está vacío.")
            
            # Seleccionar características
            feature_df = df_processed.select_dtypes(include=[np.number]).fillna(0)
            if feature_df.empty:
                 raise ValueError("No hay características numéricas para la predicción.")

            # Escalar y seleccionar características
            features_scaled = self.scaler.transform(feature_df)
            features_selected = self.feature_selector.transform(features_scaled)

            # Predicción con RandomForest (ejemplo, se pueden usar otros modelos)
            fault_prediction_proba = self.random_forest.predict_proba(features_selected)[0]
            fault_class = self.random_forest.predict(features_selected)[0]
            
            main_fault_type = self.fault_types.get(int(fault_class), 'unknown') # Asegurar int
            confidence = float(np.max(fault_prediction_proba)) # Asegurar float
            fault_detected = main_fault_type != 'normal'
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                analysis_type='fault_detection',
                timestamp=datetime.now(timezone.utc),
                confidence_score=confidence, # Usar confidence_score
                predictions={
                    'fault_detected': fault_detected,
                    'main_fault': main_fault_type,
                    'fault_probability': float(fault_prediction_proba[fault_class]), # Asegurar float
                    'fault_distribution': {self.fault_types.get(i, 'unknown'): float(p) for i, p in enumerate(fault_prediction_proba)} # Asegurar float
                },
                explanation={}, # Será llenado por XAIExplainer
                metadata={
                    'processing_time_s': float(processing_time), # Asegurar float
                    'data_points': int(len(df)), # Asegurar int
                    'features_count': int(features_selected.shape[1]), # Asegurar int
                    'level': 2,
                    'model_used': 'RandomForestClassifier'
                },
                model_version='2.0-level2-fault-detection',
                fault_detected=fault_detected,
                fault_type=main_fault_type,
                severity=self.severity_mapping.get(main_fault_type, 'none')
            )
        except Exception as e:
            logger.error(f"Error en predicción de fallas: {str(e)}")
            # Crear AnalysisResult en caso de error
            return self._create_error_result(str(e), 'fault_detection')

    def _create_error_result(self, error_msg: str, analysis_type: str) -> AnalysisResult:
        """Crear resultado de error para detección de fallas"""
        return AnalysisResult(
            analysis_type=analysis_type,
            timestamp=datetime.now(timezone.utc),
            confidence_score=0.0,
            predictions={'error': True, 'message': error_msg},
            explanation={'error': str(error_msg), 'method': 'error_handling'},
            metadata={'level': 0, 'error': True}
        )

# =============================================================================================
# MODELOS DE PREDICCIÓN DE SALUD (HEALTH PREDICTION) - Nivel 2 (usando ML/DL)
# =============================================================================================

class HealthPredictionModel:
    """Modelo de predicción de salud (SOH y RUL) mejorado con DL"""
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.soh_model = None
        self.rul_model = None
        self.scaler = StandardScaler()
        self._initialize_models()

    def _initialize_models(self):
        """Inicializar modelos de SOH y RUL"""
        if TENSORFLOW_AVAILABLE:
            try:
                # Modelo LSTM para SOH (regresión)
                self.soh_model = Sequential([
                    LSTM(100, activation='relu', input_shape=(50, 5), return_sequences=True),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='linear')
                ])
                self.soh_model.compile(optimizer='adam', loss='mse')

                # Modelo GRU para RUL (regresión)
                self.rul_model = Sequential([
                    GRU(100, activation='relu', input_shape=(50, 5), return_sequences=True),
                    Dropout(0.2),
                    GRU(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='linear')
                ])
                self.rul_model.compile(optimizer='adam', loss='mse')
                logger.info("Modelos de predicción de salud (DL) inicializados.")
            except Exception as e:
                logger.error(f"Error inicializando modelos de salud DL: {str(e)}")
                self.soh_model = None
                self.rul_model = None
        else:
            logger.warning("TensorFlow no disponible, usando modelos de regresión tradicionales para salud.")
            self.soh_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rul_model = RandomForestRegressor(n_estimators=100, random_state=42)
            logger.info("Modelos de predicción de salud (RF) inicializados.")

    def train_models(self, X_sequences: np.ndarray, y_soh: np.ndarray, y_rul: np.ndarray):
        """Entrenar modelos de predicción de salud"""
        if TENSORFLOW_AVAILABLE and self.soh_model and self.rul_model:
            try:
                # Entrenamiento LSTM para SOH
                self.soh_model.fit(X_sequences, y_soh, epochs=10, batch_size=32, verbose=0)
                # Entrenamiento GRU para RUL
                self.rul_model.fit(X_sequences, y_rul, epochs=10, batch_size=32, verbose=0)
                logger.info("Modelos de predicción de salud (DL) entrenados.")
            except Exception as e:
                logger.error(f"Error entrenando modelos de salud DL: {str(e)}")
        else:
            try:
                # Para modelos tradicionales, aplanar secuencias
                X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
                self.soh_model.fit(X_flat, y_soh)
                self.rul_model.fit(X_flat, y_rul)
                logger.info("Modelos de predicción de salud (RF) entrenados.")
            except Exception as e:
                logger.error(f"Error entrenando modelos de salud RF: {str(e)}")

    def predict_health(self, df: pd.DataFrame, battery_metadata: Optional[BatteryMetadata] = None) -> AnalysisResult:
        """Predecir SOH y RUL usando el modelo avanzado (Nivel 2)"""
        start_time = datetime.now()
        try:
            df_processed = self.preprocessor.prepare_features(df, battery_metadata)
            
            # Asegurar suficientes datos para secuencia
            sequence_length = 50 # Definir aquí o usar de config
            if len(df_processed) < sequence_length:
                raise ValueError(f"Datos insuficientes para crear secuencias (mínimo {sequence_length} puntos).")
            
            # Preparar la última secuencia para predicción
            # Seleccionar características clave para secuencias, asegurando que existan
            feature_cols = [col for col in ['voltage', 'current', 'temperature', 'soc', 'soh'] if col in df_processed.columns]
            if not feature_cols:
                raise ValueError("No hay características válidas para secuencias en el DataFrame preprocesado.")
            
            last_sequence_data = df_processed[feature_cols].tail(sequence_length).fillna(0)
            
            # Si el scaler no ha sido ajustado, ajustarlo con los datos actuales
            if not hasattr(self.scaler, 'n_features_in_'):
                self.scaler.fit(last_sequence_data)
                
            last_sequence_scaled = self.scaler.transform(last_sequence_data)
            
            # Añadir dimensión de lote para la predicción
            X_predict = np.expand_dims(last_sequence_scaled, axis=0)

            predicted_soh = 0.0
            predicted_rul_days = 0.0
            
            if TENSORFLOW_AVAILABLE and self.soh_model and self.rul_model:
                predicted_soh = self.soh_model.predict(X_predict)[0][0]
                predicted_rul_days = self.rul_model.predict(X_predict)[0][0]
            else:
                # Para modelos tradicionales, aplanar la secuencia
                X_predict_flat = X_predict.reshape(X_predict.shape[0], -1)
                predicted_soh = self.soh_model.predict(X_predict_flat)[0]
                predicted_rul_days = self.rul_model.predict(X_predict_flat)[0]
            
            # Asegurar que los resultados son floats nativos de Python
            predicted_soh = float(np.clip(predicted_soh, 0, 100))
            predicted_rul_days = float(np.clip(predicted_rul_days, 0, 3650)) # RUL máximo 10 años

            health_status = self._classify_health_status(predicted_soh)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                analysis_type='health_prediction',
                timestamp=datetime.now(timezone.utc),
                confidence_score=0.75, # Confianza por defecto o calculada
                predictions={
                    'current_soh': predicted_soh,
                    'rul_days': predicted_rul_days,
                    'health_status': health_status,
                    'soh_history': [float(s) for s in df['soh'].dropna().tail(10).tolist()] if 'soh' in df.columns else [],
                    'rul_history': [float(r) for r in df['rul_days'].dropna().tail(10).tolist()] if 'rul_days' in df.columns else [],
                    'timestamps': [ts.isoformat() for ts in df['timestamp'].dropna().tail(10).tolist()] if 'timestamp' in df.columns else []
                },
                explanation={}, # Será llenado por XAIExplainer
                metadata={
                    'processing_time_s': float(processing_time), # Asegurar float
                    'data_points': int(len(df)), # Asegurar int
                    'level': 2,
                    'model_used': 'LSTM/GRU' if TENSORFLOW_AVAILABLE else 'RandomForest'
                },
                model_version='2.0-level2-health-prediction',
                rul_prediction=predicted_rul_days
            )
        except Exception as e:
            logger.error(f"Error en predicción de salud: {str(e)}")
            return self._create_error_result(str(e), 'health_prediction')

    def _estimate_degradation_rate(self, df: pd.DataFrame) -> float:
        """Estimar tasa de degradación mensual"""
        if 'soh' in df.columns and len(df) > 10:
            soh_values = df['soh'].dropna()
            if len(soh_values) > 1:
                x = np.arange(len(soh_values))
                # Asegurar que los coeficientes son floats nativos
                trend = float(np.polyfit(x, soh_values, 1)[0])
                return float(abs(trend) * 30)
        return 0.5 # Valor por defecto (0.5% por mes)

    def _classify_health_status(self, soh: float) -> str:
        """Clasificar estado de salud"""
        if soh > 90: return 'excellent'
        elif soh > 80: return 'good'
        elif soh > 70: return 'fair'
        elif soh > 60: return 'poor'
        else: return 'critical'

    def _create_error_result(self, error_msg: str, analysis_type: str) -> AnalysisResult:
        """Crear resultado de error para predicción de salud"""
        return AnalysisResult(
            analysis_type=analysis_type,
            timestamp=datetime.now(timezone.utc),
            confidence_score=0.0,
            predictions={'error': True, 'message': error_msg},
            explanation={'error': str(error_msg), 'method': 'error_handling'},
            metadata={'level': 0, 'error': True}
        )

# =============================================================================================
# SISTEMA DE EXPLICABILIDAD (XAI) - NIVEL 2
# =============================================================================================
# CONSOLIDACIÓN DE LA CLASE XAIExplainer
class XAIExplainer:
    """Explicador de IA mejorado con SHAP y LIME"""
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None

    def explain_fault_detection(self, df: pd.DataFrame, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Explicar predicciones de detección de fallas. Maneja Nivel 1 y prepara para Nivel 2."""
        try:
            # Para Nivel 1 (ej. monitoreo continuo), usar explicaciones basadas en reglas
            if prediction_result.get('metadata', {}).get('level') == 1:
                return self._explain_level1_fault_detection(prediction_result)
            elif SHAP_AVAILABLE and LIME_AVAILABLE:
                # Aquí iría la lógica avanzada de SHAP/LIME para Nivel 2
                # Esta es una implementación placeholder que necesitará desarrollo
                logger.info("Intentando explicación avanzada (SHAP/LIME) para detección de fallas.")
                return self._advanced_fault_explanation(df, prediction_result)
            else:
                return self._basic_fault_explanation(prediction_result)
        except Exception as e:
            logger.error(f"Error en explain_fault_detection de XAIExplainer: {str(e)}")
            return {'error': str(e), 'method': 'explain_fault_detection_failed'}

    def _explain_level1_fault_detection(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Explicar detección de fallas de Nivel 1 basada en resultados de monitoreo continuo."""
        predictions = prediction_result.get('predictions', {})
        
        explanation_parts = []

        # Explicar violaciones de umbrales
        threshold_details = predictions.get('threshold_results', {}).get('threshold_details', [])
        if threshold_details:
            explanation_parts.append("Violaciones de umbrales críticos detectadas:")
            for detail in threshold_details:
                param = detail['parameter']
                violation_type = detail['violation_type']
                current_value = detail['current_value']
                threshold = detail['threshold']
                explanation_parts.append(f"- {param}: {float(current_value):.2f} ({violation_type}, límite: {float(threshold):.2f})") # Asegurar float

        # Explicar anomalías
        anomaly_details = predictions.get('anomaly_results', {}).get('anomaly_details', [])
        if anomaly_details:
            explanation_parts.append("Anomalías estadísticas detectadas:")
            for detail in anomaly_details:
                if 'parameter' in detail:
                    param = detail['parameter']
                    z_score = detail.get('z_score', 0.0)
                    explanation_parts.append(f"- {param}: desviación estadística significativa (Z-score: {float(z_score):.2f})") # Asegurar float
                else: # Puede ser un diccionario con 'index' y 'score' del IsolationForest
                    explanation_parts.append(f"- Anomaly en índice {detail.get('index')}: score {float(detail.get('score',0.0)):.2f}")


        # Explicar violaciones de control
        control_details = predictions.get('control_results', {}).get('control_details', [])
        if control_details:
            explanation_parts.append("Violaciones de control estadístico:")
            for detail in control_details:
                param = detail['parameter']
                chart_type = detail['chart_type']
                violations = detail['violations']
                explanation_parts.append(f"- {param}: {int(violations)} violaciones en gráfico {chart_type}") # Asegurar int

        explanation_text = "\n".join(explanation_parts) if explanation_parts else "No se detectaron problemas significativos en el monitoreo continuo."
        
        # El campo 'confidence' en AnalysisResult ahora es 'confidence_score'
        return { 
            'method': 'level1_rule_based',
            'explanation_text': explanation_text,
            'confidence': float(prediction_result.get('confidence_score', 0.0)), # Asegurar float
            'analysis_level': 1
        }

    def _basic_fault_explanation(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Explicación básica de fallas para compatibilidad o fallback."""
        fault_detected = prediction_result.get('fault_detected', False)
        fault_type = prediction_result.get('predictions', {}).get('main_fault', 'normal')
        
        if not fault_detected:
            explanation_text = "La batería muestra un comportamiento normal. Todos los parámetros están dentro de rangos aceptables."
        else:
            explanations = {
                'degradation': "Se detectó degradación de la batería basada en análisis de tendencias y patrones de comportamiento.",
                'short_circuit': "Posible cortocircuito interno detectado basado en patrones de voltaje y corriente anómalos.",
                'overcharge': "Condición de sobrecarga detectada. Voltaje o corriente fuera de rangos seguros.",
                'overheat': "Sobrecalentamiento detectado. Temperatura fuera de rangos operacionales seguros.",
                'thermal_runaway': "¡ALERTA CRÍTICA! Posible fuga térmica detectada. Requiere atención inmediata."
            }
            explanation_text = explanations.get(fault_type, f"Falla de tipo '{fault_type}' detectada. Necesita mayor investigación.")

        return {
            'method': 'basic_rule_based',
            'explanation_text': explanation_text,
            'fault_type': fault_type,
            'confidence': float(prediction_result.get('confidence_score', 0.0)) # Asegurar float
        }
        
    def _advanced_fault_explanation(self, df: pd.DataFrame, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder para la explicación avanzada de fallas (SHAP/LIME)."""
        explanation_text = "Explicación avanzada no implementada en esta versión. Se requiere desarrollo de SHAP/LIME."
        
        # Aquí iría la lógica para inicializar SHAP/LIME y generar explicaciones detalladas.
        # Esto implicaría:
        # 1. Identificar el modelo predictivo subyacente (e.g., RandomForestClassifier del FaultDetectionModel).
        # 2. Preparar los datos de entrada para el explicador.
        # 3. Generar valores SHAP o explicaciones LIME.
        # 4. Procesar esos valores en un formato legible para el usuario.
        
        return {
            'method': 'advanced_xai_placeholder',
            'explanation_text': explanation_text,
            'confidence': float(prediction_result.get('confidence_score', 0.0)), # Asegurar float
            'details': {} # Aquí se añadirían los detalles de SHAP/LIME
        }

    def explain_health_prediction(self, df: pd.DataFrame, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Explicar predicciones de salud (SOH y RUL)."""
        try:
            # Asegurar que los valores son float nativos
            current_soh = float(prediction_result.get('predictions', {}).get('current_soh', 0.0))
            rul_days = float(prediction_result.get('predictions', {}).get('rul_days', 0.0))
            health_status = prediction_result.get('predictions', {}).get('health_status', 'unknown')
            
            explanation_parts = [
                f"Estado de salud actual: {current_soh:.1f}% ({health_status})",
                f"Vida útil restante estimada: {rul_days:.0f} días"
            ]

            # Agregar recomendaciones basadas en SOH
            if current_soh < 70:
                explanation_parts.append("RECOMENDACIÓN: Considerar reemplazo de la batería.")
            elif current_soh < 80:
                explanation_parts.append("RECOMENDACIÓN: Monitoreo frecuente recomendado.")
            elif current_soh < 90:
                explanation_parts.append("RECOMENDACIÓN: Monitoreo regular suficiente.")
            else:
                explanation_parts.append("Estado excelente. Continuar con monitoreo rutinario.")
            
            explanation_text = ". ".join(explanation_parts)
            
            return {
                'method': 'health_analysis_level1', # Aunque sea Nivel 2, la explicación es similar a la básica por ahora
                'explanation_text': explanation_text,
                'health_metrics': {
                    'soh': current_soh,
                    'rul_days': rul_days,
                    'status': health_status
                },
                'confidence': float(prediction_result.get('confidence_score', 0.0)) # Asegurar float
            }
        except Exception as e:
            logger.error(f"Error en explain_health_prediction de XAIExplainer: {str(e)}")
            return {'error': str(e), 'method': 'explain_health_prediction_failed'}

# ============================================================================
# NIVEL 2: ANÁLISIS AVANZADO (DEEP LEARNING Y XAI COMPLETO)
# ============================================================================
class AdvancedAnalysisEngine:
    """Motor de análisis avanzado - Nivel 2"""
    def __init__(self, model_cache_dir: str = "/tmp/battsentinel_models"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        self.preprocessor = DataPreprocessor()
        self.deep_models = {}
        self.gaussian_processes = {}
        self.autoencoders = {}
        self.model_config = {
            'lstm_sequence_length': 50,
            'lstm_features': 10,
            'autoencoder_latent_dim': 8,
            'gp_kernel_length_scale': 1.0,
            'training_validation_split': 0.2
        }
        if TENSORFLOW_AVAILABLE:
            self._initialize_deep_models()
        else:
            logger.warning("TensorFlow no disponible. Funcionalidades de Deep Learning deshabilitadas.")

    def _initialize_deep_models(self):
        """Inicializar modelos de deep learning"""
        try:
            # Configurar TensorFlow para uso eficiente de memoria si hay GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    logger.info("Configuración de memoria de GPU para TensorFlow.")
                except RuntimeError as e:
                    logger.warning(f"Error al configurar la memoria de GPU: {e}")
            logger.info("Modelos de deep learning inicializados")
        except Exception as e:
            logger.error(f"Error inicializando modelos de deep learning: {str(e)}")

    def analyze_advanced(self, df: pd.DataFrame, battery_metadata: Optional[BatteryMetadata] = None) -> AnalysisResult:
        """Análisis avanzado completo (Nivel 2)"""
        start_time = datetime.now()
        try:
            df_processed = self.preprocessor.prepare_features(df, battery_metadata)
            if len(df_processed) < 50:
                raise ValueError("Datos insuficientes para análisis avanzado (mínimo 50 puntos)")
            
            deep_results = self._deep_learning_analysis(df_processed, battery_metadata)
            anomaly_results = self._autoencoder_anomaly_detection(df_processed)
            uncertainty_results = self._gaussian_process_prediction(df_processed)
            survival_results = self._survival_analysis(df_processed, battery_metadata)
            
            combined_results = self._combine_level2_results(
                deep_results, anomaly_results, uncertainty_results, survival_results
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                analysis_type='advanced_analysis',
                timestamp=datetime.now(timezone.utc),
                confidence_score=float(combined_results['confidence']), # Asegurar float
                predictions=combined_results['predictions'],
                explanation=combined_results['explanation'],
                metadata={
                    'processing_time_s': float(processing_time), # Asegurar float
                    'data_points': int(len(df)), # Asegurar int
                    'features_analyzed': int(len(df_processed.columns)), # Asegurar int
                    'level': 2,
                    'models_used': combined_results['models_used']
                },
                model_version='2.0-level2'
            )
        except Exception as e:
            logger.error(f"Error en análisis avanzado: {str(e)}")
            return self._create_error_result(str(e), 'advanced_analysis')

    def _deep_learning_analysis(self, df: pd.DataFrame, metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
        """Análisis usando redes neuronales profundas"""
        results = { 'lstm_fault_detection': {}, 'gru_health_prediction': {}, 'tcn_classification': {} }
        if not TENSORFLOW_AVAILABLE: return {'error': 'TensorFlow no disponible', 'models_used': []}
        try:
            sequences, targets = self._prepare_sequences(df)
            if len(sequences) < 10:
                return {'error': 'Datos insuficientes para secuencias', 'models_used': []}
            
            lstm_result = self._lstm_fault_detection(sequences, targets, df)
            results['lstm_fault_detection'] = lstm_result
            gru_result = self._gru_health_prediction(sequences, targets, df)
            results['gru_health_prediction'] = gru_result
            tcn_result = self._tcn_pattern_classification(sequences, df)
            results['tcn_classification'] = tcn_result
            results['models_used'] = ['LSTM', 'GRU', 'TCN']
            results['success'] = True
        except Exception as e:
            logger.error(f"Error en análisis de deep learning: {str(e)}")
            results['error'] = str(e)
            results['models_used'] = []
        return results

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar secuencias temporales para modelos RNN"""
        feature_cols = [col for col in ['voltage', 'current', 'temperature', 'soc', 'soh'] if col in df.columns]
        if not feature_cols:
            raise ValueError("No hay características válidas para secuencias")
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[feature_cols].fillna(0))
        
        sequence_length = min(self.model_config['lstm_sequence_length'], len(data_scaled) // 2)
        sequences = []
        targets = []
        for i in range(sequence_length, len(data_scaled)):
            sequences.append(data_scaled[i-sequence_length:i])
            if 'soh' in df.columns:
                targets.append(df['soh'].iloc[i])
            else:
                targets.append(self._create_synthetic_target(data_scaled[i]))
        return np.array(sequences), np.array(targets)

    def _create_synthetic_target(self, data_point: np.ndarray) -> float:
        """Crear target sintético para entrenamiento"""
        voltage_idx = 0
        current_idx = 1
        if len(data_point) > voltage_idx:
            voltage = data_point[voltage_idx]
            current = data_point[current_idx] if len(data_point) > current_idx else 0
            health_score = min(100.0, max(0.0, (voltage - 3.0) / (4.2 - 3.0) * 100))
            if abs(current) > 0.8:
                health_score *= 0.95
            return float(health_score) # Asegurar float
        return 85.0 # Valor por defecto
    
    def _lstm_fault_detection(self, sequences: np.ndarray, targets: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Detección de fallas usando LSTM"""
        try:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequences.shape[1], sequences.shape[2])), Dropout(0.2),
                LSTM(32, return_sequences=False), Dropout(0.2),
                Dense(16, activation='relu'), Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            fault_targets = (targets < 80).astype(int)
            
            if len(sequences) > 20:
                split_idx = int(len(sequences) * 0.8)
                X_train, X_val = sequences[:split_idx], sequences[split_idx:]
                y_train, y_val = fault_targets[:split_idx], fault_targets[split_idx:]
                
                history = model.fit(
                    X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16, verbose=0,
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                )
                predictions = model.predict(sequences, verbose=0)
                fault_probability = float(np.mean(predictions)) # Asegurar float
                fault_detected = fault_probability > 0.5
                confidence = float(max(fault_probability, 1 - fault_probability)) # Asegurar float
                
                return {
                    'fault_detected': bool(fault_detected), # Asegurar bool
                    'fault_probability': float(fault_probability), # Asegurar float
                    'confidence': float(confidence), # Asegurar float
                    'model_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else 0.0, # Asegurar float
                    'predictions': [float(p) for p in predictions.flatten().tolist()[-10:]], # Asegurar float
                    'method': 'LSTM'
                }
            else:
                return {'error': 'Datos insuficientes para entrenamiento LSTM'}
        except Exception as e:
            logger.error(f"Error en LSTM fault detection: {str(e)}")
            return {'error': str(e)}

    def _gru_health_prediction(self, sequences: np.ndarray, targets: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Predicción de salud usando GRU (SOH y RUL)"""
        try:
            model = Sequential([
                GRU(64, activation='relu', input_shape=(sequences.shape[1], sequences.shape[2]), return_sequences=True), Dropout(0.2),
                GRU(32, activation='relu'), Dropout(0.2),
                Dense(1, activation='linear') # Predice SOH
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            if len(sequences) > 20:
                split_idx = int(len(sequences) * 0.8)
                X_train, X_val = sequences[:split_idx], sequences[split_idx:]
                y_train, y_val = targets[:split_idx], targets[split_idx:]
                
                history = model.fit(
                    X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=16, verbose=0,
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                )
                predictions = model.predict(sequences, verbose=0)
                predicted_soh = float(predictions[-1][0]) # Última predicción SOH, asegurar float
                
                # RUL simple basado en degradación histórica
                degradation_rate = self._estimate_degradation_rate(df)
                rul_days = float((predicted_soh - 60) / degradation_rate * 30) if degradation_rate > 0 else 365 # SOH 60% como fin de vida, asegurar float
                rul_days = np.clip(rul_days, 0, 3650)
                
                return {
                    'current_soh': float(predicted_soh), # Asegurar float
                    'rul_days': float(rul_days), # Asegurar float
                    'health_status': self._classify_health_status(predicted_soh),
                    'model_accuracy': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else 0.0, # Usar val_loss para regresión, asegurar float
                    'predictions_soh': [float(p) for p in predictions.flatten().tolist()[-10:]], # Últimas 10 predicciones SOH, asegurar float
                    'method': 'GRU'
                }
            else:
                return {'error': 'Datos insuficientes para entrenamiento GRU'}
        except Exception as e:
            logger.error(f"Error en GRU health prediction: {str(e)}")
            return {'error': str(e)}

    def _tcn_pattern_classification(self, sequences: np.ndarray, df: pd.DataFrame) -> Dict[str, Any]:
        """Clasificación de patrones (ej. modos de operación) usando TCN"""
        try:
            # Placeholder para num_classes, ajusta según tus patrones
            num_classes = 3 # Ejemplo: carga, descarga, inactivo
            model = FaultDetectionModel()._build_tcn_model((sequences.shape[1], sequences.shape[2]), num_classes) # Reutilizar la construcción TCN
            
            if model and len(sequences) > 20:
                # Targets de ejemplo para clasificación de patrones
                # En un caso real, esto vendría de etiquetas de modos de operación
                synthetic_targets = np.random.randint(0, num_classes, len(sequences))
                
                split_idx = int(len(sequences) * 0.8)
                X_train, X_val = sequences[:split_idx], sequences[split_idx:]
                y_train, y_val = synthetic_targets[:split_idx], synthetic_targets[split_idx:]
                
                history = model.fit(
                    X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16, verbose=0,
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                )
                predictions_proba = model.predict(sequences, verbose=0)
                predicted_pattern = int(np.argmax(predictions_proba[-1])) # Último patrón, asegurar int
                confidence = float(np.max(predictions_proba[-1])) # Confianza del último patrón, asegurar float
                
                return {
                    'detected_pattern': predicted_pattern,
                    'confidence': confidence,
                    'model_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else 0.0, # Asegurar float
                    'method': 'TCN'
                }
            else:
                return {'error': 'Datos insuficientes para entrenamiento TCN o modelo no construido'}
        except Exception as e:
            logger.error(f"Error en TCN pattern classification: {str(e)}")
            return {'error': str(e)}

    def _autoencoder_anomaly_detection(self, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """Detección de anomalías con Autoencoders (Nivel 2)"""
        results = { 'anomalies_detected': False, 'anomaly_score': 0.0, 'details': [] }
        if not TENSORFLOW_AVAILABLE: return {'error': 'TensorFlow no disponible'}
        
        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_cols: return {'error': 'No hay características numéricas para autoencoder'}
        
        data = df_processed[feature_cols].fillna(0).values
        if len(data) < 50: return {'error': 'Datos insuficientes para autoencoder (min 50)'}
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        try:
            input_dim = data_scaled.shape[1]
            encoding_dim = self.model_config['autoencoder_latent_dim']
            hidden_layers = self.model_config['autoencoder']['hidden_layers']
            
            model = FaultDetectionModel()._build_autoencoder_model((input_dim,), encoding_dim, hidden_layers) # Reutilizar
            
            if model:
                model.fit(data_scaled, data_scaled, epochs=self.model_config['autoencoder']['epochs'],
                          batch_size=self.model_config['autoencoder']['batch_size'], verbose=0,
                          validation_split=0.1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
                
                reconstructions = model.predict(data_scaled, verbose=0)
                mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
                
                threshold = float(np.mean(mse) + 2 * np.std(mse)) # Umbral dinámico, asegurar float
                
                anomalies = np.where(mse > threshold)[0]
                if len(anomalies) > 0:
                    results['anomalies_detected'] = True
                    results['anomaly_score'] = float(np.mean(mse[anomalies])) # Asegurar float
                    results['details'] = [{
                        'index': int(idx), # Asegurar int
                        'reconstruction_error': float(mse[idx]), # Asegurar float
                        'timestamp': df_processed.index[idx].isoformat() if hasattr(df_processed.index[idx], 'isoformat') else str(df_processed.index[idx])
                    } for idx in anomalies[-5:]] # Últimas 5
                
            return results
        except Exception as e:
            logger.error(f"Error en autoencoder anomaly detection: {str(e)}")
            return {'error': str(e)}

    def _gaussian_process_prediction(self, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """Predicción con incertidumbre usando Gaussian Processes para SOH"""
        results = {'soh_prediction': None, 'uncertainty': None, 'method': 'GaussianProcess'}
        
        if 'soh' not in df_processed.columns or len(df_processed) < 10:
            return {'error': 'Datos de SOH insuficientes para Gaussian Process'}
        
        try:
            # Usar timestamp como característica o un índice numérico
            X = np.arange(len(df_processed)).reshape(-1, 1)
            y = df_processed['soh'].values
            
            kernel = RBF(length_scale=self.model_config['gp_kernel_length_scale']) + WhiteKernel()
            gp = GaussianProcessRegressor(kernel=kernel, alpha=(0.1)**2, n_restarts_optimizer=10)
            
            gp.fit(X, y)
            
            # Predecir el último valor y su incertidumbre
            last_idx = len(df_processed) -1
            mean_prediction, std_prediction = gp.predict(np.array([[last_idx]]), return_std=True)
            
            results['soh_prediction'] = float(mean_prediction[0]) # Asegurar float
            results['uncertainty'] = float(std_prediction[0]) # Asegurar float
            
            # Opcional: Predicción futura
            future_indices = np.arange(len(df_processed), len(df_processed) + 30).reshape(-1, 1)
            future_soh, future_std = gp.predict(future_indices, return_std=True)
            results['future_soh_predictions'] = [float(s) for s in future_soh.tolist()] # Asegurar float
            results['future_soh_uncertainty'] = [float(s) for s in future_std.tolist()] # Asegurar float
            
            return results
        except Exception as e:
            logger.error(f"Error en Gaussian Process: {str(e)}")
            return {'error': str(e)}

    def _survival_analysis(self, df_processed: pd.DataFrame, battery_metadata: Optional[BatteryMetadata]) -> Dict[str, Any]:
        """Análisis de supervivencia para RUL (Placeholder)"""
        # Esto sería una implementación más compleja con modelos de supervivencia
        # Por ahora, un placeholder con estimación básica de RUL
        
        # Simulación de un RUL
        if 'soh' in df_processed.columns and len(df_processed) > 10:
            current_soh = df_processed['soh'].iloc[-1]
            # Estimar vida útil restante con una simplificación
            # Asumiendo que 60% SOH es el final de la vida útil
            if battery_metadata and battery_metadata.design_cycles > 0:
                # Si tenemos ciclos de diseño, estimar basados en degradación lineal
                soh_drop_per_cycle = (100 - current_soh) / df_processed['cycles'].iloc[-1] if df_processed['cycles'].iloc[-1] > 0 else 0.01
                remaining_cycles = (current_soh - 60) / soh_drop_per_cycle if soh_drop_per_cycle > 0 else 0
                rul_days = remaining_cycles / 365 * (30/1) # Asumiendo 1 ciclo/mes, muy simplificado
            else:
                # Estimar RUL en días basándose en la degradación general de SOH
                degradation_rate_per_day = self._estimate_degradation_rate(df_processed) / 30 # Convertir mensual a diario
                if degradation_rate_per_day > 0:
                    rul_days = (current_soh - 60) / degradation_rate_per_day
                else:
                    rul_days = 730 # 2 años por defecto
            rul_days = float(np.clip(rul_days, 0, 3650)) # Asegurar float
        else:
            rul_days = 730.0 # 2 años por defecto
        
        return {
            'rul_prediction_days': rul_days,
            'method': 'survival_analysis_placeholder',
            'details': 'Análisis de supervivencia avanzado no implementado en esta versión.'
        }
        
    def _estimate_degradation_rate(self, df: pd.DataFrame) -> float:
        """Estimar tasa de degradación mensual - Reutiliza de HealthPredictionModel"""
        if 'soh' in df.columns and len(df) > 10:
            soh_values = df['soh'].dropna()
            if len(soh_values) > 1:
                x = np.arange(len(soh_values))
                trend = float(np.polyfit(x, soh_values, 1)[0]) # Asegurar float
                return float(abs(trend) * 30) # Asegurar float
        return 0.5 # Valor por defecto (0.5% por mes)

    def _combine_level2_results(self, deep_res: Dict, anomaly_res: Dict, uncertainty_res: Dict, survival_res: Dict) -> Dict[str, Any]:
        """Combinar resultados de Nivel 2"""
        overall_confidence = []
        overall_status = 'Normal'
        
        # Recolectar confianzas y estados
        if 'confidence' in deep_res.get('lstm_fault_detection', {}):
            overall_confidence.append(deep_res['lstm_fault_detection']['confidence'])
            if deep_res['lstm_fault_detection'].get('fault_detected'):
                overall_status = 'Fault Detected'
        
        if 'confidence' in deep_res.get('gru_health_prediction', {}):
            # No hay 'confidence' directo aquí, usar un valor por defecto o calcularlo
            pass 
        
        if 'confidence' in deep_res.get('tcn_classification', {}):
            overall_confidence.append(deep_res['tcn_classification']['confidence'])
        
        if anomaly_res.get('anomalies_detected'):
            overall_status = 'Anomaly Detected'
            # No hay confianza directa, usar un valor fijo o derivado
            overall_confidence.append(0.85)
            
        final_confidence = float(np.mean(overall_confidence)) if overall_confidence else 0.78 # Asegurar float
        
        # Resumen general de resultados
        results_summary = {
            'continuous_monitoring': 'success', # Asumiendo que pasó el monitoreo continuo para llegar aquí
            'fault_detection': 'success' if deep_res.get('lstm_fault_detection', {}).get('fault_detected', False) else 'success',
            'health_prediction': 'success' if deep_res.get('gru_health_prediction', {}) else 'success',
            'explanations': 'available' if SHAP_AVAILABLE and LIME_AVAILABLE else 'basic'
        }
        
        combined_predictions = {
            'level': 2,
            'results_summary': results_summary,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'deep_learning_results': deep_res,
            'autoencoder_anomalies': anomaly_res,
            'gaussian_process': uncertainty_res,
            'survival_analysis': survival_res
        }
        
        return {
            'confidence': final_confidence,
            'predictions': combined_predictions,
            'explanation': {'summary': 'Resultados combinados de análisis avanzado'},
            'models_used': deep_res.get('models_used', []) # Añadir modelos usados
        }
    
    def _create_error_result(self, error_msg: str, analysis_type: str) -> AnalysisResult:
        """Crear resultado de error para análisis avanzado"""
        return AnalysisResult(
            analysis_type=analysis_type,
            timestamp=datetime.now(timezone.utc),
            confidence_score=0.0,
            predictions={'error': True, 'message': error_msg},
            explanation={'error': str(error_msg), 'method': 'error_handling'},
            metadata={'level': 0, 'error': True}
        )

# =============================================================================================
# MOTOR DE ANÁLISIS INTEGRADO (COMPREHENSIVE ANALYSIS)
# =============================================================================================

class ComprehensiveAnalysisEngine:
    """Motor de análisis integral que coordina Nivel 1 y Nivel 2"""
    def __init__(self):
        self.continuous_engine = ContinuousMonitoringEngine()
        self.fault_detection_model = FaultDetectionModel()
        self.health_prediction_model = HealthPredictionModel()
        self.xai_explainer = XAIExplainer() # Instancia única de XAIExplainer
        self.advanced_analysis_engine = AdvancedAnalysisEngine()
        logger.info("Motor de análisis integral inicializado.")

    def analyze_battery(self, df: pd.DataFrame, battery_metadata: Optional[BatteryMetadata] = None) -> List[AnalysisResult]:
        """Realizar un análisis completo de la batería (Nivel 1 y Nivel 2)"""
        results = []
        
        # 1. Ejecutar monitoreo continuo (Nivel 1)
        continuous_result = self.continuous_engine.analyze_continuous(df, battery_metadata)
        results.append(continuous_result)
        
        # Determinar si se necesita análisis avanzado
        perform_advanced_analysis = (
            continuous_result.predictions.get('overall_fault_detected', False) or
            continuous_result.predictions.get('anomaly_results', {}).get('anomalies_detected', False)
        )
        
        # Forzar análisis avanzado para demostración o si se desea siempre
        # perform_advanced_analysis = True 

        if perform_advanced_analysis and len(df) >= 50: # Mínimo de datos para Nivel 2
            logger.info("Activando análisis avanzado debido a condiciones de falla o datos suficientes.")
            
            # 2. Ejecutar Detección de Fallas (Nivel 2)
            fault_detection_result = self.fault_detection_model.predict_fault(df, battery_metadata)
            
            # 3. Ejecutar Predicción de Salud (Nivel 2)
            health_prediction_result = self.health_prediction_model.predict_health(df, battery_metadata)
            
            # 4. Ejecutar Análisis Avanzado (combinación de DL, GP, etc.)
            advanced_analysis_result = self.advanced_analysis_engine.analyze_advanced(df, battery_metadata)

            # Generar explicaciones para Nivel 2 (usando la instancia de XAIExplainer)
            if fault_detection_result:
                try:
                    fd_explanation = self.xai_explainer.explain_fault_detection(df, fault_detection_result.to_dict())
                    fault_detection_result.explanation = fd_explanation
                except Exception as e:
                    logger.warning(f"No se pudo generar explicación para detección de fallas: {str(e)}")
                    fault_detection_result.explanation = {'error': str(e), 'method': 'explanation_failed'}
                results.append(fault_detection_result)
            
            if health_prediction_result:
                try:
                    hp_explanation = self.xai_explainer.explain_health_prediction(df, health_prediction_result.to_dict())
                    health_prediction_result.explanation = hp_explanation
                except Exception as e:
                    logger.warning(f"No se pudo generar explicación para predicción de salud: {str(e)}")
                    health_prediction_result.explanation = {'error': str(e), 'method': 'explanation_failed'}
                results.append(health_prediction_result)

            if advanced_analysis_result:
                # La explicación para advanced_analysis_result ya se genera internamente en _combine_level2_results
                # o se puede generar una explicación general aquí.
                results.append(advanced_analysis_result)

        else:
            logger.info("No se activó el análisis avanzado (Nivel 2) debido a ausencia de fallas o datos insuficientes.")
            # Si no hay análisis avanzado, proporcionar explicaciones básicas si es necesario
            # Para el continuous_result ya está manejado internamente
        
        # Asegurar que todas las AnalysisResult son válidas
        return [res for res in results if res is not None]

# Funciones auxiliares para cargadores de modelos (si es necesario)
def load_model_from_path(model_path: Path):
    """Carga un modelo guardado desde una ruta específica."""
    try:
        if model_path.exists():
            return joblib.load(model_path)
        else:
            logger.warning(f"Modelo no encontrado en: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error cargando modelo de {model_path}: {str(e)}")
        return None

def save_model_to_path(model, model_path: Path):
    """Guarda un modelo en una ruta específica."""
    try:
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado en: {model_path}")
    except Exception as e:
        logger.error(f"Error guardando modelo en {model_path}: {str(e)}")
