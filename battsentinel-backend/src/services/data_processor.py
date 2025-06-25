import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class DataProcessor:
    """Procesador de archivos de datos de batería"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'txt', 'xlsx', 'xls']
        
    def process_file(self, file_path, battery_id):
        """Procesar archivo y extraer datos de batería"""
        try:
            # Determinar tipo de archivo
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension == 'txt':
                # Intentar diferentes delimitadores
                try:
                    df = pd.read_csv(file_path, delimiter='\t')
                except:
                    try:
                        df = pd.read_csv(file_path, delimiter=';')
                    except:
                        df = pd.read_csv(file_path, delimiter=',')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Normalizar nombres de columnas
            df.columns = df.columns.str.lower().str.strip()
            column_mapping = self._get_column_mapping()
            df = df.rename(columns=column_mapping)
            
            # Validar y limpiar datos
            df = self._clean_data(df)
            
            # Convertir a lista de diccionarios para inserción en BD
            data_points = []
            for _, row in df.iterrows():
                data_point = {
                    'battery_id': battery_id,
                    'timestamp': self._parse_timestamp(row.get('timestamp')),
                    'voltage': self._safe_float(row.get('voltage')),
                    'current': self._safe_float(row.get('current')),
                    'power': self._safe_float(row.get('power')),
                    'soc': self._safe_float(row.get('soc')),
                    'soh': self._safe_float(row.get('soh')),
                    'capacity': self._safe_float(row.get('capacity')),
                    'cycles': self._safe_int(row.get('cycles')),
                    'temperature': self._safe_float(row.get('temperature')),
                    'internal_resistance': self._safe_float(row.get('internal_resistance')),
                    'data_source': 'manual'
                }
                
                # Solo agregar si tiene al menos un valor válido
                if any(v is not None for k, v in data_point.items() 
                      if k not in ['battery_id', 'timestamp', 'data_source']):
                    data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def _get_column_mapping(self):
        """Mapeo de nombres de columnas comunes"""
        return {
            # Tiempo
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp',
            'tiempo': 'timestamp',
            'fecha': 'timestamp',
            
            # Voltaje
            'voltage': 'voltage',
            'volt': 'voltage',
            'v': 'voltage',
            'voltaje': 'voltage',
            'tension': 'voltage',
            
            # Corriente
            'current': 'current',
            'curr': 'current',
            'i': 'current',
            'corriente': 'current',
            'intensidad': 'current',
            
            # Potencia
            'power': 'power',
            'pow': 'power',
            'p': 'power',
            'potencia': 'power',
            
            # Estado de carga
            'soc': 'soc',
            'state_of_charge': 'soc',
            'charge': 'soc',
            'estado_carga': 'soc',
            
            # Estado de salud
            'soh': 'soh',
            'state_of_health': 'soh',
            'health': 'soh',
            'estado_salud': 'soh',
            
            # Capacidad
            'capacity': 'capacity',
            'cap': 'capacity',
            'capacidad': 'capacity',
            
            # Ciclos
            'cycles': 'cycles',
            'cycle': 'cycles',
            'ciclos': 'cycles',
            'ciclo': 'cycles',
            
            # Temperatura
            'temperature': 'temperature',
            'temp': 'temperature',
            't': 'temperature',
            'temperatura': 'temperature',
            
            # Resistencia interna
            'internal_resistance': 'internal_resistance',
            'resistance': 'internal_resistance',
            'r': 'internal_resistance',
            'resistencia': 'internal_resistance',
            'resistencia_interna': 'internal_resistance'
        }
    
    def _clean_data(self, df):
        """Limpiar y validar datos"""
        # Remover filas completamente vacías
        df = df.dropna(how='all')
        
        # Validar rangos de valores
        if 'voltage' in df.columns:
            df.loc[df['voltage'] < 0, 'voltage'] = np.nan
            df.loc[df['voltage'] > 100, 'voltage'] = np.nan  # Voltaje máximo razonable
        
        if 'current' in df.columns:
            df.loc[abs(df['current']) > 1000, 'current'] = np.nan  # Corriente máxima razonable
        
        if 'soc' in df.columns:
            df.loc[df['soc'] < 0, 'soc'] = np.nan
            df.loc[df['soc'] > 100, 'soc'] = np.nan
        
        if 'soh' in df.columns:
            df.loc[df['soh'] < 0, 'soh'] = np.nan
            df.loc[df['soh'] > 100, 'soh'] = np.nan
        
        if 'temperature' in df.columns:
            df.loc[df['temperature'] < -50, 'temperature'] = np.nan  # Temperatura mínima
            df.loc[df['temperature'] > 200, 'temperature'] = np.nan  # Temperatura máxima
        
        return df
    
    def _parse_timestamp(self, timestamp_value):
        """Parsear timestamp de diferentes formatos"""
        if pd.isna(timestamp_value):
            return datetime.now()
        
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        
        if isinstance(timestamp_value, str):
            # Intentar diferentes formatos
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%d/%m/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_value, fmt)
                except ValueError:
                    continue
        
        # Si no se puede parsear, usar timestamp actual
        return datetime.now()
    
    def _safe_float(self, value):
        """Convertir valor a float de forma segura"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value):
        """Convertir valor a int de forma segura"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def validate_data_format(self, file_path):
        """Validar formato de archivo antes de procesarlo"""
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension not in self.supported_formats:
                return False, f"Unsupported file format: {file_extension}"
            
            # Intentar leer las primeras filas
            if file_extension == 'csv':
                df = pd.read_csv(file_path, nrows=5)
            elif file_extension == 'txt':
                df = pd.read_csv(file_path, nrows=5, delimiter='\t')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, nrows=5)
            
            if df.empty:
                return False, "File is empty"
            
            return True, "Valid format"
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def get_file_preview(self, file_path, rows=10):
        """Obtener vista previa del archivo"""
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file_path, nrows=rows)
            elif file_extension == 'txt':
                df = pd.read_csv(file_path, nrows=rows, delimiter='\t')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, nrows=rows)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Información del archivo
            info = {
                'columns': list(df.columns),
                'rows': len(df),
                'sample_data': df.to_dict('records'),
                'data_types': df.dtypes.to_dict()
            }
            
            return info
            
        except Exception as e:
            raise Exception(f"Error getting file preview: {str(e)}")
