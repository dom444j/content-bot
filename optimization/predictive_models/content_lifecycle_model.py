"""
Modelo predictivo para analizar y optimizar el ciclo de vida del contenido multimedia.
Este modelo identifica patrones de rendimiento a lo largo del tiempo y predice la longevidad
y potencial de monetización del contenido.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from .base_model import BaseModel

class ContentLifecycleModel(BaseModel):
    """
    Modelo para analizar y predecir el ciclo de vida completo del contenido multimedia.
    
    Capacidades:
    - Identificación de patrones de rendimiento temporal
    - Predicción de longevidad y vida útil del contenido
    - Análisis de factores que afectan la durabilidad
    - Recomendaciones para optimizar el ciclo de vida
    - Identificación de contenido con potencial de resurgimiento
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el modelo de ciclo de vida de contenido.
        
        Args:
            model_path: Ruta opcional al modelo pre-entrenado
        """
        super().__init__(model_name="content_lifecycle_model", model_path=model_path)
        self.longevity_model = None
        self.peak_performance_model = None
        self.decay_rate_model = None
        self.revival_potential_model = None
        self.scaler = StandardScaler()
        
        # Configuración de características
        self.content_features = [
            'content_type', 'duration', 'topic', 'quality_score', 
            'engagement_rate', 'initial_views', 'peak_views',
            'time_to_peak', 'hashtags_count', 'mentions_count',
            'cta_timing', 'cta_type', 'thumbnail_quality',
            'title_quality', 'description_quality', 'audio_quality',
            'visual_quality', 'trending_relevance', 'seasonality',
            'platform', 'posting_time', 'day_of_week',
            'creator_followers', 'creator_engagement_rate'
        ]
        
        self.categorical_features = [
            'content_type', 'topic', 'cta_type', 'platform', 
            'day_of_week', 'seasonality'
        ]
        
        self.target_features = [
            'content_lifespan', 'peak_performance', 'decay_rate', 
            'revival_potential', 'total_engagement'
        ]
        
        # Definir etapas del ciclo de vida
        self.lifecycle_stages = [
            'launch', 'growth', 'peak', 'decay', 'steady', 'revival', 'archive'
        ]
        
        # Inicializar modelos si existe la ruta
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos de contenido para el entrenamiento o predicción.
        
        Args:
            data: DataFrame con datos de contenido y métricas temporales
            
        Returns:
            DataFrame preprocesado
        """
        # Verificar columnas requeridas
        required_columns = ['content_id', 'timestamp'] + self.content_features
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Faltan columnas en los datos: {missing_columns}")
            # Agregar columnas faltantes con valores predeterminados
            for col in missing_columns:
                if col == 'content_id':
                    data[col] = [f"content_{i}" for i in range(len(data))]
                elif col == 'timestamp':
                    data[col] = datetime.now()
                elif col in self.categorical_features:
                    data[col] = 'unknown'
                else:
                    data[col] = 0.0
        
        # Convertir timestamp a datetime si es necesario
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Procesar características categóricas
        processed_data = data.copy()
        
        # One-hot encoding para características categóricas
        for col in self.categorical_features:
            if col in processed_data.columns:
                dummies = pd.get_dummies(processed_data[col], prefix=col)
                processed_data = pd.concat([processed_data, dummies], axis=1)
        
        # Calcular características derivadas
        
        # Calcular tiempo desde publicación si no existe
        if 'days_since_publish' not in processed_data.columns:
            if 'publish_date' in processed_data.columns:
                processed_data['days_since_publish'] = (
                    processed_data['timestamp'] - pd.to_datetime(processed_data['publish_date'])
                ).dt.days
            else:
                # Usar la fecha más antigua como referencia
                min_date = processed_data['timestamp'].min()
                processed_data['days_since_publish'] = (
                    processed_data['timestamp'] - min_date
                ).dt.days
        
        # Calcular ratio de retención si no existe
        if 'retention_ratio' not in processed_data.columns and 'initial_views' in processed_data.columns and 'current_views' in processed_data.columns:
            processed_data['retention_ratio'] = processed_data.apply(
                lambda x: x['current_views'] / x['initial_views'] if x['initial_views'] > 0 else 0,
                axis=1
            )
        
        # Calcular velocidad de decaimiento si no existe
        if 'decay_velocity' not in processed_data.columns and 'peak_views' in processed_data.columns and 'current_views' in processed_data.columns and 'time_to_peak' in processed_data.columns and 'days_since_publish' in processed_data.columns:
            processed_data['decay_velocity'] = processed_data.apply(
                lambda x: (x['peak_views'] - x['current_views']) / (x['days_since_publish'] - x['time_to_peak']) 
                if x['days_since_publish'] > x['time_to_peak'] and x['peak_views'] > x['current_views'] else 0,
                axis=1
            )
        
        # Calcular etapa del ciclo de vida si no existe
        if 'lifecycle_stage' not in processed_data.columns:
            processed_data['lifecycle_stage'] = processed_data.apply(
                lambda x: self._determine_lifecycle_stage(x),
                axis=1
            )
        
        # Ordenar por contenido y tiempo
        processed_data = processed_data.sort_values(['content_id', 'timestamp'])
        
        return processed_data
    
    def _determine_lifecycle_stage(self, row: pd.Series) -> str:
        """
        Determina la etapa del ciclo de vida del contenido basado en sus métricas.
        
        Args:
            row: Fila de datos con métricas de contenido
            
        Returns:
            Etapa del ciclo de vida
        """
        # Valores predeterminados para casos donde faltan datos
        days_since_publish = row.get('days_since_publish', 0)
        time_to_peak = row.get('time_to_peak', 7)  # Valor predeterminado de 7 días
        current_views = row.get('current_views', 0)
        peak_views = row.get('peak_views', 0)
        initial_views = row.get('initial_views', 0)
        
        # Lógica para determinar la etapa
        if days_since_publish <= 1:
            return 'launch'
        elif days_since_publish < time_to_peak and current_views < peak_views:
            return 'growth'
        elif abs(days_since_publish - time_to_peak) <= 2:  # Cerca del pico (+/- 2 días)
            return 'peak'
        elif days_since_publish > time_to_peak and current_views < peak_views * 0.7:
            return 'decay'
        elif days_since_publish > time_to_peak + 14 and current_views < peak_views * 0.3:
            if current_views > initial_views * 1.2:  # Resurgimiento si supera vistas iniciales
                return 'revival'
            else:
                return 'archive'
        else:
            return 'steady'
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrena el modelo de ciclo de vida con datos históricos de contenido.
        
        Args:
            training_data: DataFrame con datos históricos de contenido y métricas
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento del modelo de ciclo de vida de contenido")
        
        # Preprocesar datos
        processed_data = self.preprocess_data(training_data)
        
        # Verificar si hay suficientes datos
        if len(processed_data) < 10:
            return {"error": "Insuficientes datos para entrenar el modelo"}
        
        try:
            # Preparar características para entrenamiento
            # Seleccionar solo características numéricas
            numeric_features = [col for col in processed_data.columns 
                               if col not in ['content_id', 'timestamp', 'lifecycle_stage'] + self.categorical_features
                               and pd.api.types.is_numeric_dtype(processed_data[col])]
            
            # Entrenar modelo de longevidad (predicción de duración total del ciclo de vida)
            if 'content_lifespan' in processed_data.columns:
                X_longevity = processed_data[numeric_features].fillna(0)
                y_longevity = processed_data['content_lifespan']
                
                # Dividir datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X_longevity, y_longevity, test_size=0.2, random_state=42
                )
                
                # Normalizar características
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Entrenar modelo
                self.longevity_model = GradientBoostingRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=42
                )
                self.longevity_model.fit(X_train_scaled, y_train)
                
                # Evaluar modelo
                longevity_score = self.longevity_model.score(X_test_scaled, y_test)
                self.logger.info(f"Modelo de longevidad entrenado. R²: {longevity_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'content_lifespan' para entrenar el modelo de longevidad")
            
            # Entrenar modelo de rendimiento máximo
            if 'peak_performance' in processed_data.columns:
                X_peak = processed_data[numeric_features].fillna(0)
                y_peak = processed_data['peak_performance']
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X_peak, y_peak, test_size=0.2, random_state=42
                )
                
                # Normalizar
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Entrenar modelo
                self.peak_performance_model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42
                )
                self.peak_performance_model.fit(X_train_scaled, y_train)
                
                # Evaluar
                peak_score = self.peak_performance_model.score(X_test_scaled, y_test)
                self.logger.info(f"Modelo de rendimiento máximo entrenado. R²: {peak_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'peak_performance' para entrenar el modelo")
            
            # Entrenar modelo de tasa de decaimiento
            if 'decay_rate' in processed_data.columns:
                X_decay = processed_data[numeric_features].fillna(0)
                y_decay = processed_data['decay_rate']
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X_decay, y_decay, test_size=0.2, random_state=42
                )
                
                # Normalizar
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Entrenar modelo
                self.decay_rate_model = GradientBoostingRegressor(
                    n_estimators=100, 
                    learning_rate=0.05, 
                    max_depth=4, 
                    random_state=42
                )
                self.decay_rate_model.fit(X_train_scaled, y_train)
                
                # Evaluar
                decay_score = self.decay_rate_model.score(X_test_scaled, y_test)
                self.logger.info(f"Modelo de tasa de decaimiento entrenado. R²: {decay_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'decay_rate' para entrenar el modelo")
            
            # Entrenar modelo de potencial de resurgimiento
            if 'revival_potential' in processed_data.columns:
                X_revival = processed_data[numeric_features].fillna(0)
                y_revival = processed_data['revival_potential']
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X_revival, y_revival, test_size=0.2, random_state=42
                )
                
                # Normalizar
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Entrenar modelo
                self.revival_potential_model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=8, 
                    random_state=42
                )
                self.revival_potential_model.fit(X_train_scaled, y_train)
                
                # Evaluar
                revival_score = self.revival_potential_model.score(X_test_scaled, y_test)
                self.logger.info(f"Modelo de potencial de resurgimiento entrenado. R²: {revival_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'revival_potential' para entrenar el modelo")
            
            # Guardar modelo
            self.save_model()
            
            # Analizar patrones de ciclo de vida
            lifecycle_patterns = self._analyze_lifecycle_patterns(processed_data)
            
            # Preparar métricas de entrenamiento
            metrics = {
                "lifecycle_patterns": lifecycle_patterns
            }
            
            if self.longevity_model:
                metrics["longevity_model_r2"] = longevity_score
                metrics["longevity_feature_importance"] = dict(zip(
                    numeric_features, 
                    self.longevity_model.feature_importances_
                ))
            
            if self.peak_performance_model:
                metrics["peak_performance_model_r2"] = peak_score
                metrics["peak_feature_importance"] = dict(zip(
                    numeric_features, 
                    self.peak_performance_model.feature_importances_
                ))
            
            if self.decay_rate_model:
                metrics["decay_rate_model_r2"] = decay_score
                metrics["decay_feature_importance"] = dict(zip(
                    numeric_features, 
                    self.decay_rate_model.feature_importances_
                ))
            
            if self.revival_potential_model:
                metrics["revival_potential_model_r2"] = revival_score
                metrics["revival_feature_importance"] = dict(zip(
                    numeric_features, 
                    self.revival_potential_model.feature_importances_
                ))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento del modelo de ciclo de vida: {str(e)}")
            return {"error": f"Error en entrenamiento: {str(e)}"}
    
    def _analyze_lifecycle_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza patrones en el ciclo de vida del contenido.
        
        Args:
            data: DataFrame con datos de contenido y etapas del ciclo de vida
            
        Returns:
            Análisis de patrones del ciclo de vida
        """
        lifecycle_analysis = {}
        
        try:
            # Analizar distribución de etapas del ciclo de vida
            if 'lifecycle_stage' in data.columns:
                stage_distribution = data['lifecycle_stage'].value_counts(normalize=True).to_dict()
                lifecycle_analysis["stage_distribution"] = stage_distribution
            
            # Analizar duración promedio de cada etapa
            if 'lifecycle_stage' in data.columns and 'content_id' in data.columns and 'days_since_publish' in data.columns:
                # Agrupar por contenido y etapa, y calcular la duración de cada etapa
                stage_durations = {}
                
                # Obtener la primera y última fecha para cada contenido y etapa
                stage_ranges = data.groupby(['content_id', 'lifecycle_stage'])['days_since_publish'].agg(['min', 'max'])
                
                # Calcular la duración de cada etapa
                stage_ranges['duration'] = stage_ranges['max'] - stage_ranges['min'] + 1  # +1 para incluir el día inicial
                
                # Calcular la duración promedio por etapa
                avg_durations = stage_ranges.groupby(level=1)['duration'].mean()
                
                for stage in self.lifecycle_stages:
                    if stage in avg_durations:
                        stage_durations[stage] = float(avg_durations[stage])
                    else:
                        stage_durations[stage] = 0.0
                
                lifecycle_analysis["avg_stage_durations"] = stage_durations
            
            # Analizar factores que afectan la longevidad
            if self.longevity_model and hasattr(self.longevity_model, 'feature_importances_'):
                # Obtener las características más importantes
                feature_names = [col for col in data.columns 
                                if col not in ['content_id', 'timestamp', 'lifecycle_stage'] + self.categorical_features
                                and pd.api.types.is_numeric_dtype(data[col])]
                
                feature_importance = dict(zip(feature_names, self.longevity_model.feature_importances_))
                top_factors = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                lifecycle_analysis["longevity_factors"] = dict(top_factors)
            
            # Analizar patrones de transición entre etapas
            if 'lifecycle_stage' in data.columns and 'content_id' in data.columns:
                # Crear matriz de transición
                transition_matrix = {}
                
                for content_id in data['content_id'].unique():
                    content_data = data[data['content_id'] == content_id].sort_values('days_since_publish')
                    
                    if len(content_data) > 1:
                        stages = content_data['lifecycle_stage'].tolist()
                        
                        for i in range(len(stages) - 1):
                            from_stage = stages[i]
                            to_stage = stages[i + 1]
                            
                            if from_stage not in transition_matrix:
                                transition_matrix[from_stage] = {}
                            
                            if to_stage not in transition_matrix[from_stage]:
                                transition_matrix[from_stage][to_stage] = 0
                            
                            transition_matrix[from_stage][to_stage] += 1
                
                # Normalizar matriz de transición
                normalized_matrix = {}
                
                for from_stage, transitions in transition_matrix.items():
                    total = sum(transitions.values())
                    normalized_matrix[from_stage] = {to_stage: count / total for to_stage, count in transitions.items()}
                
                lifecycle_analysis["transition_matrix"] = normalized_matrix
            
            # Analizar patrones por tipo de contenido
            if 'content_type' in data.columns and 'lifecycle_stage' in data.columns:
                content_type_patterns = {}
                
                for content_type in data['content_type'].unique():
                    type_data = data[data['content_type'] == content_type]
                    
                    # Distribución de etapas para este tipo de contenido
                    stage_dist = type_data['lifecycle_stage'].value_counts(normalize=True).to_dict()
                    
                    # Duración promedio del ciclo de vida para este tipo
                    avg_lifespan = 0
                    if 'content_lifespan' in type_data.columns:
                        avg_lifespan = type_data['content_lifespan'].mean()
                    
                    content_type_patterns[content_type] = {
                        "stage_distribution": stage_dist,
                        "avg_lifespan": float(avg_lifespan)
                    }
                
                lifecycle_analysis["content_type_patterns"] = content_type_patterns
            
            return lifecycle_analysis
            
        except Exception as e:
            self.logger.error(f"Error al analizar patrones del ciclo de vida: {str(e)}")
            return {"error": f"Error al analizar patrones: {str(e)}"}
    
    def predict_lifecycle(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice el ciclo de vida completo de un contenido o conjunto de contenidos.
        
        Args:
            content_data: DataFrame con datos de contenido
            
        Returns:
            Predicciones del ciclo de vida
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Preprocesar datos
            processed_data = self.preprocess_data(content_data)
            
            # Preparar características para predicción
            numeric_features = [col for col in processed_data.columns 
                               if col not in ['content_id', 'timestamp', 'lifecycle_stage'] + self.categorical_features
                               and pd.api.types.is_numeric_dtype(processed_data[col])]
            
            X = processed_data[numeric_features].fillna(0)
            
            # Normalizar datos
            X_scaled = self.scaler.transform(X)
            
            # Preparar resultados
            results = []
            
            for i, row in processed_data.iterrows():
                content_result = {
                    "content_id": row["content_id"],
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "current_stage": row.get("lifecycle_stage", self._determine_lifecycle_stage(row))
                }
                
                # Predecir longevidad si el modelo está disponible
                if self.longevity_model:
                    X_longevity = X_scaled[i:i+1]
                    longevity_pred = self.longevity_model.predict(X_longevity)[0]
                    content_result["predicted_lifespan_days"] = float(longevity_pred)
                    
                    # Calcular fecha estimada de fin de ciclo de vida
                    if 'publish_date' in row:
                        publish_date = pd.to_datetime(row['publish_date'])
                        end_date = publish_date + timedelta(days=int(longevity_pred))
                        content_result["estimated_end_date"] = end_date.strftime("%Y-%m-%d")
                
                # Predecir rendimiento máximo si el modelo está disponible
                if self.peak_performance_model:
                    X_peak = X_scaled[i:i+1]
                    peak_pred = self.peak_performance_model.predict(X_peak)[0]
                    content_result["predicted_peak_performance"] = float(peak_pred)
                    
                    # Estimar tiempo hasta el pico
                    if 'time_to_peak' in numeric_features:
                        time_to_peak_idx = numeric_features.index('time_to_peak')
                        time_to_peak = X.iloc[i, time_to_peak_idx]
                        content_result["estimated_time_to_peak_days"] = float(time_to_peak)
                        
                        # Calcular fecha estimada del pico
                        if 'publish_date' in row:
                            publish_date = pd.to_datetime(row['publish_date'])
                            peak_date = publish_date + timedelta(days=int(time_to_peak))
                            content_result["estimated_peak_date"] = peak_date.strftime("%Y-%m-%d")
                
                # Predecir tasa de decaimiento si el modelo está disponible
                if self.decay_rate_model:
                    X_decay = X_scaled[i:i+1]
                    decay_pred = self.decay_rate_model.predict(X_decay)[0]
                    content_result["predicted_decay_rate"] = float(decay_pred)
                    
                    # Interpretar tasa de decaimiento
                    if decay_pred > 0.1:
                        content_result["decay_interpretation"] = "Rápido"
                    elif decay_pred > 0.05:
                        content_result["decay_interpretation"] = "Moderado"
                    else:
                        content_result["decay_interpretation"] = "Lento"
                
                # Predecir potencial de resurgimiento si el modelo está disponible
                if self.revival_potential_model:
                    X_revival = X_scaled[i:i+1]
                    revival_pred = self.revival_potential_model.predict(X_revival)[0]
                    content_result["revival_potential"] = float(revival_pred)
                    
                    # Interpretar potencial de resurgimiento
                    if revival_pred > 0.7:
                        content_result["revival_interpretation"] = "Alto"
                    elif revival_pred > 0.3:
                        content_result["revival_interpretation"] = "Moderado"
                    else:
                        content_result["revival_interpretation"] = "Bajo"
                
                # Generar proyección del ciclo de vida completo
                content_result["lifecycle_projection"] = self._generate_lifecycle_projection(row, content_result)
                
                results.append(content_result)
            
            return {"predictions": results}
            
        except Exception as e:
            self.logger.error(f"Error en predicción del ciclo de vida: {str(e)}")
            return {"error": f"Error en predicción: {str(e)}"}
    
    def _generate_lifecycle_projection(self, content_data: pd.Series, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera una proyección completa del ciclo de vida del contenido.
        
        Args:
            content_data: Serie con datos del contenido
            predictions: Diccionario con predicciones ya realizadas
            
        Returns:
            Lista de puntos de proyección del ciclo de vida
        """
        projection = []
        
        try:
            # Obtener datos necesarios para la proyección
            publish_date = None
            if 'publish_date' in content_data:
                publish_date = pd.to_datetime(content_data['publish_date'])
            else:
                # Usar fecha actual como aproximación
                publish_date = datetime.now()
            
            # Obtener predicciones
            lifespan = predictions.get("predicted_lifespan_days", 90)  # Valor predeterminado de 90 días
            time_to_peak = content_data.get("time_to_peak", lifespan * 0.2)  # 20% del ciclo de vida por defecto
            peak_performance = predictions.get("predicted_peak_performance", 100)  # Valor arbitrario
            decay_rate = predictions.get("predicted_decay_rate", 0.05)  # Valor predeterminado
            revival_potential = predictions.get("revival_potential", 0.1)  # Valor predeterminado
            
            # Valores iniciales
            initial_views = content_data.get("initial_views", peak_performance * 0.2)  # 20% del pico por defecto
            
            # Generar puntos de proyección para todo el ciclo de vida
            days = int(lifespan) + 1
            
            for day in range(days):
                # Calcular fecha
                current_date = publish_date + timedelta(days=day)
                
                # Determinar etapa del ciclo de vida
                stage = ""
                if day <= 1:
                    stage = "launch"
                elif day < time_to_peak:
                    stage = "growth"
                elif abs(day - time_to_peak) <= 2:
                    stage = "peak"
                elif day < time_to_peak + (lifespan - time_to_peak) * 0.5:
                    stage = "decay"
                elif revival_potential > 0.5 and day > lifespan * 0.7:
                    stage = "revival"
                elif day > lifespan * 0.9:
                    stage = "archive"
                else:
                    stage = "steady"
                
                # Calcular rendimiento proyectado
                performance = 0
                
                if day <= 1:  # Lanzamiento
                    performance = initial_views
                elif day < time_to_peak:  # Crecimiento
                    # Crecimiento exponencial hasta el pico
                    growth_progress = day / time_to_peak
                    performance = initial_views + (peak_performance - initial_views) * (growth_progress ** 2)
                elif abs(day - time_to_peak) <= 2:  # Pico
                    performance = peak_performance
                                elif stage == "decay":  # Decaimiento
                    # Decaimiento exponencial desde el pico
                    decay_progress = (day - time_to_peak) / (lifespan - time_to_peak)
                    performance = peak_performance * (1 - decay_rate) ** (day - time_to_peak)
                elif stage == "revival":  # Resurgimiento
                    # Pequeño repunte en la fase final
                    revival_start = lifespan * 0.7
                    revival_progress = (day - revival_start) / (lifespan * 0.2)
                    base_performance = peak_performance * (1 - decay_rate) ** (revival_start - time_to_peak)
                    revival_boost = base_performance * revival_potential * revival_progress
                    performance = base_performance + revival_boost
                else:  # Steady o Archive
                    # Rendimiento estable o residual
                    performance = peak_performance * (1 - decay_rate) ** (day - time_to_peak)
                    if stage == "archive":
                        performance = performance * 0.5  # Reducción adicional en fase de archivo
                
                # Añadir punto de proyección
                projection_point = {
                    "day": day,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "stage": stage,
                    "projected_performance": float(performance)
                }
                
                projection.append(projection_point)
            
            return projection
            
        except Exception as e:
            self.logger.error(f"Error al generar proyección del ciclo de vida: {str(e)}")
            return []
    
    def optimize_lifecycle(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera recomendaciones para optimizar el ciclo de vida del contenido.
        
        Args:
            content_data: DataFrame con datos de contenido
            
        Returns:
            Recomendaciones de optimización
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Preprocesar datos
            processed_data = self.preprocess_data(content_data)
            
            # Obtener predicciones del ciclo de vida
            lifecycle_predictions = self.predict_lifecycle(processed_data)
            
            if "error" in lifecycle_predictions:
                return lifecycle_predictions
            
            # Preparar recomendaciones
            recommendations = []
            
            for content_pred in lifecycle_predictions.get("predictions", []):
                content_id = content_pred["content_id"]
                content_row = processed_data[processed_data["content_id"] == content_id].iloc[0]
                
                # Obtener etapa actual
                current_stage = content_pred["current_stage"]
                
                # Recomendaciones específicas por etapa
                stage_recommendations = self._get_stage_recommendations(current_stage, content_pred, content_row)
                
                # Recomendaciones para extender la vida útil
                lifespan_recommendations = self._get_lifespan_extension_recommendations(content_pred, content_row)
                
                # Recomendaciones para maximizar el rendimiento en el pico
                peak_recommendations = self._get_peak_optimization_recommendations(content_pred, content_row)
                
                # Recomendaciones para potenciar resurgimiento
                revival_recommendations = self._get_revival_recommendations(content_pred, content_row)
                
                # Recomendaciones de monetización basadas en el ciclo de vida
                monetization_recommendations = self._get_lifecycle_monetization_recommendations(content_pred, content_row)
                
                # Consolidar recomendaciones
                content_recommendations = {
                    "content_id": content_id,
                    "current_stage": current_stage,
                    "stage_recommendations": stage_recommendations,
                    "lifespan_extension": lifespan_recommendations,
                    "peak_optimization": peak_recommendations,
                    "revival_strategies": revival_recommendations,
                    "monetization_strategies": monetization_recommendations
                }
                
                recommendations.append(content_recommendations)
            
            return {"recommendations": recommendations}
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones de optimización: {str(e)}")
            return {"error": f"Error al generar recomendaciones: {str(e)}"}
    
    def _get_stage_recommendations(self, stage: str, predictions: Dict[str, Any], content_data: pd.Series) -> List[str]:
        """
        Genera recomendaciones específicas para la etapa actual del ciclo de vida.
        
        Args:
            stage: Etapa actual del ciclo de vida
            predictions: Predicciones del ciclo de vida
            content_data: Datos del contenido
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        if stage == "launch":
            recommendations = [
                "Promocionar intensivamente en las primeras 24-48 horas para maximizar el impulso inicial",
                "Utilizar notificaciones push y alertas para seguidores existentes",
                "Compartir en todas las plataformas sociales disponibles",
                "Considerar una pequeña inversión en promoción pagada para amplificar el alcance inicial",
                "Responder rápidamente a los primeros comentarios para fomentar la participación"
            ]
        elif stage == "growth":
            recommendations = [
                "Fomentar compartir el contenido mediante llamados a la acción específicos",
                "Interactuar con los comentarios para aumentar el engagement y la visibilidad",
                "Crear contenido complementario que dirija tráfico al contenido principal",
                "Optimizar títulos, miniaturas y descripciones basándose en el rendimiento inicial",
                "Identificar y conectar con influencers relevantes que puedan amplificar el alcance"
            ]
        elif stage == "peak":
            recommendations = [
                "Maximizar monetización durante este período de máxima visibilidad",
                "Capturar datos de audiencia para futuras campañas",
                "Preparar contenido de seguimiento para mantener el impulso",
                "Implementar estrategias de retención para mantener la audiencia después del pico",
                "Considerar colaboraciones o crosspromotion para extender el pico"
            ]
        elif stage == "decay":
            recommendations = [
                "Ralentizar la tasa de decaimiento con actualizaciones o nuevo contexto",
                "Repackaging del contenido en nuevos formatos para audiencias diferentes",
                "Implementar estrategias de remarketing para audiencias que ya interactuaron",
                "Analizar métricas para identificar segmentos de audiencia con mayor retención",
                "Reducir gradualmente la inversión promocional y redirigirla a contenido más reciente"
            ]
        elif stage == "steady":
            recommendations = [
                "Mantener actualizaciones periódicas para sostener el interés",
                "Optimizar para búsqueda a largo plazo (SEO)",
                "Vincular desde contenido nuevo para mantener tráfico constante",
                "Evaluar oportunidades de monetización sostenible a largo plazo",
                "Utilizar como contenido de referencia en nuevas publicaciones relacionadas"
            ]
        elif stage == "revival":
            recommendations = [
                "Conectar el contenido con tendencias o eventos actuales",
                "Reintroducir a nuevas audiencias con contexto actualizado",
                "Implementar nuevas estrategias de distribución en canales no explorados previamente",
                "Crear contenido derivado que dirija tráfico al original",
                "Considerar colaboraciones con creadores que puedan aportar nuevas audiencias"
            ]
        elif stage == "archive":
            recommendations = [
                "Evaluar si el contenido debe mantenerse público o archivarse",
                "Extraer aprendizajes para futuros contenidos",
                "Considerar actualizar o reformatear completamente si el tema sigue siendo relevante",
                "Redirigir tráfico residual hacia contenido más reciente y relevante",
                "Mantener como recurso histórico si tiene valor de referencia"
            ]
        
        # Personalizar recomendaciones según el tipo de contenido
        content_type = content_data.get("content_type", "unknown")
        if content_type == "video":
            if stage in ["launch", "growth"]:
                recommendations.append("Crear clips cortos para plataformas sociales que dirijan al video completo")
            elif stage in ["decay", "revival"]:
                recommendations.append("Considerar añadir nuevos cortes o ediciones para revitalizar el contenido")
        elif content_type == "article":
            if stage in ["steady", "archive"]:
                recommendations.append("Actualizar con información reciente para mejorar SEO y relevancia")
        
        return recommendations
    
    def _get_lifespan_extension_recommendations(self, predictions: Dict[str, Any], content_data: pd.Series) -> List[str]:
        """
        Genera recomendaciones para extender la vida útil del contenido.
        
        Args:
            predictions: Predicciones del ciclo de vida
            content_data: Datos del contenido
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = [
            "Actualizar periódicamente con información nueva o relevante",
            "Crear series o secuelas que mantengan el interés en el contenido original",
            "Repackaging en diferentes formatos (video, audio, infografía, etc.)",
            "Vincular desde contenido nuevo y relevante",
            "Optimizar para búsqueda a largo plazo con palabras clave evergreen"
        ]
        
        # Personalizar según predicciones específicas
        decay_rate = predictions.get("predicted_decay_rate", 0.05)
        if decay_rate > 0.1:  # Decaimiento rápido
            recommendations.append("Implementar estrategias de remarketing agresivas para contrarrestar el rápido decaimiento")
            recommendations.append("Considerar una actualización completa del contenido para un relanzamiento")
        
        revival_potential = predictions.get("revival_potential", 0)
        if revival_potential > 0.5:  # Alto potencial de resurgimiento
            recommendations.append("Planificar estrategia de resurgimiento para el momento óptimo del ciclo")
            recommendations.append("Preparar actualizaciones significativas para cuando el contenido alcance su punto más bajo")
        
        return recommendations
    
    def _get_peak_optimization_recommendations(self, predictions: Dict[str, Any], content_data: pd.Series) -> List[str]:
        """
        Genera recomendaciones para optimizar el rendimiento en el pico del ciclo de vida.
        
        Args:
            predictions: Predicciones del ciclo de vida
            content_data: Datos del contenido
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = [
            "Maximizar opciones de monetización durante el período de pico",
            "Implementar llamados a la acción para capturar datos de audiencia",
            "Preparar contenido complementario para lanzar durante el pico",
            "Considerar colaboraciones estratégicas para amplificar el alcance máximo",
            "Optimizar la experiencia de usuario para maximizar el tiempo de permanencia"
        ]
        
        # Personalizar según predicciones específicas
        time_to_peak = predictions.get("estimated_time_to_peak_days", 7)
        if time_to_peak < 3:  # Pico muy rápido
            recommendations.append("Preparar estrategia de respuesta rápida para capitalizar el pico temprano")
            recommendations.append("Tener preparados recursos adicionales para escalar rápidamente si el contenido se viraliza")
        elif time_to_peak > 14:  # Pico tardío
            recommendations.append("Mantener promoción constante para sostener el crecimiento hasta el pico")
            recommendations.append("Implementar estrategia de contenido secuencial para mantener el impulso")
        
        return recommendations
    
    def _get_revival_recommendations(self, predictions: Dict[str, Any], content_data: pd.Series) -> List[str]:
        """
        Genera recomendaciones para potenciar el resurgimiento del contenido.
        
        Args:
            predictions: Predicciones del ciclo de vida
            content_data: Datos del contenido
            
        Returns:
            Lista de recomendaciones
        """
        revival_potential = predictions.get("revival_potential", 0)
        revival_interpretation = predictions.get("revival_interpretation", "Bajo")
        
        if revival_interpretation == "Alto":
            recommendations = [
                "Planificar relanzamiento estratégico cuando el contenido alcance la fase de steady",
                "Preparar actualizaciones significativas con nuevo ángulo o información",
                "Identificar eventos futuros que puedan hacer relevante nuevamente el contenido",
                "Crear contenido complementario que redirija a la pieza original",
                "Implementar estrategia de distribución en nuevos canales no explorados inicialmente"
            ]
        elif revival_interpretation == "Moderado":
            recommendations = [
                "Considerar actualizaciones periódicas para mantener la relevancia",
                "Vincular desde contenido nuevo relacionado",
                "Monitorear tendencias relacionadas para identificar oportunidades de resurgimiento",
                "Repackaging en formatos alternativos para nuevas audiencias",
                "Implementar estrategia de remarketing para audiencias que ya interactuaron"
            ]
        else:  # Bajo potencial
            recommendations = [
                "Extraer aprendizajes para futuros contenidos similares",
                "Considerar reformatear completamente si el tema sigue siendo relevante",
                "Evaluar si el contenido debe mantenerse público o archivarse",
                "Redirigir tráfico residual hacia contenido más reciente",
                "Utilizar partes valiosas como referencia en nuevo contenido"
            ]
        
        return recommendations
    
    def _get_lifecycle_monetization_recommendations(self, predictions: Dict[str, Any], content_data: pd.Series) -> List[Dict[str, str]]:
        """
        Genera recomendaciones de monetización basadas en el ciclo de vida.
        
        Args:
            predictions: Predicciones del ciclo de vida
            content_data: Datos del contenido
            
        Returns:
            Lista de recomendaciones de monetización
        """
        current_stage = predictions.get("current_stage", "unknown")
        
        monetization_strategies = []
        
        if current_stage == "launch":
            monetization_strategies = [
                {
                    "strategy": "Publicidad mínima",
                    "description": "Minimizar la publicidad para no afectar la experiencia inicial y maximizar el alcance"
                },
                {
                    "strategy": "Afiliados estratégicos",
                    "description": "Incluir solo enlaces de afiliados altamente relevantes y de alto valor"
                },
                {
                    "strategy": "Promoción cruzada",
                    "description": "Promocionar productos/servicios propios de forma sutil"
                }
            ]
        elif current_stage == "growth":
            monetization_strategies = [
                {
                    "strategy": "Incremento gradual de publicidad",
                    "description": "Aumentar gradualmente la densidad publicitaria sin afectar el crecimiento"
                },
                {
                    "strategy": "Afiliados expandidos",
                    "description": "Ampliar la gama de ofertas de afiliados relevantes"
                },
                {
                    "strategy": "Ofertas de tiempo limitado",
                    "description": "Introducir ofertas especiales para aprovechar el crecimiento de audiencia"
                }
            ]
        elif current_stage == "peak":
            monetization_strategies = [
                {
                    "strategy": "Monetización máxima",
                    "description": "Implementar estrategia completa de monetización durante el pico de audiencia"
                },
                {
                    "strategy": "Patrocinios premium",
                    "description": "Negociar patrocinios de alto valor aprovechando la máxima visibilidad"
                },
                {
                    "strategy": "Productos propios",
                    "description": "Promocionar agresivamente productos/servicios propios"
                },
                {
                    "strategy": "Captura de leads",
                    "description": "Priorizar la captura de datos de audiencia para monetización futura"
                }
            ]
        elif current_stage == "decay":
            monetization_strategies = [
                {
                    "strategy": "Optimización publicitaria",
                    "description": "Ajustar la estrategia publicitaria para maximizar RPM sin acelerar el decaimiento"
                },
                {
                    "strategy": "Remarketing",
                    "description": "Implementar campañas de remarketing para audiencia ya comprometida"
                },
                {
                    "strategy": "Ofertas de recuperación",
                    "description": "Introducir ofertas especiales para mantener el engagement"
                }
            ]
        elif current_stage == "steady":
            monetization_strategies = [
                {
                    "strategy": "Monetización sostenible",
                    "description": "Implementar estrategia de monetización a largo plazo"
                },
                {
                    "strategy": "Afiliados evergreen",
                    "description": "Mantener solo afiliados de alto rendimiento a largo plazo"
                },
                {
                    "strategy": "Productos de valor continuo",
                    "description": "Promocionar productos/servicios con propuesta de valor duradera"
                }
            ]
        elif current_stage == "revival":
            monetization_strategies = [
                {
                    "strategy": "Relanzamiento monetizado",
                    "description": "Implementar nueva estrategia de monetización adaptada al resurgimiento"
                },
                {
                    "strategy": "Nuevas ofertas",
                    "description": "Introducir nuevos productos/servicios relevantes para la audiencia actual"
                },
                {
                    "strategy": "Patrocinios renovados",
                    "description": "Buscar nuevos patrocinadores interesados en el resurgimiento"
                }
            ]
        elif current_stage == "archive":
            monetization_strategies = [
                {
                    "strategy": "Monetización pasiva",
                    "description": "Mantener solo monetización que no requiera mantenimiento activo"
                },
                {
                    "strategy": "Redirección estratégica",
                    "description": "Redirigir tráfico residual hacia contenido más monetizable"
                },
                {
                    "strategy": "Valor histórico",
                    "description": "Evaluar si el contenido tiene valor como recurso histórico o de referencia"
                }
            ]
        
        return monetization_strategies
    
    def visualize_lifecycle(self, content_data: pd.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera visualizaciones del ciclo de vida del contenido.
        
        Args:
            content_data: DataFrame con datos de contenido
            output_path: Ruta opcional para guardar visualizaciones
            
        Returns:
            Rutas a las visualizaciones generadas
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Obtener predicciones del ciclo de vida
            lifecycle_predictions = self.predict_lifecycle(content_data)
            
            if "error" in lifecycle_predictions:
                return lifecycle_predictions
            
            # Crear directorio para visualizaciones si no existe
            if output_path:
                os.makedirs(output_path, exist_ok=True)
            else:
                output_path = "lifecycle_visualizations"
                os.makedirs(output_path, exist_ok=True)
            
            visualization_paths = {}
            
            # Generar visualizaciones para cada contenido
            for content_pred in lifecycle_predictions.get("predictions", []):
                content_id = content_pred["content_id"]
                
                # Extraer datos de proyección
                projection = content_pred.get("lifecycle_projection", [])
                
                if not projection:
                    continue
                
                # Preparar datos para visualización
                days = [point["day"] for point in projection]
                performance = [point["projected_performance"] for point in projection]
                stages = [point["stage"] for point in projection]
                
                # Crear figura
                plt.figure(figsize=(12, 8))
                
                # Definir colores por etapa
                stage_colors = {
                    "launch": "green",
                    "growth": "blue",
                    "peak": "purple",
                    "decay": "orange",
                    "steady": "gray",
                    "revival": "red",
                    "archive": "black"
                }
                
                # Graficar por etapas con colores diferentes
                current_stage = stages[0]
                stage_start = 0
                
                for i in range(1, len(stages) + 1):
                    if i == len(stages) or stages[i] != current_stage:
                        plt.plot(
                            days[stage_start:i],
                            performance[stage_start:i],
                            color=stage_colors.get(current_stage, "blue"),
                            linewidth=3,
                            label=current_stage if stage_start == 0 or current_stage != stages[stage_start-1] else None
                        )
                        
                        if i < len(stages):
                            current_stage = stages[i]
                            stage_start = i
                
                # Añadir etiquetas y título
                plt.xlabel("Días desde publicación")
                plt.ylabel("Rendimiento")
                plt.title(f"Proyección del Ciclo de Vida - Contenido {content_id}")
                
                # Añadir leyenda
                plt.legend()
                
                # Añadir grid
                plt.grid(True, linestyle="--", alpha=0.7)
                
                # Guardar visualización
                file_path = os.path.join(output_path, f"lifecycle_{content_id}.png")
                plt.savefig(file_path)
                plt.close()
                
                visualization_paths[content_id] = file_path
            
            return {"visualization_paths": visualization_paths}
            
        except Exception as e:
            self.logger.error(f"Error al generar visualizaciones del ciclo de vida: {str(e)}")
            return {"error": f"Error al generar visualizaciones: {str(e)}"}