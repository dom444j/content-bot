"""
Modelo predictivo para analizar y optimizar el rendimiento de contenido en diferentes plataformas.
Este modelo compara métricas entre plataformas, identifica factores de éxito específicos por plataforma
y recomienda estrategias de optimización para maximizar el rendimiento multiplataforma.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
import json
from .base_model import BaseModel

class PlatformPerformanceModel(BaseModel):
    """
    Modelo para analizar y predecir el rendimiento de contenido en múltiples plataformas.
    
    Capacidades:
    - Comparación de métricas entre plataformas (YouTube, TikTok, Instagram, etc.)
    - Identificación de factores de éxito específicos por plataforma
    - Predicción de rendimiento cruzado entre plataformas
    - Recomendaciones para optimización multiplataforma
    - Análisis de tendencias específicas por plataforma
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el modelo de rendimiento de plataformas.
        
        Args:
            model_path: Ruta opcional al modelo pre-entrenado
        """
        super().__init__(model_name="platform_performance_model", model_path=model_path)
        
        # Modelos específicos por plataforma
        self.platform_models = {}
        
        # Modelo de rendimiento cruzado
        self.cross_platform_model = None
        
        # Preprocesadores
        self.preprocessors = {}
        
        # Plataformas soportadas
        self.supported_platforms = [
            'youtube', 'youtube_shorts', 'tiktok', 
            'instagram', 'instagram_reels', 'threads',
            'bluesky', 'x'
        ]
        
        # Características específicas por plataforma
        self.platform_features = {
            'youtube': [
                'duration', 'title_length', 'description_length', 'tags_count',
                'thumbnail_quality', 'cta_timing', 'cta_type', 'end_screen',
                'cards_count', 'chapters_count', 'hashtags_count'
            ],
            'youtube_shorts': [
                'duration', 'title_length', 'hashtags_count', 'cta_timing',
                'cta_type', 'vertical_ratio', 'music_used', 'text_overlay'
            ],
            'tiktok': [
                'duration', 'hashtags_count', 'music_used', 'effects_count',
                'cta_timing', 'cta_type', 'vertical_ratio', 'text_overlay',
                'trending_sound', 'duet_enabled', 'stitch_enabled'
            ],
            'instagram': [
                'caption_length', 'hashtags_count', 'mentions_count',
                'location_tagged', 'carousel_count', 'filter_used',
                'product_tags', 'cta_type'
            ],
            'instagram_reels': [
                'duration', 'hashtags_count', 'music_used', 'effects_count',
                'cta_timing', 'cta_type', 'vertical_ratio', 'text_overlay',
                'trending_sound', 'caption_length'
            ],
            'threads': [
                'text_length', 'hashtags_count', 'mentions_count',
                'media_count', 'link_included', 'cta_type'
            ],
            'bluesky': [
                'text_length', 'hashtags_count', 'mentions_count',
                'media_count', 'link_included', 'cta_type'
            ],
            'x': [
                'text_length', 'hashtags_count', 'mentions_count',
                'media_count', 'link_included', 'poll_included',
                'cta_type', 'thread_count'
            ]
        }
        
        # Métricas de rendimiento por plataforma
        self.platform_metrics = {
            'youtube': [
                'views', 'watch_time', 'average_view_duration', 'likes',
                'comments', 'shares', 'subscribers_gained', 'ctr', 
                'audience_retention', 'engagement_rate', 'rpm'
            ],
            'youtube_shorts': [
                'views', 'likes', 'comments', 'shares', 'subscribers_gained',
                'engagement_rate', 'rpm'
            ],
            'tiktok': [
                'views', 'likes', 'comments', 'shares', 'follows',
                'profile_visits', 'completion_rate', 'engagement_rate',
                'revenue'
            ],
            'instagram': [
                'impressions', 'reach', 'likes', 'comments', 'shares',
                'saves', 'follows', 'profile_visits', 'engagement_rate'
            ],
            'instagram_reels': [
                'plays', 'likes', 'comments', 'shares', 'saves',
                'follows', 'profile_visits', 'completion_rate', 'engagement_rate'
            ],
            'threads': [
                'impressions', 'likes', 'replies', 'reposts',
                'follows', 'profile_visits', 'engagement_rate'
            ],
            'bluesky': [
                'impressions', 'likes', 'replies', 'reposts',
                'follows', 'profile_visits', 'engagement_rate'
            ],
            'x': [
                'impressions', 'likes', 'replies', 'retweets',
                'quotes', 'follows', 'profile_visits', 'link_clicks',
                'engagement_rate'
            ]
        }
        
        # Características comunes entre plataformas
        self.common_features = [
            'content_type', 'topic', 'language', 'publish_time',
            'day_of_week', 'is_trending_topic', 'creator_followers',
            'creator_engagement_rate', 'content_quality_score',
            'has_cta', 'cta_type', 'emotional_tone'
        ]
        
        # Características categóricas
        self.categorical_features = [
            'content_type', 'topic', 'language', 'day_of_week',
            'cta_type', 'emotional_tone', 'platform'
        ]
        
        # Inicializar modelos si existe la ruta
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, data: pd.DataFrame, platform: str = None) -> pd.DataFrame:
        """
        Preprocesa los datos para el entrenamiento o predicción.
        
        Args:
            data: DataFrame con datos de contenido y métricas
            platform: Plataforma específica para preprocesar (opcional)
            
        Returns:
            DataFrame preprocesado
        """
        # Verificar si hay datos
        if data.empty:
            self.logger.warning("DataFrame vacío proporcionado para preprocesamiento")
            return pd.DataFrame()
        
        # Verificar columna de plataforma
        if 'platform' not in data.columns and platform is None:
            self.logger.warning("No se especificó plataforma y no hay columna 'platform' en los datos")
            return pd.DataFrame()
        
        # Usar plataforma específica o filtrar por plataforma
        if platform:
            platform_data = data.copy()
            platform_data['platform'] = platform
        else:
            platform_data = data.copy()
        
        # Convertir plataforma a minúsculas
        platform_data['platform'] = platform_data['platform'].str.lower()
        
        # Filtrar solo plataformas soportadas
        platform_data = platform_data[platform_data['platform'].isin(self.supported_platforms)]
        
        if platform_data.empty:
            self.logger.warning(f"No hay datos para plataformas soportadas: {self.supported_platforms}")
            return pd.DataFrame()
        
        # Convertir timestamp a datetime si existe
        if 'timestamp' in platform_data.columns:
            platform_data['timestamp'] = pd.to_datetime(platform_data['timestamp'])
            
            # Extraer características temporales
            platform_data['hour_of_day'] = platform_data['timestamp'].dt.hour
            platform_data['day_of_week'] = platform_data['timestamp'].dt.dayofweek
            platform_data['month'] = platform_data['timestamp'].dt.month
            platform_data['is_weekend'] = platform_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Convertir publish_time a datetime si existe
        if 'publish_time' in platform_data.columns and not pd.api.types.is_datetime64_any_dtype(platform_data['publish_time']):
            platform_data['publish_time'] = pd.to_datetime(platform_data['publish_time'])
            
            # Extraer características temporales si no existen
            if 'hour_of_day' not in platform_data.columns:
                platform_data['hour_of_day'] = platform_data['publish_time'].dt.hour
            if 'day_of_week' not in platform_data.columns:
                platform_data['day_of_week'] = platform_data['publish_time'].dt.dayofweek
            if 'month' not in platform_data.columns:
                platform_data['month'] = platform_data['publish_time'].dt.month
            if 'is_weekend' not in platform_data.columns:
                platform_data['is_weekend'] = platform_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Manejar valores faltantes
        for col in platform_data.columns:
            if col in self.categorical_features:
                platform_data[col] = platform_data[col].fillna('unknown')
            elif pd.api.types.is_numeric_dtype(platform_data[col]):
                platform_data[col] = platform_data[col].fillna(0)
        
        # Calcular características derivadas
        
        # Engagement rate si no existe
        if 'engagement_rate' not in platform_data.columns:
            for platform_name in platform_data['platform'].unique():
                platform_subset = platform_data[platform_data['platform'] == platform_name]
                
                if platform_name in ['youtube', 'youtube_shorts']:
                    if all(col in platform_subset.columns for col in ['likes', 'comments', 'shares', 'views']):
                        mask = platform_data['platform'] == platform_name
                        platform_data.loc[mask, 'engagement_rate'] = (
                            (platform_subset['likes'] + platform_subset['comments'] + platform_subset['shares']) / 
                            platform_subset['views'].clip(lower=1)
                        )
                
                elif platform_name in ['tiktok', 'instagram_reels']:
                    if all(col in platform_subset.columns for col in ['likes', 'comments', 'shares', 'views']):
                        mask = platform_data['platform'] == platform_name
                        platform_data.loc[mask, 'engagement_rate'] = (
                            (platform_subset['likes'] + platform_subset['comments'] + platform_subset['shares']) / 
                            platform_subset['views'].clip(lower=1)
                        )
                
                elif platform_name in ['instagram']:
                    if all(col in platform_subset.columns for col in ['likes', 'comments', 'impressions']):
                        mask = platform_data['platform'] == platform_name
                        platform_data.loc[mask, 'engagement_rate'] = (
                            (platform_subset['likes'] + platform_subset['comments']) / 
                            platform_subset['impressions'].clip(lower=1)
                        )
                
                elif platform_name in ['threads', 'bluesky', 'x']:
                    if all(col in platform_subset.columns for col in ['likes', 'replies', 'impressions']):
                        mask = platform_data['platform'] == platform_name
                        platform_data.loc[mask, 'engagement_rate'] = (
                            (platform_subset['likes'] + platform_subset['replies']) / 
                            platform_subset['impressions'].clip(lower=1)
                        )
        
        # Normalizar engagement_rate a porcentaje
        if 'engagement_rate' in platform_data.columns:
            platform_data['engagement_rate'] = platform_data['engagement_rate'] * 100
        
        # Calcular has_cta si no existe
        if 'has_cta' not in platform_data.columns and 'cta_type' in platform_data.columns:
            platform_data['has_cta'] = (~platform_data['cta_type'].isin(['none', 'unknown', np.nan])).astype(int)
        
        # Calcular content_quality_score si no existe (promedio de características relevantes)
        if 'content_quality_score' not in platform_data.columns:
            quality_features = []
            
            if 'visual_quality' in platform_data.columns:
                quality_features.append('visual_quality')
            if 'audio_quality' in platform_data.columns:
                quality_features.append('audio_quality')
            if 'narrative_quality' in platform_data.columns:
                quality_features.append('narrative_quality')
            if 'production_value' in platform_data.columns:
                quality_features.append('production_value')
            
            if quality_features:
                platform_data['content_quality_score'] = platform_data[quality_features].mean(axis=1)
            else:
                # Valor predeterminado si no hay características de calidad
                platform_data['content_quality_score'] = 5.0
        
        return platform_data
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrena el modelo de rendimiento de plataformas con datos históricos.
        
        Args:
            training_data: DataFrame con datos históricos de contenido y métricas
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento del modelo de rendimiento de plataformas")
        
        # Preprocesar datos
        processed_data = self.preprocess_data(training_data)
        
        if processed_data.empty:
            return {"error": "No hay datos válidos para entrenar el modelo"}
        
        # Resultados de entrenamiento
        training_results = {}
        
        try:
            # Entrenar modelos específicos por plataforma
            for platform in processed_data['platform'].unique():
                if platform not in self.supported_platforms:
                    continue
                
                platform_data = processed_data[processed_data['platform'] == platform]
                
                if len(platform_data) < 10:
                    self.logger.warning(f"Insuficientes datos para entrenar modelo de {platform}")
                    continue
                
                self.logger.info(f"Entrenando modelo para plataforma: {platform}")
                
                # Obtener características específicas de la plataforma
                platform_specific_features = self.platform_features.get(platform, [])
                
                # Combinar con características comunes
                features = self.common_features + [f for f in platform_specific_features if f in platform_data.columns]
                
                # Filtrar características disponibles
                available_features = [f for f in features if f in platform_data.columns]
                
                if not available_features:
                    self.logger.warning(f"No hay características disponibles para {platform}")
                    continue
                
                # Seleccionar métrica objetivo (engagement_rate por defecto)
                target_metrics = self.platform_metrics.get(platform, ['engagement_rate'])
                target_metric = next((m for m in target_metrics if m in platform_data.columns), None)
                
                if not target_metric:
                    self.logger.warning(f"No hay métrica objetivo disponible para {platform}")
                    continue
                
                # Preparar datos para entrenamiento
                X = platform_data[available_features]
                y = platform_data[target_metric]
                
                # Identificar características categóricas disponibles
                categorical_features = [f for f in self.categorical_features if f in available_features]
                numeric_features = [f for f in available_features if f not in categorical_features]
                
                # Crear preprocesador
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ]
                )
                
                # Dividir datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Crear pipeline con preprocesador y modelo
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', GradientBoostingRegressor(
                        n_estimators=100, 
                        learning_rate=0.1, 
                        max_depth=5, 
                        random_state=42
                    ))
                ])
                
                # Entrenar modelo
                pipeline.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = pipeline.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.logger.info(f"Modelo de {platform} entrenado. MSE: {mse:.4f}, R²: {r2:.4f}")
                
                # Guardar modelo y preprocesador
                self.platform_models[platform] = pipeline
                self.preprocessors[platform] = {
                    'features': available_features,
                    'categorical_features': categorical_features,
                    'numeric_features': numeric_features,
                    'target_metric': target_metric
                }
                
                # Guardar resultados
                training_results[platform] = {
                    'mse': float(mse),
                    'r2': float(r2),
                    'sample_size': len(platform_data),
                    'features_used': available_features,
                    'target_metric': target_metric
                }
            
            # Entrenar modelo de rendimiento cruzado entre plataformas
            if len(processed_data['platform'].unique()) > 1:
                self.logger.info("Entrenando modelo de rendimiento cruzado entre plataformas")
                
                # Preparar datos para entrenamiento
                # Usar solo características comunes entre plataformas
                common_available = [f for f in self.common_features if f in processed_data.columns]
                common_available.append('platform')  # Añadir plataforma como característica
                
                if 'engagement_rate' in processed_data.columns:
                    X_cross = processed_data[common_available]
                    y_cross = processed_data['engagement_rate']
                    
                    # Identificar características categóricas disponibles
                    categorical_features = [f for f in self.categorical_features if f in common_available]
                    numeric_features = [f for f in common_available if f not in categorical_features]
                    
                    # Crear preprocesador
                    cross_preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                        ]
                    )
                    
                    # Dividir datos en entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_cross, y_cross, test_size=0.2, random_state=42
                    )
                    
                    # Crear pipeline con preprocesador y modelo
                    cross_pipeline = Pipeline([
                        ('preprocessor', cross_preprocessor),
                        ('model', RandomForestRegressor(
                            n_estimators=100, 
                            max_depth=10, 
                            random_state=42
                        ))
                    ])
                    
                    # Entrenar modelo
                    cross_pipeline.fit(X_train, y_train)
                    
                    # Evaluar modelo
                    y_pred = cross_pipeline.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.logger.info(f"Modelo de rendimiento cruzado entrenado. MSE: {mse:.4f}, R²: {r2:.4f}")
                    
                    # Guardar modelo
                    self.cross_platform_model = cross_pipeline
                    
                    # Guardar resultados
                    training_results['cross_platform'] = {
                        'mse': float(mse),
                        'r2': float(r2),
                        'sample_size': len(processed_data),
                        'features_used': common_available,
                        'target_metric': 'engagement_rate'
                    }
            
            # Guardar modelo
            self.save_model()
            
            # Analizar factores de éxito por plataforma
            success_factors = self._analyze_success_factors(processed_data)
            training_results['success_factors'] = success_factors
            
            # Analizar diferencias entre plataformas
            platform_differences = self._analyze_platform_differences(processed_data)
            training_results['platform_differences'] = platform_differences
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento del modelo de rendimiento: {str(e)}")
            return {"error": f"Error en entrenamiento: {str(e)}"}
    
    def _analyze_success_factors(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza los factores de éxito para cada plataforma.
        
        Args:
            data: DataFrame con datos de contenido y métricas
            
        Returns:
            Análisis de factores de éxito por plataforma
        """
        success_factors = {}
        
        try:
            for platform in data['platform'].unique():
                if platform not in self.supported_platforms or platform not in self.platform_models:
                    continue
                
                platform_data = data[data['platform'] == platform]
                
                if len(platform_data) < 10:
                    continue
                
                # Obtener información del preprocesador
                preprocessor_info = self.preprocessors.get(platform, {})
                features = preprocessor_info.get('features', [])
                target_metric = preprocessor_info.get('target_metric', 'engagement_rate')
                
                if not features or target_metric not in platform_data.columns:
                    continue
                
                # Obtener modelo
                pipeline = self.platform_models[platform]
                
                # Extraer importancia de características si es posible
                if hasattr(pipeline['model'], 'feature_importances_'):
                    # Obtener nombres de características después del preprocesamiento
                    preprocessor = pipeline['preprocessor']
                    feature_names = []
                    
                    # Obtener nombres de características numéricas
                    if 'num' in preprocessor.named_transformers_:
                        num_features = preprocessor_info.get('numeric_features', [])
                        feature_names.extend(num_features)
                    
                    # Obtener nombres de características categóricas (con one-hot encoding)
                    if 'cat' in preprocessor.named_transformers_:
                        cat_features = preprocessor_info.get('categorical_features', [])
                        cat_encoder = preprocessor.named_transformers_['cat']
                        
                        if hasattr(cat_encoder, 'get_feature_names_out'):
                            cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
                            feature_names.extend(cat_feature_names)
                        else:
                            # Fallback para versiones antiguas de scikit-learn
                            for cat in cat_features:
                                unique_values = platform_data[cat].unique()
                                for val in unique_values:
                                    feature_names.append(f"{cat}_{val}")
                    
                    # Obtener importancia de características
                    feature_importances = pipeline['model'].feature_importances_
                    
                    # Ajustar longitud si es necesario
                    if len(feature_names) > len(feature_importances):
                        feature_names = feature_names[:len(feature_importances)]
                    elif len(feature_names) < len(feature_importances):
                        feature_importances = feature_importances[:len(feature_names)]
                    
                    # Crear diccionario de importancia
                    importance_dict = dict(zip(feature_names, feature_importances))
                    
                    # Ordenar por importancia
                    sorted_importance = sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    # Tomar las 10 características más importantes
                    top_factors = dict(sorted_importance[:10])
                    
                    success_factors[platform] = {
                        'top_factors': top_factors,
                        'target_metric': target_metric
                    }
                
                # Análisis de correlación
                if len(features) > 1:
                    correlation_data = platform_data[features + [target_metric]].copy()
                    
                    # Filtrar solo columnas numéricas
                    numeric_cols = correlation_data.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 1:
                        correlation_matrix = correlation_data[numeric_cols].corr()
                        
                        # Obtener correlaciones con la métrica objetivo
                        if target_metric in correlation_matrix:
                            target_correlations = correlation_matrix[target_metric].drop(target_metric)
                            
                            # Ordenar por valor absoluto de correlación
                            sorted_correlations = target_correlations.abs().sort_values(ascending=False)
                            
                            # Tomar las 10 correlaciones más fuertes
                            top_correlations = {}
                            for feature in sorted_correlations.index[:10]:
                                top_correlations[feature] = float(correlation_matrix.loc[feature, target_metric])
                            
                            if platform in success_factors:
                                success_factors[platform]['correlations'] = top_correlations
                            else:
                                success_factors[platform] = {
                                    'correlations': top_correlations,
                                    'target_metric': target_metric
                                }
            
            return success_factors
            
        except Exception as e:
            self.logger.error(f"Error al analizar factores de éxito: {str(e)}")
            return {"error": f"Error al analizar factores: {str(e)}"}
    
    def _analyze_platform_differences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza las diferencias de rendimiento entre plataformas.
        
        Args:
            data: DataFrame con datos de contenido y métricas
            
        Returns:
            Análisis de diferencias entre plataformas
        """
        platform_differences = {}
        
        try:
            # Verificar si hay suficientes plataformas para comparar
            platforms = data['platform'].unique()
            
            if len(platforms) < 2:
                return {"message": "Se necesitan al menos 2 plataformas para comparar"}
            
            # Comparar engagement rate entre plataformas
            if 'engagement_rate' in data.columns:
                engagement_by_platform = data.groupby('platform')['engagement_rate'].agg(['mean', 'median', 'std']).reset_index()
                
                # Convertir a diccionario
                engagement_stats = {}
                for _, row in engagement_by_platform.iterrows():
                    platform = row['platform']
                    engagement_stats[platform] = {
                        'mean_engagement': float(row['mean']),
                        'median_engagement': float(row['median']),
                        'std_engagement': float(row['std'])
                    }
                
                platform_differences['engagement_comparison'] = engagement_stats
            
            # Comparar factores de éxito entre plataformas
            common_features = [f for f in self.common_features if f in data.columns]
            
            if common_features and 'engagement_rate' in data.columns:
                feature_importance_by_platform = {}
                
                for platform in platforms:
                    platform_data = data[data['platform'] == platform]
                    
                    if len(platform_data) < 10:
                        continue
                    
                    # Calcular correlación con engagement_rate
                    numeric_features = platform_data[common_features + ['engagement_rate']].select_dtypes(include=['number']).columns
                    
                    if len(numeric_features) > 1 and 'engagement_rate' in numeric_features:
                        correlations = platform_data[numeric_features].corr()['engagement_rate'].drop('engagement_rate')
                        
                        # Ordenar por valor absoluto
                        sorted_correlations = correlations.abs().sort_values(ascending=False)
                        
                        # Tomar las 5 correlaciones más fuertes
                        top_correlations = {}
                        for feature in sorted_correlations.index[:5]:
                            top_correlations[feature] = float(correlations[feature])
                        
                        feature_importance_by_platform[platform] = top_correlations
                
                platform_differences['feature_importance_comparison'] = feature_importance_by_platform
            
            # Comparar métricas de rendimiento entre plataformas
            performance_metrics = {}
            
            for platform in platforms:
                platform_metrics = self.platform_metrics.get(platform, [])
                available_metrics = [m for m in platform_metrics if m in data.columns]
                
                if not available_metrics:
                    continue
                
                platform_data = data[data['platform'] == platform]
                
                if len(platform_data) < 10:
                    continue
                
                # Calcular estadísticas para cada métrica
                metric_stats = {}
                for metric in available_metrics:
                    if pd.api.types.is_numeric_dtype(platform_data[metric]):
                        stats = platform_data[metric].agg(['mean', 'median', 'std']).to_dict()
                        metric_stats[metric] = {
                            'mean': float(stats['mean']),
                            'median': float(stats['median']),
                                                        'std': float(stats['std'])
                        }
                
                performance_metrics[platform] = metric_stats
            
            platform_differences['performance_metrics_comparison'] = performance_metrics
            
            # Analizar diferencias en horarios óptimos de publicación
            if 'hour_of_day' in data.columns and 'engagement_rate' in data.columns:
                optimal_hours = {}
                
                for platform in platforms:
                    platform_data = data[data['platform'] == platform]
                    
                    if len(platform_data) < 24:  # Al menos un dato por hora
                        continue
                    
                    # Agrupar por hora y calcular engagement promedio
                    hourly_engagement = platform_data.groupby('hour_of_day')['engagement_rate'].mean().reset_index()
                    
                    # Ordenar por engagement
                    hourly_engagement = hourly_engagement.sort_values('engagement_rate', ascending=False)
                    
                    # Tomar las 3 mejores horas
                    top_hours = hourly_engagement.head(3)['hour_of_day'].tolist()
                    
                    optimal_hours[platform] = top_hours
                
                platform_differences['optimal_posting_hours'] = optimal_hours
            
            # Analizar diferencias en días óptimos de publicación
            if 'day_of_week' in data.columns and 'engagement_rate' in data.columns:
                optimal_days = {}
                day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                
                for platform in platforms:
                    platform_data = data[data['platform'] == platform]
                    
                    if len(platform_data) < 7:  # Al menos un dato por día
                        continue
                    
                    # Agrupar por día y calcular engagement promedio
                    daily_engagement = platform_data.groupby('day_of_week')['engagement_rate'].mean().reset_index()
                    
                    # Ordenar por engagement
                    daily_engagement = daily_engagement.sort_values('engagement_rate', ascending=False)
                    
                    # Tomar los 2 mejores días
                    top_days_idx = daily_engagement.head(2)['day_of_week'].tolist()
                    top_days = [day_names[idx] for idx in top_days_idx]
                    
                    optimal_days[platform] = top_days
                
                platform_differences['optimal_posting_days'] = optimal_days
            
            # Analizar diferencias en tipos de contenido más efectivos
            if 'content_type' in data.columns and 'engagement_rate' in data.columns:
                effective_content_types = {}
                
                for platform in platforms:
                    platform_data = data[data['platform'] == platform]
                    
                    if len(platform_data) < 10:
                        continue
                    
                    # Agrupar por tipo de contenido y calcular engagement promedio
                    content_engagement = platform_data.groupby('content_type')['engagement_rate'].mean().reset_index()
                    
                    # Ordenar por engagement
                    content_engagement = content_engagement.sort_values('engagement_rate', ascending=False)
                    
                    # Tomar los 3 mejores tipos de contenido
                    top_content_types = content_engagement.head(3)['content_type'].tolist()
                    
                    effective_content_types[platform] = top_content_types
                
                platform_differences['effective_content_types'] = effective_content_types
            
            return platform_differences
            
        except Exception as e:
            self.logger.error(f"Error al analizar diferencias entre plataformas: {str(e)}")
            return {"error": f"Error al analizar diferencias: {str(e)}"}
    
    def predict(self, content_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice el rendimiento de contenido en diferentes plataformas.
        
        Args:
            content_data: DataFrame con datos de contenido para predecir
            
        Returns:
            Predicciones de rendimiento por plataforma
        """
        self.logger.info("Iniciando predicción de rendimiento de contenido")
        
        # Verificar si hay modelos entrenados
        if not self.platform_models:
            return {"error": "No hay modelos entrenados disponibles"}
        
        # Preprocesar datos
        processed_data = self.preprocess_data(content_data)
        
        if processed_data.empty:
            return {"error": "No hay datos válidos para realizar predicciones"}
        
        # Resultados de predicción
        prediction_results = {}
        
        try:
            # Realizar predicciones por plataforma
            for platform in processed_data['platform'].unique():
                if platform not in self.supported_platforms or platform not in self.platform_models:
                    continue
                
                platform_data = processed_data[processed_data['platform'] == platform]
                
                if platform_data.empty:
                    continue
                
                # Obtener información del preprocesador
                preprocessor_info = self.preprocessors.get(platform, {})
                features = preprocessor_info.get('features', [])
                target_metric = preprocessor_info.get('target_metric', 'engagement_rate')
                
                if not features:
                    continue
                
                # Filtrar características disponibles
                available_features = [f for f in features if f in platform_data.columns]
                
                if not available_features:
                    continue
                
                # Preparar datos para predicción
                X = platform_data[available_features]
                
                # Obtener modelo
                pipeline = self.platform_models[platform]
                
                # Realizar predicción
                predictions = pipeline.predict(X)
                
                # Guardar predicciones
                platform_data[f'predicted_{target_metric}'] = predictions
                
                # Calcular estadísticas de predicción
                prediction_stats = {
                    'mean': float(predictions.mean()),
                    'median': float(np.median(predictions)),
                    'min': float(predictions.min()),
                    'max': float(predictions.max()),
                    'std': float(predictions.std()),
                    'target_metric': target_metric,
                    'sample_size': len(platform_data)
                }
                
                prediction_results[platform] = prediction_stats
            
            # Realizar predicciones cruzadas entre plataformas si hay modelo disponible
            if self.cross_platform_model and len(processed_data['platform'].unique()) > 1:
                self.logger.info("Realizando predicciones cruzadas entre plataformas")
                
                # Usar solo características comunes entre plataformas
                common_available = [f for f in self.common_features if f in processed_data.columns]
                common_available.append('platform')  # Añadir plataforma como característica
                
                if common_available:
                    X_cross = processed_data[common_available]
                    
                    # Realizar predicción
                    cross_predictions = self.cross_platform_model.predict(X_cross)
                    
                    # Guardar predicciones
                    processed_data['predicted_cross_platform_engagement'] = cross_predictions
                    
                    # Calcular estadísticas de predicción
                    cross_prediction_stats = {
                        'mean': float(cross_predictions.mean()),
                        'median': float(np.median(cross_predictions)),
                        'min': float(cross_predictions.min()),
                        'max': float(cross_predictions.max()),
                        'std': float(cross_predictions.std()),
                        'target_metric': 'engagement_rate',
                        'sample_size': len(processed_data)
                    }
                    
                    prediction_results['cross_platform'] = cross_prediction_stats
            
            # Generar recomendaciones basadas en predicciones
            recommendations = self._generate_recommendations(processed_data, prediction_results)
            prediction_results['recommendations'] = recommendations
            
            return prediction_results
            
        except Exception as e:
            self.logger.error(f"Error en predicción de rendimiento: {str(e)}")
            return {"error": f"Error en predicción: {str(e)}"}
    
    def _generate_recommendations(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendaciones para optimizar el rendimiento del contenido.
        
        Args:
            data: DataFrame con datos de contenido y predicciones
            predictions: Resultados de predicción por plataforma
            
        Returns:
            Recomendaciones de optimización
        """
        recommendations = {}
        
        try:
            # Recomendación de plataforma óptima
            if len(predictions) > 1 and all('mean' in pred for pred in predictions.values() if isinstance(pred, dict)):
                platform_scores = {}
                
                for platform, stats in predictions.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        platform_scores[platform] = stats['mean']
                
                if platform_scores:
                    # Ordenar plataformas por predicción media
                    sorted_platforms = sorted(
                        platform_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Recomendar las 3 mejores plataformas
                    top_platforms = [p[0] for p in sorted_platforms[:3] if p[0] != 'cross_platform']
                    
                    recommendations['optimal_platforms'] = top_platforms
            
            # Recomendaciones específicas por plataforma
            platform_recommendations = {}
            
            for platform in data['platform'].unique():
                if platform not in self.supported_platforms or platform not in self.platform_models:
                    continue
                
                platform_data = data[data['platform'] == platform]
                
                if platform_data.empty:
                    continue
                
                # Obtener información del preprocesador
                preprocessor_info = self.preprocessors.get(platform, {})
                features = preprocessor_info.get('features', [])
                
                if not features:
                    continue
                
                # Recomendaciones específicas para la plataforma
                platform_recs = {}
                
                # Recomendación de duración óptima (si aplica)
                if 'duration' in platform_data.columns and len(platform_data) > 5:
                    target_metric = preprocessor_info.get('target_metric', 'engagement_rate')
                    predicted_metric = f'predicted_{target_metric}'
                    
                    if predicted_metric in platform_data.columns:
                        # Agrupar por rangos de duración y calcular predicción promedio
                        platform_data['duration_range'] = pd.cut(
                            platform_data['duration'],
                            bins=[0, 15, 30, 60, 120, 300, 600, float('inf')],
                            labels=['0-15s', '15-30s', '30-60s', '1-2min', '2-5min', '5-10min', '10min+']
                        )
                        
                        duration_performance = platform_data.groupby('duration_range')[predicted_metric].mean().reset_index()
                        
                        # Ordenar por rendimiento
                        duration_performance = duration_performance.sort_values(predicted_metric, ascending=False)
                        
                        # Recomendar mejor rango de duración
                        if not duration_performance.empty:
                            optimal_duration = duration_performance.iloc[0]['duration_range']
                            platform_recs['optimal_duration'] = str(optimal_duration)
                
                # Recomendación de características óptimas
                for feature in ['hashtags_count', 'cta_type', 'content_type', 'emotional_tone']:
                    if feature in platform_data.columns and len(platform_data[feature].unique()) > 1:
                        target_metric = preprocessor_info.get('target_metric', 'engagement_rate')
                        predicted_metric = f'predicted_{target_metric}'
                        
                        if predicted_metric in platform_data.columns:
                            # Agrupar por característica y calcular predicción promedio
                            feature_performance = platform_data.groupby(feature)[predicted_metric].mean().reset_index()
                            
                            # Ordenar por rendimiento
                            feature_performance = feature_performance.sort_values(predicted_metric, ascending=False)
                            
                            # Recomendar mejor valor
                            if not feature_performance.empty:
                                optimal_value = feature_performance.iloc[0][feature]
                                
                                # Convertir a string si es necesario
                                if not isinstance(optimal_value, str):
                                    optimal_value = str(optimal_value)
                                
                                feature_name = feature.replace('_count', '').replace('_', ' ')
                                platform_recs[f'optimal_{feature_name}'] = optimal_value
                
                # Añadir recomendaciones para la plataforma
                if platform_recs:
                    platform_recommendations[platform] = platform_recs
            
            recommendations['platform_specific'] = platform_recommendations
            
            # Recomendaciones generales
            general_recommendations = []
            
            # Recomendar mejoras basadas en factores de éxito
            if hasattr(self, 'success_factors') and self.success_factors:
                for platform, factors in self.success_factors.items():
                    if 'top_factors' in factors:
                        top_factor = next(iter(factors['top_factors']))
                        general_recommendations.append(
                            f"Para {platform}, optimiza principalmente '{top_factor}' para mejorar el rendimiento."
                        )
            
            # Recomendar horarios óptimos si están disponibles
            if hasattr(self, 'optimal_hours') and self.optimal_hours:
                for platform, hours in self.optimal_hours.items():
                    if hours:
                        hour_str = ', '.join([f"{h}:00" for h in hours])
                        general_recommendations.append(
                            f"Para {platform}, publica preferentemente a las {hour_str} para maximizar el engagement."
                        )
            
            # Añadir recomendaciones generales
            if general_recommendations:
                recommendations['general_tips'] = general_recommendations
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones: {str(e)}")
            return {"error": f"Error al generar recomendaciones: {str(e)}"}
    
    def visualize_platform_comparison(self, data: pd.DataFrame, output_path: str = None) -> Dict[str, str]:
        """
        Genera visualizaciones comparativas del rendimiento entre plataformas.
        
        Args:
            data: DataFrame con datos de contenido y métricas
            output_path: Ruta opcional para guardar las visualizaciones
            
        Returns:
            Rutas a las visualizaciones generadas
        """
        visualization_paths = {}
        
        try:
            # Preprocesar datos
            processed_data = self.preprocess_data(data)
            
            if processed_data.empty:
                return {"error": "No hay datos válidos para visualizar"}
            
            # Crear directorio de salida si no existe
            if output_path:
                os.makedirs(output_path, exist_ok=True)
            else:
                output_path = os.path.join(os.path.dirname(self.model_path), 'visualizations')
                os.makedirs(output_path, exist_ok=True)
            
            # Visualización 1: Comparación de engagement rate entre plataformas
            if 'engagement_rate' in processed_data.columns:
                plt.figure(figsize=(12, 6))
                
                # Crear boxplot
                sns_plot = sns.boxplot(x='platform', y='engagement_rate', data=processed_data)
                sns_plot.set_title('Comparación de Engagement Rate entre Plataformas')
                sns_plot.set_xlabel('Plataforma')
                sns_plot.set_ylabel('Engagement Rate (%)')
                
                # Rotar etiquetas del eje x
                plt.xticks(rotation=45)
                
                # Ajustar diseño
                plt.tight_layout()
                
                # Guardar figura
                engagement_path = os.path.join(output_path, 'platform_engagement_comparison.png')
                plt.savefig(engagement_path)
                plt.close()
                
                visualization_paths['engagement_comparison'] = engagement_path
            
            # Visualización 2: Matriz de correlación de características por plataforma
            for platform in processed_data['platform'].unique():
                platform_data = processed_data[processed_data['platform'] == platform]
                
                if len(platform_data) < 10:
                    continue
                
                # Obtener características numéricas
                numeric_data = platform_data.select_dtypes(include=['number'])
                
                if len(numeric_data.columns) < 2:
                    continue
                
                # Calcular matriz de correlación
                correlation_matrix = numeric_data.corr()
                
                # Crear mapa de calor
                plt.figure(figsize=(12, 10))
                sns_plot = sns.heatmap(
                    correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt='.2f',
                    linewidths=0.5,
                    vmin=-1, 
                    vmax=1
                )
                sns_plot.set_title(f'Matriz de Correlación para {platform.capitalize()}')
                
                # Ajustar diseño
                plt.tight_layout()
                
                # Guardar figura
                correlation_path = os.path.join(output_path, f'{platform}_correlation_matrix.png')
                plt.savefig(correlation_path)
                plt.close()
                
                visualization_paths[f'{platform}_correlation'] = correlation_path
            
            # Visualización 3: Rendimiento por hora del día
            if 'hour_of_day' in processed_data.columns and 'engagement_rate' in processed_data.columns:
                plt.figure(figsize=(14, 7))
                
                # Agrupar por plataforma y hora
                hourly_data = processed_data.groupby(['platform', 'hour_of_day'])['engagement_rate'].mean().reset_index()
                
                # Crear gráfico de líneas
                sns_plot = sns.lineplot(
                    x='hour_of_day', 
                    y='engagement_rate', 
                    hue='platform', 
                    data=hourly_data,
                    marker='o'
                )
                sns_plot.set_title('Engagement Rate por Hora del Día')
                sns_plot.set_xlabel('Hora del Día')
                sns_plot.set_ylabel('Engagement Rate Promedio (%)')
                sns_plot.set_xticks(range(0, 24))
                
                # Añadir leyenda
                plt.legend(title='Plataforma')
                
                # Ajustar diseño
                plt.tight_layout()
                
                # Guardar figura
                hourly_path = os.path.join(output_path, 'hourly_engagement.png')
                plt.savefig(hourly_path)
                plt.close()
                
                visualization_paths['hourly_engagement'] = hourly_path
            
            # Visualización 4: Comparación de tipos de contenido
            if 'content_type' in processed_data.columns and 'engagement_rate' in processed_data.columns:
                plt.figure(figsize=(14, 8))
                
                # Agrupar por plataforma y tipo de contenido
                content_data = processed_data.groupby(['platform', 'content_type'])['engagement_rate'].mean().reset_index()
                
                # Crear gráfico de barras
                sns_plot = sns.barplot(
                    x='content_type', 
                    y='engagement_rate', 
                    hue='platform', 
                    data=content_data
                )
                sns_plot.set_title('Engagement Rate por Tipo de Contenido')
                sns_plot.set_xlabel('Tipo de Contenido')
                sns_plot.set_ylabel('Engagement Rate Promedio (%)')
                
                # Rotar etiquetas del eje x
                plt.xticks(rotation=45)
                
                # Añadir leyenda
                plt.legend(title='Plataforma')
                
                # Ajustar diseño
                plt.tight_layout()
                
                # Guardar figura
                content_path = os.path.join(output_path, 'content_type_comparison.png')
                plt.savefig(content_path)
                plt.close()
                
                visualization_paths['content_type_comparison'] = content_path
            
            return visualization_paths
            
        except Exception as e:
            self.logger.error(f"Error al generar visualizaciones: {str(e)}")
            return {"error": f"Error al generar visualizaciones: {str(e)}"}
    
    def save_model(self, model_path: str = None) -> bool:
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            model_path: Ruta opcional para guardar el modelo
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Usar ruta proporcionada o la predeterminada
            save_path = model_path or self.model_path
            
            if not save_path:
                save_path = os.path.join(os.getcwd(), 'models', 'platform_performance_model')
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Guardar modelos específicos por plataforma
            for platform, model in self.platform_models.items():
                platform_path = f"{save_path}_{platform}.joblib"
                joblib.dump(model, platform_path)
            
            # Guardar modelo de rendimiento cruzado
            if self.cross_platform_model:
                cross_platform_path = f"{save_path}_cross_platform.joblib"
                joblib.dump(self.cross_platform_model, cross_platform_path)
            
            # Guardar metadatos y preprocesadores
            metadata = {
                'supported_platforms': self.supported_platforms,
                'platform_features': self.platform_features,
                'platform_metrics': self.platform_metrics,
                'common_features': self.common_features,
                'categorical_features': self.categorical_features,
                'preprocessors': self.preprocessors,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = f"{save_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Modelo guardado correctamente en {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar modelo: {str(e)}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Carga un modelo previamente entrenado desde disco.
        
        Args:
            model_path: Ruta al modelo guardado
            
        Returns:
            True si se cargó correctamente, False en caso contrario
        """
        try:
            # Verificar si existe el directorio
            if not os.path.exists(os.path.dirname(model_path)):
                self.logger.error(f"Directorio no encontrado: {os.path.dirname(model_path)}")
                return False
            
            # Cargar metadatos
            metadata_path = f"{model_path}_metadata.json"
            if not os.path.exists(metadata_path):
                self.logger.error(f"Archivo de metadatos no encontrado: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Actualizar atributos
            self.supported_platforms = metadata.get('supported_platforms', self.supported_platforms)
            self.platform_features = metadata.get('platform_features', self.platform_features)
            self.platform_metrics = metadata.get('platform_metrics', self.platform_metrics)
            self.common_features = metadata.get('common_features', self.common_features)
            self.categorical_features = metadata.get('categorical_features', self.categorical_features)
            self.preprocessors = metadata.get('preprocessors', {})
            
            # Cargar modelos específicos por plataforma
            self.platform_models = {}
            for platform in self.supported_platforms:
                platform_path = f"{model_path}_{platform}.joblib"
                if os.path.exists(platform_path):
                    self.platform_models[platform] = joblib.load(platform_path)
            
            # Cargar modelo de rendimiento cruzado
            cross_platform_path = f"{model_path}_cross_platform.joblib"
            if os.path.exists(cross_platform_path):
                self.cross_platform_model = joblib.load(cross_platform_path)
            
            self.logger.info(f"Modelo cargado correctamente desde {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al cargar modelo: {str(e)}")
            return False