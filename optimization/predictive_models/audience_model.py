"""
Modelo predictivo para análisis y segmentación de audiencia en plataformas multimedia.
Este modelo identifica patrones demográficos, preferencias de contenido y comportamientos de engagement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os
import json
from .base_model import BaseModel

class AudienceModel(BaseModel):
    """
    Modelo para análisis, segmentación y predicción de comportamientos de audiencia.
    
    Capacidades:
    - Segmentación demográfica y psicográfica
    - Identificación de patrones de engagement
    - Predicción de respuesta a diferentes tipos de contenido
    - Análisis de retención y lealtad
    - Recomendaciones para optimización de contenido por segmento
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el modelo de audiencia.
        
        Args:
            model_path: Ruta opcional al modelo pre-entrenado
        """
        super().__init__(model_name="audience_model", model_path=model_path)
        self.segment_clusters = None
        self.engagement_model = None
        self.retention_model = None
        self.content_preference_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
        # Configuración de características
        self.audience_features = [
            'age_group', 'gender', 'location', 'device_type', 
            'watch_time', 'engagement_rate', 'subscription_status',
            'return_frequency', 'content_categories', 'time_of_day',
            'session_duration', 'click_through_rate', 'completion_rate',
            'social_activity', 'platform_preference'
        ]
        
        self.categorical_features = ['age_group', 'gender', 'location', 'device_type', 
                                    'content_categories', 'platform_preference']
        
        self.target_features = ['engagement_score', 'retention_score', 'conversion_rate']
        
        # Inicializar modelos si existe la ruta
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos de audiencia para el entrenamiento o predicción.
        
        Args:
            data: DataFrame con datos de audiencia
            
        Returns:
            DataFrame preprocesado
        """
        # Verificar columnas requeridas
        required_columns = ['user_id', 'timestamp'] + self.audience_features
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Faltan columnas en los datos: {missing_columns}")
            # Agregar columnas faltantes con valores predeterminados
            for col in missing_columns:
                if col == 'user_id':
                    data[col] = [f"user_{i}" for i in range(len(data))]
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
                # Para características que pueden tener múltiples valores (como content_categories)
                if col == 'content_categories' or col == 'platform_preference':
                    # Dividir valores separados por comas
                    unique_categories = set()
                    for categories in processed_data[col].dropna():
                        if isinstance(categories, str):
                            for category in categories.split(','):
                                unique_categories.add(category.strip())
                    
                    # Crear columnas para cada categoría
                    for category in unique_categories:
                        col_name = f"{col}_{category}"
                        processed_data[col_name] = processed_data[col].apply(
                            lambda x: 1 if isinstance(x, str) and category in x.split(',') else 0
                        )
                else:
                    # One-hot encoding estándar para otras características categóricas
                    dummies = pd.get_dummies(processed_data[col], prefix=col)
                    processed_data = pd.concat([processed_data, dummies], axis=1)
        
        # Calcular características derivadas
        
        # Calcular engagement_score si no existe
        if 'engagement_score' not in processed_data.columns:
            if all(col in processed_data.columns for col in ['engagement_rate', 'completion_rate', 'social_activity']):
                processed_data['engagement_score'] = (
                    processed_data['engagement_rate'] * 0.4 + 
                    processed_data['completion_rate'] * 0.4 + 
                    processed_data['social_activity'] * 0.2
                )
        
        # Calcular retention_score si no existe
        if 'retention_score' not in processed_data.columns:
            if all(col in processed_data.columns for col in ['return_frequency', 'session_duration']):
                processed_data['retention_score'] = (
                    processed_data['return_frequency'] * 0.6 + 
                    processed_data['session_duration'] * 0.4
                )
        
        # Ordenar por usuario y tiempo
        processed_data = processed_data.sort_values(['user_id', 'timestamp'])
        
        return processed_data
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrena el modelo de audiencia con datos históricos.
        
        Args:
            training_data: DataFrame con datos históricos de audiencia
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento del modelo de audiencia")
        
        # Preprocesar datos
        processed_data = self.preprocess_data(training_data)
        
        # Verificar si hay suficientes datos
        if len(processed_data) < 10:
            return {"error": "Insuficientes datos para entrenar el modelo"}
        
        try:
            # Preparar características para clustering
            # Seleccionar solo características numéricas para clustering
            numeric_features = [col for col in processed_data.columns 
                               if col not in ['user_id', 'timestamp'] + self.categorical_features
                               and pd.api.types.is_numeric_dtype(processed_data[col])]
            
            X_cluster = processed_data[numeric_features].fillna(0)
            
            # Normalizar datos
            X_scaled = self.scaler.fit_transform(X_cluster)
            
            # Aplicar PCA para visualización
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Aplicar clustering
            self.kmeans.fit(X_scaled)
            cluster_labels = self.kmeans.labels_
            
            # Guardar resultados de clustering
            processed_data['segment'] = cluster_labels
            self.segment_clusters = {
                'cluster_centers': self.kmeans.cluster_centers_,
                'pca_components': self.pca.components_,
                'feature_names': numeric_features
            }
            
            # Entrenar modelo de engagement
            if 'engagement_score' in processed_data.columns:
                X_engagement = processed_data[numeric_features].fillna(0)
                y_engagement = processed_data['engagement_score']
                
                self.engagement_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.engagement_model.fit(X_engagement, y_engagement)
                
                # Evaluar modelo
                engagement_score = self.engagement_model.score(X_engagement, y_engagement)
                self.logger.info(f"Modelo de engagement entrenado. R²: {engagement_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'engagement_score' para entrenar el modelo")
            
            # Entrenar modelo de retención
            if 'retention_score' in processed_data.columns:
                X_retention = processed_data[numeric_features].fillna(0)
                y_retention = processed_data['retention_score']
                
                self.retention_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.retention_model.fit(X_retention, y_retention)
                
                # Evaluar modelo
                retention_score = self.retention_model.score(X_retention, y_retention)
                self.logger.info(f"Modelo de retención entrenado. R²: {retention_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'retention_score' para entrenar el modelo")
            
            # Entrenar modelo de preferencia de contenido
            if 'content_preference' in processed_data.columns:
                X_preference = processed_data[numeric_features].fillna(0)
                y_preference = processed_data['content_preference']
                
                self.content_preference_model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.content_preference_model.fit(X_preference, y_preference)
                
                # Evaluar modelo
                preference_score = self.content_preference_model.score(X_preference, y_preference)
                self.logger.info(f"Modelo de preferencia de contenido entrenado. Precisión: {preference_score:.4f}")
            
            # Guardar modelo
            self.save_model()
            
            # Generar visualización de segmentos
            self._generate_segment_visualization(X_pca, cluster_labels, processed_data)
            
            # Analizar segmentos
            segment_analysis = self._analyze_segments(processed_data)
            
            # Preparar métricas de entrenamiento
            metrics = {
                "segments_found": self.kmeans.n_clusters,
                "segment_sizes": pd.Series(cluster_labels).value_counts().to_dict(),
                "pca_explained_variance": self.pca.explained_variance_ratio_.tolist(),
                "segment_analysis": segment_analysis
            }
            
            if self.engagement_model:
                metrics["engagement_model_r2"] = engagement_score
                metrics["engagement_feature_importance"] = dict(zip(
                    numeric_features, 
                    self.engagement_model.feature_importances_
                ))
            
            if self.retention_model:
                metrics["retention_model_r2"] = retention_score
                metrics["retention_feature_importance"] = dict(zip(
                    numeric_features, 
                    self.retention_model.feature_importances_
                ))
            
            if self.content_preference_model:
                metrics["preference_model_accuracy"] = preference_score
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento del modelo de audiencia: {str(e)}")
            return {"error": f"Error en entrenamiento: {str(e)}"}
    
    def _generate_segment_visualization(self, X_pca: np.ndarray, 
                                       cluster_labels: np.ndarray,
                                       data: pd.DataFrame) -> str:
        """
        Genera visualización de segmentos de audiencia.
        
        Args:
            X_pca: Datos reducidos con PCA
            cluster_labels: Etiquetas de cluster
            data: DataFrame original
            
        Returns:
            Ruta al archivo de visualización
        """
        try:
            plt.figure(figsize=(12, 10))
            
            # Graficar puntos coloreados por segmento
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                 cmap='viridis', alpha=0.7, s=50)
            
            # Graficar centroides
            centers_pca = self.pca.transform(self.scaler.transform(self.kmeans.cluster_centers_))
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.8, marker='X')
            
            # Añadir leyenda
            plt.colorbar(scatter, label='Segmento')
            
            # Añadir etiquetas
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.title('Segmentos de Audiencia')
            
            # Guardar visualización
            os.makedirs('outputs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_path = f"outputs/audience_segments_{timestamp}.png"
            plt.savefig(viz_path)
            plt.close()
            
            self.logger.info(f"Visualización de segmentos guardada en {viz_path}")
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error al generar visualización: {str(e)}")
            return ""
    
    def _analyze_segments(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza las características de cada segmento de audiencia.
        
        Args:
            data: DataFrame con datos de audiencia y etiquetas de segmento
            
        Returns:
            Análisis de segmentos
        """
        segment_analysis = {}
        
        try:
            # Para cada segmento
            for segment in sorted(data['segment'].unique()):
                segment_data = data[data['segment'] == segment]
                
                # Análisis demográfico
                demographics = {}
                
                # Analizar distribución de edad
                if 'age_group' in segment_data.columns:
                    demographics['age_distribution'] = segment_data['age_group'].value_counts(normalize=True).to_dict()
                
                # Analizar distribución de género
                if 'gender' in segment_data.columns:
                    demographics['gender_distribution'] = segment_data['gender'].value_counts(normalize=True).to_dict()
                
                # Analizar distribución geográfica
                if 'location' in segment_data.columns:
                    demographics['location_distribution'] = segment_data['location'].value_counts(normalize=True).head(5).to_dict()
                
                # Analizar dispositivos
                if 'device_type' in segment_data.columns:
                    demographics['device_distribution'] = segment_data['device_type'].value_counts(normalize=True).to_dict()
                
                # Análisis de comportamiento
                behavior = {}
                
                # Analizar tiempo de visualización
                if 'watch_time' in segment_data.columns:
                    behavior['avg_watch_time'] = segment_data['watch_time'].mean()
                    behavior['max_watch_time'] = segment_data['watch_time'].max()
                
                # Analizar tasa de engagement
                if 'engagement_rate' in segment_data.columns:
                    behavior['avg_engagement_rate'] = segment_data['engagement_rate'].mean()
                
                # Analizar frecuencia de retorno
                if 'return_frequency' in segment_data.columns:
                    behavior['avg_return_frequency'] = segment_data['return_frequency'].mean()
                
                # Analizar duración de sesión
                if 'session_duration' in segment_data.columns:
                    behavior['avg_session_duration'] = segment_data['session_duration'].mean()
                
                # Analizar tasa de clics
                if 'click_through_rate' in segment_data.columns:
                    behavior['avg_ctr'] = segment_data['click_through_rate'].mean()
                
                # Analizar tasa de finalización
                if 'completion_rate' in segment_data.columns:
                    behavior['avg_completion_rate'] = segment_data['completion_rate'].mean()
                
                # Analizar preferencias de contenido
                content_preferences = {}
                
                # Buscar columnas de categorías de contenido
                content_category_cols = [col for col in segment_data.columns if col.startswith('content_categories_')]
                if content_category_cols:
                    # Calcular la suma de cada categoría
                    category_sums = segment_data[content_category_cols].sum()
                    # Normalizar para obtener porcentajes
                    total = category_sums.sum()
                    if total > 0:
                        category_percentages = (category_sums / total).to_dict()
                        # Limpiar nombres de categorías
                        content_preferences['category_preferences'] = {
                            k.replace('content_categories_', ''): v 
                            for k, v in category_percentages.items()
                        }
                
                # Analizar preferencias de plataforma
                platform_preferences = {}
                
                # Buscar columnas de plataformas
                platform_cols = [col for col in segment_data.columns if col.startswith('platform_preference_')]
                if platform_cols:
                    # Calcular la suma de cada plataforma
                    platform_sums = segment_data[platform_cols].sum()
                    # Normalizar para obtener porcentajes
                    total = platform_sums.sum()
                    if total > 0:
                        platform_percentages = (platform_sums / total).to_dict()
                        # Limpiar nombres de plataformas
                        platform_preferences['platform_distribution'] = {
                            k.replace('platform_preference_', ''): v 
                            for k, v in platform_percentages.items()
                        }
                
                # Determinar nombre descriptivo para el segmento
                segment_name = f"Segmento {segment}"
                
                # Intentar asignar un nombre más descriptivo basado en características dominantes
                descriptors = []
                
                # Por edad
                if 'age_distribution' in demographics and demographics['age_distribution']:
                    top_age = max(demographics['age_distribution'].items(), key=lambda x: x[1])[0]
                    descriptors.append(top_age)
                
                # Por género
                if 'gender_distribution' in demographics and demographics['gender_distribution']:
                    top_gender = max(demographics['gender_distribution'].items(), key=lambda x: x[1])[0]
                    descriptors.append(top_gender)
                
                # Por comportamiento de engagement
                if 'avg_engagement_rate' in behavior:
                    if behavior['avg_engagement_rate'] > 0.7:
                        descriptors.append("Alto engagement")
                    elif behavior['avg_engagement_rate'] < 0.3:
                        descriptors.append("Bajo engagement")
                
                # Por preferencia de contenido
                if 'category_preferences' in content_preferences and content_preferences['category_preferences']:
                    top_category = max(content_preferences['category_preferences'].items(), key=lambda x: x[1])[0]
                    descriptors.append(f"Fan de {top_category}")
                
                # Crear nombre descriptivo si hay suficientes descriptores
                if len(descriptors) >= 2:
                    segment_name = " - ".join(descriptors)
                
                # Guardar análisis completo del segmento
                segment_analysis[segment] = {
                    "segment_name": segment_name,
                    "size": len(segment_data),
                    "percentage": len(segment_data) / len(data) * 100,
                    "demographics": demographics,
                    "behavior": behavior,
                    "content_preferences": content_preferences,
                    "platform_preferences": platform_preferences
                }
            
            return segment_analysis
            
        except Exception as e:
            self.logger.error(f"Error al analizar segmentos: {str(e)}")
            return {"error": f"Error al analizar segmentos: {str(e)}"}
    
    def predict_segment(self, user_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice el segmento al que pertenece un usuario o conjunto de usuarios.
        
        Args:
            user_data: DataFrame con datos de usuarios
            
        Returns:
            Predicciones de segmento y características
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Preprocesar datos
            processed_data = self.preprocess_data(user_data)
            
            # Preparar características para predicción
            numeric_features = [col for col in processed_data.columns 
                               if col not in ['user_id', 'timestamp'] + self.categorical_features
                               and pd.api.types.is_numeric_dtype(processed_data[col])]
            
            X = processed_data[numeric_features].fillna(0)
            
            # Normalizar datos
            X_scaled = self.scaler.transform(X)
            
            # Predecir segmento
            segment_labels = self.kmeans.predict(X_scaled)
            
            # Preparar resultados
            results = []
            
            for i, row in processed_data.iterrows():
                user_result = {
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "segment": int(segment_labels[i])
                }
                
                # Obtener nombre descriptivo del segmento si está disponible
                if hasattr(self, 'segment_analysis') and self.segment_analysis:
                    segment_key = str(int(segment_labels[i]))
                    if segment_key in self.segment_analysis:
                        user_result["segment_name"] = self.segment_analysis[segment_key]["segment_name"]
                
                # Predecir engagement si el modelo está disponible
                if self.engagement_model:
                    X_engagement = X.iloc[i:i+1]
                    engagement_pred = self.engagement_model.predict(X_engagement)[0]
                    user_result["predicted_engagement"] = float(engagement_pred)
                    
                    # Categorizar engagement
                    if engagement_pred > 0.7:
                        user_result["engagement_category"] = "Alto"
                    elif engagement_pred > 0.4:
                        user_result["engagement_category"] = "Medio"
                    else:
                        user_result["engagement_category"] = "Bajo"
                
                # Predecir retención si el modelo está disponible
                if self.retention_model:
                    X_retention = X.iloc[i:i+1]
                    retention_pred = self.retention_model.predict(X_retention)[0]
                    user_result["predicted_retention"] = float(retention_pred)
                    
                    # Categorizar retención
                    if retention_pred > 0.7:
                        user_result["retention_category"] = "Alta"
                    elif retention_pred > 0.4:
                        user_result["retention_category"] = "Media"
                    else:
                        user_result["retention_category"] = "Baja"
                
                # Predecir preferencia de contenido si el modelo está disponible
                if self.content_preference_model:
                    X_preference = X.iloc[i:i+1]
                    preference_pred = self.content_preference_model.predict(X_preference)[0]
                    user_result["content_preference"] = preference_pred
                    
                    # Obtener probabilidades de cada preferencia
                    preference_probs = self.content_preference_model.predict_proba(X_preference)[0]
                    user_result["preference_probabilities"] = dict(zip(
                        self.content_preference_model.classes_, preference_probs.tolist()
                    ))
                
                results.append(user_result)
            
            return {"predictions": results}
            
        except Exception as e:
            self.logger.error(f"Error en predicción de segmento: {str(e)}")
            return {"error": f"Error en predicción: {str(e)}"}
    
    def generate_segment_recommendations(self, segment_id: int) -> Dict[str, Any]:
        """
        Genera recomendaciones de contenido y estrategia para un segmento específico.
        
        Args:
            segment_id: ID del segmento
            
        Returns:
            Recomendaciones para el segmento
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Verificar si tenemos análisis de segmentos
            if not hasattr(self, 'segment_analysis') or not self.segment_analysis:
                return {"error": "No hay análisis de segmentos disponible"}
            
            segment_key = str(segment_id)
            if segment_key not in self.segment_analysis:
                return {"error": f"No se encontró análisis para el segmento {segment_id}"}
            
            segment_info = self.segment_analysis[segment_key]
            
            # Generar recomendaciones
            recommendations = {
                "segment_id": segment_id,
                "segment_name": segment_info.get("segment_name", f"Segmento {segment_id}"),
                "content_recommendations": [],
                "cta_recommendations": [],
                "timing_recommendations": [],
                "platform_recommendations": []
            }
            
            # Recomendaciones de contenido basadas en preferencias
            if "content_preferences" in segment_info and "category_preferences" in segment_info["content_preferences"]:
                top_categories = sorted(
                    segment_info["content_preferences"]["category_preferences"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                for category, score in top_categories:
                    recommendations["content_recommendations"].append({
                        "category": category,
                        "relevance_score": score,
                        "suggestion": f"Crear contenido de {category} para este segmento"
                    })
            
            # Recomendaciones de CTA basadas en comportamiento
            if "behavior" in segment_info:
                behavior = segment_info["behavior"]
                
                # CTAs basados en engagement
                if "avg_engagement_rate" in behavior:
                    engagement = behavior["avg_engagement_rate"]
                    if engagement > 0.7:
                        recommendations["cta_recommendations"].append({
                            "type": "Participación activa",
                            "suggestion": "Usar CTAs que fomenten la participación: 'Comenta tu experiencia', 'Comparte tu opinión'"
                        })
                    elif engagement > 0.4:
                        recommendations["cta_recommendations"].append({
                            "type": "Educación",
                            "suggestion": "Usar CTAs educativos: 'Aprende más sobre', 'Descubre cómo'"
                        })
                    else:
                        recommendations["cta_recommendations"].append({
                            "type": "Curiosidad",
                            "suggestion": "Usar CTAs que generen curiosidad: '¿Sabías que...?', 'El secreto detrás de...'"
                        })
                
                # CTAs basados en tasa de clics
                if "avg_ctr" in behavior:
                    ctr = behavior["avg_ctr"]
                    if ctr > 0.5:
                        recommendations["cta_recommendations"].append({
                            "type": "Acción directa",
                            "suggestion": "Usar CTAs directos: 'Haz clic ahora', 'Regístrate hoy'"
                        })
                    else:
                        recommendations["cta_recommendations"].append({
                            "type": "Valor percibido",
                            "suggestion": "Usar CTAs con valor: 'Obtén acceso gratuito', 'Descarga tu guía'"
                        })
            
            # Recomendaciones de timing basadas en comportamiento
            if "behavior" in segment_info:
                behavior = segment_info["behavior"]
                
                # Timing basado en tiempo de visualización
                if "avg_watch_time" in behavior:
                    watch_time = behavior["avg_watch_time"]
                    if watch_time < 10:
                        recommendations["timing_recommendations"].append({
                            "type": "CTA temprano",
                            "suggestion": "Colocar CTA principal en los primeros 3-5 segundos"
                        })
                    elif watch_time < 30:
                        recommendations["timing_recommendations"].append({
                            "type": "CTA medio",
                            "suggestion": "Colocar CTA principal entre 5-10 segundos"
                        })
                    else:
                        recommendations["timing_recommendations"].append({
                            "type": "CTA múltiple",
                            "suggestion": "Usar múltiples CTAs: uno temprano (3-5s) y otro al final"
                        })
                
                # Timing basado en tasa de finalización
                if "avg_completion_rate" in behavior:
                    completion = behavior["avg_completion_rate"]
                    if completion < 0.3:
                        recommendations["timing_recommendations"].append({
                            "type": "Contenido corto",
                            "suggestion": "Crear contenido de 15-30 segundos con CTA temprano"
                        })
                    elif completion < 0.7:
                        recommendations["timing_recommendations"].append({
                            "type": "Contenido medio",
                            "suggestion": "Crear contenido de 30-60 segundos con CTA a los 15-20 segundos"
                        })
                    else:
                        recommendations["timing_recommendations"].append({
                            "type": "Contenido largo",
                            "suggestion": "Crear contenido de 1-3 minutos con múltiples CTAs"
                        })
            
            # Recomendaciones de plataforma
            if "platform_preferences" in segment_info and "platform_distribution" in segment_info["platform_preferences"]:
                top_platforms = sorted(
                    segment_info["platform_preferences"]["platform_distribution"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                for platform, score in top_platforms:
                    recommendations["platform_recommendations"].append({
                        "platform": platform,
                        "relevance_score": score,
                        "suggestion": f"Priorizar contenido en {platform} para este segmento"
                    })
            
                        # Recomendaciones demográficas
            if "demographics" in segment_info:
                demographics = segment_info["demographics"]
                
                # Recomendaciones basadas en edad
                if "age_distribution" in demographics and demographics["age_distribution"]:
                    top_age = max(demographics["age_distribution"].items(), key=lambda x: x[1])[0]
                    
                    # Adaptar recomendaciones según grupo de edad
                    if "18-24" in top_age:
                        recommendations["demographic_recommendations"] = [
                            "Utilizar lenguaje informal y actual",
                            "Incorporar referencias a tendencias actuales",
                            "Priorizar formatos cortos y dinámicos",
                            "Enfatizar experiencias sociales compartibles"
                        ]
                    elif "25-34" in top_age:
                        recommendations["demographic_recommendations"] = [
                            "Balancear información con entretenimiento",
                            "Enfocarse en soluciones prácticas y valor",
                            "Incorporar narrativas aspiracionales",
                            "Destacar beneficios de ahorro de tiempo"
                        ]
                    elif "35-44" in top_age:
                        recommendations["demographic_recommendations"] = [
                            "Priorizar contenido de calidad sobre cantidad",
                            "Enfatizar aspectos de confiabilidad y experiencia",
                            "Incluir testimonios y casos de éxito",
                            "Balancear innovación con familiaridad"
                        ]
                    elif "45+" in top_age or "45-54" in top_age or "55+" in top_age:
                        recommendations["demographic_recommendations"] = [
                            "Utilizar lenguaje claro y directo",
                            "Enfatizar calidad, durabilidad y confiabilidad",
                            "Incluir explicaciones detalladas",
                            "Priorizar contenido informativo sobre entretenimiento"
                        ]
                    else:
                        recommendations["demographic_recommendations"] = [
                            f"Adaptar contenido al grupo de edad predominante: {top_age}",
                            "Investigar preferencias específicas de este grupo demográfico"
                        ]
            
            # Recomendaciones de formato y estilo
            recommendations["format_recommendations"] = []
            
            # Determinar recomendaciones de formato basadas en comportamiento
            if "behavior" in segment_info:
                behavior = segment_info["behavior"]
                
                # Formato basado en tiempo de visualización
                if "avg_watch_time" in behavior:
                    watch_time = behavior["avg_watch_time"]
                    if watch_time < 15:
                        recommendations["format_recommendations"].append({
                            "type": "Duración",
                            "suggestion": "Crear contenido muy corto (5-15 segundos)"
                        })
                    elif watch_time < 30:
                        recommendations["format_recommendations"].append({
                            "type": "Duración",
                            "suggestion": "Crear contenido corto (15-30 segundos)"
                        })
                    elif watch_time < 60:
                        recommendations["format_recommendations"].append({
                            "type": "Duración",
                            "suggestion": "Crear contenido medio (30-60 segundos)"
                        })
                    else:
                        recommendations["format_recommendations"].append({
                            "type": "Duración",
                            "suggestion": "Crear contenido largo (60+ segundos)"
                        })
                
                # Estilo basado en engagement
                if "avg_engagement_rate" in behavior:
                    engagement = behavior["avg_engagement_rate"]
                    if engagement > 0.7:
                        recommendations["format_recommendations"].append({
                            "type": "Estilo",
                            "suggestion": "Estilo interactivo y participativo"
                        })
                    elif engagement > 0.4:
                        recommendations["format_recommendations"].append({
                            "type": "Estilo",
                            "suggestion": "Estilo narrativo con elementos interactivos"
                        })
                    else:
                        recommendations["format_recommendations"].append({
                            "type": "Estilo",
                            "suggestion": "Estilo directo y conciso con llamados a la acción claros"
                        })
            
            # Recomendaciones de monetización
            recommendations["monetization_recommendations"] = []
            
            # Determinar potencial de monetización basado en comportamiento
            if "behavior" in segment_info:
                behavior = segment_info["behavior"]
                
                # Evaluar potencial de conversión
                conversion_potential = 0
                factors = 0
                
                if "avg_engagement_rate" in behavior:
                    conversion_potential += behavior["avg_engagement_rate"] * 0.3
                    factors += 0.3
                
                if "avg_completion_rate" in behavior:
                    conversion_potential += behavior["avg_completion_rate"] * 0.3
                    factors += 0.3
                
                if "avg_ctr" in behavior:
                    conversion_potential += behavior["avg_ctr"] * 0.4
                    factors += 0.4
                
                if factors > 0:
                    conversion_potential = conversion_potential / factors
                    
                    if conversion_potential > 0.7:
                        recommendations["monetization_recommendations"].append({
                            "type": "Potencial",
                            "suggestion": "Alto potencial de monetización. Priorizar ofertas premium y directas."
                        })
                    elif conversion_potential > 0.4:
                        recommendations["monetization_recommendations"].append({
                            "type": "Potencial",
                            "suggestion": "Potencial medio de monetización. Balancear contenido gratuito con ofertas."
                        })
                    else:
                        recommendations["monetization_recommendations"].append({
                            "type": "Potencial",
                            "suggestion": "Bajo potencial de monetización directa. Enfocarse en construir engagement."
                        })
                
                # Recomendar estrategias específicas de monetización
                if "avg_watch_time" in behavior and "avg_engagement_rate" in behavior:
                    watch_time = behavior["avg_watch_time"]
                    engagement = behavior["avg_engagement_rate"]
                    
                    if watch_time > 60 and engagement > 0.6:
                        recommendations["monetization_recommendations"].append({
                            "type": "Estrategia",
                            "suggestion": "Contenido premium o membresías exclusivas"
                        })
                    elif watch_time > 30 and engagement > 0.4:
                        recommendations["monetization_recommendations"].append({
                            "type": "Estrategia",
                            "suggestion": "Patrocinios integrados o product placement"
                        })
                    else:
                        recommendations["monetization_recommendations"].append({
                            "type": "Estrategia",
                            "suggestion": "Publicidad pre-roll o mid-roll corta"
                        })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones para segmento: {str(e)}")
            return {"error": f"Error al generar recomendaciones: {str(e)}"}
    
    def analyze_audience_trends(self, historical_data: pd.DataFrame, 
                               time_window: int = 90) -> Dict[str, Any]:
        """
        Analiza tendencias en la composición y comportamiento de la audiencia a lo largo del tiempo.
        
        Args:
            historical_data: DataFrame con datos históricos de audiencia
            time_window: Ventana de tiempo en días para el análisis
            
        Returns:
            Análisis de tendencias de audiencia
        """
        try:
            # Preprocesar datos
            processed_data = self.preprocess_data(historical_data)
            
            # Verificar si hay suficientes datos
            if len(processed_data) < 10:
                return {"error": "Insuficientes datos para analizar tendencias"}
            
            # Filtrar por ventana de tiempo
            if "timestamp" in processed_data.columns:
                end_date = processed_data["timestamp"].max()
                start_date = end_date - timedelta(days=time_window)
                processed_data = processed_data[processed_data["timestamp"] >= start_date]
            
            # Verificar si quedan suficientes datos después del filtrado
            if len(processed_data) < 5:
                return {"error": f"Insuficientes datos en la ventana de tiempo de {time_window} días"}
            
            # Agregar columna de semana para análisis temporal
            processed_data["week"] = processed_data["timestamp"].dt.isocalendar().week
            processed_data["month"] = processed_data["timestamp"].dt.month
            
            # Análisis de tendencias demográficas
            demographic_trends = {}
            
            # Analizar tendencias de edad
            if "age_group" in processed_data.columns:
                age_trends = processed_data.groupby(["month", "age_group"]).size().unstack(fill_value=0)
                age_trends = age_trends.div(age_trends.sum(axis=1), axis=0)  # Normalizar por mes
                
                # Convertir a formato más amigable
                age_trend_data = []
                for month in age_trends.index:
                    month_data = {"month": int(month)}
                    for age_group in age_trends.columns:
                        month_data[age_group] = float(age_trends.loc[month, age_group])
                    age_trend_data.append(month_data)
                
                demographic_trends["age_trends"] = age_trend_data
                
                # Identificar grupos de edad en crecimiento/declive
                if len(age_trends) > 1:
                    first_month = age_trends.iloc[0]
                    last_month = age_trends.iloc[-1]
                    
                    growing_age_groups = []
                    declining_age_groups = []
                    
                    for age_group in age_trends.columns:
                        if last_month[age_group] > first_month[age_group] * 1.1:  # 10% de crecimiento
                            growing_age_groups.append({
                                "age_group": age_group,
                                "growth_rate": (last_month[age_group] / first_month[age_group] - 1) * 100
                            })
                        elif last_month[age_group] < first_month[age_group] * 0.9:  # 10% de declive
                            declining_age_groups.append({
                                "age_group": age_group,
                                "decline_rate": (1 - last_month[age_group] / first_month[age_group]) * 100
                            })
                    
                    demographic_trends["growing_age_groups"] = growing_age_groups
                    demographic_trends["declining_age_groups"] = declining_age_groups
            
            # Analizar tendencias de género
            if "gender" in processed_data.columns:
                gender_trends = processed_data.groupby(["month", "gender"]).size().unstack(fill_value=0)
                gender_trends = gender_trends.div(gender_trends.sum(axis=1), axis=0)  # Normalizar por mes
                
                # Convertir a formato más amigable
                gender_trend_data = []
                for month in gender_trends.index:
                    month_data = {"month": int(month)}
                    for gender in gender_trends.columns:
                        month_data[gender] = float(gender_trends.loc[month, gender])
                    gender_trend_data.append(month_data)
                
                demographic_trends["gender_trends"] = gender_trend_data
            
            # Análisis de tendencias de comportamiento
            behavior_trends = {}
            
            # Analizar tendencias de engagement
            if "engagement_rate" in processed_data.columns:
                engagement_trends = processed_data.groupby("month")["engagement_rate"].mean()
                
                # Convertir a formato más amigable
                engagement_trend_data = [
                    {"month": int(month), "avg_engagement": float(rate)}
                    for month, rate in engagement_trends.items()
                ]
                
                behavior_trends["engagement_trends"] = engagement_trend_data
                
                # Calcular tendencia de engagement
                if len(engagement_trends) > 1:
                    first_month = engagement_trends.iloc[0]
                    last_month = engagement_trends.iloc[-1]
                    
                    engagement_change = (last_month / first_month - 1) * 100
                    behavior_trends["engagement_change_percent"] = float(engagement_change)
                    
                    if engagement_change > 5:
                        behavior_trends["engagement_trend"] = "creciente"
                    elif engagement_change < -5:
                        behavior_trends["engagement_trend"] = "decreciente"
                    else:
                        behavior_trends["engagement_trend"] = "estable"
            
            # Analizar tendencias de retención
            if "return_frequency" in processed_data.columns:
                retention_trends = processed_data.groupby("month")["return_frequency"].mean()
                
                # Convertir a formato más amigable
                retention_trend_data = [
                    {"month": int(month), "avg_retention": float(rate)}
                    for month, rate in retention_trends.items()
                ]
                
                behavior_trends["retention_trends"] = retention_trend_data
                
                # Calcular tendencia de retención
                if len(retention_trends) > 1:
                    first_month = retention_trends.iloc[0]
                    last_month = retention_trends.iloc[-1]
                    
                    retention_change = (last_month / first_month - 1) * 100
                    behavior_trends["retention_change_percent"] = float(retention_change)
                    
                    if retention_change > 5:
                        behavior_trends["retention_trend"] = "creciente"
                    elif retention_change < -5:
                        behavior_trends["retention_trend"] = "decreciente"
                    else:
                        behavior_trends["retention_trend"] = "estable"
            
            # Análisis de tendencias de preferencias de contenido
            content_trends = {}
            
            # Buscar columnas de categorías de contenido
            content_category_cols = [col for col in processed_data.columns if col.startswith('content_categories_')]
            if content_category_cols:
                # Analizar tendencias por categoría
                category_trends = {}
                
                for category_col in content_category_cols:
                    category_name = category_col.replace('content_categories_', '')
                    monthly_trends = processed_data.groupby("month")[category_col].mean()
                    
                    # Convertir a formato más amigable
                    trend_data = [
                        {"month": int(month), "preference_score": float(score)}
                        for month, score in monthly_trends.items()
                    ]
                    
                    category_trends[category_name] = trend_data
                
                content_trends["category_trends"] = category_trends
                
                # Identificar categorías en crecimiento/declive
                growing_categories = []
                declining_categories = []
                
                for category, trend_data in category_trends.items():
                    if len(trend_data) > 1:
                        first_month = trend_data[0]["preference_score"]
                        last_month = trend_data[-1]["preference_score"]
                        
                        if last_month > first_month * 1.1:  # 10% de crecimiento
                            growing_categories.append({
                                "category": category,
                                "growth_rate": (last_month / first_month - 1) * 100
                            })
                        elif last_month < first_month * 0.9:  # 10% de declive
                            declining_categories.append({
                                "category": category,
                                "decline_rate": (1 - last_month / first_month) * 100
                            })
                
                content_trends["growing_categories"] = growing_categories
                content_trends["declining_categories"] = declining_categories
            
            # Generar recomendaciones basadas en tendencias
            recommendations = []
            
            # Recomendaciones demográficas
            if "growing_age_groups" in demographic_trends and demographic_trends["growing_age_groups"]:
                top_growing_age = max(demographic_trends["growing_age_groups"], key=lambda x: x["growth_rate"])
                recommendations.append(
                    f"Aumentar contenido dirigido al grupo de edad {top_growing_age['age_group']} que está creciendo a un ritmo del {top_growing_age['growth_rate']:.1f}%"
                )
            
            # Recomendaciones de comportamiento
            if "engagement_trend" in behavior_trends:
                if behavior_trends["engagement_trend"] == "decreciente":
                    recommendations.append(
                        f"Atención: El engagement está disminuyendo ({behavior_trends['engagement_change_percent']:.1f}%). Revisar estrategia de contenido y CTAs."
                    )
                elif behavior_trends["engagement_trend"] == "creciente":
                    recommendations.append(
                        f"El engagement está aumentando ({behavior_trends['engagement_change_percent']:.1f}%). Continuar y amplificar estrategias actuales."
                    )
            
            # Recomendaciones de contenido
            if "growing_categories" in content_trends and content_trends["growing_categories"]:
                top_growing_category = max(content_trends["growing_categories"], key=lambda x: x["growth_rate"])
                recommendations.append(
                    f"Priorizar contenido de la categoría '{top_growing_category['category']}' que muestra un crecimiento del {top_growing_category['growth_rate']:.1f}%"
                )
            
            if "declining_categories" in content_trends and content_trends["declining_categories"]:
                top_declining_category = max(content_trends["declining_categories"], key=lambda x: x["decline_rate"])
                recommendations.append(
                    f"Reevaluar o renovar enfoque para la categoría '{top_declining_category['category']}' que muestra un declive del {top_declining_category['decline_rate']:.1f}%"
                )
            
            # Preparar resultado final
            result = {
                "time_window_days": time_window,
                "data_points_analyzed": len(processed_data),
                "demographic_trends": demographic_trends,
                "behavior_trends": behavior_trends,
                "content_trends": content_trends,
                "recommendations": recommendations
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al analizar tendencias de audiencia: {str(e)}")
            return {"error": f"Error al analizar tendencias: {str(e)}"}