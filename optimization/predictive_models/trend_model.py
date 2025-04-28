"""
Modelo predictivo para análisis y predicción de tendencias en contenido multimedia.
Este modelo identifica patrones de tendencias, predice su duración y potencial de monetización.
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os
import json
from .base_model import BaseModel

class TrendModel(BaseModel):
    """
    Modelo para análisis y predicción de tendencias en contenido multimedia.
    
    Capacidades:
    - Identificación de tendencias emergentes
    - Predicción de duración de tendencias
    - Estimación de potencial de monetización
    - Agrupamiento de tendencias similares
    - Recomendaciones para aprovechar tendencias
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el modelo de tendencias.
        
        Args:
            model_path: Ruta opcional al modelo pre-entrenado
        """
        super().__init__(model_name="trend_model", model_path=model_path)
        self.trend_clusters = None
        self.trend_duration_model = None
        self.trend_monetization_model = None
        self.trend_classifier = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
        # Configuración de características
        self.trend_features = [
            'growth_rate', 'acceleration', 'peak_volume', 'duration_days',
            'engagement_rate', 'conversion_rate', 'platform_spread',
            'demographic_spread', 'geographic_spread', 'sentiment_score',
            'related_trends_count', 'seasonality_score', 'virality_score'
        ]
        
        self.target_features = ['trend_duration', 'monetization_potential']
        
        # Inicializar modelos si existe la ruta
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos de tendencias para el entrenamiento o predicción.
        
        Args:
            data: DataFrame con datos de tendencias
            
        Returns:
            DataFrame preprocesado
        """
        # Verificar columnas requeridas
        required_columns = ['trend_name', 'timestamp'] + self.trend_features
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Faltan columnas en los datos: {missing_columns}")
            # Agregar columnas faltantes con valores predeterminados
            for col in missing_columns:
                if col == 'trend_name':
                    data[col] = 'unknown_trend'
                elif col == 'timestamp':
                    data[col] = datetime.now()
                else:
                    data[col] = 0.0
        
        # Convertir timestamp a datetime si es necesario
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Calcular características derivadas
        processed_data = data.copy()
        
        # Calcular aceleración si no existe
        if 'acceleration' not in processed_data.columns:
            processed_data['acceleration'] = processed_data.groupby('trend_name')['growth_rate'].diff().fillna(0)
        
        # Calcular propagación entre plataformas si no existe
        if 'platform_spread' not in processed_data.columns:
            if 'platforms' in processed_data.columns:
                processed_data['platform_spread'] = processed_data['platforms'].apply(
                    lambda x: len(x.split(',')) if isinstance(x, str) else 1
                )
        
        # Calcular puntuación de viralidad si no existe
        if 'virality_score' not in processed_data.columns:
            if all(col in processed_data.columns for col in ['growth_rate', 'engagement_rate']):
                processed_data['virality_score'] = (
                    processed_data['growth_rate'] * 0.7 + 
                    processed_data['engagement_rate'] * 0.3
                )
        
        # Ordenar por tendencia y tiempo
        processed_data = processed_data.sort_values(['trend_name', 'timestamp'])
        
        return processed_data
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entrena el modelo de tendencias con datos históricos.
        
        Args:
            training_data: DataFrame con datos históricos de tendencias
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento del modelo de tendencias")
        
        # Preprocesar datos
        processed_data = self.preprocess_data(training_data)
        
        # Verificar si hay suficientes datos
        if len(processed_data) < 10:
            return {"error": "Insuficientes datos para entrenar el modelo"}
        
        try:
            # Extraer características para clustering
            cluster_features = [f for f in self.trend_features if f in processed_data.columns]
            X_cluster = processed_data[cluster_features].fillna(0)
            
            # Normalizar datos
            X_scaled = self.scaler.fit_transform(X_cluster)
            
            # Aplicar PCA para visualización
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Aplicar clustering
            self.kmeans.fit(X_scaled)
            cluster_labels = self.kmeans.labels_
            
            # Guardar resultados de clustering
            processed_data['cluster'] = cluster_labels
            self.trend_clusters = {
                'cluster_centers': self.kmeans.cluster_centers_,
                'pca_components': self.pca.components_,
                'feature_names': cluster_features
            }
            
            # Entrenar modelo de duración de tendencias
            if 'trend_duration' in processed_data.columns:
                X_duration = processed_data[cluster_features].fillna(0)
                y_duration = processed_data['trend_duration']
                
                self.trend_duration_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.trend_duration_model.fit(X_duration, y_duration)
                
                # Evaluar modelo
                duration_score = self.trend_duration_model.score(X_duration, y_duration)
                self.logger.info(f"Modelo de duración de tendencias entrenado. R²: {duration_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'trend_duration' para entrenar el modelo de duración")
            
            # Entrenar modelo de potencial de monetización
            if 'monetization_potential' in processed_data.columns:
                X_monetization = processed_data[cluster_features].fillna(0)
                y_monetization = processed_data['monetization_potential']
                
                self.trend_monetization_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.trend_monetization_model.fit(X_monetization, y_monetization)
                
                # Evaluar modelo
                monetization_score = self.trend_monetization_model.score(X_monetization, y_monetization)
                self.logger.info(f"Modelo de potencial de monetización entrenado. R²: {monetization_score:.4f}")
            else:
                self.logger.warning("No se encontró la columna 'monetization_potential' para entrenar el modelo")
            
            # Entrenar clasificador de tipo de tendencia
            if 'trend_type' in processed_data.columns:
                X_type = processed_data[cluster_features].fillna(0)
                y_type = processed_data['trend_type']
                
                self.trend_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                self.trend_classifier.fit(X_type, y_type)
                
                # Evaluar modelo
                type_score = self.trend_classifier.score(X_type, y_type)
                self.logger.info(f"Clasificador de tipo de tendencia entrenado. Precisión: {type_score:.4f}")
            
            # Guardar modelo
            self.save_model()
            
            # Preparar métricas de entrenamiento
            metrics = {
                "clusters_found": self.kmeans.n_clusters,
                "cluster_sizes": pd.Series(cluster_labels).value_counts().to_dict(),
                "pca_explained_variance": self.pca.explained_variance_ratio_.tolist(),
            }
            
            if self.trend_duration_model:
                metrics["duration_model_r2"] = duration_score
                metrics["duration_feature_importance"] = dict(zip(
                    cluster_features, 
                    self.trend_duration_model.feature_importances_
                ))
            
            if self.trend_monetization_model:
                metrics["monetization_model_r2"] = monetization_score
                metrics["monetization_feature_importance"] = dict(zip(
                    cluster_features, 
                    self.trend_monetization_model.feature_importances_
                ))
            
            if self.trend_classifier:
                metrics["classifier_accuracy"] = type_score
                metrics["trend_types"] = list(self.trend_classifier.classes_)
            
            # Generar visualización de clusters
            self._generate_cluster_visualization(X_pca, cluster_labels, processed_data)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento del modelo de tendencias: {str(e)}")
            return {"error": f"Error en entrenamiento: {str(e)}"}
    
    def _generate_cluster_visualization(self, X_pca: np.ndarray, 
                                       cluster_labels: np.ndarray,
                                       data: pd.DataFrame) -> str:
        """
        Genera visualización de clusters de tendencias.
        
        Args:
            X_pca: Datos reducidos con PCA
            cluster_labels: Etiquetas de cluster
            data: DataFrame original
            
        Returns:
            Ruta al archivo de visualización
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Graficar puntos coloreados por cluster
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                 cmap='viridis', alpha=0.7, s=50)
            
            # Graficar centroides
            centers_pca = self.pca.transform(self.scaler.transform(self.kmeans.cluster_centers_))
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.8, marker='X')
            
            # Añadir leyenda
            plt.colorbar(scatter, label='Cluster')
            
            # Añadir etiquetas
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.title('Clusters de Tendencias')
            
            # Guardar visualización
            os.makedirs('outputs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_path = f"outputs/trend_clusters_{timestamp}.png"
            plt.savefig(viz_path)
            plt.close()
            
            self.logger.info(f"Visualización de clusters guardada en {viz_path}")
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error al generar visualización: {str(e)}")
            return ""
    
    def predict_trend_properties(self, trend_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice propiedades de tendencias (duración, potencial de monetización, tipo).
        
        Args:
            trend_data: DataFrame con datos de tendencias
            
        Returns:
            Predicciones de propiedades de tendencias
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Preprocesar datos
            processed_data = self.preprocess_data(trend_data)
            
            # Extraer características
            features = [f for f in self.trend_features if f in processed_data.columns]
            X = processed_data[features].fillna(0)
            
            # Normalizar datos
            X_scaled = self.scaler.transform(X)
            
            # Predecir cluster
            cluster_labels = self.kmeans.predict(X_scaled)
            
            # Preparar resultados
            results = []
            
            for i, row in processed_data.iterrows():
                trend_result = {
                    "trend_name": row["trend_name"],
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "cluster": int(cluster_labels[i])
                }
                
                # Predecir duración si el modelo está disponible
                if self.trend_duration_model:
                    X_duration = X.iloc[i:i+1]
                    duration_pred = self.trend_duration_model.predict(X_duration)[0]
                    trend_result["predicted_duration_days"] = float(duration_pred)
                    
                    # Calcular fecha estimada de fin de tendencia
                    if "timestamp" in row:
                        end_date = row["timestamp"] + timedelta(days=float(duration_pred))
                        trend_result["estimated_end_date"] = end_date.strftime("%Y-%m-%d")
                
                # Predecir potencial de monetización si el modelo está disponible
                if self.trend_monetization_model:
                    X_monetization = X.iloc[i:i+1]
                    monetization_pred = self.trend_monetization_model.predict(X_monetization)[0]
                    trend_result["monetization_potential"] = float(monetization_pred)
                    
                    # Categorizar potencial
                    if monetization_pred > 0.8:
                        trend_result["monetization_category"] = "Alto"
                    elif monetization_pred > 0.5:
                        trend_result["monetization_category"] = "Medio"
                    else:
                        trend_result["monetization_category"] = "Bajo"
                
                # Predecir tipo de tendencia si el clasificador está disponible
                if self.trend_classifier:
                    X_type = X.iloc[i:i+1]
                    type_pred = self.trend_classifier.predict(X_type)[0]
                    trend_result["trend_type"] = type_pred
                    
                    # Obtener probabilidades de cada tipo
                    type_probs = self.trend_classifier.predict_proba(X_type)[0]
                    trend_result["trend_type_probabilities"] = dict(zip(
                        self.trend_classifier.classes_, type_probs.tolist()
                    ))
                
                results.append(trend_result)
            
            return {"predictions": results}
            
        except Exception as e:
            self.logger.error(f"Error en predicción de propiedades de tendencias: {str(e)}")
            return {"error": f"Error en predicción: {str(e)}"}
    
    def identify_emerging_trends(self, trend_data: pd.DataFrame, 
                                threshold: float = 0.7) -> Dict[str, Any]:
        """
        Identifica tendencias emergentes con alto potencial.
        
        Args:
            trend_data: DataFrame con datos de tendencias
            threshold: Umbral de potencial para considerar una tendencia emergente
            
        Returns:
            Tendencias emergentes identificadas
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Obtener predicciones de propiedades
            predictions = self.predict_trend_properties(trend_data)
            
            if "error" in predictions:
                return predictions
            
            # Filtrar tendencias emergentes
            emerging_trends = []
            
            for pred in predictions["predictions"]:
                # Verificar si tiene potencial de monetización
                if "monetization_potential" in pred and pred["monetization_potential"] >= threshold:
                    # Verificar si es una tendencia reciente (últimos 7 días)
                    trend_date = datetime.strptime(pred["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if (datetime.now() - trend_date).days <= 7:
                        emerging_trends.append(pred)
            
            # Ordenar por potencial de monetización
            emerging_trends.sort(key=lambda x: x.get("monetization_potential", 0), reverse=True)
            
            return {
                "emerging_trends": emerging_trends,
                "count": len(emerging_trends),
                "threshold": threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error al identificar tendencias emergentes: {str(e)}")
            return {"error": f"Error al identificar tendencias emergentes: {str(e)}"}
    
    def generate_trend_recommendations(self, trend_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera recomendaciones para aprovechar tendencias.
        
        Args:
            trend_data: DataFrame con datos de tendencias
            
        Returns:
            Recomendaciones para tendencias
        """
        if not self.is_trained():
            return {"error": "El modelo no está entrenado"}
        
        try:
            # Obtener predicciones de propiedades
            predictions = self.predict_trend_properties(trend_data)
            
            if "error" in predictions:
                return predictions
            
            # Identificar tendencias emergentes
            emerging = self.identify_emerging_trends(trend_data)
            
            if "error" in emerging:
                emerging = {"emerging_trends": []}
            
            # Generar recomendaciones
            recommendations = []
            
            # Recomendaciones para tendencias emergentes
            for trend in emerging.get("emerging_trends", []):
                trend_name = trend["trend_name"]
                monetization = trend.get("monetization_potential", 0)
                duration = trend.get("predicted_duration_days", 0)
                trend_type = trend.get("trend_type", "Desconocido")
                
                rec = {
                    "trend_name": trend_name,
                    "priority": "Alta" if monetization > 0.8 else "Media",
                    "recommendation_type": "Tendencia emergente",
                    "actions": []
                }
                
                # Acciones recomendadas según tipo de tendencia
                if trend_type == "Viral":
                    rec["actions"].append(
                        f"Crear contenido inmediatamente para '{trend_name}' con CTAs agresivos"
                    )
                    rec["actions"].append(
                        f"Programar 3-5 piezas de contenido en las próximas 48 horas"
                    )
                    rec["actions"].append(
                        f"Utilizar hashtags: #{trend_name.replace(' ', '')}, #viral, #trending"
                    )
                elif trend_type == "Seasonal":
                    rec["actions"].append(
                        f"Preparar campaña para '{trend_name}' con 7-10 piezas de contenido"
                    )
                    rec["actions"].append(
                        f"Programar publicaciones durante los próximos {max(7, int(duration))} días"
                    )
                    rec["actions"].append(
                        f"Crear CTAs relacionados con la temporada"
                    )
                elif trend_type == "Niche":
                    rec["actions"].append(
                        f"Crear contenido especializado para '{trend_name}'"
                    )
                    rec["actions"].append(
                        f"Enfocar en audiencia específica interesada en este nicho"
                    )
                    rec["actions"].append(
                        f"Utilizar CTAs educativos y de valor agregado"
                    )
                else:
                    rec["actions"].append(
                        f"Crear 2-3 piezas de contenido para '{trend_name}'"
                    )
                    rec["actions"].append(
                        f"Monitorear rendimiento y escalar si hay buena respuesta"
                    )
                
                # Recomendaciones de plataformas
                if "platform_spread" in trend_data.columns:
                    platforms = trend_data.loc[trend_data["trend_name"] == trend_name, "platform_spread"].iloc[0]
                    if platforms > 3:
                        rec["actions"].append(
                            f"Distribuir en múltiples plataformas: YouTube, TikTok, Instagram, Threads"
                        )
                    elif platforms > 1:
                        rec["actions"].append(
                            f"Enfocar en las 2-3 plataformas principales para esta tendencia"
                        )
                    else:
                        rec["actions"].append(
                            f"Concentrar esfuerzos en la plataforma principal para esta tendencia"
                        )
                
                # Recomendaciones de monetización
                if monetization > 0.8:
                    rec["actions"].append(
                        f"Alto potencial de monetización: Implementar CTAs de afiliados y patrocinios"
                    )
                elif monetization > 0.5:
                    rec["actions"].append(
                        f"Potencial medio: Utilizar CTAs para crecimiento de audiencia y engagement"
                    )
                else:
                    rec["actions"].append(
                        f"Bajo potencial monetario: Usar para crecimiento de marca y audiencia"
                    )
                
                recommendations.append(rec)
            
            # Recomendaciones generales basadas en clusters
            if self.trend_clusters:
                cluster_counts = {}
                for pred in predictions["predictions"]:
                    cluster = pred.get("cluster", 0)
                    if cluster in cluster_counts:
                        cluster_counts[cluster] += 1
                    else:
                        cluster_counts[cluster] = 1
                
                # Identificar cluster dominante
                if cluster_counts:
                    dominant_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]
                    
                    # Recomendación general basada en cluster dominante
                    general_rec = {
                        "trend_name": "General",
                        "priority": "Media",
                        "recommendation_type": "Estrategia de cluster",
                        "actions": [
                            f"El cluster {dominant_cluster} es dominante en las tendencias actuales",
                            f"Adaptar estrategia de contenido para alinearse con características de este cluster",
                            f"Considerar redistribución de recursos hacia tendencias en este cluster"
                        ]
                    }
                    
                    recommendations.append(general_rec)
            
            return {
                "recommendations": recommendations,
                "count": len(recommendations),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones: {str(e)}")
            return {"error": f"Error al generar recomendaciones: {str(e)}"}
    
    def analyze_trend_lifecycle(self, historical_data: pd.DataFrame, 
                               trend_name: str) -> Dict[str, Any]:
        """
        Analiza el ciclo de vida completo de una tendencia específica.
        
        Args:
            historical_data: DataFrame con datos históricos
            trend_name: Nombre de la tendencia a analizar
            
        Returns:
            Análisis del ciclo de vida de la tendencia
        """
        try:
            # Filtrar datos para la tendencia específica
            trend_data = historical_data[historical_data["trend_name"] == trend_name].copy()
            
            if trend_data.empty:
                return {"error": f"No se encontraron datos para la tendencia '{trend_name}'"}
            
            # Asegurar que los datos estén ordenados por tiempo
            if "timestamp" in trend_data.columns:
                trend_data = trend_data.sort_values("timestamp")
            
            # Extraer métricas clave
            if "growth_rate" in trend_data.columns:
                growth_rates = trend_data["growth_rate"].tolist()
                peak_growth = trend_data["growth_rate"].max()
                peak_growth_date = trend_data.loc[trend_data["growth_rate"].idxmax(), "timestamp"]
            else:
                growth_rates = []
                peak_growth = None
                peak_growth_date = None
            
            if "engagement_rate" in trend_data.columns:
                engagement_rates = trend_data["engagement_rate"].tolist()
                peak_engagement = trend_data["engagement_rate"].max()
                peak_engagement_date = trend_data.loc[trend_data["engagement_rate"].idxmax(), "timestamp"]
            else:
                engagement_rates = []
                peak_engagement = None
                peak_engagement_date = None
            
            # Determinar fase actual del ciclo de vida
            if not growth_rates:
                lifecycle_phase = "Desconocida"
            elif len(growth_rates) < 3:
                lifecycle_phase = "Emergente"
            elif growth_rates[-1] > 0 and growth_rates[-1] >= growth_rates[-2]:
                lifecycle_phase = "Crecimiento"
            elif growth_rates[-1] > 0 and growth_rates[-1] < growth_rates[-2]:
                lifecycle_phase = "Madurez"
            elif growth_rates[-1] <= 0:
                lifecycle_phase = "Declive"
            else:
                lifecycle_phase = "Indeterminada"
            
            # Calcular duración total
            if "timestamp" in trend_data.columns and len(trend_data) > 1:
                start_date = trend_data["timestamp"].min()
                end_date = trend_data["timestamp"].max()
                duration_days = (end_date - start_date).days
            else:
                duration_days = None
            
            # Generar visualización
            viz_path = ""
            if "timestamp" in trend_data.columns and ("growth_rate" in trend_data.columns or "engagement_rate" in trend_data.columns):
                try:
                    plt.figure(figsize=(12, 6))
                    
                    if "growth_rate" in trend_data.columns:
                        plt.plot(trend_data["timestamp"], trend_data["growth_rate"], 
                                label="Tasa de crecimiento", color="blue")
                    
                    if "engagement_rate" in trend_data.columns:
                        plt.plot(trend_data["timestamp"], trend_data["engagement_rate"], 
                                label="Tasa de engagement", color="green")
                    
                    plt.title(f"Ciclo de vida de la tendencia: {trend_name}")
                    plt.xlabel("Fecha")
                    plt.ylabel("Tasa")
                    plt.legend()
                    plt.grid(True)
                    
                    # Marcar fases del ciclo de vida
                    if len(trend_data) > 3:
                        phases = []
                        phase_dates = []
                        
                        # Simplificación: dividir en cuartiles
                        quartiles = len(trend_data) // 4
                        if quartiles > 0:
                            phases = ["Emergente", "Crecimiento", "Madurez", "Declive"]
                            phase_dates = [
                                trend_data["timestamp"].iloc[0],
                                trend_data["timestamp"].iloc[quartiles],
                                trend_data["timestamp"].iloc[2*quartiles],
                                trend_data["timestamp"].iloc[3*quartiles]
                            ]
                            
                            for i, (phase, date) in enumerate(zip(phases, phase_dates)):
                                plt.axvline(x=date, color='red', linestyle='--', alpha=0.5)
                                plt.text(date, plt.ylim()[1]*0.9, phase, rotation=90, verticalalignment='top')
                    
                    # Guardar visualización
                    os.makedirs('outputs', exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    viz_path = f"outputs/trend_lifecycle_{trend_name.replace(' ', '_')}_{timestamp}.png"
                    plt.savefig(viz_path)
                    plt.close()
                    
                except Exception as e:
                    self.logger.error(f"Error al generar visualización de ciclo de vida: {str(e)}")
            
            # Preparar resultado
            result = {
                "trend_name": trend_name,
                "lifecycle_phase": lifecycle_phase,
                "duration_days": duration_days,
                "data_points": len(trend_data),
                "visualization_path": viz_path
            }
            
            if peak_growth is not None:
                result["peak_growth_rate"] = float(peak_growth)
                result["peak_growth_date"] = peak_growth_date.strftime("%Y-%m-%d %H:%M:%S")
            
            if peak_engagement is not None:
                result["peak_engagement_rate"] = float(peak_engagement)
                result["peak_engagement_date"] = peak_engagement_date.strftime("%Y-%m-%d %H:%M:%S")
            
            # Añadir recomendaciones según fase del ciclo de vida
            recommendations = []
            
            if lifecycle_phase == "Emergente":
                recommendations = [
                    "Crear contenido inmediatamente para capturar la tendencia temprana",
                    "Utilizar CTAs para generar expectativa: 'Sé el primero en...'",
                    "Monitorear crecimiento diariamente para ajustar estrategia"
                ]
            elif lifecycle_phase == "Crecimiento":
                recommendations = [
                    "Aumentar frecuencia de publicación para maximizar alcance",
                    "Implementar CTAs de monetización: afiliados, productos",
                    "Diversificar formatos de contenido en múltiples plataformas"
                ]
            elif lifecycle_phase == "Madurez":
                recommendations = [
                    "Optimizar CTAs para conversión y monetización",
                    "Añadir ángulos únicos para diferenciarse de la competencia",
                    "Preparar estrategia para la fase de declive"
                ]
            elif lifecycle_phase == "Declive":
                                recommendations = [
                    "Reducir frecuencia de publicación sobre esta tendencia",
                    "Enfocar en contenido evergreen relacionado con la temática",
                    "Preparar transición hacia nuevas tendencias emergentes"
                ]
            
            result["recommendations"] = recommendations
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al analizar ciclo de vida de tendencia: {str(e)}")
            return {"error": f"Error al analizar ciclo de vida: {str(e)}"}
    
    def compare_trends(self, historical_data: pd.DataFrame, 
                      trend_names: List[str]) -> Dict[str, Any]:
        """
        Compara múltiples tendencias para identificar patrones y diferencias.
        
        Args:
            historical_data: DataFrame con datos históricos
            trend_names: Lista de nombres de tendencias a comparar
            
        Returns:
            Análisis comparativo de tendencias
        """
        try:
            if not trend_names or len(trend_names) < 2:
                return {"error": "Se requieren al menos dos tendencias para comparar"}
            
            # Filtrar datos para las tendencias especificadas
            trend_data = historical_data[historical_data["trend_name"].isin(trend_names)].copy()
            
            if trend_data.empty:
                return {"error": "No se encontraron datos para las tendencias especificadas"}
            
            # Asegurar que los datos estén ordenados por tiempo
            if "timestamp" in trend_data.columns:
                trend_data = trend_data.sort_values(["trend_name", "timestamp"])
            
            # Analizar cada tendencia individualmente
            trend_analyses = {}
            for trend in trend_names:
                analysis = self.analyze_trend_lifecycle(historical_data, trend)
                if "error" not in analysis:
                    trend_analyses[trend] = analysis
            
            # Generar visualización comparativa
            viz_path = ""
            if "timestamp" in trend_data.columns and "growth_rate" in trend_data.columns:
                try:
                    plt.figure(figsize=(14, 8))
                    
                    for trend in trend_names:
                        trend_subset = trend_data[trend_data["trend_name"] == trend]
                        if not trend_subset.empty:
                            plt.plot(trend_subset["timestamp"], trend_subset["growth_rate"], 
                                    label=f"{trend}", marker='o', markersize=4)
                    
                    plt.title("Comparación de tendencias")
                    plt.xlabel("Fecha")
                    plt.ylabel("Tasa de crecimiento")
                    plt.legend()
                    plt.grid(True)
                    
                    # Guardar visualización
                    os.makedirs('outputs', exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    viz_path = f"outputs/trend_comparison_{timestamp}.png"
                    plt.savefig(viz_path)
                    plt.close()
                    
                except Exception as e:
                    self.logger.error(f"Error al generar visualización comparativa: {str(e)}")
            
            # Calcular métricas comparativas
            comparison_metrics = {}
            
            # Duración promedio
            durations = [analysis.get("duration_days", 0) for analysis in trend_analyses.values() 
                         if "duration_days" in analysis and analysis["duration_days"] is not None]
            if durations:
                comparison_metrics["avg_duration"] = sum(durations) / len(durations)
                comparison_metrics["max_duration"] = max(durations)
                comparison_metrics["min_duration"] = min(durations)
            
            # Crecimiento máximo
            peak_growths = [analysis.get("peak_growth_rate", 0) for analysis in trend_analyses.values() 
                           if "peak_growth_rate" in analysis]
            if peak_growths:
                comparison_metrics["avg_peak_growth"] = sum(peak_growths) / len(peak_growths)
                comparison_metrics["max_peak_growth"] = max(peak_growths)
                comparison_metrics["min_peak_growth"] = min(peak_growths)
            
            # Engagement máximo
            peak_engagements = [analysis.get("peak_engagement_rate", 0) for analysis in trend_analyses.values() 
                               if "peak_engagement_rate" in analysis]
            if peak_engagements:
                comparison_metrics["avg_peak_engagement"] = sum(peak_engagements) / len(peak_engagements)
                comparison_metrics["max_peak_engagement"] = max(peak_engagements)
                comparison_metrics["min_peak_engagement"] = min(peak_engagements)
            
            # Identificar tendencia más prometedora
            if peak_growths and durations:
                # Índice combinado: crecimiento * duración
                combined_indices = {}
                for trend, analysis in trend_analyses.items():
                    if "peak_growth_rate" in analysis and "duration_days" in analysis and analysis["duration_days"] is not None:
                        combined_indices[trend] = analysis["peak_growth_rate"] * analysis["duration_days"]
                
                if combined_indices:
                    most_promising = max(combined_indices.items(), key=lambda x: x[1])[0]
                    comparison_metrics["most_promising_trend"] = most_promising
            
            # Generar recomendaciones comparativas
            recommendations = []
            
            if "most_promising_trend" in comparison_metrics:
                most_promising = comparison_metrics["most_promising_trend"]
                recommendations.append(
                    f"Priorizar recursos para la tendencia '{most_promising}' que muestra el mejor balance entre crecimiento y duración"
                )
            
            if durations:
                longest_trend = max(trend_analyses.items(), key=lambda x: x[1].get("duration_days", 0) if x[1].get("duration_days") is not None else 0)[0]
                recommendations.append(
                    f"La tendencia '{longest_trend}' tiene la mayor longevidad, considerar para contenido de largo plazo"
                )
            
            if peak_growths:
                fastest_growing = max(trend_analyses.items(), key=lambda x: x[1].get("peak_growth_rate", 0))[0]
                recommendations.append(
                    f"La tendencia '{fastest_growing}' muestra el crecimiento más rápido, ideal para campañas virales"
                )
            
            # Preparar resultado
            result = {
                "trends_analyzed": list(trend_analyses.keys()),
                "comparison_metrics": comparison_metrics,
                "individual_analyses": trend_analyses,
                "recommendations": recommendations,
                "visualization_path": viz_path
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al comparar tendencias: {str(e)}")
            return {"error": f"Error al comparar tendencias: {str(e)}"}
    
    def forecast_trend_performance(self, historical_data: pd.DataFrame, 
                                  trend_name: str,
                                  forecast_days: int = 30) -> Dict[str, Any]:
        """
        Pronostica el rendimiento futuro de una tendencia específica.
        
        Args:
            historical_data: DataFrame con datos históricos
            trend_name: Nombre de la tendencia a pronosticar
            forecast_days: Número de días a pronosticar
            
        Returns:
            Pronóstico de rendimiento de la tendencia
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            return {"error": "Se requieren los paquetes statsmodels para pronósticos"}
        
        try:
            # Filtrar datos para la tendencia específica
            trend_data = historical_data[historical_data["trend_name"] == trend_name].copy()
            
            if trend_data.empty:
                return {"error": f"No se encontraron datos para la tendencia '{trend_name}'"}
            
            # Verificar que haya suficientes datos para pronóstico
            if len(trend_data) < 5:
                return {"error": f"Insuficientes datos para pronosticar la tendencia '{trend_name}'"}
            
            # Asegurar que los datos estén ordenados por tiempo
            if "timestamp" in trend_data.columns:
                trend_data = trend_data.sort_values("timestamp")
                trend_data.set_index("timestamp", inplace=True)
            else:
                return {"error": "Se requiere una columna de timestamp para realizar pronósticos"}
            
            # Preparar series temporales para pronóstico
            metrics_to_forecast = []
            if "growth_rate" in trend_data.columns:
                metrics_to_forecast.append("growth_rate")
            if "engagement_rate" in trend_data.columns:
                metrics_to_forecast.append("engagement_rate")
            
            if not metrics_to_forecast:
                return {"error": "No se encontraron métricas adecuadas para pronosticar"}
            
            # Realizar pronósticos para cada métrica
            forecasts = {}
            for metric in metrics_to_forecast:
                # Preparar serie temporal
                ts_data = trend_data[metric].asfreq('D').fillna(method='ffill')
                
                # Si hay valores faltantes al inicio, llenar hacia atrás
                if ts_data.isna().any():
                    ts_data = ts_data.fillna(method='bfill')
                
                # Intentar diferentes modelos y seleccionar el mejor
                models = {}
                
                # Modelo ARIMA
                try:
                    arima_model = ARIMA(ts_data, order=(2, 1, 2))
                    arima_results = arima_model.fit()
                    arima_forecast = arima_results.forecast(steps=forecast_days)
                    models["arima"] = {
                        "forecast": arima_forecast,
                        "aic": arima_results.aic
                    }
                except Exception as e:
                    self.logger.warning(f"Error en modelo ARIMA para {metric}: {str(e)}")
                
                # Modelo Holt-Winters
                try:
                    hw_model = ExponentialSmoothing(
                        ts_data,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=7
                    )
                    hw_results = hw_model.fit()
                    hw_forecast = hw_results.forecast(steps=forecast_days)
                    models["holt_winters"] = {
                        "forecast": hw_forecast,
                        "aic": float('inf')  # Holt-Winters no proporciona AIC
                    }
                except Exception as e:
                    self.logger.warning(f"Error en modelo Holt-Winters para {metric}: {str(e)}")
                
                # Seleccionar el mejor modelo (por ahora, preferimos ARIMA si está disponible)
                best_model = "arima" if "arima" in models else "holt_winters" if "holt_winters" in models else None
                
                if best_model:
                    # Preparar fechas para el pronóstico
                    last_date = ts_data.index[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                    
                    # Guardar pronóstico
                    forecast_values = models[best_model]["forecast"].values if hasattr(models[best_model]["forecast"], 'values') else models[best_model]["forecast"]
                    
                    forecasts[metric] = {
                        "dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
                        "values": forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(forecast_values),
                        "model": best_model
                    }
                    
                    # Generar visualización
                    try:
                        plt.figure(figsize=(12, 6))
                        
                        # Datos históricos
                        plt.plot(ts_data.index, ts_data.values, label=f"{metric} histórico", color="blue")
                        
                        # Pronóstico
                        plt.plot(forecast_dates, forecast_values, label=f"{metric} pronóstico ({best_model})", 
                                color="red", linestyle="--")
                        
                        plt.title(f"Pronóstico de {metric} para '{trend_name}'")
                        plt.xlabel("Fecha")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.grid(True)
                        
                        # Guardar visualización
                        os.makedirs('outputs', exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        viz_path = f"outputs/trend_forecast_{trend_name.replace(' ', '_')}_{metric}_{timestamp}.png"
                        plt.savefig(viz_path)
                        plt.close()
                        
                        forecasts[metric]["visualization_path"] = viz_path
                        
                    except Exception as e:
                        self.logger.error(f"Error al generar visualización de pronóstico: {str(e)}")
            
            # Analizar resultados del pronóstico
            forecast_analysis = {}
            
            if "growth_rate" in forecasts:
                growth_forecast = forecasts["growth_rate"]["values"]
                
                # Determinar tendencia futura
                future_trend = "ascendente" if growth_forecast[-1] > growth_forecast[0] else "descendente" if growth_forecast[-1] < growth_forecast[0] else "estable"
                forecast_analysis["future_trend"] = future_trend
                
                # Calcular crecimiento promedio pronosticado
                avg_growth = sum(growth_forecast) / len(growth_forecast)
                forecast_analysis["avg_growth_forecast"] = avg_growth
                
                # Estimar duración restante de la tendencia
                if all(g <= 0 for g in growth_forecast[-3:]):
                    forecast_analysis["estimated_remaining_days"] = "menos de 30 días"
                    forecast_analysis["trend_status"] = "en declive"
                elif all(g > 0 for g in growth_forecast):
                    forecast_analysis["estimated_remaining_days"] = "más de 30 días"
                    forecast_analysis["trend_status"] = "fuerte"
                else:
                    # Encontrar el punto donde el crecimiento se vuelve negativo
                    for i, g in enumerate(growth_forecast):
                        if g <= 0:
                            forecast_analysis["estimated_remaining_days"] = i
                            forecast_analysis["trend_status"] = "debilitándose"
                            break
                    else:
                        forecast_analysis["estimated_remaining_days"] = "indeterminado"
                        forecast_analysis["trend_status"] = "fluctuante"
            
            # Generar recomendaciones basadas en el pronóstico
            recommendations = []
            
            if "future_trend" in forecast_analysis:
                if forecast_analysis["future_trend"] == "ascendente":
                    recommendations.append(
                        f"La tendencia '{trend_name}' seguirá creciendo. Aumentar inversión en contenido relacionado."
                    )
                elif forecast_analysis["future_trend"] == "descendente":
                    recommendations.append(
                        f"La tendencia '{trend_name}' está en declive. Reducir gradualmente la producción de contenido."
                    )
                else:
                    recommendations.append(
                        f"La tendencia '{trend_name}' se mantendrá estable. Mantener ritmo actual de contenido."
                    )
            
            if "trend_status" in forecast_analysis:
                if forecast_analysis["trend_status"] == "fuerte":
                    recommendations.append(
                        "Tendencia con potencial a largo plazo. Considerar desarrollo de contenido premium o productos."
                    )
                elif forecast_analysis["trend_status"] == "debilitándose":
                    recommendations.append(
                        "Tendencia debilitándose. Extraer valor máximo ahora y preparar transición."
                    )
                elif forecast_analysis["trend_status"] == "en declive":
                    recommendations.append(
                        "Tendencia en declive acelerado. Minimizar nuevas inversiones y redirigir recursos."
                    )
            
            # Preparar resultado final
            result = {
                "trend_name": trend_name,
                "forecast_days": forecast_days,
                "forecasts": forecasts,
                "analysis": forecast_analysis,
                "recommendations": recommendations
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al pronosticar tendencia: {str(e)}")
            return {"error": f"Error al pronosticar tendencia: {str(e)}"}