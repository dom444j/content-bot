"""
Learning Repository - Repositorio de aprendizaje para el Content Bot

Este módulo proporciona funcionalidades para almacenar, recuperar y analizar
datos de aprendizaje basados en el rendimiento de contenido.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LearningRepository")

class LearningRepository:
    """
    Repositorio para almacenar y analizar datos de aprendizaje
    basados en el rendimiento de contenido.
    """
    
    def __init__(self, config_path: str = "config/learning_config.json"):
        """
        Inicializa el repositorio de aprendizaje.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.models_dir = "models"
        self.data_dir = "data/learning_data"
        self.reports_dir = "reports/learning"
        
        # Crear directorios necesarios
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Inicializar modelos
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        logger.info("Repositorio de aprendizaje inicializado")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde un archivo JSON."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Configuración por defecto
                default_config = {
                    "features": {
                        "numeric": [
                            "duration", 
                            "title_length", 
                            "description_length", 
                            "hashtag_count"
                        ],
                        "categorical": [
                            "day_of_week", 
                            "hour_of_day", 
                            "category", 
                            "format", 
                            "has_hashtags", 
                            "has_cta"
                        ]
                    },
                    "target_metrics": [
                        "views", 
                        "likes", 
                        "comments", 
                        "shares", 
                        "engagement_score"
                    ],
                    "model_params": {
                        "random_forest": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "min_samples_split": 5,
                            "min_samples_leaf": 2,
                            "random_state": 42
                        },
                        "kmeans": {
                            "n_clusters": 3,
                            "random_state": 42
                        }
                    },
                    "min_samples_for_training": 20,
                    "test_size": 0.2,
                    "update_frequency": 7,  # días
                    "platforms": ["youtube", "tiktok", "instagram", "threads", "bluesky", "x"]
                }
                
                # Guardar configuración por defecto
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                
                return default_config
        except Exception as e:
            logger.error(f"Error cargando configuración: {str(e)}")
            return {
                "features": {
                    "numeric": ["duration", "title_length"],
                    "categorical": ["day_of_week", "category"]
                },
                "target_metrics": ["views", "engagement_score"],
                "model_params": {
                    "random_forest": {"n_estimators": 100, "random_state": 42},
                    "kmeans": {"n_clusters": 3, "random_state": 42}
                },
                "min_samples_for_training": 20,
                "test_size": 0.2,
                "update_frequency": 7,
                "platforms": ["youtube", "tiktok", "instagram"]
            }
    
    def store_content_data(self, 
                          channel_id: str, 
                          platform: str, 
                          content_data: List[Dict[str, Any]]) -> bool:
        """
        Almacena datos de contenido para aprendizaje.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            content_data: Lista de datos de contenido
            
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        try:
            if not content_data:
                logger.warning("No hay datos para almacenar")
                return False
            
            # Crear directorio para el canal si no existe
            channel_dir = f"{self.data_dir}/{channel_id}"
            os.makedirs(channel_dir, exist_ok=True)
            
            # Ruta del archivo de datos
            data_file = f"{channel_dir}/{platform}_content_data.json"
            
            # Cargar datos existentes si existen
            existing_data = []
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Combinar datos existentes con nuevos
            content_ids = set(item.get("content_id") for item in existing_data)
            new_data_count = 0
            
            for item in content_data:
                content_id = item.get("content_id")
                if content_id and content_id not in content_ids:
                    existing_data.append(item)
                    content_ids.add(content_id)
                    new_data_count += 1
            
            # Guardar datos combinados
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Almacenados {new_data_count} nuevos datos de contenido para {channel_id} en {platform}")
            
            # Verificar si es necesario actualizar modelos
            self._check_model_update(channel_id, platform)
            
            return True
            
        except Exception as e:
            logger.error(f"Error almacenando datos de contenido: {str(e)}")
            return False
    
    def _check_model_update(self, channel_id: str, platform: str) -> None:
        """Verifica si es necesario actualizar los modelos."""
        try:
            # Ruta del archivo de modelo
            model_file = f"{self.models_dir}/{channel_id}_{platform}_model.joblib"
            
            # Verificar si el modelo existe
            if not os.path.exists(model_file):
                # Entrenar modelo si no existe
                self.train_models(channel_id, platform)
                return
            
            # Verificar fecha de última actualización
            last_update = datetime.fromtimestamp(os.path.getmtime(model_file))
            days_since_update = (datetime.now() - last_update).days
            
            if days_since_update >= self.config["update_frequency"]:
                # Actualizar modelo si ha pasado el tiempo configurado
                self.train_models(channel_id, platform)
                
        except Exception as e:
            logger.error(f"Error verificando actualización de modelo: {str(e)}")
    
    def train_models(self, channel_id: str, platform: str) -> Dict[str, Any]:
        """
        Entrena modelos predictivos para un canal y plataforma.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            
        Returns:
            Resultados del entrenamiento
        """
        try:
            # Cargar datos
            data = self._load_content_data(channel_id, platform)
            
            if not data or len(data) < self.config["min_samples_for_training"]:
                return {
                    "error": f"Datos insuficientes para entrenamiento. Se requieren al menos {self.config['min_samples_for_training']} muestras."
                }
            
            # Convertir a DataFrame
            df = pd.DataFrame(data)
            
            # Preparar características
            X, feature_names = self._prepare_features(df)
            
            if X.shape[0] == 0 or X.shape[1] == 0:
                return {"error": "No se pudieron extraer características válidas"}
            
            # Entrenar modelos para cada métrica objetivo
            results = {}
            for target in self.config["target_metrics"]:
                if target in df.columns:
                    # Preparar datos de entrenamiento
                    y = df[target].values
                    
                    # Dividir en conjuntos de entrenamiento y prueba
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=self.config["test_size"], random_state=42
                    )
                    
                    # Escalar características
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                                        # Entrenar modelo Random Forest
                    model_params = self.config["model_params"]["random_forest"]
                    model = RandomForestRegressor(**model_params)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluar modelo
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calcular importancia de características
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                    
                    # Guardar modelo y scaler
                    model_key = f"{channel_id}_{platform}_{target}"
                    self.models[model_key] = model
                    self.scalers[model_key] = scaler
                    self.feature_importance[model_key] = feature_importance
                    
                    # Guardar en disco
                    model_path = f"{self.models_dir}/{model_key}_model.joblib"
                    scaler_path = f"{self.models_dir}/{model_key}_scaler.joblib"
                    importance_path = f"{self.models_dir}/{model_key}_importance.json"
                    
                    joblib.dump(model, model_path)
                    joblib.dump(scaler, scaler_path)
                    
                    with open(importance_path, 'w', encoding='utf-8') as f:
                        json.dump(feature_importance, f, indent=2)
                    
                    # Generar visualización de importancia de características
                    viz_path = self._generate_feature_importance_viz(
                        feature_importance,
                        f"{channel_id}_{platform}_{target}_importance",
                        f"Importancia de características para {target} en {platform}"
                    )
                    
                    # Almacenar resultados
                    results[target] = {
                        "mse": float(mse),
                        "r2": float(r2),
                        "feature_importance": feature_importance,
                        "visualization": viz_path,
                        "samples": len(y),
                        "model_path": model_path
                    }
            
            # Realizar clustering de contenido
            if X.shape[0] >= self.config["model_params"]["kmeans"]["n_clusters"]:
                cluster_results = self._perform_clustering(X, df, channel_id, platform)
                results["clustering"] = cluster_results
            
            # Generar informe
            report_path = self._generate_training_report(results, channel_id, platform)
            results["report"] = report_path
            
            logger.info(f"Modelos entrenados para {channel_id} en {platform}")
            return results
            
        except Exception as e:
            logger.error(f"Error entrenando modelos: {str(e)}")
            return {"error": str(e)}
    
    def _load_content_data(self, channel_id: str, platform: str) -> List[Dict[str, Any]]:
        """Carga datos de contenido para entrenamiento."""
        try:
            data_file = f"{self.data_dir}/{channel_id}/{platform}_content_data.json"
            
            if not os.path.exists(data_file):
                logger.warning(f"No existen datos para {channel_id} en {platform}")
                return []
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Cargados {len(data)} registros para {channel_id} en {platform}")
            return data
            
        except Exception as e:
            logger.error(f"Error cargando datos de contenido: {str(e)}")
            return []
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepara características para entrenamiento."""
        try:
            # Verificar columnas disponibles
            numeric_features = [f for f in self.config["features"]["numeric"] if f in df.columns]
            categorical_features = [f for f in self.config["features"]["categorical"] if f in df.columns]
            
            if not numeric_features and not categorical_features:
                logger.warning("No se encontraron características válidas en los datos")
                return np.array([]), []
            
            # Procesar características numéricas
            X_numeric = df[numeric_features].fillna(0).values if numeric_features else np.array([]).reshape(len(df), 0)
            
            # Procesar características categóricas
            X_categorical = np.array([]).reshape(len(df), 0)
            categorical_names = []
            
            for feature in categorical_features:
                # One-hot encoding
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=False)
                categorical_names.extend(dummies.columns.tolist())
                
                if X_categorical.size == 0:
                    X_categorical = dummies.values
                else:
                    X_categorical = np.hstack((X_categorical, dummies.values))
            
            # Combinar características
            if X_numeric.size > 0 and X_categorical.size > 0:
                X = np.hstack((X_numeric, X_categorical))
                feature_names = numeric_features + categorical_names
            elif X_numeric.size > 0:
                X = X_numeric
                feature_names = numeric_features
            else:
                X = X_categorical
                feature_names = categorical_names
            
            return X, feature_names
            
        except Exception as e:
            logger.error(f"Error preparando características: {str(e)}")
            return np.array([]), []
    
    def _perform_clustering(self, 
                           X: np.ndarray, 
                           df: pd.DataFrame, 
                           channel_id: str, 
                           platform: str) -> Dict[str, Any]:
        """Realiza clustering de contenido."""
        try:
            # Escalar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Configurar modelo
            kmeans_params = self.config["model_params"]["kmeans"]
            kmeans = KMeans(**kmeans_params)
            
            # Entrenar modelo
            clusters = kmeans.fit_predict(X_scaled)
            
            # Añadir clusters al DataFrame
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = clusters
            
            # Analizar clusters
            cluster_analysis = {}
            for cluster_id in range(kmeans_params["n_clusters"]):
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                
                # Calcular estadísticas por cluster
                stats = {}
                for metric in self.config["target_metrics"]:
                    if metric in cluster_data.columns:
                        stats[metric] = {
                            "mean": float(cluster_data[metric].mean()),
                            "median": float(cluster_data[metric].median()),
                            "min": float(cluster_data[metric].min()),
                            "max": float(cluster_data[metric].max()),
                            "std": float(cluster_data[metric].std()) if len(cluster_data) > 1 else 0
                        }
                
                # Características más comunes
                common_features = {}
                for feature in self.config["features"]["categorical"]:
                    if feature in cluster_data.columns:
                        value_counts = cluster_data[feature].value_counts()
                        if not value_counts.empty:
                            most_common = value_counts.index[0]
                            frequency = int(value_counts.iloc[0])
                            percentage = float(frequency / len(cluster_data) * 100)
                            
                            common_features[feature] = {
                                "most_common": most_common,
                                "frequency": frequency,
                                "percentage": percentage
                            }
                
                # Guardar análisis
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "count": len(cluster_data),
                    "percentage": float(len(cluster_data) / len(df) * 100),
                    "metrics": stats,
                    "common_features": common_features
                }
            
            # Guardar modelo
            model_key = f"{channel_id}_{platform}_kmeans"
            model_path = f"{self.models_dir}/{model_key}_model.joblib"
            scaler_path = f"{self.models_dir}/{model_key}_scaler.joblib"
            
            joblib.dump(kmeans, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Generar visualización
            viz_path = self._generate_cluster_visualization(
                df_with_clusters,
                f"{channel_id}_{platform}_clusters",
                f"Clusters de contenido para {channel_id} en {platform}"
            )
            
            return {
                "n_clusters": kmeans_params["n_clusters"],
                "analysis": cluster_analysis,
                "model_path": model_path,
                "visualization": viz_path
            }
            
        except Exception as e:
            logger.error(f"Error realizando clustering: {str(e)}")
            return {"error": str(e)}
    
    def _generate_feature_importance_viz(self,
                                        feature_importance: Dict[str, float],
                                        filename: str,
                                        title: str) -> str:
        """Genera visualización de importancia de características."""
        try:
            # Ordenar características por importancia
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Limitar a las 15 características más importantes
            top_features = sorted_features[:15]
            
            # Extraer nombres y valores
            feature_names = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            # Crear figura
            plt.figure(figsize=(10, 8))
            
            # Crear barras horizontales
            bars = plt.barh(range(len(feature_names)), importance_values, align='center')
            
            # Añadir etiquetas
            plt.yticks(range(len(feature_names)), feature_names)
            plt.xlabel('Importancia')
            plt.title(title)
            
            # Añadir valores
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height()/2,
                    f"{importance_values[i]:.4f}",
                    va='center'
                )
            
            # Guardar figura
            output_path = f"{self.reports_dir}/{filename}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Visualización de importancia generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización de importancia: {str(e)}")
            return ""
    
    def _generate_cluster_visualization(self,
                                       df: pd.DataFrame,
                                       filename: str,
                                       title: str) -> str:
        """Genera visualización de clusters."""
        try:
            # Verificar si hay métricas para visualizar
            available_metrics = [m for m in self.config["target_metrics"] if m in df.columns]
            
            if len(available_metrics) < 2:
                logger.warning("No hay suficientes métricas para visualización de clusters")
                return ""
            
            # Seleccionar las dos métricas principales
            x_metric = available_metrics[0]
            y_metric = available_metrics[1]
            
            # Crear figura
            plt.figure(figsize=(10, 8))
            
            # Graficar clusters
            scatter = plt.scatter(
                df[x_metric],
                df[y_metric],
                c=df['cluster'],
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            
            # Añadir etiquetas
            plt.xlabel(x_metric.capitalize())
            plt.ylabel(y_metric.capitalize())
            plt.title(title)
            
            # Añadir leyenda
            legend = plt.legend(*scatter.legend_elements(), title="Clusters")
            plt.gca().add_artist(legend)
            
            # Guardar figura
            output_path = f"{self.reports_dir}/{filename}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Visualización de clusters generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización de clusters: {str(e)}")
            return ""
    
    def _generate_training_report(self,
                                 results: Dict[str, Any],
                                 channel_id: str,
                                 platform: str) -> str:
        """Genera un informe de entrenamiento."""
        try:
            # Crear informe
            report = {
                "channel_id": channel_id,
                "platform": platform,
                "timestamp": datetime.now().isoformat(),
                "models": {}
            }
            
            # Añadir resultados de modelos
            for target, model_results in results.items():
                if target != "clustering":
                    report["models"][target] = {
                        "performance": {
                            "mse": model_results.get("mse"),
                            "r2": model_results.get("r2")
                        },
                        "top_features": self._get_top_features(model_results.get("feature_importance", {})),
                        "visualization": model_results.get("visualization")
                    }
            
            # Añadir resultados de clustering
            if "clustering" in results:
                report["clustering"] = results["clustering"]
            
            # Guardar informe
            report_path = f"{self.reports_dir}/{channel_id}_{platform}_training_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Informe de entrenamiento generado: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generando informe de entrenamiento: {str(e)}")
            return ""
    
    def _get_top_features(self, feature_importance: Dict[str, float], top_n: int = 5) -> List[Dict[str, Any]]:
        """Obtiene las características más importantes."""
        try:
            # Ordenar características
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Limitar a top_n
            top_features = sorted_features[:top_n]
            
            # Formatear resultado
            result = []
            for feature, importance in top_features:
                result.append({
                    "feature": feature,
                    "importance": float(importance)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo características principales: {str(e)}")
            return []
    
    def predict_performance(self,
                           channel_id: str,
                           platform: str,
                           content_features: Dict[str, Any],
                           target_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predice el rendimiento de un contenido basado en sus características.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            content_features: Características del contenido
            target_metrics: Métricas objetivo a predecir (opcional)
            
        Returns:
            Predicciones de rendimiento
        """
        try:
            if not target_metrics:
                target_metrics = self.config["target_metrics"]
            
            # Verificar si existen modelos
            predictions = {}
            for target in target_metrics:
                model_key = f"{channel_id}_{platform}_{target}"
                
                # Cargar modelo si no está en memoria
                if model_key not in self.models:
                    model_path = f"{self.models_dir}/{model_key}_model.joblib"
                    scaler_path = f"{self.models_dir}/{model_key}_scaler.joblib"
                    
                    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                        predictions[target] = {"error": "Modelo no disponible"}
                        continue
                    
                    self.models[model_key] = joblib.load(model_path)
                    self.scalers[model_key] = joblib.load(scaler_path)
                
                # Preparar características
                X, _ = self._prepare_content_features(content_features)
                
                if X.size == 0:
                    predictions[target] = {"error": "No se pudieron extraer características válidas"}
                    continue
                
                # Escalar características
                X_scaled = self.scalers[model_key].transform([X])
                
                # Realizar predicción
                prediction = float(self.models[model_key].predict(X_scaled)[0])
                
                # Guardar predicción
                predictions[target] = {
                    "prediction": prediction,
                    "confidence": self._calculate_prediction_confidence(model_key, X_scaled)
                }
            
            # Predecir cluster
            cluster_prediction = self._predict_cluster(channel_id, platform, content_features)
            if cluster_prediction:
                predictions["cluster"] = cluster_prediction
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error prediciendo rendimiento: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_content_features(self, content_features: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Prepara características de un contenido para predicción."""
        try:
            # Crear DataFrame con una fila
            df = pd.DataFrame([content_features])
            
            # Preparar características
            X, feature_names = self._prepare_features(df)
            
            if X.shape[0] == 0:
                return np.array([]), []
            
            return X[0], feature_names
            
        except Exception as e:
            logger.error(f"Error preparando características de contenido: {str(e)}")
            return np.array([]), []
    
    def _calculate_prediction_confidence(self, model_key: str, X_scaled: np.ndarray) -> float:
        """Calcula la confianza de una predicción."""
        try:
            # Para Random Forest, podemos usar la desviación estándar de las predicciones de los árboles
            if model_key in self.models and hasattr(self.models[model_key], "estimators_"):
                predictions = [tree.predict(X_scaled)[0] for tree in self.models[model_key].estimators_]
                std_dev = np.std(predictions)
                
                # Normalizar a un valor de confianza (0-1)
                mean_pred = np.mean(predictions)
                if mean_pred != 0:
                    confidence = 1 - (std_dev / abs(mean_pred))
                else:
                    confidence = 0
                
                return float(max(0, min(1, confidence)))
            
            return 0.5  # Valor por defecto
            
        except Exception as e:
            logger.error(f"Error calculando confianza de predicción: {str(e)}")
            return 0.5
    
    def _predict_cluster(self, 
                        channel_id: str, 
                        platform: str, 
                        content_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predice el cluster al que pertenecería un contenido."""
        try:
            model_key = f"{channel_id}_{platform}_kmeans"
            model_path = f"{self.models_dir}/{model_key}_model.joblib"
            scaler_path = f"{self.models_dir}/{model_key}_scaler.joblib"
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None
            
            # Cargar modelo y scaler
            kmeans = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Preparar características
            X, _ = self._prepare_content_features(content_features)
            
            if X.size == 0:
                return None
            
            # Escalar características
            X_scaled = scaler.transform([X])
            
            # Predecir cluster
            cluster = int(kmeans.predict(X_scaled)[0])
            
            # Calcular distancia al centroide (para confianza)
            centroid = kmeans.cluster_centers_[cluster]
            distance = np.linalg.norm(X_scaled[0] - centroid)
            
            # Normalizar distancia a confianza
            max_distance = np.max([np.linalg.norm(X_scaled[0] - c) for c in kmeans.cluster_centers_])
            confidence = 1 - (distance / max_distance) if max_distance > 0 else 0
            
            return {
                "cluster": cluster,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error prediciendo cluster: {str(e)}")
            return None
    
    def get_content_recommendations(self,
                                   channel_id: str,
                                   platform: str,
                                   target_metric: str = "engagement_score",
                                   top_n: int = 5) -> Dict[str, Any]:
        """
        Obtiene recomendaciones para optimizar contenido.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            target_metric: Métrica objetivo a optimizar
            top_n: Número de recomendaciones
            
        Returns:
            Recomendaciones para optimización de contenido
        """
        try:
            # Verificar si existe modelo
            model_key = f"{channel_id}_{platform}_{target_metric}"
            importance_path = f"{self.models_dir}/{model_key}_importance.json"
            
            if not os.path.exists(importance_path):
                return {
                    "error": f"No hay modelo disponible para {target_metric} en {platform}"
                }
            
            # Cargar importancia de características
            with open(importance_path, 'r', encoding='utf-8') as f:
                feature_importance = json.load(f)
            
            # Obtener características más importantes
            top_features = self._get_top_features(feature_importance, top_n=top_n)
            
            # Cargar datos históricos
            data = self._load_content_data(channel_id, platform)
            
            if not data:
                return {
                    "top_features": top_features,
                    "recommendations": []
                }
            
            # Convertir a DataFrame
            df = pd.DataFrame(data)
            
            # Generar recomendaciones
            recommendations = []
            
            for feature_info in top_features:
                feature = feature_info["feature"]
                base_feature = feature.split('_')[0] if '_' in feature else feature
                
                recommendation = {
                    "feature": base_feature,
                    "importance": feature_info["importance"]
                }
                
                # Analizar característica
                if base_feature in self.config["features"]["numeric"]:
                    # Para características numéricas
                    if base_feature in df.columns and target_metric in df.columns:
                        # Encontrar valor óptimo
                        correlation = df[base_feature].corr(df[target_metric])
                        
                        if correlation > 0:
                            # Correlación positiva: valores más altos son mejores
                            optimal_value = df.loc[df[target_metric].idxmax()][base_feature]
                            recommendation["suggestion"] = f"Aumentar {base_feature} cerca de {optimal_value}"
                            recommendation["direction"] = "increase"
                        else:
                            # Correlación negativa: valores más bajos son mejores
                            optimal_value = df.loc[df[target_metric].idxmax()][base_feature]
                            recommendation["suggestion"] = f"Reducir {base_feature} cerca de {optimal_value}"
                            recommendation["direction"] = "decrease"
                        
                        recommendation["optimal_value"] = float(optimal_value)
                        recommendation["correlation"] = float(correlation)
                
                elif base_feature in self.config["features"]["categorical"]:
                    # Para características categóricas
                    if base_feature in df.columns and target_metric in df.columns:
                        # Encontrar categoría óptima
                        category_performance = df.groupby(base_feature)[target_metric].mean()
                        
                        if not category_performance.empty:
                            best_category = category_performance.idxmax()
                            recommendation["suggestion"] = f"Utilizar {base_feature}: {best_category}"
                            recommendation["optimal_value"] = best_category
                            recommendation["category_performance"] = {
                                cat: float(val) for cat, val in category_performance.items()
                            }
                
                # Si es una característica one-hot
                elif '_' in feature:
                    category, value = feature.split('_', 1)
                    
                    if category in self.config["features"]["categorical"]:
                        recommendation["feature"] = category
                        recommendation["suggestion"] = f"Utilizar {category}: {value}"
                        recommendation["optimal_value"] = value
                
                recommendations.append(recommendation)
            
            return {
                "target_metric": target_metric,
                "top_features": top_features,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo recomendaciones: {str(e)}")
            return {"error": str(e)}