"""
Analytics Engine - Sistema de análisis de datos para el Content Bot

Este módulo proporciona funcionalidades para recopilar, procesar y analizar
métricas de rendimiento de contenido en múltiples plataformas.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pymongo
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/analytics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnalyticsEngine")

class AnalyticsEngine:
    """
    Motor de análisis para procesar y visualizar métricas de rendimiento
    de contenido en múltiples plataformas.
    """
    
    def __init__(self, config_path: str = "config/analytics_config.json"):
        """
        Inicializa el motor de análisis.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.db_client = self._connect_database()
        self.metrics_cache = {}
        self.last_update = {}
        self.visualization_dir = "reports/visualizations"
        os.makedirs(self.visualization_dir, exist_ok=True)
        logger.info("Motor de análisis inicializado")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde un archivo JSON."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Configuración por defecto
                default_config = {
                    "database": {
                        "uri": "mongodb://localhost:27017/",
                        "name": "content_bot_analytics"
                    },
                    "metrics": {
                        "engagement": ["views", "likes", "comments", "shares", "saves"],
                        "retention": ["watch_time", "completion_rate", "return_rate"],
                        "conversion": ["click_through_rate", "subscription_rate", "cta_conversion"],
                        "monetization": ["revenue", "rpm", "cpm", "affiliate_clicks", "affiliate_conversions"]
                    },
                    "platforms": ["youtube", "tiktok", "instagram", "threads", "bluesky", "x"],
                    "update_interval": 3600,  # segundos
                    "cache_ttl": 86400,  # segundos
                    "visualization": {
                        "default_style": "darkgrid",
                        "color_palette": "viridis",
                        "interactive": True
                    }
                }
                
                # Guardar configuración por defecto
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                
                return default_config
        except Exception as e:
            logger.error(f"Error cargando configuración: {str(e)}")
            return {
                "database": {"uri": "mongodb://localhost:27017/", "name": "content_bot_analytics"},
                "metrics": {"engagement": ["views", "likes", "comments"]},
                "platforms": ["youtube", "tiktok", "instagram"],
                "update_interval": 3600,
                "cache_ttl": 86400
            }
    
    def _connect_database(self) -> MongoClient:
        """Conecta a la base de datos MongoDB."""
        try:
            client = MongoClient(self.config["database"]["uri"])
            # Verificar conexión
            client.server_info()
            logger.info("Conexión a base de datos establecida")
            return client
        except Exception as e:
            logger.error(f"Error conectando a la base de datos: {str(e)}")
            logger.warning("Operando sin persistencia de datos")
            return None
    
    def store_metrics(self, 
                     channel_id: str, 
                     platform: str, 
                     content_id: str, 
                     metrics: Dict[str, Any],
                     timestamp: Optional[datetime] = None) -> bool:
        """
        Almacena métricas de contenido en la base de datos.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, etc.)
            content_id: ID del contenido
            metrics: Diccionario de métricas
            timestamp: Marca de tiempo (opcional)
            
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        if not timestamp:
            timestamp = datetime.now()
            
        try:
            if self.db_client:
                db = self.db_client[self.config["database"]["name"]]
                collection = db["content_metrics"]
                
                # Documento a insertar
                doc = {
                    "channel_id": channel_id,
                    "platform": platform,
                    "content_id": content_id,
                    "metrics": metrics,
                    "timestamp": timestamp,
                    "date": timestamp.strftime("%Y-%m-%d")
                }
                
                # Insertar o actualizar
                result = collection.update_one(
                    {
                        "channel_id": channel_id,
                        "platform": platform,
                        "content_id": content_id,
                        "date": timestamp.strftime("%Y-%m-%d")
                    },
                    {"$set": doc},
                    upsert=True
                )
                
                # Actualizar caché
                cache_key = f"{channel_id}_{platform}_{content_id}"
                self.metrics_cache[cache_key] = {
                    "metrics": metrics,
                    "timestamp": timestamp
                }
                self.last_update[cache_key] = datetime.now()
                
                logger.info(f"Métricas almacenadas para {platform}/{content_id}")
                return True
            else:
                # Solo actualizar caché
                cache_key = f"{channel_id}_{platform}_{content_id}"
                self.metrics_cache[cache_key] = {
                    "metrics": metrics,
                    "timestamp": timestamp
                }
                self.last_update[cache_key] = datetime.now()
                
                logger.warning(f"Métricas almacenadas solo en caché para {platform}/{content_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error almacenando métricas: {str(e)}")
            return False
    
    def get_metrics(self, 
                   channel_id: str, 
                   platform: str, 
                   content_id: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   metrics_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Obtiene métricas de contenido.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            content_id: ID del contenido (opcional)
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            metrics_filter: Lista de métricas a incluir
            
        Returns:
            Diccionario con métricas
        """
        try:
            # Preparar filtros
            query = {
                "channel_id": channel_id,
                "platform": platform
            }
            
            if content_id:
                query["content_id"] = content_id
                
            if start_date:
                if not end_date:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    
                query["date"] = {
                    "$gte": start_date,
                    "$lte": end_date
                }
            
            # Verificar caché para consultas simples
            if content_id and not start_date and not end_date:
                cache_key = f"{channel_id}_{platform}_{content_id}"
                if cache_key in self.metrics_cache:
                    cache_time = self.last_update.get(cache_key)
                    if cache_time and (datetime.now() - cache_time).total_seconds() < self.config["cache_ttl"]:
                        return self.metrics_cache[cache_key]
            
            # Consultar base de datos
            if self.db_client:
                db = self.db_client[self.config["database"]["name"]]
                collection = db["content_metrics"]
                
                # Ejecutar consulta
                cursor = collection.find(query).sort("timestamp", -1)
                
                # Procesar resultados
                results = []
                for doc in cursor:
                    # Filtrar métricas si es necesario
                    if metrics_filter:
                        filtered_metrics = {k: v for k, v in doc["metrics"].items() if k in metrics_filter}
                        doc["metrics"] = filtered_metrics
                    
                    # Convertir ObjectId a string para serialización
                    doc["_id"] = str(doc["_id"])
                    results.append(doc)
                
                if not results:
                    return {"error": "No se encontraron métricas", "query": query}
                
                # Si es un solo contenido, devolver el más reciente
                if content_id and not start_date:
                    return results[0]
                
                return {"results": results, "count": len(results)}
            else:
                # Buscar en caché
                if content_id:
                    cache_key = f"{channel_id}_{platform}_{content_id}"
                    if cache_key in self.metrics_cache:
                        return self.metrics_cache[cache_key]
                
                return {"error": "Base de datos no disponible y datos no encontrados en caché"}
                
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {str(e)}")
            return {"error": str(e)}
    
    def analyze_performance(self, 
                           channel_id: str, 
                           platform: str,
                           period: str = "7d",
                           metric_type: str = "engagement") -> Dict[str, Any]:
        """
        Analiza el rendimiento de un canal en una plataforma.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            period: Período de análisis (1d, 7d, 30d, 90d)
            metric_type: Tipo de métrica (engagement, retention, conversion, monetization)
            
        Returns:
            Análisis de rendimiento
        """
        try:
            # Determinar fechas
            end_date = datetime.now()
            
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "7d":
                start_date = end_date - timedelta(days=7)
            elif period == "30d":
                start_date = end_date - timedelta(days=30)
            elif period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=7)  # Default
            
            # Obtener métricas relevantes
            metrics_filter = self.config["metrics"].get(metric_type, [])
            
            # Consultar métricas
            metrics_data = self.get_metrics(
                channel_id=channel_id,
                platform=platform,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                metrics_filter=metrics_filter
            )
            
            if "error" in metrics_data:
                return {"error": metrics_data["error"]}
            
            # Procesar resultados
            results = metrics_data.get("results", [])
            
            if not results:
                return {"error": "No hay datos suficientes para análisis"}
            
            # Agrupar por fecha
            daily_metrics = {}
            for result in results:
                date = result.get("date")
                content_id = result.get("content_id")
                
                if date not in daily_metrics:
                    daily_metrics[date] = {}
                
                for metric_name, value in result.get("metrics", {}).items():
                    if metric_name not in daily_metrics[date]:
                        daily_metrics[date][metric_name] = []
                    
                    daily_metrics[date][metric_name].append(value)
            
            # Calcular promedios diarios
            daily_averages = {}
            for date, metrics in daily_metrics.items():
                daily_averages[date] = {}
                for metric_name, values in metrics.items():
                    daily_averages[date][metric_name] = sum(values) / len(values)
            
            # Convertir a DataFrame para análisis
            dates = []
            data = {metric: [] for metric in metrics_filter}
            
            for date in sorted(daily_averages.keys()):
                dates.append(date)
                for metric in metrics_filter:
                    data[metric].append(daily_averages[date].get(metric, 0))
            
            df = pd.DataFrame(data, index=dates)
            
            # Calcular estadísticas
            stats_result = {}
            for metric in metrics_filter:
                if len(df[metric]) > 0:
                    stats_result[metric] = {
                        "mean": float(df[metric].mean()),
                        "median": float(df[metric].median()),
                        "min": float(df[metric].min()),
                        "max": float(df[metric].max()),
                        "std": float(df[metric].std()) if len(df[metric]) > 1 else 0,
                        "trend": self._calculate_trend(df[metric])
                    }
            
            # Generar visualización
            if self.config["visualization"].get("interactive", False):
                viz_path = self._generate_interactive_visualization(
                    df, 
                    f"{channel_id}_{platform}_{metric_type}_{period}",
                    f"Análisis de {metric_type} para {channel_id} en {platform} ({period})"
                )
            else:
                viz_path = self._generate_visualization(
                    df, 
                    f"{channel_id}_{platform}_{metric_type}_{period}",
                    f"Análisis de {metric_type} para {channel_id} en {platform} ({period})"
                )
            
            # Preparar resultado
            analysis = {
                "channel_id": channel_id,
                "platform": platform,
                "period": period,
                "metric_type": metric_type,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "metrics": stats_result,
                "visualization": viz_path,
                "data_points": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando rendimiento: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Calcula la tendencia de una serie temporal."""
        if len(series) < 2:
            return {"direction": "stable", "slope": 0, "confidence": 0}
        
        try:
            # Calcular regresión lineal
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Determinar dirección
            if abs(slope) < 0.01 * series.mean():
                direction = "stable"
            elif slope > 0:
                direction = "up"
            else:
                direction = "down"
            
            # Calcular confianza (r²)
            confidence = r_value ** 2
            
            return {
                "direction": direction,
                "slope": float(slope),
                "confidence": float(confidence),
                "p_value": float(p_value)
            }
        except Exception as e:
            logger.error(f"Error calculando tendencia: {str(e)}")
            return {"direction": "unknown", "slope": 0, "confidence": 0}
    
    def _generate_visualization(self, 
                               df: pd.DataFrame, 
                               filename: str, 
                               title: str) -> str:
        """Genera una visualización estática de métricas."""
        try:
            # Configurar estilo
            sns.set_style(self.config["visualization"].get("default_style", "darkgrid"))
            
            # Crear figura
            plt.figure(figsize=(12, 8))
            
            # Graficar cada métrica
            for column in df.columns:
                plt.plot(df.index, df[column], marker='o', linewidth=2, label=column)
            
            # Añadir etiquetas y leyenda
            plt.title(title, fontsize=16)
            plt.xlabel("Fecha", fontsize=12)
            plt.ylabel("Valor", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Guardar figura
            output_path = f"{self.visualization_dir}/{filename}.png"
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Visualización generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización: {str(e)}")
            return ""
    
    def _generate_interactive_visualization(self, 
                                           df: pd.DataFrame, 
                                           filename: str, 
                                           title: str) -> str:
        """Genera una visualización interactiva de métricas con Plotly."""
        try:
            # Preparar datos
            df_melted = df.reset_index().melt(id_vars='index', var_name='Métrica', value_name='Valor')
            
            # Crear figura
            fig = px.line(
                df_melted, 
                x='index', 
                y='Valor', 
                color='Métrica',
                title=title,
                labels={'index': 'Fecha', 'Valor': 'Valor'},
                line_shape='spline',
                markers=True
            )
            
            # Personalizar diseño
            fig.update_layout(
                template='plotly_dark',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            # Guardar como HTML
            output_path = f"{self.visualization_dir}/{filename}.html"
            fig.write_html(output_path)
            
            logger.info(f"Visualización interactiva generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización interactiva: {str(e)}")
            return ""
    
    def compare_platforms(self, 
                         channel_id: str, 
                         platforms: List[str],
                         metric: str,
                         period: str = "30d") -> Dict[str, Any]:
        """
        Compara el rendimiento entre plataformas para un canal.
        
        Args:
            channel_id: ID del canal
            platforms: Lista de plataformas a comparar
            metric: Métrica a comparar
            period: Período de análisis
            
        Returns:
            Comparación de plataformas
        """
        try:
            results = {}
            platform_data = {}
            
            # Obtener datos para cada plataforma
            for platform in platforms:
                analysis = self.analyze_performance(
                    channel_id=channel_id,
                    platform=platform,
                    period=period,
                    metric_type=self._get_metric_type(metric)
                )
                
                if "error" not in analysis:
                    metric_stats = analysis.get("metrics", {}).get(metric)
                    if metric_stats:
                        results[platform] = metric_stats
                        
                        # Obtener datos diarios para gráfico
                        metrics_data = self.get_metrics(
                            channel_id=channel_id,
                            platform=platform,
                            start_date=analysis["start_date"],
                            end_date=analysis["end_date"],
                            metrics_filter=[metric]
                        )
                        
                        if "error" not in metrics_data:
                            # Procesar datos diarios
                            daily_data = self._process_daily_metrics(
                                metrics_data.get("results", []),
                                metric
                            )
                            platform_data[platform] = daily_data
            
            if not results:
                return {"error": "No hay datos suficientes para comparación"}
            
            # Generar visualización comparativa
            viz_path = self._generate_platform_comparison(
                platform_data,
                metric,
                f"{channel_id}_platform_comparison_{metric}_{period}",
                f"Comparación de {metric} entre plataformas para {channel_id} ({period})"
            )
            
            # Calcular ranking
            ranking = sorted(
                results.keys(),
                key=lambda x: results[x].get("mean", 0),
                reverse=True
            )
            
            # Preparar resultado
            comparison = {
                "channel_id": channel_id,
                "metric": metric,
                "period": period,
                "platforms": platforms,
                "results": results,
                "ranking": ranking,
                "best_platform": ranking[0] if ranking else None,
                "visualization": viz_path,
                "timestamp": datetime.now().isoformat()
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparando plataformas: {str(e)}")
            return {"error": str(e)}
    
    def _get_metric_type(self, metric: str) -> str:
        """Determina el tipo de una métrica."""
        for metric_type, metrics in self.config["metrics"].items():
            if metric in metrics:
                return metric_type
        return "engagement"  # Default
    
    def _process_daily_metrics(self, 
                              results: List[Dict[str, Any]], 
                              metric: str) -> Dict[str, float]:
        """Procesa métricas diarias para visualización."""
        daily_metrics = {}
        
        for result in results:
            date = result.get("date")
            value = result.get("metrics", {}).get(metric, 0)
            
            if date not in daily_metrics:
                daily_metrics[date] = []
            
            daily_metrics[date].append(value)
        
        # Calcular promedios diarios
        daily_averages = {}
        for date, values in daily_metrics.items():
            daily_averages[date] = sum(values) / len(values)
        
        return daily_averages
    
    def _generate_platform_comparison(self,
                                     platform_data: Dict[str, Dict[str, float]],
                                     metric: str,
                                     filename: str,
                                     title: str) -> str:
        """Genera una visualización comparativa entre plataformas."""
        try:
            if self.config["visualization"].get("interactive", False):
                # Crear DataFrame para Plotly
                data = []
                for platform, daily_metrics in platform_data.items():
                    for date, value in daily_metrics.items():
                        data.append({
                            "Fecha": date,
                            "Valor": value,
                            "Plataforma": platform
                        })
                
                df = pd.DataFrame(data)
                
                # Crear figura interactiva
                fig = px.line(
                    df,
                    x="Fecha",
                    y="Valor",
                    color="Plataforma",
                    title=title,
                    labels={"Valor": metric.capitalize()},
                    line_shape="spline",
                    markers=True
                )
                
                # Personalizar diseño
                fig.update_layout(
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                # Guardar como HTML
                output_path = f"{self.visualization_dir}/{filename}.html"
                fig.write_html(output_path)
            else:
                # Crear figura estática
                plt.figure(figsize=(12, 8))
                
                # Graficar cada plataforma
                for platform, daily_metrics in platform_data.items():
                    dates = sorted(daily_metrics.keys())
                    values = [daily_metrics[date] for date in dates]
                    plt.plot(dates, values, marker='o', linewidth=2, label=platform)
                
                # Añadir etiquetas y leyenda
                plt.title(title, fontsize=16)
                plt.xlabel("Fecha", fontsize=12)
                plt.ylabel(metric.capitalize(), fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Guardar figura
                output_path = f"{self.visualization_dir}/{filename}.png"
                plt.savefig(output_path, dpi=300)
                plt.close()
            
            logger.info(f"Comparación de plataformas generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando comparación de plataformas: {str(e)}")
            return ""
    
    def analyze_cta_performance(self,
                               channel_id: str,
                               platform: str,
                               days: int = 30) -> Dict[str, Any]:
        """
        Analiza el rendimiento de diferentes CTAs.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            days: Número de días a analizar
            
        Returns:
            Análisis de rendimiento de CTAs
        """
        try:
            # Determinar fechas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Consultar métricas con CTAs
            query = {
                "channel_id": channel_id,
                "platform": platform,
                "date": {
                    "$gte": start_date.strftime("%Y-%m-%d"),
                    "$lte": end_date.strftime("%Y-%m-%d")
                }
            }
            
            if self.db_client:
                db = self.db_client[self.config["database"]["name"]]
                collection = db["content_metrics"]
                
                # Obtener documentos con información de CTA
                cursor = collection.find(query)
                
                # Procesar resultados
                cta_performance = {}
                content_count = 0
                
                for doc in cursor:
                    content_id = doc.get("content_id")
                    metrics = doc.get("metrics", {})
                    cta_info = metrics.get("cta_info", {})
                    
                    if not cta_info:
                        continue
                    
                    content_count += 1
                    cta_type = cta_info.get("type", "unknown")
                    cta_position = cta_info.get("position", "unknown")
                    cta_text = cta_info.get("text", "unknown")
                    
                    # Métricas de conversión
                    ctr = metrics.get("click_through_rate", 0)
                    conversion = metrics.get("cta_conversion", 0)
                    
                    # Agrupar por tipo de CTA
                    if cta_type not in cta_performance:
                        cta_performance[cta_type] = {
                            "count": 0,
                            "ctr_sum": 0,
                            "conversion_sum": 0,
                            "positions": {},
                            "examples": []
                        }
                    
                    cta_data = cta_performance[cta_type]
                    cta_data["count"] += 1
                    cta_data["ctr_sum"] += ctr
                    cta_data["conversion_sum"] += conversion
                    
                    # Agrupar por posición
                    if cta_position not in cta_data["positions"]:
                        cta_data["positions"][cta_position] = {
                            "count": 0,
                            "ctr_sum": 0,
                            "conversion_sum": 0
                        }
                    
                    pos_data = cta_data["positions"][cta_position]
                    pos_data["count"] += 1
                    pos_data["ctr_sum"] += ctr
                    pos_data["conversion_sum"] += conversion
                    
                    # Guardar ejemplos
                    if len(cta_data["examples"]) < 5:
                        cta_data["examples"].append({
                            "text": cta_text,
                            "ctr": ctr,
                            "conversion": conversion,
                            "position": cta_position,
                            "content_id": content_id
                        })
                
                if not cta_performance:
                    return {"error": "No se encontraron CTAs para analizar"}
                
                # Calcular promedios
                for cta_type, data in cta_performance.items():
                    if data["count"] > 0:
                        data["avg_ctr"] = data["ctr_sum"] / data["count"]
                        data["avg_conversion"] = data["conversion_sum"] / data["count"]
                        
                        # Calcular promedios por posición
                        for pos, pos_data in data["positions"].items():
                            if pos_data["count"] > 0:
                                pos_data["avg_ctr"] = pos_data["ctr_sum"] / pos_data["count"]
                                pos_data["avg_conversion"] = pos_data["conversion_sum"] / pos_data["count"]
                
                # Determinar mejor tipo de CTA
                best_cta = max(
                    cta_performance.keys(),
                    key=lambda x: cta_performance[x].get("avg_conversion", 0)
                )
                
                                # Determinar mejor posición
                best_positions = {}
                for cta_type, data in cta_performance.items():
                    if data["positions"]:
                        best_position = max(
                            data["positions"].keys(),
                            key=lambda x: data["positions"][x].get("avg_conversion", 0)
                        )
                        best_positions[cta_type] = best_position
                
                # Generar visualización
                viz_path = self._generate_cta_performance_visualization(
                    cta_performance,
                    f"{channel_id}_{platform}_cta_performance_{days}d",
                    f"Rendimiento de CTAs para {channel_id} en {platform} (últimos {days} días)"
                )
                
                # Preparar resultado
                analysis = {
                    "channel_id": channel_id,
                    "platform": platform,
                    "period": f"{days}d",
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "content_count": content_count,
                    "cta_types": list(cta_performance.keys()),
                    "best_cta_type": best_cta,
                    "best_positions": best_positions,
                    "cta_performance": cta_performance,
                    "visualization": viz_path,
                    "timestamp": datetime.now().isoformat()
                }
                
                return analysis
            else:
                return {"error": "Base de datos no disponible"}
                
        except Exception as e:
            logger.error(f"Error analizando rendimiento de CTAs: {str(e)}")
            return {"error": str(e)}
    
    def _generate_cta_performance_visualization(self,
                                              cta_performance: Dict[str, Any],
                                              filename: str,
                                              title: str) -> str:
        """Genera una visualización del rendimiento de CTAs."""
        try:
            if self.config["visualization"].get("interactive", False):
                # Crear datos para visualización
                cta_types = []
                ctr_values = []
                conversion_values = []
                
                for cta_type, data in cta_performance.items():
                    if "avg_ctr" in data and "avg_conversion" in data:
                        cta_types.append(cta_type)
                        ctr_values.append(data["avg_ctr"] * 100)  # Convertir a porcentaje
                        conversion_values.append(data["avg_conversion"] * 100)  # Convertir a porcentaje
                
                # Crear figura
                fig = go.Figure()
                
                # Añadir barras para CTR
                fig.add_trace(go.Bar(
                    x=cta_types,
                    y=ctr_values,
                    name='CTR (%)',
                    marker_color='royalblue'
                ))
                
                # Añadir barras para Conversión
                fig.add_trace(go.Bar(
                    x=cta_types,
                    y=conversion_values,
                    name='Conversión (%)',
                    marker_color='firebrick'
                ))
                
                # Personalizar diseño
                fig.update_layout(
                    title=title,
                    xaxis_title='Tipo de CTA',
                    yaxis_title='Porcentaje (%)',
                    barmode='group',
                    template='plotly_dark',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Guardar como HTML
                output_path = f"{self.visualization_dir}/{filename}.html"
                fig.write_html(output_path)
            else:
                # Crear datos para visualización
                cta_types = []
                ctr_values = []
                conversion_values = []
                
                for cta_type, data in cta_performance.items():
                    if "avg_ctr" in data and "avg_conversion" in data:
                        cta_types.append(cta_type)
                        ctr_values.append(data["avg_ctr"] * 100)  # Convertir a porcentaje
                        conversion_values.append(data["avg_conversion"] * 100)  # Convertir a porcentaje
                
                # Crear figura
                plt.figure(figsize=(12, 8))
                
                # Configurar ancho de barras
                x = np.arange(len(cta_types))
                width = 0.35
                
                # Crear barras
                plt.bar(x - width/2, ctr_values, width, label='CTR (%)', color='royalblue')
                plt.bar(x + width/2, conversion_values, width, label='Conversión (%)', color='firebrick')
                
                # Añadir etiquetas y leyenda
                plt.title(title, fontsize=16)
                plt.xlabel('Tipo de CTA', fontsize=12)
                plt.ylabel('Porcentaje (%)', fontsize=12)
                plt.xticks(x, cta_types, rotation=45)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                
                # Guardar figura
                output_path = f"{self.visualization_dir}/{filename}.png"
                plt.savefig(output_path, dpi=300)
                plt.close()
            
            logger.info(f"Visualización de rendimiento de CTAs generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización de CTAs: {str(e)}")
            return ""
    
    def analyze_content_patterns(self,
                                channel_id: str,
                                platform: str,
                                days: int = 90,
                                min_content: int = 10) -> Dict[str, Any]:
        """
        Analiza patrones en el contenido para identificar factores de éxito.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            days: Número de días a analizar
            min_content: Mínimo de contenidos para análisis
            
        Returns:
            Análisis de patrones de contenido
        """
        try:
            # Determinar fechas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Consultar métricas
            metrics_data = self.get_metrics(
                channel_id=channel_id,
                platform=platform,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            if "error" in metrics_data:
                return {"error": metrics_data["error"]}
            
            # Procesar resultados
            results = metrics_data.get("results", [])
            
            if len(results) < min_content:
                return {"error": f"Insuficientes datos para análisis. Se requieren al menos {min_content} contenidos."}
            
            # Extraer características y métricas
            content_data = []
            for result in results:
                content_id = result.get("content_id")
                metrics = result.get("metrics", {})
                metadata = metrics.get("metadata", {})
                
                if not metadata:
                    continue
                
                # Extraer métricas de rendimiento
                engagement = (
                    metrics.get("views", 0) * 0.3 +
                    metrics.get("likes", 0) * 0.3 +
                    metrics.get("comments", 0) * 0.2 +
                    metrics.get("shares", 0) * 0.2
                )
                
                # Extraer características del contenido
                content_data.append({
                    "content_id": content_id,
                    "engagement": engagement,
                    "duration": metadata.get("duration", 0),
                    "publish_time": metadata.get("publish_time", ""),
                    "day_of_week": metadata.get("day_of_week", ""),
                    "has_hashtags": metadata.get("has_hashtags", False),
                    "hashtag_count": metadata.get("hashtag_count", 0),
                    "title_length": metadata.get("title_length", 0),
                    "description_length": metadata.get("description_length", 0),
                    "has_cta": "cta_info" in metrics,
                    "category": metadata.get("category", "unknown"),
                    "format": metadata.get("format", "unknown")
                })
            
            if not content_data:
                return {"error": "No se encontraron metadatos de contenido para análisis"}
            
            # Convertir a DataFrame para análisis
            df = pd.DataFrame(content_data)
            
            # Análisis por duración
            duration_analysis = self._analyze_numeric_factor(df, "duration", "engagement")
            
            # Análisis por día de la semana
            day_analysis = self._analyze_categorical_factor(df, "day_of_week", "engagement")
            
            # Análisis por formato
            format_analysis = self._analyze_categorical_factor(df, "format", "engagement")
            
            # Análisis por categoría
            category_analysis = self._analyze_categorical_factor(df, "category", "engagement")
            
            # Análisis por longitud de título
            title_length_analysis = self._analyze_numeric_factor(df, "title_length", "engagement")
            
            # Análisis por uso de hashtags
            hashtag_analysis = {
                "with_hashtags": {
                    "count": int(df[df["has_hashtags"] == True].shape[0]),
                    "avg_engagement": float(df[df["has_hashtags"] == True]["engagement"].mean())
                },
                "without_hashtags": {
                    "count": int(df[df["has_hashtags"] == False].shape[0]),
                    "avg_engagement": float(df[df["has_hashtags"] == False]["engagement"].mean())
                }
            }
            
            # Análisis por cantidad de hashtags
            hashtag_count_analysis = self._analyze_numeric_factor(df, "hashtag_count", "engagement")
            
            # Generar visualizaciones
            viz_paths = {}
            
            # Visualización de duración
            if len(df) >= 5:
                viz_paths["duration"] = self._generate_scatter_visualization(
                    df, 
                    "duration", 
                    "engagement",
                    f"{channel_id}_{platform}_duration_analysis_{days}d",
                    f"Relación entre duración y engagement para {channel_id} en {platform}"
                )
                
                # Visualización de día de la semana
                viz_paths["day_of_week"] = self._generate_categorical_visualization(
                    day_analysis,
                    f"{channel_id}_{platform}_day_analysis_{days}d",
                    f"Engagement por día de la semana para {channel_id} en {platform}"
                )
                
                # Visualización de formato
                viz_paths["format"] = self._generate_categorical_visualization(
                    format_analysis,
                    f"{channel_id}_{platform}_format_analysis_{days}d",
                    f"Engagement por formato para {channel_id} en {platform}"
                )
            
            # Preparar resultado
            analysis = {
                "channel_id": channel_id,
                "platform": platform,
                "period": f"{days}d",
                "content_count": len(content_data),
                "duration_analysis": duration_analysis,
                "day_analysis": day_analysis,
                "format_analysis": format_analysis,
                "category_analysis": category_analysis,
                "title_length_analysis": title_length_analysis,
                "hashtag_analysis": hashtag_analysis,
                "hashtag_count_analysis": hashtag_count_analysis,
                "visualizations": viz_paths,
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando patrones de contenido: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_numeric_factor(self, 
                               df: pd.DataFrame, 
                               factor: str, 
                               metric: str) -> Dict[str, Any]:
        """Analiza la relación entre un factor numérico y una métrica."""
        try:
            if df.empty or factor not in df.columns or metric not in df.columns:
                return {"error": "Datos insuficientes para análisis"}
            
            # Calcular correlación
            correlation = df[factor].corr(df[metric])
            
            # Agrupar por rangos
            factor_min = df[factor].min()
            factor_max = df[factor].max()
            
            # Determinar número de bins
            n_bins = min(5, len(df[factor].unique()))
            if n_bins < 2:
                n_bins = 2
                
            # Crear bins
            bins = np.linspace(factor_min, factor_max, n_bins + 1)
            labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
            
            # Agrupar
            df["bin"] = pd.cut(df[factor], bins=bins, labels=labels)
            grouped = df.groupby("bin")[metric].agg(["mean", "count"]).reset_index()
            
            # Convertir a diccionario
            ranges = {}
            for _, row in grouped.iterrows():
                ranges[row["bin"]] = {
                    "count": int(row["count"]),
                    "avg_metric": float(row["mean"])
                }
            
            # Determinar mejor rango
            if grouped.shape[0] > 0:
                best_range = grouped.loc[grouped["mean"].idxmax()]["bin"]
            else:
                best_range = None
            
            return {
                "correlation": float(correlation),
                "min": float(factor_min),
                "max": float(factor_max),
                "ranges": ranges,
                "best_range": best_range
            }
            
        except Exception as e:
            logger.error(f"Error analizando factor numérico: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_categorical_factor(self, 
                                   df: pd.DataFrame, 
                                   factor: str, 
                                   metric: str) -> Dict[str, Any]:
        """Analiza la relación entre un factor categórico y una métrica."""
        try:
            if df.empty or factor not in df.columns or metric not in df.columns:
                return {"error": "Datos insuficientes para análisis"}
            
            # Agrupar por categoría
            grouped = df.groupby(factor)[metric].agg(["mean", "count"]).reset_index()
            
            # Convertir a diccionario
            categories = {}
            for _, row in grouped.iterrows():
                categories[row[factor]] = {
                    "count": int(row["count"]),
                    "avg_metric": float(row["mean"])
                }
            
            # Determinar mejor categoría
            if grouped.shape[0] > 0:
                best_category = grouped.loc[grouped["mean"].idxmax()][factor]
            else:
                best_category = None
            
            return {
                "categories": categories,
                "best_category": best_category
            }
            
        except Exception as e:
            logger.error(f"Error analizando factor categórico: {str(e)}")
            return {"error": str(e)}
    
    def _generate_scatter_visualization(self,
                                       df: pd.DataFrame,
                                       x_col: str,
                                       y_col: str,
                                       filename: str,
                                       title: str) -> str:
        """Genera una visualización de dispersión."""
        try:
            if self.config["visualization"].get("interactive", False):
                # Crear figura interactiva
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    labels={x_col: x_col.capitalize(), y_col: y_col.capitalize()},
                    trendline="ols"
                )
                
                # Personalizar diseño
                fig.update_layout(
                    template="plotly_dark",
                    hovermode="closest"
                )
                
                # Guardar como HTML
                output_path = f"{self.visualization_dir}/{filename}.html"
                fig.write_html(output_path)
            else:
                # Crear figura estática
                plt.figure(figsize=(10, 6))
                
                # Graficar dispersión
                plt.scatter(df[x_col], df[y_col], alpha=0.7)
                
                # Añadir línea de tendencia
                z = np.polyfit(df[x_col], df[y_col], 1)
                p = np.poly1d(z)
                plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.7)
                
                # Añadir etiquetas
                plt.title(title, fontsize=14)
                plt.xlabel(x_col.capitalize(), fontsize=12)
                plt.ylabel(y_col.capitalize(), fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Guardar figura
                output_path = f"{self.visualization_dir}/{filename}.png"
                plt.savefig(output_path, dpi=300)
                plt.close()
            
            logger.info(f"Visualización de dispersión generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización de dispersión: {str(e)}")
            return ""
    
    def _generate_categorical_visualization(self,
                                          category_data: Dict[str, Any],
                                          filename: str,
                                          title: str) -> str:
        """Genera una visualización para factores categóricos."""
        try:
            categories = category_data.get("categories", {})
            
            if not categories:
                return ""
            
            # Preparar datos
            labels = list(categories.keys())
            values = [categories[label]["avg_metric"] for label in labels]
            counts = [categories[label]["count"] for label in labels]
            
            if self.config["visualization"].get("interactive", False):
                # Crear figura interactiva
                fig = go.Figure()
                
                # Añadir barras
                fig.add_trace(go.Bar(
                    x=labels,
                    y=values,
                    text=counts,
                    textposition="auto",
                    name="Engagement promedio"
                ))
                
                # Personalizar diseño
                fig.update_layout(
                    title=title,
                    xaxis_title="Categoría",
                    yaxis_title="Engagement promedio",
                    template="plotly_dark"
                )
                
                # Guardar como HTML
                output_path = f"{self.visualization_dir}/{filename}.html"
                fig.write_html(output_path)
            else:
                # Crear figura estática
                plt.figure(figsize=(12, 6))
                
                # Crear barras
                bars = plt.bar(labels, values, alpha=0.8)
                
                # Añadir etiquetas de conteo
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f"n={counts[i]}",
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )
                
                # Añadir etiquetas
                plt.title(title, fontsize=14)
                plt.xlabel("Categoría", fontsize=12)
                plt.ylabel("Engagement promedio", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                
                # Guardar figura
                output_path = f"{self.visualization_dir}/{filename}.png"
                plt.savefig(output_path, dpi=300)
                plt.close()
            
            logger.info(f"Visualización categórica generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización categórica: {str(e)}")
            return ""