import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import seaborn as sns
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/cohort_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cohort_analyzer")

class CohortAnalyzer:
    """
    Analiza cohortes de audiencia para entender patrones de retención,
    engagement y monetización a lo largo del tiempo.
    """
    
    def __init__(self, data_path: str = "data/analysis"):
        """
        Inicializa el analizador de cohortes.
        
        Args:
            data_path: Ruta para almacenar datos de análisis
        """
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Cargar configuración
        self.config_path = "config/platforms.json"
        self.config = self._load_config()
        
        # Inicializar almacenamiento de datos
        self.cohort_data = self._load_data("cohort_data.json", {})
        self.retention_metrics = self._load_data("retention_metrics.json", {})
        self.engagement_metrics = self._load_data("engagement_metrics.json", {})
        self.monetization_metrics = self._load_data("monetization_metrics.json", {})
        
        # Importar adaptadores de plataforma
        self.platform_adapters = {}
        self._load_platform_adapters()
        
        # Configurar visualización
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de plataformas"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _load_data(self, filename: str, default: Any) -> Any:
        """Carga datos desde un archivo JSON"""
        filepath = os.path.join(self.data_path, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error al cargar {filename}: {str(e)}")
            return default
    
    def _save_data(self, filename: str, data: Any) -> bool:
        """Guarda datos en un archivo JSON"""
        filepath = os.path.join(self.data_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error al guardar {filename}: {str(e)}")
            return False
    
    def _load_platform_adapters(self):
        """Carga los adaptadores de plataforma disponibles"""
        try:
            # Importar adaptadores dinámicamente
            from platform_adapters.youtube_adapter import YouTubeAdapter
            from platform_adapters.tiktok_adapter import TikTokAdapter
            from platform_adapters.instagram_adapter import InstagramAdapter
            
            # Inicializar adaptadores con configuración
            self.platform_adapters["youtube"] = YouTubeAdapter(self.config.get("youtube", {}))
            self.platform_adapters["tiktok"] = TikTokAdapter(self.config.get("tiktok", {}))
            self.platform_adapters["instagram"] = InstagramAdapter(self.config.get("instagram", {}))
            
            logger.info(f"Adaptadores de plataforma cargados: {list(self.platform_adapters.keys())}")
        except ImportError as e:
            logger.warning(f"No se pudieron cargar todos los adaptadores: {str(e)}")
        except Exception as e:
            logger.error(f"Error al inicializar adaptadores: {str(e)}")
    
    def create_cohort(self, platform: str, channel_id: str, cohort_name: str, 
                     start_date: str, end_date: str, segment_by: str = "acquisition_date") -> Dict[str, Any]:
        """
        Crea una nueva cohorte de audiencia para análisis.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram)
            channel_id: ID del canal
            cohort_name: Nombre de la cohorte
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            segment_by: Criterio de segmentación ('acquisition_date', 'content_type', 'source')
            
        Returns:
            Información de la cohorte creada
        """
        if platform not in self.platform_adapters:
            logger.error(f"Plataforma no soportada: {platform}")
            return {
                "status": "error",
                "message": f"Plataforma no soportada: {platform}"
            }
        
        try:
            # Validar fechas
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            if end_dt < start_dt:
                return {
                    "status": "error",
                    "message": "La fecha de fin debe ser posterior a la fecha de inicio"
                }
            
            # Obtener datos de audiencia
            adapter = self.platform_adapters[platform]
            audience_data = adapter.get_audience_data(channel_id, start_date, end_date)
            
            if not audience_data:
                return {
                    "status": "error",
                    "message": f"No se pudieron obtener datos de audiencia para el canal: {channel_id}"
                }
            
            # Crear cohorte
            cohort_id = f"{platform}_{channel_id}_{cohort_name}_{start_date}_{end_date}"
            
            # Segmentar audiencia según el criterio
            segments = self._segment_audience(audience_data, segment_by)
            
            # Crear estructura de cohorte
            cohort_info = {
                "cohort_id": cohort_id,
                "platform": platform,
                "channel_id": channel_id,
                "name": cohort_name,
                "start_date": start_date,
                "end_date": end_date,
                "segment_by": segment_by,
                "segments": segments,
                "creation_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metrics": {
                    "total_users": sum(len(users) for users in segments.values()),
                    "segments_count": len(segments)
                }
            }
            
            # Guardar cohorte
            self.cohort_data[cohort_id] = cohort_info
            self._save_data("cohort_data.json", self.cohort_data)
            
            return {
                "status": "success",
                "cohort_id": cohort_id,
                "message": f"Cohorte '{cohort_name}' creada exitosamente",
                "cohort": cohort_info
            }
            
        except Exception as e:
            logger.error(f"Error al crear cohorte: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al crear cohorte: {str(e)}"
            }
    
    def _segment_audience(self, audience_data: List[Dict[str, Any]], segment_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Segmenta la audiencia según el criterio especificado.
        
        Args:
            audience_data: Datos de audiencia
            segment_by: Criterio de segmentación
            
        Returns:
            Audiencia segmentada
        """
        segments = defaultdict(list)
        
        for user in audience_data:
            if segment_by == "acquisition_date":
                # Segmentar por semana de adquisición
                if "first_seen" in user:
                    try:
                        date = datetime.fromisoformat(user["first_seen"]).strftime("%Y-%U")  # Año-Semana
                        segments[f"Week_{date}"].append(user)
                    except (ValueError, TypeError):
                        segments["Unknown"].append(user)
                else:
                    segments["Unknown"].append(user)
                    
            elif segment_by == "content_type":
                # Segmentar por tipo de contenido que atrajo al usuario
                if "acquisition_content_type" in user:
                    segments[user["acquisition_content_type"]].append(user)
                else:
                    segments["Unknown"].append(user)
                    
            elif segment_by == "source":
                # Segmentar por fuente de adquisición
                if "acquisition_source" in user:
                    segments[user["acquisition_source"]].append(user)
                else:
                    segments["Unknown"].append(user)
                    
            else:
                # Segmentación por defecto
                segments["All"].append(user)
        
        return dict(segments)
    
    def analyze_retention(self, cohort_id: str, periods: int = 12, period_type: str = "week") -> Dict[str, Any]:
        """
        Analiza la retención de una cohorte a lo largo del tiempo.
        
        Args:
            cohort_id: ID de la cohorte
            periods: Número de períodos a analizar
            period_type: Tipo de período ('day', 'week', 'month')
            
        Returns:
            Análisis de retención
        """
        if cohort_id not in self.cohort_data:
            logger.error(f"Cohorte no encontrada: {cohort_id}")
            return {
                "status": "error",
                "message": f"Cohorte no encontrada: {cohort_id}"
            }
        
        try:
            # Obtener información de la cohorte
            cohort_info = self.cohort_data[cohort_id]
            platform = cohort_info["platform"]
            channel_id = cohort_info["channel_id"]
            start_date = datetime.fromisoformat(cohort_info["start_date"])
            end_date = datetime.fromisoformat(cohort_info["end_date"])
            
            # Calcular fechas de períodos
            period_dates = self._calculate_period_dates(start_date, periods, period_type)
            
            # Obtener datos de actividad
            adapter = self.platform_adapters[platform]
            activity_data = adapter.get_audience_activity(
                channel_id, 
                start_date.isoformat(), 
                (start_date + timedelta(days=periods * self._get_period_days(period_type))).isoformat()
            )
            
            if not activity_data:
                return {
                    "status": "error",
                    "message": f"No se pudieron obtener datos de actividad para el canal: {channel_id}"
                }
            
            # Calcular retención por segmento y período
            retention_matrix = {}
            segments = cohort_info["segments"]
            
            for segment_name, users in segments.items():
                retention_matrix[segment_name] = self._calculate_segment_retention(
                    users, activity_data, period_dates, period_type
                )
            
            # Calcular retención promedio
            avg_retention = self._calculate_average_retention(retention_matrix)
            
            # Generar visualización
            visualization_path = self._generate_retention_heatmap(
                retention_matrix, 
                avg_retention, 
                cohort_id, 
                period_type
            )
            
            # Guardar resultados
            retention_analysis = {
                "cohort_id": cohort_id,
                "analysis_date": datetime.now().isoformat(),
                "periods": periods,
                "period_type": period_type,
                "retention_matrix": retention_matrix,
                "average_retention": avg_retention,
                "visualization_path": visualization_path
            }
            
            self.retention_metrics[cohort_id] = retention_analysis
            self._save_data("retention_metrics.json", self.retention_metrics)
            
            return {
                "status": "success",
                "cohort_id": cohort_id,
                "analysis": retention_analysis
            }
            
        except Exception as e:
            logger.error(f"Error al analizar retención: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar retención: {str(e)}"
            }
    
    def _calculate_period_dates(self, start_date: datetime, periods: int, period_type: str) -> List[datetime]:
        """
        Calcula las fechas de los períodos para el análisis.
        
        Args:
            start_date: Fecha de inicio
            periods: Número de períodos
            period_type: Tipo de período ('day', 'week', 'month')
            
        Returns:
            Lista de fechas de períodos
        """
        period_dates = [start_date]
        days_per_period = self._get_period_days(period_type)
        
        for i in range(1, periods + 1):
            period_dates.append(start_date + timedelta(days=i * days_per_period))
        
        return period_dates
    
    def _get_period_days(self, period_type: str) -> int:
        """
        Obtiene el número de días por período.
        
        Args:
            period_type: Tipo de período ('day', 'week', 'month')
            
        Returns:
            Número de días
        """
        if period_type == "day":
            return 1
        elif period_type == "week":
            return 7
        elif period_type == "month":
            return 30
        else:
            return 7  # Por defecto, semana
    
    def _calculate_segment_retention(self, users: List[Dict[str, Any]], 
                                    activity_data: List[Dict[str, Any]],
                                    period_dates: List[datetime],
                                    period_type: str) -> List[float]:
        """
        Calcula la retención de un segmento a lo largo del tiempo.
        
        Args:
            users: Lista de usuarios del segmento
            activity_data: Datos de actividad
            period_dates: Fechas de los períodos
            period_type: Tipo de período
            
        Returns:
            Lista de tasas de retención por período
        """
        # Extraer IDs de usuarios
        user_ids = [user.get("user_id") for user in users if "user_id" in user]
        total_users = len(user_ids)
        
        if total_users == 0:
            return [0.0] * len(period_dates)
        
        # Calcular retención por período
        retention_rates = [100.0]  # El primer período siempre es 100%
        
        for i in range(1, len(period_dates)):
            period_start = period_dates[i-1]
            period_end = period_dates[i]
            
            # Contar usuarios activos en este período
            active_users = 0
            for activity in activity_data:
                if (activity.get("user_id") in user_ids and 
                    "activity_date" in activity):
                    try:
                        activity_date = datetime.fromisoformat(activity["activity_date"])
                        if period_start <= activity_date < period_end:
                            active_users += 1
                            break  # Contar cada usuario una vez por período
                    except (ValueError, TypeError):
                        continue
            
            # Calcular tasa de retención
            retention_rate = (active_users / total_users) * 100 if total_users > 0 else 0
            retention_rates.append(round(retention_rate, 2))
        
        return retention_rates
    
    def _calculate_average_retention(self, retention_matrix: Dict[str, List[float]]) -> List[float]:
        """
        Calcula la retención promedio de todos los segmentos.
        
        Args:
            retention_matrix: Matriz de retención por segmento
            
        Returns:
            Lista de tasas de retención promedio
        """
        if not retention_matrix:
            return []
        
        # Obtener el número de períodos
        num_periods = len(next(iter(retention_matrix.values())))
        
        # Calcular promedio por período
        avg_retention = []
        for period in range(num_periods):
            period_sum = sum(rates[period] for rates in retention_matrix.values() if len(rates) > period)
            period_count = sum(1 for rates in retention_matrix.values() if len(rates) > period)
            
            if period_count > 0:
                avg_retention.append(round(period_sum / period_count, 2))
            else:
                avg_retention.append(0.0)
        
        return avg_retention
    
    def _generate_retention_heatmap(self, retention_matrix: Dict[str, List[float]], 
                                   avg_retention: List[float], 
                                   cohort_id: str, 
                                   period_type: str) -> str:
        """
        Genera un mapa de calor de retención.
        
        Args:
            retention_matrix: Matriz de retención por segmento
            avg_retention: Retención promedio
            cohort_id: ID de la cohorte
            period_type: Tipo de período
            
        Returns:
            Ruta del archivo de visualización
        """
        # Crear directorio para visualizaciones
        viz_dir = os.path.join(self.data_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Preparar datos para el mapa de calor
        segments = list(retention_matrix.keys())
        periods = [f"{period_type.capitalize()} {i}" for i in range(len(avg_retention))]
        
        # Crear DataFrame
        data = []
        for segment in segments:
            for i, rate in enumerate(retention_matrix[segment]):
                data.append({
                    "Segment": segment,
                    "Period": periods[i],
                    "Retention": rate
                })
        
        df = pd.DataFrame(data)
        pivot_table = df.pivot(index="Segment", columns="Period", values="Retention")
        
        # Crear visualización
        plt.figure(figsize=(14, 10))
        
        # Mapa de calor
        ax = sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".1f", 
                         linewidths=.5, cbar_kws={"label": "Retention %"})
        
        plt.title(f"Cohort Retention Analysis - {cohort_id}", fontsize=16)
        plt.ylabel("Segment", fontsize=12)
        plt.xlabel(f"Period ({period_type}s)", fontsize=12)
        
        # Añadir línea de retención promedio
        plt.figure(figsize=(14, 6))
        plt.plot(periods, avg_retention, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title(f"Average Retention - {cohort_id}", fontsize=16)
        plt.ylabel("Retention %", fontsize=12)
        plt.xlabel(f"Period ({period_type}s)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 105)
        
        for i, rate in enumerate(avg_retention):
            plt.text(i, rate + 2, f"{rate}%", ha='center')
        
        # Guardar visualizaciones
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_path = os.path.join(viz_dir, f"retention_heatmap_{cohort_id}_{timestamp}.png")
        line_path = os.path.join(viz_dir, f"retention_line_{cohort_id}_{timestamp}.png")
        
        plt.savefig(line_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return {
            "heatmap": heatmap_path,
            "line_chart": line_path
        }
    
    def analyze_engagement(self, cohort_id: str, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analiza el engagement de una cohorte a lo largo del tiempo.
        
        Args:
            cohort_id: ID de la cohorte
            metrics: Lista de métricas a analizar (views, likes, comments, shares, ctr, watch_time)
            
        Returns:
            Análisis de engagement
        """
        if cohort_id not in self.cohort_data:
            logger.error(f"Cohorte no encontrada: {cohort_id}")
            return {
                "status": "error",
                "message": f"Cohorte no encontrada: {cohort_id}"
            }
        
        # Métricas por defecto si no se especifican
        if metrics is None:
            metrics = ["views", "likes", "comments", "shares", "ctr", "watch_time"]
        
        try:
            # Obtener información de la cohorte
            cohort_info = self.cohort_data[cohort_id]
            platform = cohort_info["platform"]
            channel_id = cohort_info["channel_id"]
            start_date = cohort_info["start_date"]
            end_date = cohort_info["end_date"]
            
            # Obtener datos de engagement
            adapter = self.platform_adapters[platform]
            engagement_data = adapter.get_engagement_metrics(
                channel_id, start_date, end_date, metrics
            )
            
            if not engagement_data:
                return {
                    "status": "error",
                    "message": f"No se pudieron obtener datos de engagement para el canal: {channel_id}"
                }
            
            # Calcular engagement por segmento
            segments = cohort_info["segments"]
            segment_engagement = {}
            
            for segment_name, users in segments.items():
                segment_engagement[segment_name] = self._calculate_segment_engagement(
                    users, engagement_data, metrics
                )
            
            # Calcular engagement promedio
            avg_engagement = self._calculate_average_engagement(segment_engagement, metrics)
            
            # Generar visualización
            visualization_path = self._generate_engagement_charts(
                segment_engagement, 
                avg_engagement, 
                cohort_id, 
                metrics
            )
            
            # Guardar resultados
            engagement_analysis = {
                "cohort_id": cohort_id,
                "analysis_date": datetime.now().isoformat(),
                "metrics": metrics,
                "segment_engagement": segment_engagement,
                "average_engagement": avg_engagement,
                "visualization_path": visualization_path
            }
            
            self.engagement_metrics[cohort_id] = engagement_analysis
            self._save_data("engagement_metrics.json", self.engagement_metrics)
            
            return {
                "status": "success",
                "cohort_id": cohort_id,
                "analysis": engagement_analysis
            }
            
        except Exception as e:
            logger.error(f"Error al analizar engagement: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar engagement: {str(e)}"
            }
    
    def _calculate_segment_engagement(self, users: List[Dict[str, Any]], 
                                     engagement_data: List[Dict[str, Any]],
                                     metrics: List[str]) -> Dict[str, Any]:
        """
        Calcula el engagement de un segmento.
        
        Args:
            users: Lista de usuarios del segmento
            engagement_data: Datos de engagement
            metrics: Lista de métricas a analizar
            
        Returns:
            Métricas de engagement del segmento
        """
        # Extraer IDs de usuarios
        user_ids = [user.get("user_id") for user in users if "user_id" in user]
        
        if not user_ids:
            return {metric: 0 for metric in metrics}
        
        # Filtrar datos de engagement para los usuarios del segmento
        segment_data = [
            data for data in engagement_data 
            if data.get("user_id") in user_ids
        ]
        
        # Calcular métricas de engagement
        result = {}
        
        for metric in metrics:
            if metric == "ctr":
                # Click-through rate
                impressions = sum(data.get("impressions", 0) for data in segment_data)
                clicks = sum(data.get("clicks", 0) for data in segment_data)
                result[metric] = round((clicks / impressions) * 100, 2) if impressions > 0 else 0
                
            elif metric == "watch_time":
                # Tiempo de visualización promedio (en segundos)
                total_watch_time = sum(data.get("watch_time", 0) for data in segment_data)
                total_views = sum(data.get("views", 0) for data in segment_data)
                result[metric] = round(total_watch_time / total_views, 2) if total_views > 0 else 0
                
            else:
                # Métricas simples (vistas, likes, comentarios, compartidos)
                result[metric] = sum(data.get(metric, 0) for data in segment_data)
        
        # Calcular engagement rate
        total_interactions = sum(result.get(m, 0) for m in ["likes", "comments", "shares"])
        total_views = result.get("views", 0)
        
        result["engagement_rate"] = round((total_interactions / total_views) * 100, 2) if total_views > 0 else 0
        
        return result
    
    def _calculate_average_engagement(self, segment_engagement: Dict[str, Dict[str, Any]], 
                                     metrics: List[str]) -> Dict[str, float]:
        """
        Calcula el engagement promedio de todos los segmentos.
        
        Args:
            segment_engagement: Engagement por segmento
            metrics: Lista de métricas
            
        Returns:
            Engagement promedio
        """
        if not segment_engagement:
            return {metric: 0 for metric in metrics + ["engagement_rate"]}
        
        # Calcular promedio por métrica
        avg_engagement = {}
        
        for metric in metrics + ["engagement_rate"]:
            metric_sum = sum(segment.get(metric, 0) for segment in segment_engagement.values())
            metric_count = sum(1 for segment in segment_engagement.values() if metric in segment)
            
            if metric_count > 0:
                avg_engagement[metric] = round(metric_sum / metric_count, 2)
            else:
                avg_engagement[metric] = 0
        
        return avg_engagement
    
    def _generate_engagement_charts(self, segment_engagement: Dict[str, Dict[str, Any]], 
                                   avg_engagement: Dict[str, float], 
                                   cohort_id: str, 
                                   metrics: List[str]) -> Dict[str, str]:
        """
        Genera gráficos de engagement.
        
        Args:
            segment_engagement: Engagement por segmento
            avg_engagement: Engagement promedio
            cohort_id: ID de la cohorte
            metrics: Lista de métricas
            
        Returns:
            Rutas de los archivos de visualización
        """
        # Crear directorio para visualizaciones
        viz_dir = os.path.join(self.data_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Timestamp para nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Diccionario para almacenar rutas
        visualization_paths = {}
        
        # Gráfico de barras para engagement rate
        plt.figure(figsize=(14, 8))
        segments = list(segment_engagement.keys())
        engagement_rates = [segment.get("engagement_rate", 0) for segment in segment_engagement.values()]
        
        bars = plt.bar(segments, engagement_rates, color='skyblue')
        plt.axhline(y=avg_engagement.get("engagement_rate", 0), color='r', linestyle='-', label='Average')
        
        plt.title(f"Engagement Rate by Segment - {cohort_id}", fontsize=16)
        plt.ylabel("Engagement Rate (%)", fontsize=12)
        plt.xlabel("Segment", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}%', ha='center', va='bottom')
        
        # Guardar gráfico
        engagement_rate_path = os.path.join(viz_dir, f"engagement_rate_{cohort_id}_{timestamp}.png")
        plt.savefig(engagement_rate_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["engagement_rate"] = engagement_rate_path
        
        # Gráfico de radar para todas las métricas
        metrics_to_plot = [m for m in metrics if m not in ["views"]]  # Excluir vistas para mejor escala
        
        if metrics_to_plot:
            # Preparar datos para el gráfico de radar
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Número de variables
            N = len(metrics_to_plot)
            
            # Ángulos para cada variable
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Cerrar el polígono
            
                        # Normalizar valores para el radar
            max_values = {
                metric: max([segment.get(metric, 0) for segment in segment_engagement.values()])
                for metric in metrics_to_plot
            }
            
            # Añadir valores normalizados para cada segmento
            for segment_name, segment_data in segment_engagement.items():
                values = []
                for metric in metrics_to_plot:
                    # Normalizar valor (0-1)
                    if max_values[metric] > 0:
                        values.append(segment_data.get(metric, 0) / max_values[metric])
                    else:
                        values.append(0)
                
                # Cerrar el polígono
                values += values[:1]
                
                # Dibujar el polígono
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=segment_name)
                ax.fill(angles, values, alpha=0.1)
            
            # Añadir etiquetas
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_to_plot)
            
            # Añadir título y leyenda
            plt.title(f"Engagement Metrics by Segment - {cohort_id}", fontsize=16)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Guardar gráfico
            radar_path = os.path.join(viz_dir, f"engagement_radar_{cohort_id}_{timestamp}.png")
            plt.savefig(radar_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            visualization_paths["radar"] = radar_path
        
        return visualization_paths
    
    def analyze_monetization(self, cohort_id: str, revenue_metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analiza la monetización de una cohorte a lo largo del tiempo.
        
        Args:
            cohort_id: ID de la cohorte
            revenue_metrics: Lista de métricas de ingresos a analizar (ad_revenue, subscription, donations, etc.)
            
        Returns:
            Análisis de monetización
        """
        if cohort_id not in self.cohort_data:
            logger.error(f"Cohorte no encontrada: {cohort_id}")
            return {
                "status": "error",
                "message": f"Cohorte no encontrada: {cohort_id}"
            }
        
        # Métricas por defecto si no se especifican
        if revenue_metrics is None:
            revenue_metrics = ["ad_revenue", "subscription", "donations", "merchandise", "sponsorships"]
        
        try:
            # Obtener información de la cohorte
            cohort_info = self.cohort_data[cohort_id]
            platform = cohort_info["platform"]
            channel_id = cohort_info["channel_id"]
            start_date = cohort_info["start_date"]
            end_date = cohort_info["end_date"]
            
            # Obtener datos de monetización
            adapter = self.platform_adapters[platform]
            monetization_data = adapter.get_monetization_metrics(
                channel_id, start_date, end_date, revenue_metrics
            )
            
            if not monetization_data:
                return {
                    "status": "error",
                    "message": f"No se pudieron obtener datos de monetización para el canal: {channel_id}"
                }
            
            # Calcular monetización por segmento
            segments = cohort_info["segments"]
            segment_monetization = {}
            
            for segment_name, users in segments.items():
                segment_monetization[segment_name] = self._calculate_segment_monetization(
                    users, monetization_data, revenue_metrics
                )
            
            # Calcular LTV (Lifetime Value) por segmento
            ltv_by_segment = self._calculate_ltv_by_segment(segment_monetization)
            
            # Calcular ARPU (Average Revenue Per User) por segmento
            arpu_by_segment = self._calculate_arpu_by_segment(segment_monetization, segments)
            
            # Generar visualización
            visualization_path = self._generate_monetization_charts(
                segment_monetization, 
                ltv_by_segment,
                arpu_by_segment,
                cohort_id, 
                revenue_metrics
            )
            
            # Guardar resultados
            monetization_analysis = {
                "cohort_id": cohort_id,
                "analysis_date": datetime.now().isoformat(),
                "revenue_metrics": revenue_metrics,
                "segment_monetization": segment_monetization,
                "ltv_by_segment": ltv_by_segment,
                "arpu_by_segment": arpu_by_segment,
                "visualization_path": visualization_path
            }
            
            self.monetization_metrics[cohort_id] = monetization_analysis
            self._save_data("monetization_metrics.json", self.monetization_metrics)
            
            return {
                "status": "success",
                "cohort_id": cohort_id,
                "analysis": monetization_analysis
            }
            
        except Exception as e:
            logger.error(f"Error al analizar monetización: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar monetización: {str(e)}"
            }
    
    def _calculate_segment_monetization(self, users: List[Dict[str, Any]], 
                                       monetization_data: List[Dict[str, Any]],
                                       revenue_metrics: List[str]) -> Dict[str, Any]:
        """
        Calcula la monetización de un segmento.
        
        Args:
            users: Lista de usuarios del segmento
            monetization_data: Datos de monetización
            revenue_metrics: Lista de métricas de ingresos
            
        Returns:
            Métricas de monetización del segmento
        """
        # Extraer IDs de usuarios
        user_ids = [user.get("user_id") for user in users if "user_id" in user]
        
        if not user_ids:
            return {metric: 0 for metric in revenue_metrics + ["total_revenue"]}
        
        # Filtrar datos de monetización para los usuarios del segmento
        segment_data = [
            data for data in monetization_data 
            if data.get("user_id") in user_ids
        ]
        
        # Calcular métricas de monetización
        result = {}
        
        for metric in revenue_metrics:
            result[metric] = sum(data.get(metric, 0) for data in segment_data)
        
        # Calcular ingresos totales
        result["total_revenue"] = sum(result.get(metric, 0) for metric in revenue_metrics)
        
        return result
    
    def _calculate_ltv_by_segment(self, segment_monetization: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calcula el LTV (Lifetime Value) por segmento.
        
        Args:
            segment_monetization: Monetización por segmento
            
        Returns:
            LTV por segmento
        """
        ltv_by_segment = {}
        
        for segment_name, monetization in segment_monetization.items():
            ltv_by_segment[segment_name] = monetization.get("total_revenue", 0)
        
        return ltv_by_segment
    
    def _calculate_arpu_by_segment(self, segment_monetization: Dict[str, Dict[str, Any]], 
                                  segments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calcula el ARPU (Average Revenue Per User) por segmento.
        
        Args:
            segment_monetization: Monetización por segmento
            segments: Segmentos de usuarios
            
        Returns:
            ARPU por segmento
        """
        arpu_by_segment = {}
        
        for segment_name, monetization in segment_monetization.items():
            users_count = len(segments.get(segment_name, []))
            total_revenue = monetization.get("total_revenue", 0)
            
            if users_count > 0:
                arpu_by_segment[segment_name] = round(total_revenue / users_count, 2)
            else:
                arpu_by_segment[segment_name] = 0
        
        return arpu_by_segment
    
    def _generate_monetization_charts(self, segment_monetization: Dict[str, Dict[str, Any]], 
                                     ltv_by_segment: Dict[str, float],
                                     arpu_by_segment: Dict[str, float],
                                     cohort_id: str, 
                                     revenue_metrics: List[str]) -> Dict[str, str]:
        """
        Genera gráficos de monetización.
        
        Args:
            segment_monetization: Monetización por segmento
            ltv_by_segment: LTV por segmento
            arpu_by_segment: ARPU por segmento
            cohort_id: ID de la cohorte
            revenue_metrics: Lista de métricas de ingresos
            
        Returns:
            Rutas de los archivos de visualización
        """
        # Crear directorio para visualizaciones
        viz_dir = os.path.join(self.data_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Timestamp para nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Diccionario para almacenar rutas
        visualization_paths = {}
        
        # Gráfico de barras apiladas para fuentes de ingresos por segmento
        plt.figure(figsize=(14, 8))
        
        segments = list(segment_monetization.keys())
        bottom = np.zeros(len(segments))
        
        for metric in revenue_metrics:
            values = [segment_monetization[segment].get(metric, 0) for segment in segments]
            plt.bar(segments, values, bottom=bottom, label=metric)
            bottom += values
        
        plt.title(f"Revenue Sources by Segment - {cohort_id}", fontsize=16)
        plt.ylabel("Revenue", fontsize=12)
        plt.xlabel("Segment", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Guardar gráfico
        revenue_path = os.path.join(viz_dir, f"revenue_sources_{cohort_id}_{timestamp}.png")
        plt.savefig(revenue_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["revenue_sources"] = revenue_path
        
        # Gráfico de barras para LTV por segmento
        plt.figure(figsize=(14, 8))
        
        segments = list(ltv_by_segment.keys())
        ltv_values = [ltv_by_segment[segment] for segment in segments]
        
        bars = plt.bar(segments, ltv_values, color='green')
        
        plt.title(f"Lifetime Value (LTV) by Segment - {cohort_id}", fontsize=16)
        plt.ylabel("LTV", fontsize=12)
        plt.xlabel("Segment", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # Guardar gráfico
        ltv_path = os.path.join(viz_dir, f"ltv_{cohort_id}_{timestamp}.png")
        plt.savefig(ltv_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["ltv"] = ltv_path
        
        # Gráfico de barras para ARPU por segmento
        plt.figure(figsize=(14, 8))
        
        segments = list(arpu_by_segment.keys())
        arpu_values = [arpu_by_segment[segment] for segment in segments]
        
        bars = plt.bar(segments, arpu_values, color='purple')
        
        plt.title(f"Average Revenue Per User (ARPU) by Segment - {cohort_id}", fontsize=16)
        plt.ylabel("ARPU", fontsize=12)
        plt.xlabel("Segment", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # Guardar gráfico
        arpu_path = os.path.join(viz_dir, f"arpu_{cohort_id}_{timestamp}.png")
        plt.savefig(arpu_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        visualization_paths["arpu"] = arpu_path
        
        return visualization_paths
    
    def compare_cohorts(self, cohort_ids: List[str], metric_type: str = "retention") -> Dict[str, Any]:
        """
        Compara múltiples cohortes según un tipo de métrica.
        
        Args:
            cohort_ids: Lista de IDs de cohortes a comparar
            metric_type: Tipo de métrica ('retention', 'engagement', 'monetization')
            
        Returns:
            Comparación de cohortes
        """
        # Verificar que todas las cohortes existan
        for cohort_id in cohort_ids:
            if cohort_id not in self.cohort_data:
                logger.error(f"Cohorte no encontrada: {cohort_id}")
                return {
                    "status": "error",
                    "message": f"Cohorte no encontrada: {cohort_id}"
                }
        
        try:
            # Obtener datos de métricas según el tipo
            if metric_type == "retention":
                metrics_data = self.retention_metrics
                comparison_key = "average_retention"
                chart_title = "Retention Comparison"
                y_label = "Retention %"
            elif metric_type == "engagement":
                metrics_data = self.engagement_metrics
                comparison_key = "average_engagement"
                chart_title = "Engagement Comparison"
                y_label = "Engagement Rate %"
            elif metric_type == "monetization":
                metrics_data = self.monetization_metrics
                comparison_key = "arpu_by_segment"
                chart_title = "ARPU Comparison"
                y_label = "ARPU ($)"
            else:
                return {
                    "status": "error",
                    "message": f"Tipo de métrica no válido: {metric_type}"
                }
            
            # Verificar que haya datos de métricas para todas las cohortes
            for cohort_id in cohort_ids:
                if cohort_id not in metrics_data:
                    logger.warning(f"No hay datos de {metric_type} para la cohorte: {cohort_id}")
                    return {
                        "status": "warning",
                        "message": f"No hay datos de {metric_type} para la cohorte: {cohort_id}"
                    }
            
            # Preparar datos para comparación
            comparison_data = {}
            
            for cohort_id in cohort_ids:
                cohort_name = self.cohort_data[cohort_id]["name"]
                comparison_data[cohort_name] = metrics_data[cohort_id][comparison_key]
            
            # Generar visualización
            visualization_path = self._generate_comparison_chart(
                comparison_data, 
                chart_title, 
                y_label, 
                metric_type
            )
            
            return {
                "status": "success",
                "comparison_type": metric_type,
                "cohorts": [self.cohort_data[cohort_id]["name"] for cohort_id in cohort_ids],
                "comparison_data": comparison_data,
                "visualization_path": visualization_path
            }
            
        except Exception as e:
            logger.error(f"Error al comparar cohortes: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al comparar cohortes: {str(e)}"
            }
    
    def _generate_comparison_chart(self, comparison_data: Dict[str, Any], 
                                  chart_title: str, 
                                  y_label: str, 
                                  metric_type: str) -> str:
        """
        Genera un gráfico de comparación entre cohortes.
        
        Args:
            comparison_data: Datos de comparación
            chart_title: Título del gráfico
            y_label: Etiqueta del eje Y
            metric_type: Tipo de métrica
            
        Returns:
            Ruta del archivo de visualización
        """
        # Crear directorio para visualizaciones
        viz_dir = os.path.join(self.data_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Timestamp para nombres de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear visualización según el tipo de métrica
        if metric_type == "retention":
            # Gráfico de líneas para retención
            plt.figure(figsize=(14, 8))
            
            for cohort_name, retention_data in comparison_data.items():
                periods = [f"Period {i}" for i in range(len(retention_data))]
                plt.plot(periods, retention_data, marker='o', linewidth=2, label=cohort_name)
            
            plt.title(chart_title, fontsize=16)
            plt.ylabel(y_label, fontsize=12)
            plt.xlabel("Period", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Guardar gráfico
            chart_path = os.path.join(viz_dir, f"retention_comparison_{timestamp}.png")
            
        elif metric_type == "engagement":
            # Gráfico de barras para engagement
            plt.figure(figsize=(14, 8))
            
            cohorts = list(comparison_data.keys())
            engagement_rates = [data.get("engagement_rate", 0) for data in comparison_data.values()]
            
            bars = plt.bar(cohorts, engagement_rates, color='skyblue')
            
            plt.title(chart_title, fontsize=16)
            plt.ylabel(y_label, fontsize=12)
            plt.xlabel("Cohort", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height}%', ha='center', va='bottom')
            
            # Guardar gráfico
            chart_path = os.path.join(viz_dir, f"engagement_comparison_{timestamp}.png")
            
        elif metric_type == "monetization":
            # Gráfico de barras para ARPU
            plt.figure(figsize=(14, 8))
            
            # Preparar datos
            all_segments = set()
            for arpu_data in comparison_data.values():
                all_segments.update(arpu_data.keys())
            
            all_segments = sorted(list(all_segments))
            
            # Configurar gráfico
            x = np.arange(len(all_segments))
            width = 0.8 / len(comparison_data)
            
            # Dibujar barras para cada cohorte
            for i, (cohort_name, arpu_data) in enumerate(comparison_data.items()):
                arpu_values = [arpu_data.get(segment, 0) for segment in all_segments]
                offset = i * width - (len(comparison_data) - 1) * width / 2
                plt.bar(x + offset, arpu_values, width, label=cohort_name)
            
            plt.title(chart_title, fontsize=16)
            plt.ylabel(y_label, fontsize=12)
            plt.xlabel("Segment", fontsize=12)
            plt.xticks(x, all_segments, rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Guardar gráfico
            chart_path = os.path.join(viz_dir, f"monetization_comparison_{timestamp}.png")
            
        else:
            # Tipo no soportado
            return ""
        
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return chart_path
    
    def export_cohort_data(self, cohort_id: str, format_type: str = "json") -> Dict[str, Any]:
        """
        Exporta los datos de una cohorte para uso externo.
        
        Args:
            cohort_id: ID de la cohorte
            format_type: Formato de exportación (json, csv)
            
        Returns:
            Resultado de la exportación
        """
        if cohort_id not in self.cohort_data:
            logger.error(f"Cohorte no encontrada: {cohort_id}")
            return {
                "status": "error",
                "message": f"Cohorte no encontrada: {cohort_id}"
            }
        
        valid_formats = ["json", "csv"]
        if format_type not in valid_formats:
            return {
                "status": "error",
                "message": f"Formato no válido. Debe ser uno de: {', '.join(valid_formats)}"
            }
        
        try:
            # Crear directorio de exportación
            export_dir = os.path.join(self.data_path, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            # Obtener datos de la cohorte
            cohort_info = self.cohort_data[cohort_id]
            
            # Obtener métricas si existen
            retention_data = self.retention_metrics.get(cohort_id, {})
            engagement_data = self.engagement_metrics.get(cohort_id, {})
            monetization_data = self.monetization_metrics.get(cohort_id, {})
            
            # Preparar datos para exportación
            export_data = {
                "cohort_info": cohort_info,
                "retention": retention_data,
                "engagement": engagement_data,
                "monetization": monetization_data
            }
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cohort_{cohort_id}_{timestamp}.{format_type}"
            filepath = os.path.join(export_dir, filename)
            
            # Exportar datos
            if format_type == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=4)
            
            elif format_type == "csv":
                # Para CSV, exportar tablas separadas
                import csv
                
                # Exportar información de la cohorte
                info_filepath = os.path.join(export_dir, f"cohort_info_{cohort_id}_{timestamp}.csv")
                with open(info_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["cohort_id", "platform", "channel_id", "name", "start_date", "end_date", "segment_by", "total_users", "segments_count"])
                    writer.writerow([
                        cohort_info.get("cohort_id", ""),
                        cohort_info.get("platform", ""),
                        cohort_info.get("channel_id", ""),
                        cohort_info.get("name", ""),
                        cohort_info.get("start_date", ""),
                        cohort_info.get("end_date", ""),
                        cohort_info.get("segment_by", ""),
                        cohort_info.get("metrics", {}).get("total_users", 0),
                        cohort_info.get("metrics", {}).get("segments_count", 0)
                    ])
                
                # Exportar datos de retención
                if retention_data:
                    retention_filepath = os.path.join(export_dir, f"cohort_retention_{cohort_id}_{timestamp}.csv")
                    with open(retention_filepath, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        
                        # Encabezados
                        periods = [f"Period_{i}" for i in range(len(retention_data.get("average_retention", [])))]
                        writer.writerow(["segment"] + periods)
                        
                        # Datos de retención por segmento
                        for segment, rates in retention_data.get("retention_matrix", {}).items():
                            writer.writerow([segment] + rates)
                        
                        # Promedio
                        writer.writerow(["average"] + retention_data.get("average_retention", []))
                
                # Actualizar filepath para incluir todos los archivos
                filepath = export_dir
            
            logger.info(f"Datos de cohorte exportados: {filepath}")
            
            return {
                "status": "success",
                "message": "Datos exportados correctamente",
                "filepath": filepath,
                "format": format_type
            }
            
        except Exception as e:
            logger.error(f"Error al exportar datos: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al exportar datos: {str(e)}"
            }