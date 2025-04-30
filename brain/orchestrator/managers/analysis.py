"""
AnalysisManager - Gestor de análisis de contenido y rendimiento

Este módulo se encarga de analizar el rendimiento del contenido publicado,
identificar patrones, tendencias y oportunidades de optimización.
"""

import logging
import datetime
import uuid
import threading
import json
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum, auto
from collections import defaultdict

# Configuración de logging
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Tipos de análisis disponibles"""
    PERFORMANCE = auto()
    ENGAGEMENT = auto()
    AUDIENCE = auto()
    CONTENT = auto()
    MONETIZATION = auto()
    TREND = auto()
    COMPETITOR = auto()
    SENTIMENT = auto()
    CUSTOM = auto()

class AnalysisStatus(Enum):
    """Estados posibles de un análisis"""
    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class AnalysisManager:
    """
    Gestor de análisis de contenido y rendimiento.
    
    Esta clase se encarga de realizar análisis sobre el rendimiento del contenido,
    identificar patrones, tendencias y oportunidades de optimización.
    """
    
    def __init__(self, config_manager=None, content_manager=None, persistence=None):
        """
        Inicializa el gestor de análisis.
        
        Args:
            config_manager: Gestor de configuración
            content_manager: Gestor de contenido
            persistence: Sistema de persistencia
        """
        self.config_manager = config_manager
        self.content_manager = content_manager
        self.persistence = persistence
        
        # Diccionario para almacenar análisis
        self.analyses = {}
        
        # Configuración
        self.config = {}
        if self.config_manager:
            self.config = self.config_manager.get("analysis", {})
        
        # Programador de análisis
        self.scheduler_thread = None
        self.scheduler_stop_event = threading.Event()
        
        logger.info("AnalysisManager inicializado")
    
    def create_analysis(self, analysis_type: AnalysisType, params: Dict[str, Any], 
                       schedule_time: Optional[datetime.datetime] = None,
                       callback: Optional[Callable] = None) -> str:
        """
        Crea un nuevo análisis.
        
        Args:
            analysis_type: Tipo de análisis a realizar
            params: Parámetros específicos para el análisis
            schedule_time: Tiempo programado para ejecutar el análisis (opcional)
            callback: Función de callback a llamar cuando el análisis se complete (opcional)
            
        Returns:
            str: ID del análisis creado
        """
        analysis_id = str(uuid.uuid4())
        
        now = datetime.datetime.now()
        
        analysis = {
            "id": analysis_id,
            "type": analysis_type,
            "params": params,
            "status": AnalysisStatus.SCHEDULED if schedule_time else AnalysisStatus.PENDING,
            "schedule_time": schedule_time,
            "created_at": now,
            "updated_at": now,
            "callback": callback,
            "result": None,
            "error": None
        }
        
        self.analyses[analysis_id] = analysis
        
        # Guardar en persistencia
        if self.persistence:
            serializable_analysis = self._prepare_analysis_for_serialization(analysis)
            self.persistence.save_analysis(analysis_id, serializable_analysis)
        
        # Si está programado, asegurar que el programador esté activo
        if schedule_time:
            self._start_scheduler()
        # Si es inmediato, ejecutar en un hilo separado
        else:
            threading.Thread(
                target=self._execute_analysis_task,
                args=(analysis_id,)
            ).start()
        
        logger.info(f"Análisis creado: {analysis_id}, tipo: {analysis_type.name}")
        return analysis_id
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información sobre un análisis específico.
        
        Args:
            analysis_id: ID del análisis
            
        Returns:
            Optional[Dict[str, Any]]: Información del análisis o None si no existe
        """
        if analysis_id in self.analyses:
            return self.analyses[analysis_id]
        
        # Intentar cargar desde persistencia
        if self.persistence:
            analysis_data = self.persistence.load_analysis(analysis_id)
            if analysis_data:
                # Convertir strings a enumeraciones
                if "type" in analysis_data and isinstance(analysis_data["type"], str):
                    try:
                        analysis_data["type"] = AnalysisType[analysis_data["type"]]
                    except KeyError:
                        pass
                
                if "status" in analysis_data and isinstance(analysis_data["status"], str):
                    try:
                        analysis_data["status"] = AnalysisStatus[analysis_data["status"]]
                    except KeyError:
                        pass
                
                # Convertir strings ISO a datetime
                for field in ["created_at", "updated_at", "schedule_time"]:
                    if field in analysis_data and isinstance(analysis_data[field], str):
                        try:
                            analysis_data[field] = datetime.datetime.fromisoformat(analysis_data[field])
                        except ValueError:
                            pass
                
                self.analyses[analysis_id] = analysis_data
                return analysis_data
        
        return None
    
    def cancel_analysis(self, analysis_id: str) -> bool:
        """
        Cancela un análisis en curso o programado.
        
        Args:
            analysis_id: ID del análisis a cancelar
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            logger.warning(f"Intento de cancelar análisis inexistente: {analysis_id}")
            return False
        
        if analysis["status"] in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED]:
            logger.warning(f"Intento de cancelar análisis ya finalizado: {analysis_id}")
            return False
        
        analysis["status"] = AnalysisStatus.CANCELLED
        analysis["updated_at"] = datetime.datetime.now()
        
        # Guardar en persistencia
        if self.persistence:
            serializable_analysis = self._prepare_analysis_for_serialization(analysis)
            self.persistence.save_analysis(analysis_id, serializable_analysis)
        
        logger.info(f"Análisis cancelado: {analysis_id}")
        return True
    
    def get_analyses_by_status(self, status: AnalysisStatus) -> List[Dict[str, Any]]:
        """
        Obtiene todos los análisis con un estado específico.
        
        Args:
            status: Estado de los análisis a buscar
            
        Returns:
            List[Dict[str, Any]]: Lista de análisis con el estado especificado
        """
        return [a for a in self.analyses.values() if a["status"] == status]
    
    def get_analyses_by_type(self, analysis_type: AnalysisType) -> List[Dict[str, Any]]:
        """
        Obtiene todos los análisis de un tipo específico.
        
        Args:
            analysis_type: Tipo de análisis a buscar
            
        Returns:
            List[Dict[str, Any]]: Lista de análisis del tipo especificado
        """
        return [a for a in self.analyses.values() if a["type"] == analysis_type]
    
    def analyze_performance(self, channel_id: str = None, platform: str = None, 
                           time_period: str = "last_30_days", 
                           content_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analiza el rendimiento del contenido.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            content_ids: Lista de IDs de contenido específicos (opcional)
            
        Returns:
            Dict[str, Any]: Resultados del análisis de rendimiento
        """
        logger.info(f"Analizando rendimiento: canal={channel_id}, plataforma={platform}, período={time_period}")
        
        try:
            # Obtener publicaciones
            publications = self._get_publications(channel_id, platform, time_period, content_ids)
            
            if not publications:
                logger.warning("No se encontraron publicaciones para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron publicaciones para analizar"
                }
            
            # Análisis de rendimiento
            performance_metrics = self._calculate_performance_metrics(publications)
            top_performing = self._identify_top_performing(publications)
            underperforming = self._identify_underperforming(publications)
            performance_by_time = self._analyze_performance_by_time(publications)
            
            # Generar insights y recomendaciones
            insights = self._generate_performance_insights({
                "metrics": performance_metrics,
                "top_performing": top_performing,
                "underperforming": underperforming,
                "performance_by_time": performance_by_time
            })
            
            recommendations = self._generate_performance_recommendations(insights)
            
            return {
                "status": "success",
                "metrics": performance_metrics,
                "top_performing": top_performing,
                "underperforming": underperforming,
                "performance_by_time": performance_by_time,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar rendimiento: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar rendimiento: {str(e)}"
            }
    
    def analyze_engagement(self, channel_id: str = None, platform: str = None, 
                          time_period: str = "last_30_days",
                          content_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analiza el engagement del contenido.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            content_ids: Lista de IDs de contenido específicos (opcional)
            
        Returns:
            Dict[str, Any]: Resultados del análisis de engagement
        """
        logger.info(f"Analizando engagement: canal={channel_id}, plataforma={platform}, período={time_period}")
        
        try:
            # Obtener publicaciones
            publications = self._get_publications(channel_id, platform, time_period, content_ids)
            
            if not publications:
                logger.warning("No se encontraron publicaciones para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron publicaciones para analizar"
                }
            
            # Análisis de engagement
            engagement_metrics = self._calculate_engagement_metrics(publications)
            engagement_by_content_type = self._analyze_engagement_by_content_type(publications)
            engagement_by_time = self._analyze_engagement_by_time(publications)
            audience_segments = self._analyze_audience_segments(publications)
            
            # Generar insights y recomendaciones
            insights = self._generate_engagement_insights({
                "metrics": engagement_metrics,
                "by_content_type": engagement_by_content_type,
                "by_time": engagement_by_time,
                "audience_segments": audience_segments
            })
            
            recommendations = self._generate_engagement_recommendations(insights)
            
            return {
                "status": "success",
                "metrics": engagement_metrics,
                "by_content_type": engagement_by_content_type,
                "by_time": engagement_by_time,
                "audience_segments": audience_segments,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar engagement: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar engagement: {str(e)}"
            }
    
    def analyze_audience(self, channel_id: str = None, platform: str = None,
                        time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza la audiencia del canal.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Resultados del análisis de audiencia
        """
        logger.info(f"Analizando audiencia: canal={channel_id}, plataforma={platform}, período={time_period}")
        
        try:
            # Obtener datos de audiencia
            audience_data = self._get_audience_data(channel_id, platform, time_period)
            
            if not audience_data:
                logger.warning("No se encontraron datos de audiencia para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron datos de audiencia para analizar"
                }
            
            # Análisis de audiencia
            demographics = self._analyze_demographics(audience_data)
            growth = self._analyze_audience_growth(audience_data)
            retention = self._analyze_audience_retention(audience_data)
            interests = self._analyze_audience_interests(audience_data)
            
            # Generar insights y recomendaciones
            insights = self._generate_audience_insights({
                "demographics": demographics,
                "growth": growth,
                "retention": retention,
                "interests": interests
            })
            
            recommendations = self._generate_audience_recommendations(insights)
            
            return {
                "status": "success",
                "demographics": demographics,
                "growth": growth,
                "retention": retention,
                "interests": interests,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar audiencia: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar audiencia: {str(e)}"
            }
    
    def analyze_content(self, channel_id: str = None, platform: str = None,
                       time_period: str = "last_30_days",
                       content_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analiza el contenido publicado.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            content_ids: Lista de IDs de contenido específicos (opcional)
            
        Returns:
            Dict[str, Any]: Resultados del análisis de contenido
        """
        logger.info(f"Analizando contenido: canal={channel_id}, plataforma={platform}, período={time_period}")
        
        try:
            # Obtener publicaciones
            publications = self._get_publications(channel_id, platform, time_period, content_ids)
            
            if not publications:
                logger.warning("No se encontraron publicaciones para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron publicaciones para analizar"
                }
            
            # Análisis de contenido
            content_types = self._analyze_content_types(publications)
            successful_elements = self._identify_successful_elements(publications)
            underperforming_elements = self._identify_underperforming_elements(publications)
            content_patterns = self._identify_content_patterns(publications)
            
            # Generar insights y recomendaciones
            insights = self._generate_content_insights({
                "content_types": content_types,
                "successful_elements": successful_elements,
                "underperforming_elements": underperforming_elements,
                "content_patterns": content_patterns
            })
            
            recommendations = self._generate_content_recommendations(insights)
            
            return {
                "status": "success",
                "content_types": content_types,
                "successful_elements": successful_elements,
                "underperforming_elements": underperforming_elements,
                "content_patterns": content_patterns,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar contenido: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar contenido: {str(e)}"
            }
    
    def analyze_monetization(self, channel_id: str = None, platform: str = None,
                            time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza la monetización del canal.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Resultados del análisis de monetización
        """
        logger.info(f"Analizando monetización: canal={channel_id}, plataforma={platform}, período={time_period}")
        
        try:
            # Obtener datos de monetización
            monetization_data = self._get_monetization_data(channel_id, platform, time_period)
            
            if not monetization_data:
                logger.warning("No se encontraron datos de monetización para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron datos de monetización para analizar"
                }
            
            # Análisis de monetización
            revenue = self._analyze_revenue(monetization_data)
            revenue_by_source = self._analyze_revenue_by_source(monetization_data)
            revenue_trends = self._analyze_revenue_trends(monetization_data, time_period)
            
            # Calcular ROI si hay datos de costos
            roi = None
            if "costs" in monetization_data:
                roi = self._calculate_roi(revenue["total"], monetization_data["costs"])
            
            # Generar insights y recomendaciones
            insights = self._generate_monetization_insights({
                "revenue": revenue,
                "revenue_by_source": revenue_by_source,
                "revenue_trends": revenue_trends,
                "roi": roi
            })
            
            recommendations = self._generate_monetization_recommendations(insights)
            
            return {
                "status": "success",
                "revenue": revenue,
                "revenue_by_source": revenue_by_source,
                "revenue_trends": revenue_trends,
                "roi": roi,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar monetización: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar monetización: {str(e)}"
            }
    
    def analyze_trends(self, niche: str = None, keywords: List[str] = None,
                      time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza tendencias relevantes para el canal.
        
        Args:
            niche: Nicho específico (opcional)
            keywords: Lista de palabras clave (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Resultados del análisis de tendencias
        """
        logger.info(f"Analizando tendencias: nicho={niche}, período={time_period}")
        
        try:
            # Obtener datos de tendencias
            trend_data = self._get_trend_data(niche, keywords, time_period)
            
            if not trend_data:
                logger.warning("No se encontraron datos de tendencias para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron datos de tendencias para analizar"
                }
            
            # Análisis de tendencias
            rising_topics = self._identify_rising_topics(trend_data)
            trending_hashtags = self._identify_trending_hashtags(trend_data)
            trending_formats = self._identify_trending_formats(trend_data)
            seasonal_trends = self._identify_seasonal_trends(trend_data)
            
            # Generar insights y recomendaciones
            insights = self._generate_trend_insights({
                "rising_topics": rising_topics,
                "trending_hashtags": trending_hashtags,
                "trending_formats": trending_formats,
                "seasonal_trends": seasonal_trends
            })
            
            recommendations = self._generate_trend_recommendations(insights)
            
            return {
                "status": "success",
                "rising_topics": rising_topics,
                "trending_hashtags": trending_hashtags,
                "trending_formats": trending_formats,
                "seasonal_trends": seasonal_trends,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar tendencias: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar tendencias: {str(e)}"
            }
    
    def analyze_competitors(self, channel_id: str, competitors: List[str] = None,
                           time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza competidores del canal.
        
        Args:
            channel_id: ID del canal
            competitors: Lista de IDs de competidores (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Resultados del análisis de competidores
        """
        logger.info(f"Analizando competidores: canal={channel_id}, período={time_period}")
        
        try:
            # Obtener datos de competidores
            competitor_data = self._get_competitor_data(channel_id, competitors, time_period)
            
            if not competitor_data:
                logger.warning("No se encontraron datos de competidores para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron datos de competidores para analizar"
                }
            
            # Análisis de competidores
            competitor_performance = self._analyze_competitor_performance(competitor_data)
            content_comparison = self._analyze_content_comparison(competitor_data)
            audience_overlap = self._analyze_audience_overlap(competitor_data)
            competitive_advantage = self._identify_competitive_advantage(competitor_data)
            
            # Generar insights y recomendaciones
            insights = self._generate_competitor_insights({
                "competitor_performance": competitor_performance,
                "content_comparison": content_comparison,
                "audience_overlap": audience_overlap,
                "competitive_advantage": competitive_advantage
            })
            
            recommendations = self._generate_competitor_recommendations(insights)
            
            return {
                "status": "success",
                "competitor_performance": competitor_performance,
                "content_comparison": content_comparison,
                "audience_overlap": audience_overlap,
                "competitive_advantage": competitive_advantage,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar competidores: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar competidores: {str(e)}"
            }
    
    def analyze_sentiment(self, channel_id: str = None, content_ids: List[str] = None,
                         time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza el sentimiento de los comentarios y reacciones.
        
        Args:
            channel_id: ID del canal (opcional)
            content_ids: Lista de IDs de contenido específicos (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Resultados del análisis de sentimiento
        """
        logger.info(f"Analizando sentimiento: canal={channel_id}, período={time_period}")
        
        try:
            # Obtener datos de comentarios y reacciones
            sentiment_data = self._get_sentiment_data(channel_id, content_ids, time_period)
            
            if not sentiment_data:
                logger.warning("No se encontraron datos de sentimiento para analizar")
                return {
                    "status": "empty",
                    "message": "No se encontraron datos de sentimiento para analizar"
                }
            
            # Análisis de sentimiento
            overall_sentiment = self._analyze_overall_sentiment(sentiment_data)
            sentiment_by_topic = self._analyze_sentiment_by_topic(sentiment_data)
            sentiment_trends = self._analyze_sentiment_trends(sentiment_data)
            key_feedback = self._extract_key_feedback(sentiment_data)
            
            # Generar insights y recomendaciones
            insights = self._generate_sentiment_insights({
                "overall_sentiment": overall_sentiment,
                "sentiment_by_topic": sentiment_by_topic,
                "sentiment_trends": sentiment_trends,
                "key_feedback": key_feedback
            })
            
            recommendations = self._generate_sentiment_recommendations(insights)
            
            return {
                "status": "success",
                "overall_sentiment": overall_sentiment,
                "sentiment_by_topic": sentiment_by_topic,
                "sentiment_trends": sentiment_trends,
                "key_feedback": key_feedback,
                "insights": insights,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error al analizar sentimiento: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar sentimiento: {str(e)}"
            }
    
    def run_custom_analysis(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un análisis personalizado según la configuración proporcionada.
        
        Args:
            analysis_config: Configuración del análisis personalizado
            
        Returns:
            Dict[str, Any]: Resultados del análisis personalizado
        """
        logger.info(f"Ejecutando análisis personalizado: {analysis_config.get('name', 'sin nombre')}")
        
        try:
            analysis_type = analysis_config.get("type", "custom")
            metrics = analysis_config.get("metrics", [])
            dimensions = analysis_config.get("dimensions", [])
            filters = analysis_config.get("filters", {})
            time_period = analysis_config.get("time_period", "last_30_days")
            
            # Obtener datos según configuración
            data = self._get_custom_analysis_data(analysis_type, metrics, dimensions, filters, time_period)
            
            if not data:
                logger.warning("No se encontraron datos para el análisis personalizado")
                return {
                    "status": "empty",
                    "message": "No se encontraron datos para el análisis personalizado"
                }
            
            # Procesar datos según configuración
            result = self._process_custom_analysis(data, analysis_config)
            
            return {
                "status": "success",
                "name": analysis_config.get("name", "Análisis personalizado"),
                "config": analysis_config,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error al ejecutar análisis personalizado: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al ejecutar análisis personalizado: {str(e)}"
            }
    
    # Métodos auxiliares para obtener datos
    
    def _get_publications(self, channel_id: str = None, platform: str = None,
                         time_period: str = "last_30_days",
                         content_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene publicaciones para análisis.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            content_ids: Lista de IDs de contenido específicos (opcional)
            
        Returns:
            List[Dict[str, Any]]: Lista de publicaciones
        """
        # Implementación simulada para pruebas
        # En una implementación real, se obtendría de la base de datos o APIs
        
        # Si hay content_ids específicos, filtrar por ellos
        if content_ids:
            publications = []
            for content_id in content_ids:
                if self.content_manager:
                    content = self.content_manager.get_content(content_id)
                    if content:
                        # Simular métricas para pruebas
                        publications.append({
                            "content_id": content_id,
                            "platform": content.get("platform", "unknown"),
                            "channel_id": content.get("channel_id", channel_id),
                            "published_at": content.get("published_at", datetime.datetime.now() - datetime.timedelta(days=10)),
                            "metrics": {
                                "views": random.randint(100, 10000),
                                "likes": random.randint(10, 1000),
                                "comments": random.randint(5, 200),
                                "shares": random.randint(1, 100)
                            }
                        })
            return publications
        
        # Simulación de publicaciones para pruebas
        # En una implementación real, se obtendría de la base de datos o APIs
        publications = []
        
        # Determinar rango de fechas según time_period
        end_date = datetime.datetime.now()
        if time_period == "last_7_days":
            start_date = end_date - datetime.timedelta(days=7)
        elif time_period == "last_30_days":
            start_date = end_date - datetime.timedelta(days=30)
        elif time_period == "last_90_days":
            start_date = end_date - datetime.timedelta(days=90)
        else:
            start_date = end_date - datetime.timedelta(days=30)  # Default
        
        # Simular publicaciones en el rango de fechas
        current_date = start_date
        while current_date <= end_date:
            # Simular 1-3 publicaciones por día
            for _ in range(random.randint(1, 3)):
                # Si se especificó plataforma, usar solo esa
                pub_platform = platform if platform else random.choice(["youtube", "tiktok", "instagram", "threads"])
                
                # Crear publicación simulada
                                publication = {
                    "content_id": str(uuid.uuid4()),
                    "platform": pub_platform,
                    "channel_id": channel_id if channel_id else f"channel_{random.randint(1, 5)}",
                    "published_at": current_date + datetime.timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                    "content_type": random.choice(["video", "image", "carousel", "text"]),
                    "title": f"Contenido simulado {current_date.strftime('%Y-%m-%d')}",
                    "description": "Descripción de contenido simulado para pruebas",
                    "tags": [f"tag{i}" for i in range(1, random.randint(3, 8))],
                    "metrics": {
                        "views": random.randint(100, 10000),
                        "likes": random.randint(10, 1000),
                        "comments": random.randint(5, 200),
                        "shares": random.randint(1, 100),
                        "saves": random.randint(0, 50),
                        "clicks": random.randint(0, 300),
                        "watch_time": random.randint(10, 300),  # en segundos
                        "completion_rate": random.uniform(0.1, 0.9),
                        "ctr": random.uniform(0.01, 0.1)
                    }
                }
                
                publications.append(publication)
            
            current_date += datetime.timedelta(days=1)
        
        return publications
    
    def _get_audience_data(self, channel_id: str = None, platform: str = None,
                          time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Obtiene datos de audiencia para análisis.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Datos de audiencia
        """
        # Simulación de datos de audiencia para pruebas
        # En una implementación real, se obtendría de la base de datos o APIs
        
        # Determinar rango de fechas según time_period
        end_date = datetime.datetime.now()
        if time_period == "last_7_days":
            start_date = end_date - datetime.timedelta(days=7)
            data_points = 7
        elif time_period == "last_30_days":
            start_date = end_date - datetime.timedelta(days=30)
            data_points = 30
        elif time_period == "last_90_days":
            start_date = end_date - datetime.timedelta(days=90)
            data_points = 90
        else:
            start_date = end_date - datetime.timedelta(days=30)  # Default
            data_points = 30
        
        # Generar datos demográficos simulados
        demographics = {
            "age_groups": {
                "13-17": random.uniform(0.05, 0.15),
                "18-24": random.uniform(0.2, 0.4),
                "25-34": random.uniform(0.25, 0.4),
                "35-44": random.uniform(0.1, 0.2),
                "45-54": random.uniform(0.05, 0.1),
                "55+": random.uniform(0.01, 0.05)
            },
            "gender": {
                "male": random.uniform(0.4, 0.6),
                "female": random.uniform(0.4, 0.6),
                "other": random.uniform(0.01, 0.05)
            },
            "locations": {
                "United States": random.uniform(0.2, 0.4),
                "Mexico": random.uniform(0.1, 0.2),
                "Spain": random.uniform(0.1, 0.2),
                "Argentina": random.uniform(0.05, 0.15),
                "Colombia": random.uniform(0.05, 0.15),
                "Chile": random.uniform(0.02, 0.1),
                "Peru": random.uniform(0.02, 0.1),
                "Other": random.uniform(0.05, 0.1)
            },
            "languages": {
                "Spanish": random.uniform(0.7, 0.9),
                "English": random.uniform(0.1, 0.3),
                "Other": random.uniform(0.01, 0.05)
            },
            "devices": {
                "mobile": random.uniform(0.6, 0.8),
                "desktop": random.uniform(0.1, 0.3),
                "tablet": random.uniform(0.05, 0.1),
                "tv": random.uniform(0.01, 0.05),
                "other": random.uniform(0.01, 0.03)
            }
        }
        
        # Generar datos de crecimiento simulados
        growth_data = []
        followers_base = random.randint(1000, 10000)
        daily_growth_rate = random.uniform(0.005, 0.02)
        
        current_followers = followers_base
        current_date = start_date
        
        for _ in range(data_points):
            # Simular crecimiento con algo de variabilidad
            growth_factor = random.uniform(0.8, 1.2)
            new_followers = int(current_followers * daily_growth_rate * growth_factor)
            unfollowers = int(new_followers * random.uniform(0.1, 0.3))
            
            growth_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "followers": current_followers,
                "new_followers": new_followers,
                "unfollowers": unfollowers,
                "net_growth": new_followers - unfollowers
            })
            
            current_followers += (new_followers - unfollowers)
            current_date += datetime.timedelta(days=1)
        
        # Generar datos de retención simulados
        retention_data = {
            "cohorts": [
                {
                    "cohort_date": (start_date + datetime.timedelta(days=i*7)).strftime("%Y-%m-%d"),
                    "initial_size": random.randint(100, 500),
                    "retention_rates": {
                        "day_1": random.uniform(0.7, 0.9),
                        "day_3": random.uniform(0.5, 0.7),
                        "day_7": random.uniform(0.3, 0.5),
                        "day_14": random.uniform(0.2, 0.4),
                        "day_30": random.uniform(0.1, 0.3)
                    }
                } for i in range(min(int(data_points/7), 4))  # Crear hasta 4 cohortes
            ],
            "average_retention": {
                "day_1": random.uniform(0.7, 0.9),
                "day_3": random.uniform(0.5, 0.7),
                "day_7": random.uniform(0.3, 0.5),
                "day_14": random.uniform(0.2, 0.4),
                "day_30": random.uniform(0.1, 0.3)
            }
        }
        
        # Generar datos de intereses simulados
        interests = {
            "categories": {
                "Tecnología": random.uniform(0.1, 0.3),
                "Entretenimiento": random.uniform(0.1, 0.3),
                "Educación": random.uniform(0.1, 0.2),
                "Estilo de vida": random.uniform(0.05, 0.2),
                "Deportes": random.uniform(0.05, 0.15),
                "Música": random.uniform(0.05, 0.15),
                "Videojuegos": random.uniform(0.05, 0.2),
                "Moda": random.uniform(0.05, 0.15),
                "Comida": random.uniform(0.05, 0.15),
                "Viajes": random.uniform(0.05, 0.15)
            },
            "topics": [
                {"name": "Inteligencia Artificial", "interest_score": random.uniform(0.5, 1.0)},
                {"name": "Desarrollo Web", "interest_score": random.uniform(0.3, 0.9)},
                {"name": "Videojuegos", "interest_score": random.uniform(0.3, 0.9)},
                {"name": "Películas", "interest_score": random.uniform(0.3, 0.8)},
                {"name": "Música", "interest_score": random.uniform(0.3, 0.8)},
                {"name": "Fitness", "interest_score": random.uniform(0.2, 0.7)},
                {"name": "Cocina", "interest_score": random.uniform(0.2, 0.7)},
                {"name": "Moda", "interest_score": random.uniform(0.2, 0.6)},
                {"name": "Viajes", "interest_score": random.uniform(0.2, 0.6)},
                {"name": "Finanzas", "interest_score": random.uniform(0.2, 0.6)}
            ]
        }
        
        # Combinar todos los datos
        return {
            "channel_id": channel_id if channel_id else f"channel_{random.randint(1, 5)}",
            "platform": platform if platform else random.choice(["youtube", "tiktok", "instagram", "threads"]),
            "time_period": time_period,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "demographics": demographics,
            "growth": growth_data,
            "retention": retention_data,
            "interests": interests
        }
    
    def _get_monetization_data(self, channel_id: str = None, platform: str = None,
                              time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Obtiene datos de monetización para análisis.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Datos de monetización
        """
        # Simulación de datos de monetización para pruebas
        # En una implementación real, se obtendría de la base de datos o APIs
        
        # Determinar rango de fechas según time_period
        end_date = datetime.datetime.now()
        if time_period == "last_7_days":
            start_date = end_date - datetime.timedelta(days=7)
            data_points = 7
        elif time_period == "last_30_days":
            start_date = end_date - datetime.timedelta(days=30)
            data_points = 30
        elif time_period == "last_90_days":
            start_date = end_date - datetime.timedelta(days=90)
            data_points = 90
        else:
            start_date = end_date - datetime.timedelta(days=30)  # Default
            data_points = 30
        
        # Generar datos de ingresos diarios simulados
        daily_revenue = []
        current_date = start_date
        
        for _ in range(data_points):
            # Simular diferentes fuentes de ingresos
            ad_revenue = random.uniform(1.0, 20.0)
            affiliate_revenue = random.uniform(0.5, 15.0)
            subscription_revenue = random.uniform(0.0, 10.0)
            donation_revenue = random.uniform(0.0, 5.0)
            product_revenue = random.uniform(0.0, 25.0)
            
            daily_revenue.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "total": ad_revenue + affiliate_revenue + subscription_revenue + donation_revenue + product_revenue,
                "sources": {
                    "ads": ad_revenue,
                    "affiliate": affiliate_revenue,
                    "subscriptions": subscription_revenue,
                    "donations": donation_revenue,
                    "products": product_revenue
                }
            })
            
            current_date += datetime.timedelta(days=1)
        
        # Generar datos de costos simulados
        costs = {
            "hosting": random.uniform(5.0, 20.0),
            "tools": random.uniform(10.0, 50.0),
            "marketing": random.uniform(0.0, 30.0),
            "content_creation": random.uniform(20.0, 100.0),
            "other": random.uniform(5.0, 15.0)
        }
        
        # Calcular totales por fuente
        revenue_by_source = {
            "ads": sum(day["sources"]["ads"] for day in daily_revenue),
            "affiliate": sum(day["sources"]["affiliate"] for day in daily_revenue),
            "subscriptions": sum(day["sources"]["subscriptions"] for day in daily_revenue),
            "donations": sum(day["sources"]["donations"] for day in daily_revenue),
            "products": sum(day["sources"]["products"] for day in daily_revenue)
        }
        
        # Calcular total general
        total_revenue = sum(day["total"] for day in daily_revenue)
        total_costs = sum(costs.values())
        
        # Combinar todos los datos
        return {
            "channel_id": channel_id if channel_id else f"channel_{random.randint(1, 5)}",
            "platform": platform if platform else random.choice(["youtube", "tiktok", "instagram", "threads"]),
            "time_period": time_period,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily_revenue": daily_revenue,
            "revenue_by_source": revenue_by_source,
            "total_revenue": total_revenue,
            "costs": costs,
            "total_costs": total_costs,
            "profit": total_revenue - total_costs,
            "roi": (total_revenue - total_costs) / total_costs if total_costs > 0 else 0
        }
    
    def _get_trend_data(self, niche: str = None, keywords: List[str] = None,
                       time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Obtiene datos de tendencias para análisis.
        
        Args:
            niche: Nicho específico (opcional)
            keywords: Lista de palabras clave (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Datos de tendencias
        """
        # Simulación de datos de tendencias para pruebas
        # En una implementación real, se obtendría de APIs externas
        
        # Determinar rango de fechas según time_period
        end_date = datetime.datetime.now()
        if time_period == "last_7_days":
            start_date = end_date - datetime.timedelta(days=7)
            data_points = 7
        elif time_period == "last_30_days":
            start_date = end_date - datetime.timedelta(days=30)
            data_points = 30
        elif time_period == "last_90_days":
            start_date = end_date - datetime.timedelta(days=90)
            data_points = 90
        else:
            start_date = end_date - datetime.timedelta(days=30)  # Default
            data_points = 30
        
        # Generar temas en tendencia simulados
        rising_topics = [
            {"name": "Inteligencia Artificial Generativa", "growth_rate": random.uniform(1.2, 3.0)},
            {"name": "Realidad Virtual", "growth_rate": random.uniform(1.1, 2.5)},
            {"name": "Finanzas Personales", "growth_rate": random.uniform(1.1, 2.0)},
            {"name": "Sostenibilidad", "growth_rate": random.uniform(1.1, 1.8)},
            {"name": "Bienestar Mental", "growth_rate": random.uniform(1.1, 1.7)},
            {"name": "Recetas Saludables", "growth_rate": random.uniform(1.0, 1.5)},
            {"name": "Tutoriales de Programación", "growth_rate": random.uniform(1.0, 1.5)},
            {"name": "Reseñas de Tecnología", "growth_rate": random.uniform(1.0, 1.4)},
            {"name": "Rutinas de Ejercicio", "growth_rate": random.uniform(1.0, 1.3)},
            {"name": "Consejos de Productividad", "growth_rate": random.uniform(1.0, 1.3)}
        ]
        
        # Generar hashtags en tendencia simulados
        trending_hashtags = [
            {"name": "#IA", "volume": random.randint(10000, 50000), "growth_rate": random.uniform(1.2, 2.5)},
            {"name": "#Tecnología", "volume": random.randint(8000, 40000), "growth_rate": random.uniform(1.1, 2.0)},
            {"name": "#Finanzas", "volume": random.randint(5000, 30000), "growth_rate": random.uniform(1.1, 1.8)},
            {"name": "#Bienestar", "volume": random.randint(5000, 25000), "growth_rate": random.uniform(1.1, 1.7)},
            {"name": "#Programación", "volume": random.randint(3000, 20000), "growth_rate": random.uniform(1.0, 1.5)},
            {"name": "#Fitness", "volume": random.randint(3000, 18000), "growth_rate": random.uniform(1.0, 1.4)},
            {"name": "#Recetas", "volume": random.randint(2000, 15000), "growth_rate": random.uniform(1.0, 1.3)},
            {"name": "#Productividad", "volume": random.randint(2000, 12000), "growth_rate": random.uniform(1.0, 1.3)},
            {"name": "#Viajes", "volume": random.randint(1000, 10000), "growth_rate": random.uniform(0.9, 1.2)},
            {"name": "#Moda", "volume": random.randint(1000, 8000), "growth_rate": random.uniform(0.9, 1.1)}
        ]
        
        # Generar formatos en tendencia simulados
        trending_formats = [
            {"name": "Tutoriales cortos", "popularity": random.uniform(0.7, 1.0)},
            {"name": "Reacciones", "popularity": random.uniform(0.6, 0.9)},
            {"name": "Día en la vida", "popularity": random.uniform(0.6, 0.9)},
            {"name": "Storytelling", "popularity": random.uniform(0.5, 0.8)},
            {"name": "Explicaciones rápidas", "popularity": random.uniform(0.5, 0.8)},
            {"name": "Desafíos", "popularity": random.uniform(0.4, 0.7)},
            {"name": "Entrevistas", "popularity": random.uniform(0.4, 0.7)},
            {"name": "Reseñas", "popularity": random.uniform(0.3, 0.6)},
            {"name": "Detrás de escenas", "popularity": random.uniform(0.3, 0.6)},
            {"name": "Preguntas y respuestas", "popularity": random.uniform(0.2, 0.5)}
        ]
        
        # Generar tendencias estacionales simuladas
        current_month = datetime.datetime.now().month
        seasonal_trends = []
        
        # Tendencias según la época del año
        if 1 <= current_month <= 2:  # Inicio de año
            seasonal_trends.extend([
                {"name": "Propósitos de año nuevo", "seasonality": "Enero-Febrero", "relevance": random.uniform(0.7, 1.0)},
                {"name": "Planificación anual", "seasonality": "Enero-Febrero", "relevance": random.uniform(0.6, 0.9)},
                {"name": "Dietas y fitness", "seasonality": "Enero-Marzo", "relevance": random.uniform(0.6, 0.9)}
            ])
        elif 3 <= current_month <= 5:  # Primavera
            seasonal_trends.extend([
                {"name": "Jardinería", "seasonality": "Marzo-Mayo", "relevance": random.uniform(0.7, 1.0)},
                {"name": "Limpieza de primavera", "seasonality": "Marzo-Abril", "relevance": random.uniform(0.6, 0.9)},
                {"name": "Actividades al aire libre", "seasonality": "Abril-Junio", "relevance": random.uniform(0.6, 0.9)}
            ])
        elif 6 <= current_month <= 8:  # Verano
            seasonal_trends.extend([
                {"name": "Vacaciones", "seasonality": "Junio-Agosto", "relevance": random.uniform(0.7, 1.0)},
                {"name": "Recetas de verano", "seasonality": "Junio-Agosto", "relevance": random.uniform(0.6, 0.9)},
                {"name": "Actividades acuáticas", "seasonality": "Junio-Septiembre", "relevance": random.uniform(0.6, 0.9)}
            ])
        elif 9 <= current_month <= 10:  # Otoño
            seasonal_trends.extend([
                {"name": "Regreso a clases", "seasonality": "Agosto-Septiembre", "relevance": random.uniform(0.7, 1.0)},
                {"name": "Halloween", "seasonality": "Septiembre-Octubre", "relevance": random.uniform(0.6, 0.9)},
                {"name": "Decoración de otoño", "seasonality": "Septiembre-Noviembre", "relevance": random.uniform(0.6, 0.9)}
            ])
        else:  # Invierno/Navidad
            seasonal_trends.extend([
                {"name": "Navidad", "seasonality": "Noviembre-Diciembre", "relevance": random.uniform(0.7, 1.0)},
                {"name": "Regalos", "seasonality": "Noviembre-Diciembre", "relevance": random.uniform(0.6, 0.9)},
                {"name": "Recetas festivas", "seasonality": "Noviembre-Diciembre", "relevance": random.uniform(0.6, 0.9)}
            ])
        
        # Añadir algunas tendencias generales todo el año
        seasonal_trends.extend([
            {"name": "Cumpleaños", "seasonality": "Todo el año", "relevance": random.uniform(0.4, 0.7)},
            {"name": "Eventos deportivos", "seasonality": "Variable", "relevance": random.uniform(0.4, 0.7)},
            {"name": "Lanzamientos tecnológicos", "seasonality": "Variable", "relevance": random.uniform(0.3, 0.6)}
        ])
        
        # Combinar todos los datos
        return {
            "niche": niche,
            "keywords": keywords,
            "time_period": time_period,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "rising_topics": rising_topics,
            "trending_hashtags": trending_hashtags,
            "trending_formats": trending_formats,
            "seasonal_trends": seasonal_trends
        }
    
    def _get_competitor_data(self, channel_id: str, competitors: List[str] = None,
                            time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Obtiene datos de competidores para análisis.
        
        Args:
            channel_id: ID del canal
            competitors: Lista de IDs de competidores (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Datos de competidores
        """
        # Simulación de datos de competidores para pruebas
        # En una implementación real, se obtendría de APIs externas
        
        # Si no se proporcionan competidores, generar algunos simulados
        if not competitors:
            competitors = [f"competitor_{i}" for i in range(1, 6)]
        
        # Generar datos de rendimiento para cada competidor
        competitor_performance = []
        
        for competitor_id in competitors:
            # Simular métricas de rendimiento
            followers = random.randint(1000, 100000)
            engagement_rate = random.uniform(0.01, 0.1)
            avg_views = random.randint(500, 50000)
            avg_likes = int(avg_views * random.uniform(0.05, 0.2))
            avg_comments = int(avg_views * random.uniform(0.01, 0.05))
            
            # Simular crecimiento
            growth_rate = random.uniform(-0.05, 0.2)  # Puede ser negativo
            
            # Simular frecuencia de publicación
            posting_frequency = random.uniform(0.2, 3.0)  # Publicaciones por día
            
            competitor_performance.append({
                "competitor_id": competitor_id,
                "name": f"Competidor {competitor_id.split('_')[1]}",
                "platform": random.choice(["youtube", "tiktok", "instagram", "threads"]),
                "followers": followers,
                "engagement_rate": engagement_rate,
                "avg_views": avg_views,
                "avg_likes": avg_likes,
                "avg_comments": avg_comments,
                "growth_rate": growth_rate,
                "posting_frequency": posting_frequency
            })
        
        # Generar comparación de contenido
        content_comparison = {
            "topics": [
                {"name": "Tecnología", "our_volume": random.randint(5, 30), "competitor_avg_volume": random.randint(5, 30)},
                {"name": "Tutoriales", "our_volume": random.randint(5, 30), "competitor_avg_volume": random.randint(5, 30)},
                {"name": "Lifestyle", "our_volume": random.randint(5, 30), "competitor_avg_volume": random.randint(5, 30)},
                {"name": "Entretenimiento", "our_volume": random.randint(5, 30), "competitor_avg_volume": random.randint(5, 30)},
                {"name": "Noticias", "our_volume": random.randint(5, 30), "competitor_avg_volume": random.randint(5, 30)}
            ],
            "formats": [
                {"name": "Videos cortos", "our_usage": random.uniform(0, 1), "competitor_avg_usage": random.uniform(0, 1)},
                {"name": "Tutoriales", "our_usage": random.uniform(0, 1), "competitor_avg_usage": random.uniform(0, 1)},
                {"name": "Vlogs", "our_usage": random.uniform(0, 1), "competitor_avg_usage": random.uniform(0, 1)},
                {"name": "Entrevistas", "our_usage": random.uniform(0, 1), "competitor_avg_usage": random.uniform(0, 1)},
                {"name": "Reseñas", "our_usage": random.uniform(0, 1), "competitor_avg_usage": random.uniform(0, 1)}
            ],
            "posting_times": {
                "our_peak_times": [
                    {"day": "Lunes", "hour": random.randint(8, 22)},
                    {"day": "Miércoles", "hour": random.randint(8, 22)},
                    {"day": "Viernes", "hour": random.randint(8, 22)},
                    {"day": "Domingo", "hour": random.randint(8, 22)}
                ],
                "competitor_peak_times": [
                    {"day": "Lunes", "hour": random.randint(8, 22)},
                    {"day": "Martes", "hour": random.randint(8, 22)},
                    {"day": "Jueves", "hour": random.randint(8, 22)},
                    {"day": "Sábado", "hour": random.randint(8, 22)}
                ]
            }
        }
        
        # Generar solapamiento de audiencia
        audience_overlap = {
            "overall_overlap": random.uniform(0.1, 0.5),
            "by_competitor": [
                {"competitor_id": comp["competitor_id"], "overlap_percentage": random.uniform(0.05, 0.6)}
                for comp in competitor_performance
            ],
            "demographics_similarity": random.uniform(0.3, 0.8),
            "interests_similarity": random.uniform(0.3, 0.8)
        }
        
        # Generar ventajas competitivas
        competitive_advantage = {
            "our_strengths": [
                {"factor": "Calidad de producción", "score": random.uniform(0.5, 1.0), "avg_competitor_score": random.uniform(0.3, 0.9)},
                {"factor": "Engagement", "score": random.uniform(0.5, 1.0), "avg_competitor_score": random.uniform(0.3, 0.9)},
                {"factor": "Frecuencia", "score": random.uniform(0.5, 1.0), "avg_competitor_score": random.uniform(0.3, 0.9)},
                {"factor": "Originalidad", "score": random.uniform(0.5, 1.0), "avg_competitor_score": random.uniform(0.3, 0.9)},
                {"factor": "Monetización", "score": random.uniform(0.5, 1.0), "avg_competitor_score": random.uniform(0.3, 0.9)}
            ],
            "competitor_strengths": [
                                {"competitor_id": comp["competitor_id"], 
                 "name": comp["name"],
                 "strengths": [
                    {"factor": "Calidad de producción", "score": random.uniform(0.3, 0.9)},
                    {"factor": "Engagement", "score": random.uniform(0.3, 0.9)},
                    {"factor": "Frecuencia", "score": random.uniform(0.3, 0.9)},
                    {"factor": "Originalidad", "score": random.uniform(0.3, 0.9)},
                    {"factor": "Monetización", "score": random.uniform(0.3, 0.9)}
                 ]
                } for comp in competitor_performance
            ]
        }
        
        # Generar oportunidades y amenazas
        opportunities_threats = {
            "opportunities": [
                {"description": "Nicho emergente no cubierto por competidores", "impact_score": random.uniform(0.5, 1.0)},
                {"description": "Formato de contenido poco explorado", "impact_score": random.uniform(0.5, 0.9)},
                {"description": "Colaboraciones potenciales", "impact_score": random.uniform(0.4, 0.8)},
                {"description": "Tendencias recientes no aprovechadas", "impact_score": random.uniform(0.4, 0.8)},
                {"description": "Monetización alternativa", "impact_score": random.uniform(0.3, 0.7)}
            ],
            "threats": [
                {"description": "Saturación del nicho", "impact_score": random.uniform(0.5, 1.0)},
                {"description": "Competidor dominante en crecimiento", "impact_score": random.uniform(0.5, 0.9)},
                {"description": "Cambios en algoritmos de plataformas", "impact_score": random.uniform(0.4, 0.8)},
                {"description": "Nuevos competidores emergentes", "impact_score": random.uniform(0.4, 0.8)},
                {"description": "Cambios en preferencias de audiencia", "impact_score": random.uniform(0.3, 0.7)}
            ]
        }
        
        # Combinar todos los datos
        return {
            "channel_id": channel_id,
            "time_period": time_period,
            "competitors": competitors,
            "competitor_performance": competitor_performance,
            "content_comparison": content_comparison,
            "audience_overlap": audience_overlap,
            "competitive_advantage": competitive_advantage,
            "opportunities_threats": opportunities_threats
        }
    
    def _get_compliance_data(self, channel_id: str = None, platform: str = None) -> Dict[str, Any]:
        """
        Obtiene datos de cumplimiento normativo para análisis.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            
        Returns:
            Dict[str, Any]: Datos de cumplimiento normativo
        """
        # Simulación de datos de cumplimiento para pruebas
        # En una implementación real, se obtendría de APIs externas y análisis
        
        # Generar datos de políticas de plataforma
        platform_policies = {
            "youtube": {
                "content_restrictions": [
                    {"policy": "Contenido para adultos", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Discurso de odio", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Violencia gráfica", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Desinformación", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Copyright", "risk_level": random.uniform(0.1, 0.9)}
                ],
                "monetization_requirements": [
                    {"requirement": "1000 suscriptores", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "4000 horas de visualización", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "Cuenta AdSense vinculada", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "Cumplimiento de políticas", "status": random.choice(["cumplido", "pendiente"])}
                ]
            },
            "tiktok": {
                "content_restrictions": [
                    {"policy": "Contenido para adultos", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Discurso de odio", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Violencia gráfica", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Desinformación", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Copyright", "risk_level": random.uniform(0.1, 0.9)}
                ],
                "monetization_requirements": [
                    {"requirement": "10000 seguidores", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "100000 visualizaciones en 30 días", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "18+ años", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "Cumplimiento de políticas", "status": random.choice(["cumplido", "pendiente"])}
                ]
            },
            "instagram": {
                "content_restrictions": [
                    {"policy": "Contenido para adultos", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Discurso de odio", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Violencia gráfica", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Desinformación", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Copyright", "risk_level": random.uniform(0.1, 0.9)}
                ],
                "monetization_requirements": [
                    {"requirement": "10000 seguidores", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "Cuenta profesional", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "Cumplimiento de políticas", "status": random.choice(["cumplido", "pendiente"])}
                ]
            },
            "threads": {
                "content_restrictions": [
                    {"policy": "Contenido para adultos", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Discurso de odio", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Violencia gráfica", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Desinformación", "risk_level": random.uniform(0.1, 0.9)},
                    {"policy": "Copyright", "risk_level": random.uniform(0.1, 0.9)}
                ],
                "monetization_requirements": [
                    {"requirement": "Vinculado a Instagram", "status": random.choice(["cumplido", "pendiente"])},
                    {"requirement": "Cumplimiento de políticas", "status": random.choice(["cumplido", "pendiente"])}
                ]
            }
        }
        
        # Seleccionar plataforma específica o generar para todas
        if platform:
            selected_platforms = {platform: platform_policies.get(platform, {})}
        else:
            selected_platforms = platform_policies
        
        # Generar historial de infracciones simulado
        violation_history = []
        for _ in range(random.randint(0, 5)):
            violation_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 180))
            violation_history.append({
                "date": violation_date.strftime("%Y-%m-%d"),
                "platform": random.choice(list(platform_policies.keys())),
                "policy_violated": random.choice(["Contenido para adultos", "Discurso de odio", "Copyright", "Spam", "Desinformación"]),
                "severity": random.choice(["baja", "media", "alta"]),
                "status": random.choice(["resuelta", "en apelación", "activa"]),
                "impact": random.choice(["ninguno", "demonetización temporal", "restricción de alcance", "advertencia"])
            })
        
        # Generar análisis de riesgo de contenido
        content_risk_analysis = {
            "overall_risk_score": random.uniform(0.1, 0.9),
            "risk_factors": [
                {"factor": "Lenguaje utilizado", "risk_level": random.uniform(0.1, 0.9)},
                {"factor": "Temas tratados", "risk_level": random.uniform(0.1, 0.9)},
                {"factor": "Imágenes y visuales", "risk_level": random.uniform(0.1, 0.9)},
                {"factor": "Música y audio", "risk_level": random.uniform(0.1, 0.9)},
                {"factor": "Referencias a marcas", "risk_level": random.uniform(0.1, 0.9)},
                {"factor": "Contenido de terceros", "risk_level": random.uniform(0.1, 0.9)}
            ],
            "recommendations": [
                "Revisar uso de lenguaje en descripciones",
                "Verificar derechos de música utilizada",
                "Moderar comentarios regularmente",
                "Incluir disclaimers en contenido sensible",
                "Diversificar temas para reducir riesgo de shadowban"
            ]
        }
        
        # Generar datos de shadowban
        shadowban_analysis = {
            "shadowban_probability": random.uniform(0, 1),
            "affected_platforms": [
                {"platform": "youtube", "probability": random.uniform(0, 0.5)},
                {"platform": "tiktok", "probability": random.uniform(0, 0.5)},
                {"platform": "instagram", "probability": random.uniform(0, 0.5)},
                {"platform": "threads", "probability": random.uniform(0, 0.5)}
            ],
            "indicators": [
                {"indicator": "Reducción repentina de alcance", "severity": random.uniform(0, 1)},
                {"indicator": "Ausencia en búsquedas", "severity": random.uniform(0, 1)},
                {"indicator": "Reducción de engagement", "severity": random.uniform(0, 1)},
                {"indicator": "Contenido no aparece en feeds", "severity": random.uniform(0, 1)},
                {"indicator": "Hashtags no funcionan", "severity": random.uniform(0, 1)}
            ],
            "recovery_plan": [
                "Pausar publicaciones por 48 horas",
                "Revisar y eliminar contenido potencialmente problemático",
                "Publicar contenido seguro y no controversial",
                "Evitar uso excesivo de hashtags",
                "Interactuar genuinamente con otros creadores",
                "Contactar soporte de la plataforma"
            ]
        }
        
        # Combinar todos los datos
        return {
            "channel_id": channel_id if channel_id else f"channel_{random.randint(1, 5)}",
            "platform_policies": selected_platforms,
            "violation_history": violation_history,
            "content_risk_analysis": content_risk_analysis,
            "shadowban_analysis": shadowban_analysis
        }
    
    def analyze_content_performance(self, channel_id: str = None, platform: str = None,
                                   time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza el rendimiento del contenido publicado.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Análisis de rendimiento del contenido
        """
        # Obtener datos de publicaciones
        publications = self._get_publication_data(channel_id, platform, time_period)
        
        # Calcular métricas generales
        total_publications = len(publications)
        if total_publications == 0:
            return {"error": "No hay publicaciones en el período seleccionado"}
        
        # Calcular promedios de métricas
        avg_views = sum(pub["metrics"]["views"] for pub in publications) / total_publications
        avg_likes = sum(pub["metrics"]["likes"] for pub in publications) / total_publications
        avg_comments = sum(pub["metrics"]["comments"] for pub in publications) / total_publications
        avg_shares = sum(pub["metrics"]["shares"] for pub in publications) / total_publications
        
        # Calcular engagement rate promedio
        avg_engagement_rate = sum((pub["metrics"]["likes"] + pub["metrics"]["comments"] + 
                                  pub["metrics"]["shares"]) / pub["metrics"]["views"] 
                                 for pub in publications if pub["metrics"]["views"] > 0) / total_publications
        
        # Identificar mejores y peores publicaciones
        publications_sorted_by_views = sorted(publications, key=lambda x: x["metrics"]["views"], reverse=True)
        publications_sorted_by_engagement = sorted(
            [p for p in publications if p["metrics"]["views"] > 0],
            key=lambda x: (x["metrics"]["likes"] + x["metrics"]["comments"] + x["metrics"]["shares"]) / x["metrics"]["views"],
            reverse=True
        )
        
        best_publications = {
            "by_views": publications_sorted_by_views[:3] if len(publications_sorted_by_views) >= 3 else publications_sorted_by_views,
            "by_engagement": publications_sorted_by_engagement[:3] if len(publications_sorted_by_engagement) >= 3 else publications_sorted_by_engagement
        }
        
        worst_publications = {
            "by_views": publications_sorted_by_views[-3:] if len(publications_sorted_by_views) >= 3 else publications_sorted_by_views[-1:],
            "by_engagement": publications_sorted_by_engagement[-3:] if len(publications_sorted_by_engagement) >= 3 else publications_sorted_by_engagement[-1:]
        }
        
        # Analizar rendimiento por tipo de contenido
        content_types = {}
        for pub in publications:
            content_type = pub["content_type"]
            if content_type not in content_types:
                content_types[content_type] = {
                    "count": 0,
                    "total_views": 0,
                    "total_likes": 0,
                    "total_comments": 0,
                    "total_shares": 0
                }
            
            content_types[content_type]["count"] += 1
            content_types[content_type]["total_views"] += pub["metrics"]["views"]
            content_types[content_type]["total_likes"] += pub["metrics"]["likes"]
            content_types[content_type]["total_comments"] += pub["metrics"]["comments"]
            content_types[content_type]["total_shares"] += pub["metrics"]["shares"]
        
        # Calcular promedios por tipo de contenido
        for content_type, data in content_types.items():
            count = data["count"]
            data["avg_views"] = data["total_views"] / count
            data["avg_likes"] = data["total_likes"] / count
            data["avg_comments"] = data["total_comments"] / count
            data["avg_shares"] = data["total_shares"] / count
            data["avg_engagement"] = (data["total_likes"] + data["total_comments"] + data["total_shares"]) / data["total_views"] if data["total_views"] > 0 else 0
        
        # Analizar rendimiento por día de la semana y hora
        day_performance = {day: {"count": 0, "total_views": 0, "total_engagement": 0} 
                          for day in ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]}
        
        hour_performance = {hour: {"count": 0, "total_views": 0, "total_engagement": 0} 
                           for hour in range(24)}
        
        days_map = {
            0: "Lunes",
            1: "Martes",
            2: "Miércoles",
            3: "Jueves",
            4: "Viernes",
            5: "Sábado",
            6: "Domingo"
        }
        
        for pub in publications:
            pub_date = datetime.datetime.fromisoformat(pub["published_at"].replace("Z", "+00:00"))
            day = days_map[pub_date.weekday()]
            hour = pub_date.hour
            
            engagement = pub["metrics"]["likes"] + pub["metrics"]["comments"] + pub["metrics"]["shares"]
            
            day_performance[day]["count"] += 1
            day_performance[day]["total_views"] += pub["metrics"]["views"]
            day_performance[day]["total_engagement"] += engagement
            
            hour_performance[hour]["count"] += 1
            hour_performance[hour]["total_views"] += pub["metrics"]["views"]
            hour_performance[hour]["total_engagement"] += engagement
        
        # Calcular promedios por día y hora
        for day, data in day_performance.items():
            if data["count"] > 0:
                data["avg_views"] = data["total_views"] / data["count"]
                data["avg_engagement"] = data["total_engagement"] / data["count"]
            else:
                data["avg_views"] = 0
                data["avg_engagement"] = 0
        
        for hour, data in hour_performance.items():
            if data["count"] > 0:
                data["avg_views"] = data["total_views"] / data["count"]
                data["avg_engagement"] = data["total_engagement"] / data["count"]
            else:
                data["avg_views"] = 0
                data["avg_engagement"] = 0
        
        # Identificar mejores días y horas
        best_days = sorted(day_performance.items(), key=lambda x: x[1]["avg_views"], reverse=True)
        best_hours = sorted(hour_performance.items(), key=lambda x: x[1]["avg_views"], reverse=True)
        
        # Generar recomendaciones basadas en el análisis
        recommendations = []
        
        # Recomendación de tipo de contenido
        best_content_type = max(content_types.items(), key=lambda x: x[1]["avg_engagement"])
        recommendations.append(f"Crear más contenido de tipo '{best_content_type[0]}' que tiene el mejor engagement promedio")
        
        # Recomendación de día y hora
        if best_days and best_hours:
            recommendations.append(f"Publicar preferentemente los {best_days[0][0]} a las {best_hours[0][0]}:00 para maximizar alcance")
        
        # Recomendaciones basadas en análisis de contenido
        if publications_sorted_by_engagement:
            top_pub = publications_sorted_by_engagement[0]
            recommendations.append(f"Analizar elementos exitosos de la publicación con ID {top_pub['content_id']} para replicar su éxito")
        
        if worst_publications["by_engagement"]:
            bottom_pub = worst_publications["by_engagement"][-1]
            recommendations.append(f"Revisar y mejorar elementos de la publicación con ID {bottom_pub['content_id']} que tuvo bajo engagement")
        
        # Combinar todos los datos de análisis
        return {
            "channel_id": channel_id if channel_id else publications[0]["channel_id"] if publications else None,
            "platform": platform if platform else "multiple",
            "time_period": time_period,
            "total_publications": total_publications,
            "average_metrics": {
                "views": avg_views,
                "likes": avg_likes,
                "comments": avg_comments,
                "shares": avg_shares,
                "engagement_rate": avg_engagement_rate
            },
            "best_publications": best_publications,
            "worst_publications": worst_publications,
            "content_type_performance": content_types,
            "day_performance": day_performance,
            "hour_performance": hour_performance,
            "best_posting_times": {
                "days": [{"day": day, "avg_views": data["avg_views"]} for day, data in best_days[:3]],
                "hours": [{"hour": hour, "avg_views": data["avg_views"]} for hour, data in best_hours[:3]]
            },
            "recommendations": recommendations
        }
    
    def analyze_audience(self, channel_id: str = None, platform: str = None,
                        time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Analiza la audiencia del canal.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Análisis de audiencia
        """
        # Obtener datos de audiencia
        audience_data = self._get_audience_data(channel_id, platform, time_period)
        
        # Analizar datos demográficos
        demographics = audience_data["demographics"]
        
        # Identificar segmentos principales
        primary_age_group = max(demographics["age_groups"].items(), key=lambda x: x[1])
        primary_gender = max(demographics["gender"].items(), key=lambda x: x[1])
        primary_location = max(demographics["locations"].items(), key=lambda x: x[1])
        primary_language = max(demographics["languages"].items(), key=lambda x: x[1])
        primary_device = max(demographics["devices"].items(), key=lambda x: x[1])
        
        # Analizar crecimiento
        growth_data = audience_data["growth"]
        
        # Calcular métricas de crecimiento
        initial_followers = growth_data[0]["followers"] if growth_data else 0
        final_followers = growth_data[-1]["followers"] if growth_data else 0
        total_growth = final_followers - initial_followers
        growth_percentage = (total_growth / initial_followers * 100) if initial_followers > 0 else 0
        
        # Calcular promedio de nuevos seguidores diarios
        avg_new_followers = sum(day["new_followers"] for day in growth_data) / len(growth_data) if growth_data else 0
        
        # Calcular tasa de deserción
        churn_rate = sum(day["unfollowers"] for day in growth_data) / sum(day["new_followers"] for day in growth_data) if sum(day["new_followers"] for day in growth_data) > 0 else 0
        
        # Analizar retención
        retention_data = audience_data["retention"]
        avg_retention = retention_data["average_retention"]
        
        # Analizar intereses
        interests = audience_data["interests"]
        top_categories = sorted(interests["categories"].items(), key=lambda x: x[1], reverse=True)[:5]
        top_topics = sorted(interests["topics"], key=lambda x: x["interest_score"], reverse=True)[:5]
        
        # Generar recomendaciones basadas en el análisis
        recommendations = []
        
        # Recomendaciones demográficas
        recommendations.append(f"Optimizar contenido para el grupo de edad {primary_age_group[0]} que representa {primary_age_group[1]*100:.1f}% de la audiencia")
        recommendations.append(f"Considerar crear contenido en {primary_language[0]} para el {primary_language[1]*100:.1f}% de la audiencia")
        recommendations.append(f"Optimizar experiencia para dispositivos móviles usados por el {primary_device[1]*100:.1f}% de la audiencia")
        
        # Recomendaciones de crecimiento
        if growth_percentage < 5:
            recommendations.append("Implementar estrategias para acelerar el crecimiento de seguidores")
        
        if churn_rate > 0.3:
            recommendations.append("Mejorar retención de seguidores con contenido más relevante y consistente")
        
        # Recomendaciones de contenido basadas en intereses
        if top_topics:
            recommendations.append(f"Crear más contenido sobre '{top_topics[0]['name']}' que tiene alto interés en la audiencia")
        
        # Recomendaciones de retención
        if avg_retention["day_7"] < 0.3:
            recommendations.append("Mejorar retención a 7 días con series de contenido y llamados a la acción efectivos")
        
        # Combinar todos los datos de análisis
        return {
            "channel_id": audience_data["channel_id"],
            "platform": audience_data["platform"],
            "time_period": time_period,
            "audience_size": final_followers,
            "growth_metrics": {
                "total_growth": total_growth,
                "growth_percentage": growth_percentage,
                "avg_new_followers_daily": avg_new_followers,
                "churn_rate": churn_rate
            },
            "primary_segments": {
                "age_group": {"segment": primary_age_group[0], "percentage": primary_age_group[1]},
                "gender": {"segment": primary_gender[0], "percentage": primary_gender[1]},
                "location": {"segment": primary_location[0], "percentage": primary_location[1]},
                "language": {"segment": primary_language[0], "percentage": primary_language[1]},
                "device": {"segment": primary_device[0], "percentage": primary_device[1]}
            },
            "retention_metrics": avg_retention,
            "top_interests": {
                "categories": [{"category": cat, "score": score} for cat, score in top_categories],
                "topics": top_topics
            },
            "recommendations": recommendations
        }