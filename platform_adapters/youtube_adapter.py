"""
YouTube Adapter - Adaptador para YouTube y YouTube Shorts

Este módulo proporciona una interfaz unificada para interactuar con la API de YouTube,
gestionando la publicación de contenido, análisis de métricas, gestión de comentarios,
y otras funcionalidades específicas de YouTube y YouTube Shorts.
"""

import os
import sys
import json
import logging
import time
import requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Importar el cargador de configuraciones
from utils.config_loader import get_platform_credentials

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/youtube_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YouTubeAdapter")

class YouTubeAdapter:
    """
    Adaptador para interactuar con la API de YouTube
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el adaptador de YouTube
        
        Args:
            config_path: Ruta al archivo de configuración con credenciales
        """
        # Usar get_platform_credentials en lugar de cargar directamente el archivo JSON
        self.config = get_platform_credentials('youtube')
        self.credentials = None
        self.youtube = None
        self.youtube_analytics = None
        self.quota_usage = 0
        self.quota_reset_time = datetime.now() + timedelta(days=1)
        self.initialize_api()
    
    # Eliminar el método _load_config ya que ahora usamos get_platform_credentials
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la API de YouTube
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            # Verificar si hay credenciales
            if not self.config:
                logger.error("No se encontró configuración para YouTube")
                return False
            
            # Crear credenciales OAuth2
            self.credentials = google.oauth2.credentials.Credentials(
                token=None,
                refresh_token=self.config.get('refresh_token'),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.config.get('client_id'),
                client_secret=self.config.get('client_secret')
            )
            
            # Construir servicio de YouTube
            self.youtube = build('youtube', 'v3', credentials=self.credentials)
            
            # Construir servicio de YouTube Analytics
            self.youtube_analytics = build('youtubeAnalytics', 'v2', credentials=self.credentials)
            
            logger.info("API de YouTube inicializada correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar API de YouTube: {str(e)}")
            return False
    
    def _check_quota(self, cost: int = 1) -> bool:
        """
        Verifica si hay cuota disponible para la operación
        
        Args:
            cost: Costo en unidades de cuota de la operación
            
        Returns:
            True si hay cuota disponible, False en caso contrario
        """
        # Verificar si es tiempo de reiniciar la cuota
        if datetime.now() > self.quota_reset_time:
            self.quota_usage = 0
            self.quota_reset_time = datetime.now() + timedelta(days=1)
            logger.info("Cuota de YouTube API reiniciada")
        
        # Verificar si hay cuota disponible
        quota_limit = self.config.get('quota_limit', 10000)
        if self.quota_usage + cost > quota_limit:
            logger.warning(f"Límite de cuota alcanzado: {self.quota_usage}/{quota_limit}")
            return False
        
        # Incrementar uso de cuota
        self.quota_usage += cost
        return True
    
    def upload_video(self, video_path: str, title: str, description: str, 
                    tags: List[str] = None, category_id: str = "22", 
                    privacy_status: str = "public", is_shorts: bool = False,
                    thumbnail_path: str = None, language: str = "es", 
                    notify_subscribers: bool = True) -> Dict[str, Any]:
        """
        Sube un video a YouTube
        
        Args:
            video_path: Ruta al archivo de video
            title: Título del video
            description: Descripción del video
            tags: Lista de etiquetas
            category_id: ID de categoría (22 = People & Blogs)
            privacy_status: Estado de privacidad (public, private, unlisted)
            is_shorts: Si es un video para YouTube Shorts
            thumbnail_path: Ruta a la miniatura personalizada
            language: Código de idioma
            notify_subscribers: Si se notifica a los suscriptores
            
        Returns:
            Información del video subido o error
        """
        if not self._check_quota(cost=1600):  # Subir video cuesta ~1600 unidades
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            if not os.path.exists(video_path):
                return {"status": "error", "message": f"Archivo no encontrado: {video_path}"}
            
            # Preparar metadatos del video
            if tags is None:
                tags = []
            
            # Configuración específica para Shorts
            if is_shorts:
                # Añadir #Shorts en título y descripción para mejor distribución
                if "#Shorts" not in title:
                    title = f"{title} #Shorts"
                if "#Shorts" not in description:
                    description = f"{description}\n\n#Shorts"
                # Añadir etiqueta de Shorts
                if "Shorts" not in tags:
                    tags.append("Shorts")
            
            # Crear cuerpo de la solicitud
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": category_id,
                    "defaultLanguage": language
                },
                "status": {
                    "privacyStatus": privacy_status,
                    "selfDeclaredMadeForKids": False,
                    "notifySubscribers": notify_subscribers
                }
            }
            
            # Crear objeto MediaFileUpload para el video
            media = MediaFileUpload(
                video_path,
                mimetype="video/*",
                resumable=True
            )
            
            # Ejecutar solicitud de inserción
            request = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )
            
            # Subir video con manejo de progreso
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Subiendo video: {int(status.progress() * 100)}%")
            
            video_id = response.get("id")
            logger.info(f"Video subido correctamente: {video_id}")
            
            # Subir miniatura personalizada si se proporciona
            if thumbnail_path and os.path.exists(thumbnail_path):
                self.set_thumbnail(video_id, thumbnail_path)
            
            return {
                "status": "success",
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "shorts_url": f"https://www.youtube.com/shorts/{video_id}" if is_shorts else None
            }
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al subir video: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al subir video: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> Dict[str, Any]:
        """
        Establece una miniatura personalizada para un video
        
        Args:
            video_id: ID del video
            thumbnail_path: Ruta a la imagen de miniatura
            
        Returns:
            Resultado de la operación
        """
        if not self._check_quota(cost=50):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            if not os.path.exists(thumbnail_path):
                return {"status": "error", "message": f"Archivo no encontrado: {thumbnail_path}"}
            
            # Crear objeto MediaFileUpload para la miniatura
            media = MediaFileUpload(
                thumbnail_path,
                mimetype="image/jpeg" if thumbnail_path.endswith(".jpg") or thumbnail_path.endswith(".jpeg") else "image/png"
            )
            
            # Ejecutar solicitud de actualización
            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            ).execute()
            
            logger.info(f"Miniatura establecida para video {video_id}")
            return {"status": "success", "message": "Miniatura establecida correctamente"}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al establecer miniatura: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al establecer miniatura: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_video_metrics(self, video_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas básicas de un video
        
        Args:
            video_id: ID del video
            
        Returns:
            Métricas del video
        """
        if not self._check_quota(cost=5):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Obtener información del video
            response = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            ).execute()
            
            if not response.get("items"):
                return {"status": "error", "message": "Video no encontrado"}
            
            video_info = response["items"][0]
            snippet = video_info.get("snippet", {})
            statistics = video_info.get("statistics", {})
            content_details = video_info.get("contentDetails", {})
            
            # Extraer métricas
            metrics = {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "tags": snippet.get("tags", []),
                "category_id": snippet.get("categoryId", ""),
                "duration": content_details.get("duration", ""),
                "views": int(statistics.get("viewCount", 0)),
                "likes": int(statistics.get("likeCount", 0)),
                "comments": int(statistics.get("commentCount", 0)),
                "favorites": int(statistics.get("favoriteCount", 0)),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "metrics": metrics}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al obtener métricas: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al obtener métricas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_advanced_analytics(self, video_id: str, start_date: str = None, 
                              end_date: str = None) -> Dict[str, Any]:
        """
        Obtiene analíticas avanzadas de un video
        
        Args:
            video_id: ID del video
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            
        Returns:
            Analíticas avanzadas del video
        """
        if not self._check_quota(cost=20):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Configurar fechas
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            # Obtener analíticas
            response = self.youtube_analytics.reports().query(
                ids=f"channel=={self.config.get('channel_id')}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost,likes,dislikes,shares,comments",
                dimensions="day",
                filters=f"video=={video_id}"
            ).execute()
            
            # Procesar resultados
            analytics = {
                "video_id": video_id,
                "start_date": start_date,
                "end_date": end_date,
                "data": response.get("rows", []),
                "column_headers": [h.get("name") for h in response.get("columnHeaders", [])],
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "analytics": analytics}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al obtener analíticas: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al obtener analíticas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_comments(self, video_id: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Obtiene comentarios de un video
        
        Args:
            video_id: ID del video
            max_results: Número máximo de comentarios a obtener
            
        Returns:
            Lista de comentarios
        """
        if not self._check_quota(cost=5):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Obtener comentarios
            response = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_results, 100),
                order="relevance"
            ).execute()
            
            # Procesar comentarios
            comments = []
            for item in response.get("items", []):
                comment = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                comments.append({
                    "comment_id": item.get("id", ""),
                    "author": comment.get("authorDisplayName", ""),
                    "author_channel_id": comment.get("authorChannelId", {}).get("value", ""),
                    "text": comment.get("textDisplay", ""),
                    "like_count": comment.get("likeCount", 0),
                    "published_at": comment.get("publishedAt", ""),
                    "updated_at": comment.get("updatedAt", "")
                })
            
            return {"status": "success", "comments": comments}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al obtener comentarios: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al obtener comentarios: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def reply_to_comment(self, comment_id: str, text: str) -> Dict[str, Any]:
        """
        Responde a un comentario
        
        Args:
            comment_id: ID del comentario
            text: Texto de la respuesta
            
        Returns:
            Resultado de la operación
        """
        if not self._check_quota(cost=50):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Crear cuerpo de la solicitud
            body = {
                "snippet": {
                    "parentId": comment_id,
                    "textOriginal": text
                }
            }
            
            # Ejecutar solicitud
            response = self.youtube.comments().insert(
                part="snippet",
                body=body
            ).execute()
            
            logger.info(f"Respuesta enviada al comentario {comment_id}")
            return {
                "status": "success",
                "comment_id": response.get("id", ""),
                "text": response.get("snippet", {}).get("textDisplay", "")
            }
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al responder comentario: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al responder comentario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_trending_videos(self, region_code: str = "ES", category_id: str = None, 
                           max_results: int = 50) -> Dict[str, Any]:
        """
        Obtiene videos en tendencia
        
        Args:
            region_code: Código de región (ES, US, MX, etc.)
            category_id: ID de categoría (opcional)
            max_results: Número máximo de resultados
            
        Returns:
            Lista de videos en tendencia
        """
        if not self._check_quota(cost=5):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Preparar parámetros
            params = {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": min(max_results, 50)
            }
            
            if category_id:
                params["videoCategoryId"] = category_id
            
            # Ejecutar solicitud
            response = self.youtube.videos().list(**params).execute()
            
            # Procesar resultados
            trending_videos = []
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                
                trending_videos.append({
                    "video_id": item.get("id", ""),
                    "title": snippet.get("title", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "views": int(statistics.get("viewCount", 0)),
                    "likes": int(statistics.get("likeCount", 0)),
                    "comments": int(statistics.get("commentCount", 0))
                })
            
            return {"status": "success", "trending_videos": trending_videos}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al obtener tendencias: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al obtener tendencias: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def search_videos(self, query: str, max_results: int = 50, 
                     order: str = "relevance", published_after: str = None) -> Dict[str, Any]:
        """
        Busca videos en YouTube
        
        Args:
            query: Término de búsqueda
            max_results: Número máximo de resultados
            order: Orden de resultados (relevance, date, viewCount, rating)
            published_after: Fecha mínima de publicación (RFC 3339)
            
        Returns:
            Lista de videos encontrados
        """
        if not self._check_quota(cost=100):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Preparar parámetros
            params = {
                "part": "snippet",
                "q": query,
                "maxResults": min(max_results, 50),
                "order": order,
                "type": "video"
            }
            
            if published_after:
                params["publishedAfter"] = published_after
            
            # Ejecutar solicitud
            response = self.youtube.search().list(**params).execute()
            
            # Procesar resultados
            search_results = []
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                
                search_results.append({
                    "video_id": item.get("id", {}).get("videoId", ""),
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
                })
            
            return {"status": "success", "search_results": search_results}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al buscar videos: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al buscar videos: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def update_video(self, video_id: str, title: str = None, description: str = None,
                    tags: List[str] = None, category_id: str = None,
                    privacy_status: str = None) -> Dict[str, Any]:
        """
        Actualiza metadatos de un video
        
        Args:
            video_id: ID del video
            title: Nuevo título (opcional)
            description: Nueva descripción (opcional)
            tags: Nuevas etiquetas (opcional)
            category_id: Nueva categoría (opcional)
            privacy_status: Nuevo estado de privacidad (opcional)
            
        Returns:
            Resultado de la operación
        """
        if not self._check_quota(cost=50):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Obtener información actual del video
            response = self.youtube.videos().list(
                part="snippet,status",
                id=video_id
            ).execute()
            
            if not response.get("items"):
                return {"status": "error", "message": "Video no encontrado"}
            
            video_info = response["items"][0]
            snippet = video_info.get("snippet", {})
            status = video_info.get("status", {})
            
            # Actualizar campos si se proporcionan
            if title is not None:
                snippet["title"] = title
            if description is not None:
                snippet["description"] = description
            if tags is not None:
                snippet["tags"] = tags
            if category_id is not None:
                snippet["categoryId"] = category_id
            if privacy_status is not None:
                status["privacyStatus"] = privacy_status
            
            # Crear cuerpo de la solicitud
            body = {
                "id": video_id,
                "snippet": snippet,
                "status": status
            }
            
            # Ejecutar solicitud
            self.youtube.videos().update(
                part="snippet,status",
                body=body
            ).execute()
            
            logger.info(f"Video {video_id} actualizado correctamente")
            return {"status": "success", "message": "Video actualizado correctamente"}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al actualizar video: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al actualizar video: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def delete_video(self, video_id: str) -> Dict[str, Any]:
        """
        Elimina un video
        
        Args:
            video_id: ID del video
            
        Returns:
            Resultado de la operación
        """
        if not self._check_quota(cost=50):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Ejecutar solicitud
            self.youtube.videos().delete(id=video_id).execute()
            
            logger.info(f"Video {video_id} eliminado correctamente")
            return {"status": "success", "message": "Video eliminado correctamente"}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al eliminar video: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al eliminar video: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_channel_info(self) -> Dict[str, Any]:
        """
        Obtiene información del canal
        
        Returns:
            Información del canal
        """
        if not self._check_quota(cost=5):
            return {"status": "error", "message": "Límite de cuota alcanzado"}
        
        try:
            # Obtener información del canal
            response = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=self.config.get('channel_id')
            ).execute()
            
            if not response.get("items"):
                return {"status": "error", "message": "Canal no encontrado"}
            
            channel_info = response["items"][0]
            snippet = channel_info.get("snippet", {})
            statistics = channel_info.get("statistics", {})
            content_details = channel_info.get("contentDetails", {})
            
            # Extraer información
            info = {
                "channel_id": channel_info.get("id", ""),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "custom_url": snippet.get("customUrl", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "country": snippet.get("country", ""),
                "view_count": int(statistics.get("viewCount", 0)),
                "subscriber_count": int(statistics.get("subscriberCount", 0)),
                "video_count": int(statistics.get("videoCount", 0)),
                "uploads_playlist": content_details.get("relatedPlaylists", {}).get("uploads", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "channel_info": info}
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al obtener información del canal: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al obtener información del canal: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_optimal_upload_time(self) -> Dict[str, Any]:
        """
        Determina el momento óptimo para publicar basado en analíticas
        
        Returns:
            Hora y día óptimos para publicar
        """
        try:
            # Obtener analíticas de los últimos 90 días
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            
            response = self.youtube_analytics.reports().query(
                ids=f"channel=={self.config.get('channel_id')}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched",
                dimensions="day,hour"
            ).execute()
            
            if not response.get("rows"):
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para determinar el momento óptimo",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "hour": 19  # 7 PM
                    }
                }
            
            # Procesar datos para encontrar el mejor momento
            day_hour_views = {}
            for row in response.get("rows", []):
                day_str, hour_str, views, watch_time = row
                
                                # Convertir fecha a día de la semana (0-6, lunes-domingo)
                date_obj = datetime.strptime(day_str, "%Y-%m-%d")
                day_of_week = date_obj.weekday()
                hour = int(hour_str)
                
                # Acumular vistas y tiempo de visualización por día y hora
                key = (day_of_week, hour)
                if key not in day_hour_views:
                    day_hour_views[key] = {"views": 0, "watch_time": 0, "count": 0}
                
                day_hour_views[key]["views"] += int(views)
                day_hour_views[key]["watch_time"] += float(watch_time)
                day_hour_views[key]["count"] += 1
            
            # Calcular promedios y encontrar el mejor momento
            best_time = None
            best_score = 0
            
            for (day, hour), data in day_hour_views.items():
                # Calcular promedio de vistas y tiempo de visualización
                avg_views = data["views"] / data["count"]
                avg_watch_time = data["watch_time"] / data["count"]
                
                # Calcular puntuación combinada (50% vistas, 50% tiempo de visualización)
                # Normalizar valores para que estén en escalas comparables
                max_views = max([d["views"] / d["count"] for d in day_hour_views.values()])
                max_watch_time = max([d["watch_time"] / d["count"] for d in day_hour_views.values()])
                
                norm_views = avg_views / max_views if max_views > 0 else 0
                norm_watch_time = avg_watch_time / max_watch_time if max_watch_time > 0 else 0
                
                score = (norm_views * 0.5) + (norm_watch_time * 0.5)
                
                # Actualizar mejor momento si la puntuación es mayor
                if score > best_score:
                    best_score = score
                    best_time = (day, hour)
            
            # Mapear día de la semana a nombre
            day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            
            if best_time:
                best_day, best_hour = best_time
                return {
                    "status": "success",
                    "recommendation": {
                        "day": best_day,
                        "day_name": day_names[best_day],
                        "hour": best_hour,
                        "hour_formatted": f"{best_hour}:00",
                        "score": best_score
                    },
                    "message": f"El mejor momento para publicar es {day_names[best_day]} a las {best_hour}:00"
                }
            else:
                # Valor predeterminado si no se puede determinar
                return {
                    "status": "warning",
                    "message": "No se pudo determinar el momento óptimo con los datos disponibles",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "day_name": "Viernes",
                        "hour": 19,  # 7 PM
                        "hour_formatted": "19:00"
                    }
                }
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            logger.error(f"Error de API al obtener datos para tiempo óptimo: {error_content}")
            return {"status": "error", "message": error_content.get("error", {}).get("message", str(e))}
        except Exception as e:
            logger.error(f"Error al determinar tiempo óptimo: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "recommendation": {
                    "day": 5,  # Viernes
                    "day_name": "Viernes",
                    "hour": 19,  # 7 PM
                    "hour_formatted": "19:00"
                }
            }
    
    def check_shadowban(self, video_id: str, threshold_days: int = 7) -> Dict[str, Any]:
        """
        Verifica si un video podría estar bajo shadowban
        
        Args:
            video_id: ID del video
            threshold_days: Días para analizar tendencia
            
        Returns:
            Resultado del análisis de shadowban
        """
        try:
            # Obtener métricas del video
            metrics_response = self.get_video_metrics(video_id)
            if metrics_response["status"] != "success":
                return {"status": "error", "message": "No se pudieron obtener métricas del video"}
            
            metrics = metrics_response["metrics"]
            published_date = datetime.fromisoformat(metrics["published_at"].replace("Z", "+00:00"))
            video_age_days = (datetime.now(published_date.tzinfo) - published_date).days
            
            # Si el video es muy reciente, no podemos determinar shadowban
            if video_age_days < threshold_days:
                return {
                    "status": "warning",
                    "message": f"El video tiene menos de {threshold_days} días, análisis no concluyente",
                    "is_shadowbanned": False,
                    "confidence": 0.0
                }
            
            # Obtener analíticas para analizar tendencia
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=threshold_days)).strftime("%Y-%m-%d")
            
            analytics_response = self.get_advanced_analytics(video_id, start_date, end_date)
            if analytics_response["status"] != "success":
                return {"status": "error", "message": "No se pudieron obtener analíticas del video"}
            
            analytics = analytics_response["analytics"]
            
            # Analizar tendencia de vistas
            if not analytics["data"]:
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para analizar",
                    "is_shadowbanned": False,
                    "confidence": 0.0
                }
            
            # Extraer índices de columnas
            headers = analytics["column_headers"]
            day_idx = headers.index("day") if "day" in headers else 0
            views_idx = headers.index("views") if "views" in headers else 1
            
            # Calcular tendencia de vistas
            daily_views = [(row[day_idx], int(row[views_idx])) for row in analytics["data"]]
            daily_views.sort(key=lambda x: x[0])  # Ordenar por fecha
            
            # Calcular pendiente de la tendencia
            if len(daily_views) >= 3:
                # Usar los últimos 3 días para determinar tendencia
                recent_views = [views for _, views in daily_views[-3:]]
                trend = (recent_views[-1] - recent_views[0]) / 2
                
                # Calcular promedio de vistas diarias
                avg_views = sum(views for _, views in daily_views) / len(daily_views)
                
                # Calcular ratio de engagement
                engagement_ratio = metrics["likes"] / metrics["views"] if metrics["views"] > 0 else 0
                
                # Determinar si hay shadowban basado en múltiples factores
                is_shadowbanned = False
                confidence = 0.0
                
                # Factor 1: Caída abrupta de vistas
                if trend < -0.5 * avg_views:
                    is_shadowbanned = True
                    confidence += 0.4
                
                # Factor 2: Engagement alto pero vistas bajas
                expected_views = metrics["subscriber_count"] * 0.01 if "subscriber_count" in metrics else 100
                if engagement_ratio > 0.1 and metrics["views"] < expected_views:
                    is_shadowbanned = True
                    confidence += 0.3
                
                # Factor 3: Comentarios desactivados o limitados
                if metrics["comments"] == 0 and metrics["views"] > 100:
                    is_shadowbanned = True
                    confidence += 0.3
                
                # Limitar confianza a 1.0
                confidence = min(confidence, 1.0)
                
                return {
                    "status": "success",
                    "is_shadowbanned": is_shadowbanned,
                    "confidence": confidence,
                    "metrics": {
                        "trend": trend,
                        "avg_views": avg_views,
                        "engagement_ratio": engagement_ratio
                    },
                    "message": f"Posible shadowban: {'Sí' if is_shadowbanned else 'No'} (Confianza: {confidence:.2f})"
                }
            else:
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para determinar tendencia",
                    "is_shadowbanned": False,
                    "confidence": 0.0
                }
        except Exception as e:
            logger.error(f"Error al verificar shadowban: {str(e)}")
            return {"status": "error", "message": str(e)}