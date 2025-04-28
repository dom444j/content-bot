"""
TikTok Adapter - Adaptador para TikTok

Este módulo proporciona una interfaz unificada para interactuar con la API de TikTok,
gestionando la publicación de contenido, análisis de métricas, gestión de comentarios,
y otras funcionalidades específicas de TikTok.
"""

import os
import sys
import json
import logging
import time
import requests
import hmac
import hashlib
import base64
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/tiktok_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TikTokAdapter")

class TikTokAdapter:
    """
    Adaptador para interactuar con la API de TikTok
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el adaptador de TikTok
        
        Args:
            config_path: Ruta al archivo de configuración con credenciales
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.access_token = None
        self.token_expiry = None
        self.api_base_url = "https://open.tiktokapis.com/v2"
        self.rate_limit_remaining = 100  # Valor inicial estimado
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        self.initialize_api()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo
        
        Returns:
            Configuración de TikTok
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('tiktok', {})
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la API de TikTok
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            # Verificar si hay credenciales
            if not self.config:
                logger.error("No se encontró configuración para TikTok")
                return False
            
            # Verificar si el token actual es válido
            if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
                logger.info("Token de acceso actual válido hasta " + self.token_expiry.isoformat())
                return True
            
            # Obtener nuevo token de acceso
            client_key = self.config.get('client_key')
            client_secret = self.config.get('client_secret')
            
            if not client_key or not client_secret:
                logger.error("Faltan credenciales de cliente en la configuración")
                return False
            
            # Obtener token usando credenciales de cliente
            response = self._get_client_token(client_key, client_secret)
            
            if response.get("status") == "error":
                logger.error(f"Error al obtener token: {response.get('message')}")
                return False
            
            self.access_token = response.get("access_token")
            expires_in = response.get("expires_in", 86400)  # Por defecto 24 horas
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info(f"API de TikTok inicializada correctamente, token válido hasta {self.token_expiry.isoformat()}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar API de TikTok: {str(e)}")
            return False
    
    def _get_client_token(self, client_key: str, client_secret: str) -> Dict[str, Any]:
        """
        Obtiene un token de acceso usando credenciales de cliente
        
        Args:
            client_key: Clave de cliente
            client_secret: Secreto de cliente
            
        Returns:
            Información del token o error
        """
        try:
            url = "https://open-api.tiktok.com/oauth/client_token/"
            
            # Preparar datos
            data = {
                "client_key": client_key,
                "client_secret": client_secret,
                "grant_type": "client_credentials"
            }
            
            # Realizar solicitud
            response = requests.post(url, data=data)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Error HTTP {response.status_code}: {response.text}"
                }
            
            result = response.json()
            data = result.get("data", {})
            
            if not data or "access_token" not in data:
                return {
                    "status": "error",
                    "message": f"Respuesta inválida: {result}"
                }
            
            return {
                "status": "success",
                "access_token": data.get("access_token"),
                "expires_in": data.get("expires_in", 86400)
            }
        except Exception as e:
            logger.error(f"Error al obtener token de cliente: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _check_rate_limit(self) -> bool:
        """
        Verifica si hay límite de tasa disponible
        
        Returns:
            True si hay límite disponible, False en caso contrario
        """
        # Verificar si es tiempo de reiniciar el límite
        if datetime.now() > self.rate_limit_reset:
            self.rate_limit_remaining = 100  # Valor estimado
            self.rate_limit_reset = datetime.now() + timedelta(hours=1)
            logger.info("Límite de tasa de TikTok API reiniciado")
        
        # Verificar si hay límite disponible
        if self.rate_limit_remaining <= 0:
            logger.warning(f"Límite de tasa alcanzado, se reiniciará en {(self.rate_limit_reset - datetime.now()).total_seconds()} segundos")
            return False
        
        # Decrementar límite
        self.rate_limit_remaining -= 1
        return True
    
    def _make_api_request(self, method: str, endpoint: str, params: Dict = None, 
                         data: Dict = None, files: Dict = None) -> Dict[str, Any]:
        """
        Realiza una solicitud a la API de TikTok
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint de la API
            params: Parámetros de consulta
            data: Datos para enviar en el cuerpo
            files: Archivos para subir
            
        Returns:
            Respuesta de la API o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar token
            if not self.access_token or (self.token_expiry and datetime.now() >= self.token_expiry):
                if not self.initialize_api():
                    return {"status": "error", "message": "No se pudo inicializar la API"}
            
            # Preparar URL
            url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
            
            # Preparar headers
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Realizar solicitud
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                if files:
                    # Si hay archivos, no usar Content-Type: application/json
                    headers.pop("Content-Type", None)
                    response = requests.post(url, headers=headers, data=data, files=files)
                else:
                    response = requests.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                return {"status": "error", "message": f"Método no soportado: {method}"}
            
            # Actualizar límites de tasa si están en los headers
            if "X-RateLimit-Remaining" in response.headers:
                self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", "0"))
            if "X-RateLimit-Reset" in response.headers:
                reset_time = int(response.headers.get("X-RateLimit-Reset", "0"))
                self.rate_limit_reset = datetime.fromtimestamp(reset_time)
            
            # Procesar respuesta
            if response.status_code >= 400:
                logger.error(f"Error HTTP {response.status_code}: {response.text}")
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
            
            # Intentar parsear como JSON
            try:
                result = response.json()
                return {
                    "status": "success",
                    "data": result
                }
            except ValueError:
                # Si no es JSON, devolver texto
                return {
                    "status": "success",
                    "data": response.text
                }
        except Exception as e:
            logger.error(f"Error en solicitud a API de TikTok: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def upload_video(self, video_path: str, title: str, description: str = None, 
                    tags: List[str] = None, privacy_level: str = "PUBLIC",
                    disable_comments: bool = False, disable_duet: bool = False,
                    disable_stitch: bool = False) -> Dict[str, Any]:
        """
        Sube un video a TikTok
        
        Args:
            video_path: Ruta al archivo de video
            title: Título del video
            description: Descripción del video
            tags: Lista de hashtags
            privacy_level: Nivel de privacidad (PUBLIC, SELF_ONLY, MUTUAL_FOLLOW_FRIENDS)
            disable_comments: Si se deshabilitan los comentarios
            disable_duet: Si se deshabilita el dueto
            disable_stitch: Si se deshabilita el stitch
            
        Returns:
            Información del video subido o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not os.path.exists(video_path):
                return {"status": "error", "message": f"Archivo no encontrado: {video_path}"}
            
            # Preparar hashtags
            hashtags = []
            if tags:
                for tag in tags:
                    # Asegurar que cada tag tenga el formato correcto
                    if not tag.startswith("#"):
                        tag = f"#{tag}"
                    hashtags.append(tag)
            
            # Construir descripción con hashtags
            full_description = title
            if description:
                full_description = f"{title}\n\n{description}"
            
            if hashtags:
                hashtag_text = " ".join(hashtags)
                full_description = f"{full_description}\n\n{hashtag_text}"
            
            # Paso 1: Iniciar la carga
            init_response = self._make_api_request(
                "POST",
                "/video/init",
                data={
                    "post_info": {
                        "title": title,
                        "description": full_description,
                        "privacy_level": privacy_level,
                        "disable_comment": disable_comments,
                        "disable_duet": disable_duet,
                        "disable_stitch": disable_stitch
                    }
                }
            )
            
            if init_response.get("status") != "success":
                return init_response
            
            upload_id = init_response.get("data", {}).get("upload_id")
            if not upload_id:
                return {"status": "error", "message": "No se pudo obtener ID de carga"}
            
            # Paso 2: Subir el video
            with open(video_path, "rb") as video_file:
                upload_response = self._make_api_request(
                    "POST",
                    "/video/upload",
                    data={"upload_id": upload_id},
                    files={"video": video_file}
                )
            
            if upload_response.get("status") != "success":
                return upload_response
            
            # Paso 3: Finalizar la carga
            finish_response = self._make_api_request(
                "POST",
                "/video/publish",
                data={"upload_id": upload_id}
            )
            
            if finish_response.get("status") != "success":
                return finish_response
            
            # Obtener información del video publicado
            video_id = finish_response.get("data", {}).get("video_id")
            
            logger.info(f"Video subido correctamente: {video_id}")
            return {
                "status": "success",
                "video_id": video_id,
                "title": title,
                "description": full_description,
                "privacy_level": privacy_level,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al subir video: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_video_metrics(self, video_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas básicas de un video
        
        Args:
            video_id: ID del video
            
        Returns:
            Métricas del video
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                "/video/info",
                params={"video_id": video_id}
            )
            
            if response.get("status") != "success":
                return response
            
            video_info = response.get("data", {}).get("video_info", {})
            
            # Extraer métricas
            metrics = {
                "video_id": video_id,
                "title": video_info.get("title", ""),
                "description": video_info.get("description", ""),
                "create_time": video_info.get("create_time", ""),
                "cover_image_url": video_info.get("cover_image_url", ""),
                "share_url": video_info.get("share_url", ""),
                "view_count": video_info.get("statistics", {}).get("view_count", 0),
                "like_count": video_info.get("statistics", {}).get("like_count", 0),
                "comment_count": video_info.get("statistics", {}).get("comment_count", 0),
                "share_count": video_info.get("statistics", {}).get("share_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "metrics": metrics}
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
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Configurar fechas
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            response = self._make_api_request(
                "GET",
                "/video/analytics",
                params={
                    "video_id": video_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "metrics": "views,likes,comments,shares,profile_views,follows"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            analytics_data = response.get("data", {}).get("analytics", {})
            
            # Procesar resultados
            analytics = {
                "video_id": video_id,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": analytics_data.get("metrics", {}),
                "daily_data": analytics_data.get("daily_data", []),
                "audience": analytics_data.get("audience", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "analytics": analytics}
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
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                "/video/comments",
                params={
                    "video_id": video_id,
                    "limit": min(max_results, 100)
                }
            )
            
            if response.get("status") != "success":
                return response
            
            comments_data = response.get("data", {}).get("comments", [])
            
            # Procesar comentarios
            comments = []
            for comment in comments_data:
                comments.append({
                    "comment_id": comment.get("id", ""),
                    "text": comment.get("text", ""),
                    "create_time": comment.get("create_time", ""),
                    "like_count": comment.get("like_count", 0),
                    "user": {
                        "id": comment.get("user", {}).get("id", ""),
                        "username": comment.get("user", {}).get("username", ""),
                        "display_name": comment.get("user", {}).get("display_name", "")
                    }
                })
            
            return {"status": "success", "comments": comments}
        except Exception as e:
            logger.error(f"Error al obtener comentarios: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def reply_to_comment(self, video_id: str, comment_id: str, text: str) -> Dict[str, Any]:
        """
        Responde a un comentario
        
        Args:
            video_id: ID del video
            comment_id: ID del comentario
            text: Texto de la respuesta
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "POST",
                "/video/comment",
                data={
                    "video_id": video_id,
                    "comment_id": comment_id,  # Si se proporciona, es una respuesta
                    "text": text
                }
            )
            
            if response.get("status") != "success":
                return response
            
            comment_data = response.get("data", {}).get("comment", {})
            
            return {
                "status": "success",
                "comment_id": comment_data.get("id", ""),
                "text": comment_data.get("text", ""),
                "create_time": comment_data.get("create_time", "")
            }
        except Exception as e:
            logger.error(f"Error al responder comentario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_trending_videos(self, category: str = None, max_results: int = 50) -> Dict[str, Any]:
        """
        Obtiene videos en tendencia
        
        Args:
            category: Categoría de tendencias (opcional)
            max_results: Número máximo de resultados
            
        Returns:
            Lista de videos en tendencia
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            params = {
                "limit": min(max_results, 50)
            }
            
            if category:
                params["category"] = category
            
            response = self._make_api_request(
                "GET",
                "/trending/videos",
                params=params
            )
            
            if response.get("status") != "success":
                return response
            
            videos_data = response.get("data", {}).get("videos", [])
            
            # Procesar resultados
            trending_videos = []
            for video in videos_data:
                trending_videos.append({
                    "video_id": video.get("id", ""),
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "create_time": video.get("create_time", ""),
                    "cover_image_url": video.get("cover_image_url", ""),
                    "share_url": video.get("share_url", ""),
                    "view_count": video.get("statistics", {}).get("view_count", 0),
                    "like_count": video.get("statistics", {}).get("like_count", 0),
                    "comment_count": video.get("statistics", {}).get("comment_count", 0),
                    "share_count": video.get("statistics", {}).get("share_count", 0),
                    "user": {
                        "id": video.get("user", {}).get("id", ""),
                        "username": video.get("user", {}).get("username", ""),
                        "display_name": video.get("user", {}).get("display_name", "")
                    }
                })
            
            return {"status": "success", "trending_videos": trending_videos}
        except Exception as e:
            logger.error(f"Error al obtener videos en tendencia: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def search_videos(self, query: str, max_results: int = 50) -> Dict[str, Any]:
        """
        Busca videos en TikTok
        
        Args:
            query: Término de búsqueda
            max_results: Número máximo de resultados
            
        Returns:
            Lista de videos encontrados
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                "/search/videos",
                params={
                    "query": query,
                    "limit": min(max_results, 50)
                }
            )
            
            if response.get("status") != "success":
                return response
            
            videos_data = response.get("data", {}).get("videos", [])
            
            # Procesar resultados
            search_results = []
            for video in videos_data:
                search_results.append({
                    "video_id": video.get("id", ""),
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "create_time": video.get("create_time", ""),
                    "cover_image_url": video.get("cover_image_url", ""),
                    "share_url": video.get("share_url", ""),
                    "view_count": video.get("statistics", {}).get("view_count", 0),
                    "like_count": video.get("statistics", {}).get("like_count", 0),
                    "comment_count": video.get("statistics", {}).get("comment_count", 0),
                    "share_count": video.get("statistics", {}).get("share_count", 0),
                    "user": {
                        "id": video.get("user", {}).get("id", ""),
                        "username": video.get("user", {}).get("username", ""),
                        "display_name": video.get("user", {}).get("display_name", "")
                    }
                })
            
            return {"status": "success", "search_results": search_results}
        except Exception as e:
            logger.error(f"Error al buscar videos: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def delete_video(self, video_id: str) -> Dict[str, Any]:
        """
        Elimina un video
        
        Args:
            video_id: ID del video
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "DELETE",
                "/video",
                params={"video_id": video_id}
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Video {video_id} eliminado correctamente")
            return {"status": "success", "message": "Video eliminado correctamente"}
        except Exception as e:
            logger.error(f"Error al eliminar video: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Obtiene información del usuario autenticado
        
        Returns:
            Información del usuario
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                "/user/info"
            )
            
            if response.get("status") != "success":
                return response
            
            user_data = response.get("data", {}).get("user", {})
            
            # Extraer información
            info = {
                "user_id": user_data.get("id", ""),
                "username": user_data.get("username", ""),
                "display_name": user_data.get("display_name", ""),
                "bio": user_data.get("bio", ""),
                "avatar_url": user_data.get("avatar_url", ""),
                "follower_count": user_data.get("follower_count", 0),
                "following_count": user_data.get("following_count", 0),
                "video_count": user_data.get("video_count", 0),
                "like_count": user_data.get("like_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "user_info": info}
        except Exception as e:
            logger.error(f"Error al obtener información del usuario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_optimal_upload_time(self) -> Dict[str, Any]:
        """
        Determina el momento óptimo para publicar basado en analíticas
        
        Returns:
            Hora y día óptimos para publicar
        """
        try:
            # Obtener analíticas de los últimos 30 días
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            response = self._make_api_request(
                "GET",
                "/user/analytics",
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "metrics": "views,profile_views,follows",
                    "dimensions": "day,hour"
                }
            )
            
            if response.get("status") != "success" or not response.get("data", {}).get("analytics", {}).get("daily_data"):
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para determinar el momento óptimo",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "hour": 20  # 8 PM
                    }
                }
            
            analytics_data = response.get("data", {}).get("analytics", {})
            daily_data = analytics_data.get("daily_data", [])
            
            # Procesar datos para encontrar el mejor momento
            day_hour_views = {}
            for day_data in daily_data:
                date_str = day_data.get("date")
                hourly_data = day_data.get("hourly_data", [])
                
                # Convertir fecha a día de la semana (0-6, lunes-domingo)
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                day_of_week = date_obj.weekday()
                
                for hour_data in hourly_data:
                    hour = hour_data.get("hour")
                    views = hour_data.get("views", 0)
                    profile_views = hour_data.get("profile_views", 0)
                    follows = hour_data.get("follows", 0)
                    
                                        # Acumular métricas por día y hora
                    key = (day_of_week, hour)
                    if key not in day_hour_views:
                        day_hour_views[key] = {"views": 0, "profile_views": 0, "follows": 0, "count": 0}
                    
                    day_hour_views[key]["views"] += views
                    day_hour_views[key]["profile_views"] += profile_views
                    day_hour_views[key]["follows"] += follows
                    day_hour_views[key]["count"] += 1
            
            # Calcular promedios y encontrar el mejor momento
            best_time = None
            best_score = 0
            
            for (day, hour), data in day_hour_views.items():
                # Calcular promedio de métricas
                avg_views = data["views"] / data["count"]
                avg_profile_views = data["profile_views"] / data["count"]
                avg_follows = data["follows"] / data["count"]
                
                # Calcular puntuación combinada (40% vistas, 30% visitas al perfil, 30% nuevos seguidores)
                # Normalizar valores para que estén en escalas comparables
                max_views = max([d["views"] / d["count"] for d in day_hour_views.values()] or [1])
                max_profile_views = max([d["profile_views"] / d["count"] for d in day_hour_views.values()] or [1])
                max_follows = max([d["follows"] / d["count"] for d in day_hour_views.values()] or [1])
                
                norm_views = avg_views / max_views if max_views > 0 else 0
                norm_profile_views = avg_profile_views / max_profile_views if max_profile_views > 0 else 0
                norm_follows = avg_follows / max_follows if max_follows > 0 else 0
                
                score = (norm_views * 0.4) + (norm_profile_views * 0.3) + (norm_follows * 0.3)
                
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
                        "hour": 20,  # 8 PM
                        "hour_formatted": "20:00"
                    }
                }
        except Exception as e:
            logger.error(f"Error al determinar tiempo óptimo: {str(e)}")
            return {
                "status": "error", 
                "message": str(e),
                "recommendation": {
                    "day": 5,  # Viernes
                    "day_name": "Viernes",
                    "hour": 20,  # 8 PM
                    "hour_formatted": "20:00"
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
            create_time = metrics.get("create_time", "")
            
            # Convertir create_time a objeto datetime
            if create_time:
                try:
                    published_date = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                except ValueError:
                    # Intentar otro formato si el anterior falla
                    published_date = datetime.strptime(create_time, "%Y-%m-%dT%H:%M:%S")
            else:
                # Si no hay fecha de creación, usar fecha actual menos 30 días
                published_date = datetime.now() - timedelta(days=30)
            
            video_age_days = (datetime.now() - published_date).days
            
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
            daily_data = analytics.get("daily_data", [])
            if not daily_data:
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para analizar",
                    "is_shadowbanned": False,
                    "confidence": 0.0
                }
            
            # Extraer vistas diarias
            daily_views = []
            for day_data in daily_data:
                date_str = day_data.get("date")
                views = day_data.get("views", 0)
                daily_views.append((date_str, views))
            
            # Ordenar por fecha
            daily_views.sort(key=lambda x: x[0])
            
            # Calcular tendencia de vistas
            if len(daily_views) >= 3:
                # Usar los últimos 3 días para determinar tendencia
                recent_views = [views for _, views in daily_views[-3:]]
                trend = (recent_views[-1] - recent_views[0]) / 2
                
                # Calcular promedio de vistas diarias
                avg_views = sum(views for _, views in daily_views) / len(daily_views)
                
                # Calcular ratio de engagement
                engagement_ratio = metrics["like_count"] / metrics["view_count"] if metrics["view_count"] > 0 else 0
                
                # Determinar si hay shadowban basado en múltiples factores
                is_shadowbanned = False
                confidence = 0.0
                
                # Factor 1: Caída abrupta de vistas
                if trend < -0.5 * avg_views:
                    is_shadowbanned = True
                    confidence += 0.4
                
                # Factor 2: Engagement alto pero vistas bajas
                follower_count = self.get_user_info().get("user_info", {}).get("follower_count", 1000)
                expected_views = follower_count * 0.01  # Esperamos al menos 1% de seguidores vean el video
                if engagement_ratio > 0.1 and metrics["view_count"] < expected_views:
                    is_shadowbanned = True
                    confidence += 0.3
                
                # Factor 3: Comentarios desactivados o limitados
                if metrics["comment_count"] == 0 and metrics["view_count"] > 100:
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