"""
TikTok Adapter - Adaptador para TikTok

Este módulo proporciona una interfaz unificada para interactuar con la TikTok Business API,
gestionando la publicación de contenido, análisis de métricas, gestión de comentarios,
optimización de hashtags, y otras funcionalidades específicas de TikTok.
"""

import os
import sys
import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Importar el cargador de configuraciones
from utils.config_loader import get_platform_credentials

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
        self.config = get_platform_credentials('tiktok')
        self.access_token = None
        self.token_expiry = None
        self.api_base_url = "https://open.tiktokapis.com/v2"
        self.rate_limit_remaining = 100
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        self.scopes = [
            "video.publish",
            "video.list",
            "user.info.basic",
            "video.insights",
            "comment.list",
            "comment.publish",
            "user.insights"
        ]
        self.initialize_api()
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la TikTok Business API usando OAuth 2.0
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            if not self.config:
                logger.error("No se encontró configuración para TikTok")
                return False
            
            client_key = self.config.get('client_key')
            client_secret = self.config.get('client_secret')
            access_token = self.config.get('access_token')
            refresh_token = self.config.get('refresh_token')
            redirect_uri = self.config.get('redirect_uri', 'http://localhost:8080')
            
            if not all([client_key, client_secret]):
                logger.error("Faltan credenciales OAuth 2.0 esenciales (client_key, client_secret)")
                return False
            
            if not access_token or not refresh_token:
                logger.info("No se encontró access_token o refresh_token, iniciando flujo OAuth 2.0")
                auth_url = (
                    f"https://www.tiktok.com/v2/auth/authorize/"
                    f"?client_key={client_key}"
                    f"&response_type=code"
                    f"&scope={','.join(self.scopes)}"
                    f"&redirect_uri={redirect_uri}"
                    f"&state=auth_state_{int(time.time())}"
                )
                
                logger.info(f"Por favor, visita esta URL para autorizar la aplicación: {auth_url}")
                auth_code = input("Ingresa el código de autorización recibido: ")
                
                token_response = self._exchange_code_for_tokens(auth_code, client_key, client_secret, redirect_uri)
                if token_response.get("status") != "success":
                    logger.error(f"Error al obtener tokens: {token_response.get('message')}")
                    return False
                
                self.access_token = token_response["access_token"]
                self.token_expiry = datetime.now() + timedelta(seconds=token_response["expires_in"])
                self.config['access_token'] = self.access_token
                self.config['refresh_token'] = token_response["refresh_token"]
                self.config['token_expiry'] = self.token_expiry.isoformat()
                self._save_credentials()
            else:
                self.access_token = access_token
                self.token_expiry = datetime.fromisoformat(self.config.get('token_expiry')) if self.config.get('token_expiry') else datetime.now() + timedelta(hours=1)
                
                if datetime.now() >= self.token_expiry:
                    logger.info("Access token expirado, intentando refrescar")
                    if not self._refresh_access_token():
                        return False
            
            logger.info("API de TikTok inicializada correctamente con OAuth 2.0")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar la API de TikTok: {str(e)}")
            return False
    
    def _exchange_code_for_tokens(self, auth_code: str, client_key: str, client_secret: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Intercambia el código de autorización por access_token y refresh_token
        
        Args:
            auth_code: Código de autorización recibido
            client_key: Clave del cliente
            client_secret: Secreto del cliente
            redirect_uri: URI de redirección
            
        Returns:
            Diccionario con los tokens o error
        """
        try:
            token_url = f"{self.api_base_url}/oauth/token/"
            payload = {
                "client_key": client_key,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": redirect_uri
            }
            
            response = requests.post(token_url, data=payload)
            response.raise_for_status()
            token_data = response.json()
            
            if "access_token" not in token_data:
                return {"status": "error", "message": token_data.get("error_description", "Error desconocido")}
            
            return {
                "status": "success",
                "access_token": token_data["access_token"],
                "refresh_token": token_data["refresh_token"],
                "expires_in": token_data["expires_in"],
                "scope": token_data["scope"]
            }
        except Exception as e:
            logger.error(f"Error al intercambiar código por tokens: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _refresh_access_token(self) -> bool:
        """
        Refresca el access_token usando el refresh_token
        
        Returns:
            True si se refrescó correctamente, False en caso contrario
        """
        try:
            refresh_token = self.config.get('refresh_token')
            if not refresh_token:
                logger.error("No se encontró refresh_token para refrescar")
                return False
            
            token_url = f"{self.api_base_url}/oauth/token/"
            payload = {
                "client_key": self.config.get('client_key'),
                "client_secret": self.config.get('client_secret'),
                "grant_type": "refresh_token",
                "refresh_token": refresh_token
            }
            
            response = requests.post(token_url, data=payload)
            response.raise_for_status()
            token_data = response.json()
            
            if "access_token" not in token_data:
                logger.error(f"Error al refrescar token: {token_data.get('error_description', 'Error desconocido')}")
                return False
            
            self.access_token = token_data["access_token"]
            self.token_expiry = datetime.now() + timedelta(seconds=token_data["expires_in"])
            self.config['access_token'] = self.access_token
            self.config['refresh_token'] = token_data["refresh_token"]
            self.config['token_expiry'] = self.token_expiry.isoformat()
            self._save_credentials()
            
            logger.info("Access token refrescado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al refrescar access token: {str(e)}")
            return False
    
    def _save_credentials(self) -> None:
        """
        Guarda las credenciales actualizadas de forma segura
        """
        try:
            config_file = "config/platforms.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    all_configs = json.load(f)
                all_configs['tiktok'] = self.config
                with open(config_file, 'w') as f:
                    json.dump(all_configs, f, indent=2)
                logger.info("Credenciales actualizadas guardadas correctamente")
            else:
                logger.warning("No se pudo guardar las credenciales: archivo de configuración no encontrado")
        except Exception as e:
            logger.error(f"Error al guardar credenciales: {str(e)}")
    
    def _check_rate_limit(self, cost: int = 1) -> bool:
        """
        Verifica si hay cuota disponible para la operación
        
        Args:
            cost: Costo estimado de la operación
            
        Returns:
            True si hay cuota disponible, False en caso contrario
        """
        if datetime.now() > self.rate_limit_reset:
            self.rate_limit_remaining = 100
            self.rate_limit_reset = datetime.now() + timedelta(hours=1)
            logger.info("Límite de tasa de TikTok API reiniciado")
        
        if self.rate_limit_remaining < cost:
            logger.warning(f"Límite de tasa alcanzado: {self.rate_limit_remaining}/100")
            return False
        
        self.rate_limit_remaining -= cost
        return True
    
    def _update_rate_limit(self, response: requests.Response) -> None:
        """
        Actualiza el estado del límite de tasa basado en los encabezados de la respuesta
        
        Args:
            response: Respuesta HTTP de la API
        """
        try:
            # Nota: TikTok API no proporciona encabezados estándar de rate limit; esto es un placeholder
            remaining = response.headers.get('X-Rate-Limit-Remaining', self.rate_limit_remaining)
            reset_time = response.headers.get('X-Rate-Limit-Reset')
            self.rate_limit_remaining = int(remaining)
            if reset_time:
                self.rate_limit_reset = datetime.fromtimestamp(float(reset_time))
        except Exception as e:
            logger.warning(f"Error al actualizar límite de tasa: {str(e)}")

    def upload_video(self, video_path: str, title: str, description: str, 
                    tags: List[str] = None, privacy_level: str = "PUBLIC", 
                    disable_comments: bool = False, disable_duet: bool = False,
                    disable_stitch: bool = False) -> Dict[str, Any]:
        """
        Sube un video a TikTok
        
        Args:
            video_path: Ruta al archivo de video
            title: Título del video
            description: Descripción del video
            tags: Lista de etiquetas
            privacy_level: Nivel de privacidad (PUBLIC, MUTUAL_FOLLOW_FRIENDS, FOLLOWER_OF_CREATOR, SELF_ONLY)
            disable_comments: Deshabilitar comentarios
            disable_duet: Deshabilitar duetos
            disable_stitch: Deshabilitar stitch
            
        Returns:
            Información del video subido o error
        """
        if not self._check_rate_limit(cost=10):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not os.path.exists(video_path):
                return {"status": "error", "message": f"Archivo no encontrado: {video_path}"}
            
            if tags is None:
                tags = []
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            init_response = requests.post(
                f"{self.api_base_url}/video/init/",
                headers=headers,
                json={}
            )
            init_response.raise_for_status()
            init_data = init_response.json()
            
            if "upload_url" not in init_data.get("data", {}):
                return {"status": "error", "message": "No se pudo obtener URL de carga"}
            
            upload_url = init_data["data"]["upload_url"]
            
            with open(video_path, 'rb') as video_file:
                files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
                upload_response = requests.put(upload_url, files=files)
                upload_response.raise_for_status()
            
            publish_data = {
                "post_info": {
                    "title": title,
                    "description": description,
                    "privacy_level": privacy_level,
                    "disable_comment": disable_comments,
                    "disable_duet": disable_duet,
                    "disable_stitch": disable_stitch,
                    "video_tags": tags
                }
            }
            
            publish_response = requests.post(
                f"{self.api_base_url}/video/publish/",
                headers=headers,
                json=publish_data
            )
            publish_response.raise_for_status()
            publish_data = publish_response.json()
            
            video_id = publish_data.get("data", {}).get("video_id")
            if not video_id:
                return {"status": "error", "message": "No se pudo obtener ID del video"}
            
            logger.info(f"Video subido correctamente: {video_id}")
            self._update_rate_limit(publish_response)
            
            return {
                "status": "success",
                "video_id": video_id,
                "url": f"https://www.tiktok.com/@{self.config.get('open_id')}/video/{video_id}"
            }
        except requests.HTTPError as e:
            logger.error(f"Error de API al subir video: {str(e)}")
            return {"status": "error", "message": str(e)}
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
        if not self._check_rate_limit(cost=2):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/video/list/?fields=video_id,title,description,create_time,view_count,like_count,comment_count,share_count",
                headers=headers,
                params={"video_ids": [video_id]}
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            if not data.get("data", {}).get("videos"):
                return {"status": "error", "message": "Video no encontrado"}
            
            video_info = data["data"]["videos"][0]
            
            metrics = {
                "video_id": video_info.get("video_id", ""),
                "title": video_info.get("title", ""),
                "description": video_info.get("description", ""),
                "create_time": datetime.fromtimestamp(video_info.get("create_time", 0)).isoformat(),
                "views": video_info.get("view_count", 0),
                "likes": video_info.get("like_count", 0),
                "comments": video_info.get("comment_count", 0),
                "shares": video_info.get("share_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "metrics": metrics}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener métricas: {str(e)}")
            return {"status": "error", "message": str(e)}
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
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/video/insights/?video_id={video_id}",
                headers=headers,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "fields": "views,likes,comments,shares,average_watch_time,reach,engagement_rate"
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            analytics = {
                "video_id": video_id,
                "start_date": start_date,
                "end_date": end_date,
                "data": data.get("data", {}).get("insights", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "analytics": analytics}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener analíticas: {str(e)}")
            return {"status": "error", "message": str(e)}
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
        if not self._check_rate_limit(cost=2):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/comment/list/?video_id={video_id}&max_count={min(max_results, 100)}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            comments = []
            for comment in data.get("data", {}).get("comments", []):
                comments.append({
                    "comment_id": comment.get("id", ""),
                    "author": comment.get("user", {}).get("nickname", ""),
                    "author_id": comment.get("user", {}).get("unique_id", ""),
                    "text": comment.get("text", ""),
                    "like_count": comment.get("like_count", 0),
                    "create_time": datetime.fromtimestamp(comment.get("create_time", 0)).isoformat(),
                    "reply_count": comment.get("reply_count", 0)
                })
            
            return {"status": "success", "comments": comments}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener comentarios: {str(e)}")
            return {"status": "error", "message": str(e)}
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
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.post(
                f"{self.api_base_url}/comment/reply/",
                headers=headers,
                json={
                    "comment_id": comment_id,
                    "text": text
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            logger.info(f"Respuesta enviada al comentario {comment_id}")
            return {
                "status": "success",
                "comment_id": data.get("data", {}).get("reply_id", ""),
                "text": text
            }
        except requests.HTTPError as e:
            logger.error(f"Error de API al responder comentario: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al responder comentario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_trending_videos(self, region_code: str = "ES", max_results: int = 50) -> Dict[str, Any]:
        """
        Obtiene videos en tendencia
        
        Args:
            region_code: Código de región (ES, US, MX, etc.)
            max_results: Número máximo de resultados
            
        Returns:
            Lista de videos en tendencia
        """
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/video/query/?fields=video_id,title,description,create_time,view_count,like_count,cover_image_url",
                headers=headers,
                params={
                    "region_code": region_code,
                    "max_count": min(max_results, 50)
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            trending_videos = []
            for video in data.get("data", {}).get("videos", []):
                trending_videos.append({
                    "video_id": video.get("video_id", ""),
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "create_time": datetime.fromtimestamp(video.get("create_time", 0)).isoformat(),
                    "views": video.get("view_count", 0),
                    "likes": video.get("like_count", 0),
                    "thumbnail": video.get("cover_image_url", "")
                })
            
            return {"status": "success", "trending_videos": trending_videos}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener tendencias: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al obtener tendencias: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def search_videos(self, query: str, max_results: int = 50, region_code: str = "ES") -> Dict[str, Any]:
        """
        Busca videos en TikTok
        
        Args:
            query: Término de búsqueda
            max_results: Número máximo de resultados
            region_code: Código de región
            
        Returns:
            Lista de videos encontrados
        """
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/video/search/?fields=video_id,title,description,create_time,view_count,cover_image_url",
                headers=headers,
                params={
                    "query": query,
                    "max_count": min(max_results, 50),
                    "region_code": region_code
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            search_results = []
            for video in data.get("data", {}).get("videos", []):
                search_results.append({
                    "video_id": video.get("video_id", ""),
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "create_time": datetime.fromtimestamp(video.get("create_time", 0)).isoformat(),
                    "views": video.get("view_count", 0),
                    "thumbnail": video.get("cover_image_url", "")
                })
            
            return {"status": "success", "search_results": search_results}
        except requests.HTTPError as e:
            logger.error(f"Error de API al buscar videos: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al buscar videos: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def update_video(self, video_id: str, description: str = None, privacy_level: str = None,
                    disable_comments: bool = None, disable_duet: bool = None,
                    disable_stitch: bool = None) -> Dict[str, Any]:
        """
        Actualiza metadatos de un video
        
        Args:
            video_id: ID del video
            description: Nueva descripción (opcional)
            privacy_level: Nuevo nivel de privacidad (opcional)
            disable_comments: Deshabilitar comentarios (opcional)
            disable_duet: Deshabilitar duetos (opcional)
            disable_stitch: Deshabilitar stitch (opcional)
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            update_data = {}
            if description is not None:
                update_data["description"] = description
            if privacy_level is not None:
                update_data["privacy_level"] = privacy_level
            if disable_comments is not None:
                update_data["disable_comment"] = disable_comments
            if disable_duet is not None:
                update_data["disable_duet"] = disable_duet
            if disable_stitch is not None:
                update_data["disable_stitch"] = disable_stitch
            
            if not update_data:
                return {"status": "error", "message": "No se proporcionaron datos para actualizar"}
            
            response = requests.post(
                f"{self.api_base_url}/video/update/",
                headers=headers,
                json={
                    "video_id": video_id,
                    "post_info": update_data
                }
            )
            response.raise_for_status()
            
            self._update_rate_limit(response)
            
            logger.info(f"Video {video_id} actualizado correctamente")
            return {"status": "success", "message": "Video actualizado correctamente"}
        except requests.HTTPError as e:
            logger.error(f"Error de API al actualizar video: {str(e)}")
            return {"status": "error", "message": str(e)}
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
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.post(
                f"{self.api_base_url}/video/delete/",
                headers=headers,
                json={"video_id": video_id}
            )
            response.raise_for_status()
            
            self._update_rate_limit(response)
            
            logger.info(f"Video {video_id} eliminado correctamente")
            return {"status": "success", "message": "Video eliminado correctamente"}
        except requests.HTTPError as e:
            logger.error(f"Error de API al eliminar video: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al eliminar video: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Obtiene información del usuario
        
        Returns:
            Información del usuario
        """
        if not self._check_rate_limit(cost=1):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/user/info/?fields=open_id,nickname,avatar_url,follower_count,following_count,video_count",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            user_info = data.get("data", {}).get("user", {})
            info = {
                "open_id": user_info.get("open_id", ""),
                "nickname": user_info.get("nickname", ""),
                "avatar_url": user_info.get("avatar_url", ""),
                "follower_count": user_info.get("follower_count", 0),
                "following_count": user_info.get("following_count", 0),
                "video_count": user_info.get("video_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "user_info": info}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener información del usuario: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al obtener información del usuario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_optimal_upload_time(self) -> Dict[str, Any]:
        """
        Determina el momento óptimo para publicar basado en analíticas
        
        Returns:
            Hora y día óptimos para publicar
        """
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/user/insights/?fields=views,engagement_rate",
                headers=headers,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "dimensions": "day,hour"
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            if not data.get("data", {}).get("insights"):
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para determinar el momento óptimo",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "hour": 19  # 7 PM
                    }
                }
            
            day_hour_metrics = {}
            for insight in data["data"]["insights"]:
                day_str = insight.get("day")
                hour = int(insight.get("hour", 0))
                views = insight.get("views", 0)
                engagement = insight.get("engagement_rate", 0)
                
                date_obj = datetime.strptime(day_str, "%Y-%m-%d")
                day_of_week = date_obj.weekday()
                
                key = (day_of_week, hour)
                if key not in day_hour_metrics:
                    day_hour_metrics[key] = {"views": 0, "engagement": 0, "count": 0}
                
                day_hour_metrics[key]["views"] += views
                day_hour_metrics[key]["engagement"] += engagement
                day_hour_metrics[key]["count"] += 1
            
            best_time = None
            best_score = 0
            
            for (day, hour), metrics in day_hour_metrics.items():
                avg_views = metrics["views"] / metrics["count"]
                avg_engagement = metrics["engagement"] / metrics["count"]
                
                max_views = max([m["views"] / m["count"] for m in day_hour_metrics.values()])
                max_engagement = max([m["engagement"] / m["count"] for m in day_hour_metrics.values()])
                
                norm_views = avg_views / max_views if max_views > 0 else 0
                norm_engagement = avg_engagement / max_engagement if max_engagement > 0 else 0
                
                score = (norm_views * 0.5) + (norm_engagement * 0.5)
                
                if score > best_score:
                    best_score = score
                    best_time = (day, hour)
            
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
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener datos para tiempo óptimo: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "recommendation": {
                    "day": 5,
                    "day_name": "Viernes",
                    "hour": 19,
                    "hour_formatted": "19:00"
                }
            }
        except Exception as e:
            logger.error(f"Error al determinar tiempo óptimo: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "recommendation": {
                    "day": 5,
                    "day_name": "Viernes",
                    "hour": 19,
                    "hour_formatted": "19:00"
                }
            }
    
    def manage_hashtags(self, video_id: str = None, region_code: str = "ES") -> Dict[str, Any]:
        """
        Gestiona y optimiza hashtags para un video o devuelve hashtags en tendencia
        
        Args:
            video_id: ID del video para actualizar hashtags (opcional)
            region_code: Código de región para tendencias
            
        Returns:
            Lista de hashtags recomendados o resultado de actualización
        """
        if not self._check_rate_limit(cost=3):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            if video_id:
                # Actualizar hashtags de un video existente
                trending_hashtags = self.get_trending_hashtags(region_code=region_code)
                if trending_hashtags["status"] != "success":
                    return trending_hashtags
                
                new_hashtags = [h["name"] for h in trending_hashtags["hashtags"][:5]]
                
                response = requests.post(
                    f"{self.api_base_url}/video/update/",
                    headers=headers,
                    json={
                        "video_id": video_id,
                        "post_info": {
                            "video_tags": new_hashtags
                        }
                    }
                )
                response.raise_for_status()
                
                self._update_rate_limit(response)
                
                logger.info(f"Hashtags actualizados para video {video_id}: {new_hashtags}")
                return {
                    "status": "success",
                    "video_id": video_id,
                    "hashtags": new_hashtags,
                    "message": "Hashtags actualizados correctamente"
                }
            else:
                # Obtener hashtags en tendencia
                return self.get_trending_hashtags(region_code=region_code)
        except requests.HTTPError as e:
            logger.error(f"Error de API al gestionar hashtags: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al gestionar hashtags: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_trending_hashtags(self, region_code: str = "ES", max_results: int = 20) -> Dict[str, Any]:
        """
        Obtiene hashtags en tendencia
        
        Args:
            region_code: Código de región
            max_results: Número máximo de hashtags
            
        Returns:
            Lista de hashtags en tendencia
        """
        if not self._check_rate_limit(cost=3):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/hashtag/query/?fields=name,video_count,view_count",
                headers=headers,
                params={
                    "region_code": region_code,
                    "max_count": min(max_results, 50)
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            hashtags = []
            for hashtag in data.get("data", {}).get("hashtags", []):
                hashtags.append({
                    "name": hashtag.get("name", ""),
                    "video_count": hashtag.get("video_count", 0),
                    "view_count": hashtag.get("view_count", 0)
                })
            
            return {"status": "success", "hashtags": hashtags}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener hashtags: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al obtener hashtags: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_engagement(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Obtiene métricas de engagement del usuario
        
        Args:
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            
        Returns:
            Métricas de engagement
        """
        if not self._check_rate_limit(cost=5):
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(
                f"{self.api_base_url}/user/insights/?fields=views,likes,comments,shares,engagement_rate",
                headers=headers,
                params={
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._update_rate_limit(response)
            
            engagement = {
                "start_date": start_date,
                "end_date": end_date,
                "views": data.get("data", {}).get("views", 0),
                "likes": data.get("data", {}).get("likes", 0),
                "comments": data.get("data", {}).get("comments", 0),
                "shares": data.get("data", {}).get("shares", 0),
                "engagement_rate": data.get("data", {}).get("engagement_rate", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "engagement": engagement}
        except requests.HTTPError as e:
            logger.error(f"Error de API al obtener engagement: {str(e)}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error al obtener engagement: {str(e)}")
            return {"status": "error", "message": str(e)}
    
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
            metrics_response = self.get_video_metrics(video_id)
            if metrics_response["status"] != "success":
                return {"status": "error", "message": "No se pudieron obtener métricas del video"}
            
            metrics = metrics_response["metrics"]
            published_date = datetime.fromisoformat(metrics["create_time"])
            video_age_days = (datetime.now() - published_date).days
            
            if video_age_days < threshold_days:
                return {
                    "status": "warning",
                    "message": f"El video tiene menos de {threshold_days} días, análisis no concluyente",
                    "is_shadowbanned": False,
                    "confidence": 0.0
                }
            
            analytics_response = self.get_advanced_analytics(video_id)
            if analytics_response["status"] != "success":
                return {"status": "error", "message": "No se pudieron obtener analíticas del video"}
            
            analytics = analytics_response["analytics"]
            
            engagement_ratio = metrics["likes"] / metrics["views"] if metrics["views"] > 0 else 0
            expected_views = analytics["data"].get("reach", 1000) * 0.05
            
            is_shadowbanned = False
            confidence = 0.0
            
            if metrics["views"] < expected_views:
                is_shadowbanned = True
                confidence += 0.4
            
            if engagement_ratio < 0.02:
                is_shadowbanned = True
                confidence += 0.3
            
            if metrics["comments"] == 0 and metrics["views"] > 100:
                is_shadowbanned = True
                confidence += 0.3
            
            confidence = min(confidence, 1.0)
            
            return {
                "status": "success",
                "is_shadowbanned": is_shadowbanned,
                "confidence": confidence,
                "metrics": {
                    "engagement_ratio": engagement_ratio,
                    "expected_views": expected_views
                },
                "message": f"Posible shadowban: {'Sí' if is_shadowbanned else 'No'} (Confianza: {confidence:.2f})"
            }
        except Exception as e:
            logger.error(f"Error al verificar shadowban: {str(e)}")
            return {"status": "error", "message": str(e)}