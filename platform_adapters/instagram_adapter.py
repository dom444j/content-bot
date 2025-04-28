"""
Instagram Adapter - Adaptador para Instagram Reels

Este módulo proporciona una interfaz unificada para interactuar con la API de Instagram,
gestionando la publicación de Reels, análisis de métricas, gestión de comentarios,
y otras funcionalidades específicas de Instagram.
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
        logging.FileHandler("logs/instagram_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("InstagramAdapter")

class InstagramAdapter:
    """
    Adaptador para interactuar con la API de Instagram
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el adaptador de Instagram
        
        Args:
            config_path: Ruta al archivo de configuración con credenciales
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.access_token = None
        self.token_expiry = None
        self.api_base_url = "https://graph.facebook.com/v18.0"
        self.rate_limit_remaining = 200  # Valor inicial estimado
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)
        self.initialize_api()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo
        
        Returns:
            Configuración de Instagram
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('instagram', {})
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la API de Instagram
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            # Verificar si hay credenciales
            if not self.config:
                logger.error("No se encontró configuración para Instagram")
                return False
            
            # Verificar si el token actual es válido
            if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
                logger.info("Token de acceso actual válido hasta " + self.token_expiry.isoformat())
                return True
            
            # Obtener nuevo token de acceso
            app_id = self.config.get('app_id')
            app_secret = self.config.get('app_secret')
            long_lived_token = self.config.get('long_lived_token')
            
            if not app_id or not app_secret or not long_lived_token:
                logger.error("Faltan credenciales en la configuración")
                return False
            
            # Verificar validez del token actual
            response = self._make_api_request(
                "GET",
                f"debug_token",
                params={"input_token": long_lived_token, "access_token": f"{app_id}|{app_secret}"}
            )
            
            if response.get("status") != "success":
                logger.error(f"Error al verificar token: {response.get('message')}")
                return False
            
            token_data = response.get("data", {}).get("data", {})
            is_valid = token_data.get("is_valid", False)
            
            if not is_valid:
                logger.error("Token no válido, se requiere renovación manual")
                return False
            
            # Establecer token y fecha de expiración
            self.access_token = long_lived_token
            expires_at = token_data.get("expires_at", 0)
            
            if expires_at == 0:  # Token que no expira
                self.token_expiry = datetime.now() + timedelta(days=60)  # Verificar cada 60 días
            else:
                self.token_expiry = datetime.fromtimestamp(expires_at)
            
            logger.info(f"API de Instagram inicializada correctamente, token válido hasta {self.token_expiry.isoformat()}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar API de Instagram: {str(e)}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """
        Verifica si hay límite de tasa disponible
        
        Returns:
            True si hay límite disponible, False en caso contrario
        """
        # Verificar si es tiempo de reiniciar el límite
        if datetime.now() > self.rate_limit_reset:
            self.rate_limit_remaining = 200  # Valor estimado
            self.rate_limit_reset = datetime.now() + timedelta(hours=1)
            logger.info("Límite de tasa de Instagram API reiniciado")
        
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
        Realiza una solicitud a la API de Instagram
        
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
            
            # Asegurar que params incluya el token de acceso
            if params is None:
                params = {}
            
            if 'access_token' not in params:
                params['access_token'] = self.access_token
            
            # Preparar headers
            headers = {
                "Accept": "application/json"
            }
            
            if not files and method.upper() in ["POST", "PUT"]:
                headers["Content-Type"] = "application/json"
            
            # Realizar solicitud
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                if files:
                    # Si hay archivos, no usar Content-Type: application/json
                    response = requests.post(url, headers=headers, params=params, data=data, files=files)
                else:
                    response = requests.post(url, headers=headers, params=params, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                return {"status": "error", "message": f"Método no soportado: {method}"}
            
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
            logger.error(f"Error en solicitud a API de Instagram: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_pages(self) -> Dict[str, Any]:
        """
        Obtiene las páginas de Facebook/Instagram asociadas al usuario
        
        Returns:
            Lista de páginas disponibles
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                "me/accounts"
            )
            
            if response.get("status") != "success":
                return response
            
            pages_data = response.get("data", {}).get("data", [])
            
            # Procesar páginas
            pages = []
            for page in pages_data:
                pages.append({
                    "id": page.get("id", ""),
                    "name": page.get("name", ""),
                    "category": page.get("category", ""),
                    "access_token": page.get("access_token", "")
                })
            
            return {"status": "success", "pages": pages}
        except Exception as e:
            logger.error(f"Error al obtener páginas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_instagram_accounts(self, page_id: str = None) -> Dict[str, Any]:
        """
        Obtiene las cuentas de Instagram asociadas a una página de Facebook
        
        Args:
            page_id: ID de la página de Facebook (opcional, usa la primera si no se especifica)
            
        Returns:
            Lista de cuentas de Instagram
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Si no se proporciona page_id, obtener la primera página
            if not page_id:
                pages_response = self.get_user_pages()
                if pages_response.get("status") != "success" or not pages_response.get("pages"):
                    return {"status": "error", "message": "No se encontraron páginas de Facebook"}
                
                page_id = pages_response.get("pages")[0].get("id")
            
            response = self._make_api_request(
                "GET",
                f"{page_id}/instagram_accounts"
            )
            
            if response.get("status") != "success":
                return response
            
            accounts_data = response.get("data", {}).get("data", [])
            
            # Procesar cuentas
            accounts = []
            for account in accounts_data:
                accounts.append({
                    "id": account.get("id", ""),
                    "username": account.get("username", ""),
                    "name": account.get("name", "")
                })
            
            return {"status": "success", "accounts": accounts}
        except Exception as e:
            logger.error(f"Error al obtener cuentas de Instagram: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_instagram_account_id(self) -> str:
        """
        Obtiene el ID de la cuenta de Instagram configurada
        
        Returns:
            ID de la cuenta de Instagram o cadena vacía
        """
        # Verificar si ya está en la configuración
        if self.config.get("instagram_account_id"):
            return self.config.get("instagram_account_id")
        
        # Obtener de la API
        accounts_response = self.get_instagram_accounts()
        if accounts_response.get("status") != "success" or not accounts_response.get("accounts"):
            logger.error("No se pudo obtener ID de cuenta de Instagram")
            return ""
        
        return accounts_response.get("accounts")[0].get("id", "")
    
    def upload_reel(self, video_path: str, caption: str, thumbnail_path: str = None,
                   share_to_feed: bool = True, location_id: str = None,
                   user_tags: List[Dict] = None, hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Sube un Reel a Instagram
        
        Args:
            video_path: Ruta al archivo de video
            caption: Descripción del Reel
            thumbnail_path: Ruta a la miniatura (opcional)
            share_to_feed: Si se comparte en el feed
            location_id: ID de ubicación (opcional)
            user_tags: Lista de usuarios etiquetados (opcional)
            hashtags: Lista de hashtags (opcional)
            
        Returns:
            Información del Reel subido o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not os.path.exists(video_path):
                return {"status": "error", "message": f"Archivo no encontrado: {video_path}"}
            
            if thumbnail_path and not os.path.exists(thumbnail_path):
                return {"status": "error", "message": f"Miniatura no encontrada: {thumbnail_path}"}
            
            # Obtener ID de la cuenta de Instagram
            instagram_account_id = self.get_instagram_account_id()
            if not instagram_account_id:
                return {"status": "error", "message": "No se pudo obtener ID de cuenta de Instagram"}
            
            # Preparar hashtags
            if hashtags:
                hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags])
                caption = f"{caption}\n\n{hashtag_text}"
            
            # Paso 1: Iniciar la carga del contenedor
            container_response = self._make_api_request(
                "POST",
                f"{instagram_account_id}/media",
                data={
                    "media_type": "REELS",
                    "video_url": "https://www.example.com/placeholder.mp4",  # Placeholder, se reemplazará
                    "caption": caption,
                    "share_to_feed": share_to_feed
                }
            )
            
            if container_response.get("status") != "success":
                return container_response
            
            container_id = container_response.get("data", {}).get("id")
            if not container_id:
                return {"status": "error", "message": "No se pudo crear contenedor para el Reel"}
            
            # Paso 2: Subir el video
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()
            
            # Obtener URL de carga
            upload_url_response = self._make_api_request(
                "POST",
                f"{container_id}",
                data={
                    "upload_phase": "start"
                }
            )
            
            if upload_url_response.get("status") != "success":
                return upload_url_response
            
            upload_url = upload_url_response.get("data", {}).get("video_url")
            if not upload_url:
                return {"status": "error", "message": "No se pudo obtener URL de carga"}
            
            # Subir video a la URL proporcionada
            upload_response = requests.post(
                upload_url,
                data=video_data,
                headers={"Content-Type": "application/octet-stream"}
            )
            
            if upload_response.status_code >= 400:
                return {
                    "status": "error",
                    "message": f"Error al subir video: {upload_response.text}"
                }
            
            # Finalizar carga
            finish_response = self._make_api_request(
                "POST",
                f"{container_id}",
                data={
                    "upload_phase": "finish"
                }
            )
            
            if finish_response.get("status") != "success":
                return finish_response
            
            # Paso 3: Subir miniatura (opcional)
            if thumbnail_path:
                with open(thumbnail_path, "rb") as thumbnail_file:
                    thumbnail_data = thumbnail_file.read()
                
                thumbnail_response = self._make_api_request(
                    "POST",
                    f"{container_id}",
                    data={
                        "thumbnail_url": "https://www.example.com/placeholder.jpg"  # Placeholder
                    }
                )
                
                if thumbnail_response.get("status") != "success":
                    logger.warning(f"Error al subir miniatura: {thumbnail_response.get('message')}")
            
            # Paso 4: Publicar el Reel
            publish_response = self._make_api_request(
                "POST",
                f"{instagram_account_id}/media_publish",
                data={
                    "creation_id": container_id
                }
            )
            
            if publish_response.get("status") != "success":
                return publish_response
            
            media_id = publish_response.get("data", {}).get("id")
            
            logger.info(f"Reel subido correctamente: {media_id}")
            return {
                "status": "success",
                "media_id": media_id,
                "caption": caption,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al subir Reel: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_media_info(self, media_id: str) -> Dict[str, Any]:
        """
        Obtiene información de un medio (Reel, post, etc.)
        
        Args:
            media_id: ID del medio
            
        Returns:
            Información del medio
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"{media_id}",
                params={
                    "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,username,like_count,comments_count"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            media_data = response.get("data", {})
            
            # Procesar información
            media_info = {
                "media_id": media_data.get("id", ""),
                "caption": media_data.get("caption", ""),
                "media_type": media_data.get("media_type", ""),
                "media_url": media_data.get("media_url", ""),
                "permalink": media_data.get("permalink", ""),
                "thumbnail_url": media_data.get("thumbnail_url", ""),
                "timestamp": media_data.get("timestamp", ""),
                "username": media_data.get("username", ""),
                "like_count": media_data.get("like_count", 0),
                "comments_count": media_data.get("comments_count", 0)
            }
            
            return {"status": "success", "media_info": media_info}
        except Exception as e:
            logger.error(f"Error al obtener información del medio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_media_insights(self, media_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas de un medio (Reel, post, etc.)
        
        Args:
            media_id: ID del medio
            
        Returns:
            Métricas del medio
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"{media_id}/insights",
                params={
                    "metric": "engagement,impressions,reach,saved,video_views"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            insights_data = response.get("data", {}).get("data", [])
            
            # Procesar métricas
            insights = {}
            for metric in insights_data:
                name = metric.get("name", "")
                value = metric.get("values", [{}])[0].get("value", 0)
                insights[name] = value
            
            return {
                "status": "success",
                "media_id": media_id,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al obtener métricas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_account_insights(self, period: str = "day", days: int = 30) -> Dict[str, Any]:
        """
        Obtiene métricas de la cuenta
        
        Args:
            period: Período (day, week, month)
            days: Número de días para analizar
            
        Returns:
            Métricas de la cuenta
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Obtener ID de la cuenta de Instagram
            instagram_account_id = self.get_instagram_account_id()
            if not instagram_account_id:
                return {"status": "error", "message": "No se pudo obtener ID de cuenta de Instagram"}
            
            # Calcular fechas
            until = datetime.now()
            since = until - timedelta(days=days)
            
            response = self._make_api_request(
                "GET",
                f"{instagram_account_id}/insights",
                params={
                    "metric": "impressions,reach,profile_views,follower_count",
                    "period": period,
                    "since": int(since.timestamp()),
                    "until": int(until.timestamp())
                }
            )
            
            if response.get("status") != "success":
                return response
            
            insights_data = response.get("data", {}).get("data", [])
            
            # Procesar métricas
            insights = {}
            for metric in insights_data:
                name = metric.get("name", "")
                values = metric.get("values", [])
                insights[name] = values
            
            return {
                "status": "success",
                "account_id": instagram_account_id,
                "period": period,
                "days": days,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al obtener métricas de cuenta: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_comments(self, media_id: str, limit: int = 50) -> Dict[str, Any]:
        """
        Obtiene comentarios de un medio
        
        Args:
            media_id: ID del medio
            limit: Número máximo de comentarios
            
        Returns:
            Lista de comentarios
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"{media_id}/comments",
                params={
                    "fields": "id,text,username,timestamp,like_count",
                    "limit": limit
                }
            )
            
            if response.get("status") != "success":
                return response
            
            comments_data = response.get("data", {}).get("data", [])
            
            # Procesar comentarios
            comments = []
            for comment in comments_data:
                comments.append({
                    "comment_id": comment.get("id", ""),
                    "text": comment.get("text", ""),
                    "username": comment.get("username", ""),
                    "timestamp": comment.get("timestamp", ""),
                    "like_count": comment.get("like_count", 0)
                })
            
            return {"status": "success", "comments": comments}
        except Exception as e:
            logger.error(f"Error al obtener comentarios: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def reply_to_comment(self, comment_id: str, message: str) -> Dict[str, Any]:
        """
        Responde a un comentario
        
        Args:
            comment_id: ID del comentario
            message: Texto de la respuesta
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "POST",
                f"{comment_id}/replies",
                data={
                    "message": message
                }
            )
            
            if response.get("status") != "success":
                return response
            
            reply_data = response.get("data", {})
            
            return {
                "status": "success",
                "comment_id": reply_data.get("id", ""),
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al responder comentario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def delete_media(self, media_id: str) -> Dict[str, Any]:
        """
        Elimina un medio (Reel, post, etc.)
        
        Args:
            media_id: ID del medio
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "DELETE",
                f"{media_id}"
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Medio {media_id} eliminado correctamente")
            return {"status": "success", "message": "Medio eliminado correctamente"}
        except Exception as e:
            logger.error(f"Error al eliminar medio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Obtiene información del usuario/cuenta
        
        Returns:
            Información del usuario
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Obtener ID de la cuenta de Instagram
            instagram_account_id = self.get_instagram_account_id()
            if not instagram_account_id:
                return {"status": "error", "message": "No se pudo obtener ID de cuenta de Instagram"}
            
            response = self._make_api_request(
                "GET",
                f"{instagram_account_id}",
                params={
                    "fields": "id,username,name,biography,profile_picture_url,website,followers_count,follows_count,media_count"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            user_data = response.get("data", {})
            
            # Procesar información
            user_info = {
                "user_id": user_data.get("id", ""),
                "username": user_data.get("username", ""),
                "name": user_data.get("name", ""),
                "biography": user_data.get("biography", ""),
                "profile_picture_url": user_data.get("profile_picture_url", ""),
                "website": user_data.get("website", ""),
                "followers_count": user_data.get("followers_count", 0),
                "follows_count": user_data.get("follows_count", 0),
                "media_count": user_data.get("media_count", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "user_info": user_info}
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
            insights_response = self.get_account_insights(period="day", days=30)
            
            if insights_response.get("status") != "success" or not insights_response.get("insights", {}).get("impressions"):
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para determinar el momento óptimo",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "hour": 18  # 6 PM
                    }
                }
            
            # Obtener datos de impresiones por día
            impressions_data = insights_response.get("insights", {}).get("impressions", [])
            
            # Procesar datos para encontrar el mejor momento
            day_hour_impressions = {}
            
            # Obtener últimos 10 posts para analizar su rendimiento por hora y día
            media_response = self._make_api_request(
                "GET",
                f"{self.get_instagram_account_id()}/media",
                params={
                    "fields": "id,timestamp,insights.metric(impressions)",
                    "limit": 10
                }
            )
            
            if media_response.get("status") != "success":
                return {
                    "status": "warning",
                    "message": "No se pudieron obtener datos de posts recientes",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "hour": 18  # 6 PM
                    }
                }
            
            media_data = media_response.get("data", {}).get("data", [])
            
            for media in media_data:
                                timestamp_str = media.get("timestamp", "")
                if not timestamp_str:
                    continue
                
                # Convertir timestamp a objeto datetime
                try:
                    post_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    # Intentar otro formato si el anterior falla
                    post_time = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")
                
                # Extraer día de la semana (0-6, lunes-domingo) y hora
                day_of_week = post_time.weekday()
                hour = post_time.hour
                
                # Obtener impresiones del post
                insights = media.get("insights", {}).get("data", [])
                impressions = 0
                for insight in insights:
                    if insight.get("name") == "impressions":
                        impressions = insight.get("values", [{}])[0].get("value", 0)
                        break
                
                # Acumular impresiones por día y hora
                key = (day_of_week, hour)
                if key not in day_hour_impressions:
                    day_hour_impressions[key] = {"impressions": 0, "count": 0}
                
                day_hour_impressions[key]["impressions"] += impressions
                day_hour_impressions[key]["count"] += 1
            
            # Si no hay suficientes datos, usar valores predeterminados
            if not day_hour_impressions:
                return {
                    "status": "warning",
                    "message": "No hay suficientes datos para determinar el momento óptimo",
                    "recommendation": {
                        "day": 5,  # Viernes
                        "day_name": "Viernes",
                        "hour": 18,  # 6 PM
                        "hour_formatted": "18:00"
                    }
                }
            
            # Calcular promedios y encontrar el mejor momento
            best_time = None
            best_score = 0
            
            for (day, hour), data in day_hour_impressions.items():
                # Calcular promedio de impresiones
                avg_impressions = data["impressions"] / data["count"] if data["count"] > 0 else 0
                
                # Actualizar mejor momento si las impresiones son mayores
                if avg_impressions > best_score:
                    best_score = avg_impressions
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
                        "hour": 18,  # 6 PM
                        "hour_formatted": "18:00"
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
                    "hour": 18,  # 6 PM
                    "hour_formatted": "18:00"
                }
            }
    
    def check_shadowban(self, media_id: str, threshold_days: int = 7) -> Dict[str, Any]:
        """
        Verifica si un Reel podría estar bajo shadowban
        
        Args:
            media_id: ID del Reel
            threshold_days: Días para analizar tendencia
            
        Returns:
            Resultado del análisis de shadowban
        """
        try:
            # Obtener métricas del Reel
            metrics_response = self.get_media_insights(media_id)
            if metrics_response["status"] != "success":
                return {"status": "error", "message": "No se pudieron obtener métricas del Reel"}
            
            # Obtener información básica del Reel
            info_response = self.get_media_info(media_id)
            if info_response["status"] != "success":
                return {"status": "error", "message": "No se pudo obtener información del Reel"}
            
            metrics = metrics_response["insights"]
            media_info = info_response["media_info"]
            
            # Obtener fecha de publicación
            timestamp_str = media_info.get("timestamp", "")
            if timestamp_str:
                try:
                    published_date = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except ValueError:
                    # Intentar otro formato si el anterior falla
                    published_date = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")
            else:
                # Si no hay fecha de publicación, usar fecha actual menos 30 días
                published_date = datetime.now() - timedelta(days=30)
            
            video_age_days = (datetime.now() - published_date).days
            
            # Si el video es muy reciente, no podemos determinar shadowban
            if video_age_days < threshold_days:
                return {
                    "status": "warning",
                    "message": f"El Reel tiene menos de {threshold_days} días, análisis no concluyente",
                    "is_shadowbanned": False,
                    "confidence": 0.0
                }
            
            # Analizar métricas para detectar shadowban
            impressions = metrics.get("impressions", 0)
            reach = metrics.get("reach", 0)
            engagement = metrics.get("engagement", 0)
            saved = metrics.get("saved", 0)
            video_views = metrics.get("video_views", 0)
            
            # Obtener información del usuario para comparar con promedios
            user_info_response = self.get_user_info()
            if user_info_response["status"] != "success":
                follower_count = 1000  # Valor predeterminado si no se puede obtener
            else:
                follower_count = user_info_response["user_info"].get("followers_count", 1000)
            
            # Calcular métricas esperadas basadas en seguidores
            expected_reach = follower_count * 0.1  # Esperamos al menos 10% de seguidores
            expected_views = follower_count * 0.05  # Esperamos al menos 5% de seguidores
            
            # Calcular ratio de engagement
            engagement_ratio = engagement / reach if reach > 0 else 0
            
            # Determinar si hay shadowban basado en múltiples factores
            is_shadowbanned = False
            confidence = 0.0
            
            # Factor 1: Alcance muy bajo comparado con seguidores
            if reach < expected_reach * 0.3:  # Menos del 30% del alcance esperado
                is_shadowbanned = True
                confidence += 0.4
            
            # Factor 2: Vistas muy bajas comparadas con seguidores
            if video_views < expected_views * 0.3:  # Menos del 30% de las vistas esperadas
                is_shadowbanned = True
                confidence += 0.3
            
            # Factor 3: Engagement alto pero alcance bajo
            if engagement_ratio > 0.2 and reach < expected_reach * 0.5:
                is_shadowbanned = True
                confidence += 0.3
            
            # Factor 4: Impresiones mucho menores que el alcance
            if impressions < reach * 1.2:  # Normalmente las impresiones son mayores que el alcance
                is_shadowbanned = True
                confidence += 0.2
            
            # Limitar confianza a 1.0
            confidence = min(confidence, 1.0)
            
            return {
                "status": "success",
                "is_shadowbanned": is_shadowbanned,
                "confidence": confidence,
                "metrics": {
                    "impressions": impressions,
                    "reach": reach,
                    "engagement": engagement,
                    "engagement_ratio": engagement_ratio,
                    "video_views": video_views,
                    "expected_reach": expected_reach,
                    "expected_views": expected_views
                },
                "message": f"Posible shadowban: {'Sí' if is_shadowbanned else 'No'} (Confianza: {confidence:.2f})"
            }
        except Exception as e:
            logger.error(f"Error al verificar shadowban: {str(e)}")
            return {"status": "error", "message": str(e)}