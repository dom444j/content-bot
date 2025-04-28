"""
Threads Adapter - Adaptador para Meta Threads

Este módulo proporciona una interfaz unificada para interactuar con la API de Threads,
gestionando la publicación de contenido, análisis de métricas, gestión de comentarios,
y otras funcionalidades específicas de Threads.

Nota: Como Threads comparte infraestructura con Instagram, este adaptador extiende
algunas funcionalidades del Instagram Adapter y utiliza la Graph API de Meta.
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
        logging.FileHandler("logs/threads_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ThreadsAdapter")

class ThreadsAdapter:
    """
    Adaptador para interactuar con la API de Threads
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el adaptador de Threads
        
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
            Configuración de Threads
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Threads usa las mismas credenciales que Instagram
                return config.get('threads', config.get('instagram', {}))
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la API de Threads
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            # Verificar si hay credenciales
            if not self.config:
                logger.error("No se encontró configuración para Threads")
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
            
            logger.info(f"API de Threads inicializada correctamente, token válido hasta {self.token_expiry.isoformat()}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar API de Threads: {str(e)}")
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
            logger.info("Límite de tasa de Threads API reiniciado")
        
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
        Realiza una solicitud a la API de Threads
        
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
            logger.error(f"Error en solicitud a API de Threads: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_threads_account_id(self) -> str:
        """
        Obtiene el ID de la cuenta de Threads configurada
        
        Returns:
            ID de la cuenta de Threads o cadena vacía
        """
        # Verificar si ya está en la configuración
        if self.config.get("threads_account_id"):
            return self.config.get("threads_account_id")
        
        # Obtener de la API (mismo que Instagram)
        instagram_account_id = self.config.get("instagram_account_id", "")
        if not instagram_account_id:
            # Intentar obtener de la API
            try:
                # Obtener páginas de Facebook
                response = self._make_api_request(
                    "GET",
                    "me/accounts"
                )
                
                if response.get("status") != "success":
                    logger.error("No se pudieron obtener páginas de Facebook")
                    return ""
                
                pages_data = response.get("data", {}).get("data", [])
                if not pages_data:
                    logger.error("No se encontraron páginas de Facebook")
                    return ""
                
                # Obtener cuentas de Instagram asociadas a la primera página
                page_id = pages_data[0].get("id", "")
                if not page_id:
                    logger.error("No se pudo obtener ID de página de Facebook")
                    return ""
                
                instagram_response = self._make_api_request(
                    "GET",
                    f"{page_id}/instagram_accounts"
                )
                
                if instagram_response.get("status") != "success":
                    logger.error("No se pudieron obtener cuentas de Instagram")
                    return ""
                
                accounts_data = instagram_response.get("data", {}).get("data", [])
                if not accounts_data:
                    logger.error("No se encontraron cuentas de Instagram")
                    return ""
                
                instagram_account_id = accounts_data[0].get("id", "")
                
                # Guardar en configuración para futuras llamadas
                self.config["instagram_account_id"] = instagram_account_id
                self.config["threads_account_id"] = instagram_account_id  # Mismo ID
                
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        full_config = json.load(f)
                    
                    if 'instagram' in full_config:
                        full_config['instagram']['instagram_account_id'] = instagram_account_id
                    
                    if 'threads' in full_config:
                        full_config['threads']['threads_account_id'] = instagram_account_id
                    else:
                        full_config['threads'] = {'threads_account_id': instagram_account_id}
                    
                    with open(self.config_path, 'w', encoding='utf-8') as f:
                        json.dump(full_config, f, indent=2)
                except Exception as e:
                    logger.warning(f"No se pudo guardar ID en configuración: {str(e)}")
                
                return instagram_account_id
            except Exception as e:
                logger.error(f"Error al obtener ID de cuenta: {str(e)}")
                return ""
        
        return instagram_account_id
    
    def publish_thread(self, text: str, media_ids: List[str] = None) -> Dict[str, Any]:
        """
        Publica un thread en Threads
        
        Args:
            text: Texto del thread
            media_ids: Lista de IDs de medios a adjuntar (opcional)
            
        Returns:
            Información del thread publicado o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Obtener ID de la cuenta de Threads
            account_id = self.get_threads_account_id()
            if not account_id:
                return {"status": "error", "message": "No se pudo obtener ID de cuenta de Threads"}
            
            # Preparar datos
            data = {
                "message": text
            }
            
            # Añadir medios si existen
            if media_ids and len(media_ids) > 0:
                data["media_ids"] = ",".join(media_ids)
            
            # Publicar thread
            response = self._make_api_request(
                "POST",
                f"{account_id}/threads",
                data=data
            )
            
            if response.get("status") != "success":
                return response
            
            thread_id = response.get("data", {}).get("id", "")
            
            logger.info(f"Thread publicado correctamente: {thread_id}")
            return {
                "status": "success",
                "thread_id": thread_id,
                "text": text,
                "media_ids": media_ids,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al publicar thread: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def upload_media(self, media_path: str, caption: str = "") -> Dict[str, Any]:
        """
        Sube un medio (imagen o video) para usar en threads
        
        Args:
            media_path: Ruta al archivo de medio
            caption: Descripción del medio (opcional)
            
        Returns:
            ID del medio subido o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not os.path.exists(media_path):
                return {"status": "error", "message": f"Archivo no encontrado: {media_path}"}
            
            # Obtener ID de la cuenta de Threads
            account_id = self.get_threads_account_id()
            if not account_id:
                return {"status": "error", "message": "No se pudo obtener ID de cuenta de Threads"}
            
            # Determinar tipo de medio
            media_type = "IMAGE"
            if media_path.lower().endswith((".mp4", ".mov", ".avi")):
                media_type = "VIDEO"
            
            # Paso 1: Iniciar la carga del contenedor
            container_response = self._make_api_request(
                "POST",
                f"{account_id}/media",
                data={
                    "media_type": media_type,
                    "caption": caption
                }
            )
            
            if container_response.get("status") != "success":
                return container_response
            
            container_id = container_response.get("data", {}).get("id")
            if not container_id:
                return {"status": "error", "message": "No se pudo crear contenedor para el medio"}
            
            # Paso 2: Subir el medio
            with open(media_path, "rb") as media_file:
                media_data = media_file.read()
            
            if media_type == "IMAGE":
                # Subir imagen directamente
                upload_response = self._make_api_request(
                    "POST",
                    f"{container_id}",
                    files={"source": (os.path.basename(media_path), media_data)}
                )
            else:
                # Para videos, obtener URL de carga
                upload_url_response = self._make_api_request(
                    "POST",
                    f"{container_id}",
                    data={
                        "upload_phase": "start"
                    }
                )
                
                if upload_url_response.get("status") != "success":
                    return upload_url_response
                
                upload_url = upload_url_response.get("data", {}).get("upload_url")
                if not upload_url:
                    return {"status": "error", "message": "No se pudo obtener URL de carga"}
                
                # Subir video a la URL proporcionada
                upload_response = requests.post(
                    upload_url,
                    data=media_data,
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
            
            logger.info(f"Medio subido correctamente: {container_id}")
            return {
                "status": "success",
                "media_id": container_id,
                "media_type": media_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al subir medio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_thread_info(self, thread_id: str) -> Dict[str, Any]:
        """
        Obtiene información de un thread
        
        Args:
            thread_id: ID del thread
            
        Returns:
            Información del thread
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"{thread_id}",
                params={
                    "fields": "id,message,created_time,permalink_url,likes.summary(true),comments.summary(true)"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            thread_data = response.get("data", {})
            
            # Procesar información
            thread_info = {
                "thread_id": thread_data.get("id", ""),
                "message": thread_data.get("message", ""),
                "created_time": thread_data.get("created_time", ""),
                "permalink_url": thread_data.get("permalink_url", ""),
                "likes_count": thread_data.get("likes", {}).get("summary", {}).get("total_count", 0),
                "comments_count": thread_data.get("comments", {}).get("summary", {}).get("total_count", 0)
            }
            
            return {"status": "success", "thread_info": thread_info}
        except Exception as e:
            logger.error(f"Error al obtener información del thread: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_thread_comments(self, thread_id: str, limit: int = 50) -> Dict[str, Any]:
        """
        Obtiene comentarios de un thread
        
        Args:
            thread_id: ID del thread
            limit: Número máximo de comentarios
            
        Returns:
            Lista de comentarios
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"{thread_id}/comments",
                params={
                    "fields": "id,message,from,created_time,like_count",
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
                    "message": comment.get("message", ""),
                    "from": {
                        "id": comment.get("from", {}).get("id", ""),
                        "name": comment.get("from", {}).get("name", "")
                    },
                    "created_time": comment.get("created_time", ""),
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
    
    def delete_thread(self, thread_id: str) -> Dict[str, Any]:
        """
        Elimina un thread
        
        Args:
            thread_id: ID del thread
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "DELETE",
                f"{thread_id}"
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Thread {thread_id} eliminado correctamente")
            return {"status": "success", "message": "Thread eliminado correctamente"}
        except Exception as e:
            logger.error(f"Error al eliminar thread: {str(e)}")
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
            # Obtener ID de la cuenta de Threads
            account_id = self.get_threads_account_id()
            if not account_id:
                return {"status": "error", "message": "No se pudo obtener ID de cuenta de Threads"}
            
            response = self._make_api_request(
                "GET",
                f"{account_id}",
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