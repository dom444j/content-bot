"""
X Adapter (Twitter) - Adaptador para Twitter/X

Este módulo proporciona una interfaz unificada para interactuar con la API de Twitter/X,
gestionando la publicación de contenido, análisis de métricas, gestión de comentarios,
y otras funcionalidades específicas de Twitter/X.
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
import uuid

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/x_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("XAdapter")

class XAdapter:
    """
    Adaptador para interactuar con la API de Twitter/X
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el adaptador de Twitter/X
        
        Args:
            config_path: Ruta al archivo de configuración con credenciales
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.api_base_url = "https://api.twitter.com/2"
        self.upload_url = "https://upload.twitter.com/1.1/media/upload.json"
        self.oauth_url = "https://api.twitter.com/oauth2/token"
        self.bearer_token = None
        self.access_token = None
        self.access_token_secret = None
        self.consumer_key = None
        self.consumer_secret = None
        self.token_expiry = None
        self.user_id = None
        self.username = None
        self.rate_limit_remaining = 300  # Valor inicial estimado
        self.rate_limit_reset = datetime.now() + timedelta(minutes=15)
        self.initialize_api()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo
        
        Returns:
            Configuración de Twitter/X
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('twitter', {})
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la API de Twitter/X
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            # Verificar si hay credenciales
            if not self.config:
                logger.error("No se encontró configuración para Twitter/X")
                return False
            
            # Obtener credenciales
            self.consumer_key = self.config.get('consumer_key')
            self.consumer_secret = self.config.get('consumer_secret')
            self.access_token = self.config.get('access_token')
            self.access_token_secret = self.config.get('access_token_secret')
            self.bearer_token = self.config.get('bearer_token')
            
            if not self.consumer_key or not self.consumer_secret:
                logger.error("Faltan credenciales de API en la configuración")
                return False
            
            # Si no hay bearer token o está expirado, obtener uno nuevo
            if not self.bearer_token or (self.token_expiry and datetime.now() >= self.token_expiry):
                if not self._get_bearer_token():
                    return False
            
            # Verificar token de acceso para publicación
            if not self.access_token or not self.access_token_secret:
                logger.warning("Faltan tokens de acceso para publicación. Algunas funciones estarán limitadas.")
            
            # Obtener información del usuario
            if self.access_token and self.access_token_secret:
                user_info = self._get_user_info()
                if user_info.get("status") == "success":
                    self.user_id = user_info.get("data", {}).get("id")
                    self.username = user_info.get("data", {}).get("username")
                    logger.info(f"API de Twitter/X inicializada para usuario: @{self.username}")
                else:
                    logger.warning("No se pudo obtener información del usuario")
            
            logger.info("API de Twitter/X inicializada correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar API de Twitter/X: {str(e)}")
            return False
    
    def _get_bearer_token(self) -> bool:
        """
        Obtiene un token de portador (bearer token) para autenticación
        
        Returns:
            True si se obtuvo correctamente, False en caso contrario
        """
        try:
            # Si ya hay un bearer token en la configuración, usarlo
            if self.config.get('bearer_token'):
                self.bearer_token = self.config.get('bearer_token')
                self.token_expiry = datetime.now() + timedelta(days=30)  # Los bearer tokens duran mucho tiempo
                return True
            
            # Si no, obtener uno nuevo con las credenciales
            credentials = f"{urllib.parse.quote(self.consumer_key)}:{urllib.parse.quote(self.consumer_secret)}"
            encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
            
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
            }
            
            data = "grant_type=client_credentials"
            
            response = requests.post(self.oauth_url, headers=headers, data=data)
            
            if response.status_code != 200:
                logger.error(f"Error al obtener bearer token: {response.text}")
                return False
            
            token_data = response.json()
            self.bearer_token = token_data.get("access_token")
            
            # Establecer expiración (normalmente no expiran, pero por seguridad)
            self.token_expiry = datetime.now() + timedelta(days=30)
            
            # Guardar en configuración
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    full_config = json.load(f)
                
                if 'twitter' in full_config:
                    full_config['twitter']['bearer_token'] = self.bearer_token
                else:
                    full_config['twitter'] = {'bearer_token': self.bearer_token}
                
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(full_config, f, indent=2)
            except Exception as e:
                logger.warning(f"No se pudo guardar bearer token en configuración: {str(e)}")
            
            logger.info("Bearer token obtenido correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al obtener bearer token: {str(e)}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """
        Verifica si hay límite de tasa disponible
        
        Returns:
            True si hay límite disponible, False en caso contrario
        """
        # Verificar si es tiempo de reiniciar el límite
        if datetime.now() > self.rate_limit_reset:
            self.rate_limit_remaining = 300  # Valor estimado
            self.rate_limit_reset = datetime.now() + timedelta(minutes=15)
            logger.info("Límite de tasa de Twitter/X API reiniciado")
        
        # Verificar si hay límite disponible
        if self.rate_limit_remaining <= 0:
            logger.warning(f"Límite de tasa alcanzado, se reiniciará en {(self.rate_limit_reset - datetime.now()).total_seconds()} segundos")
            return False
        
        # Decrementar límite
        self.rate_limit_remaining -= 1
        return True
    
    def _create_oauth1_header(self, method: str, url: str, params: Dict = None) -> Dict[str, str]:
        """
        Crea un encabezado de autenticación OAuth 1.0a para solicitudes que requieren autenticación de usuario
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            url: URL completa de la solicitud
            params: Parámetros adicionales para la firma
            
        Returns:
            Encabezado de autenticación
        """
        if not self.access_token or not self.access_token_secret:
            logger.error("No hay tokens de acceso para crear encabezado OAuth")
            return {}
        
        # Parámetros base de OAuth
        oauth_params = {
            'oauth_consumer_key': self.consumer_key,
            'oauth_nonce': uuid.uuid4().hex,
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_token': self.access_token,
            'oauth_version': '1.0'
        }
        
        # Combinar con parámetros adicionales
        if params:
            signature_params = {**oauth_params, **params}
        else:
            signature_params = oauth_params
        
        # Crear cadena base para firma
        param_string = '&'.join([f"{urllib.parse.quote(k)}={urllib.parse.quote(str(v))}" for k, v in sorted(signature_params.items())])
        base_string = f"{method.upper()}&{urllib.parse.quote(url)}&{urllib.parse.quote(param_string)}"
        
        # Crear clave de firma
        signing_key = f"{urllib.parse.quote(self.consumer_secret)}&{urllib.parse.quote(self.access_token_secret)}"
        
        # Calcular firma
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode('utf-8'),
                base_string.encode('utf-8'),
                hashlib.sha1
            ).digest()
        ).decode('utf-8')
        
        # Añadir firma a parámetros OAuth
        oauth_params['oauth_signature'] = signature
        
        # Crear encabezado de autorización
        auth_header = 'OAuth ' + ', '.join([f'{urllib.parse.quote(k)}="{urllib.parse.quote(str(v))}"' for k, v in oauth_params.items()])
        
        return {"Authorization": auth_header}
    
    def _make_api_request(self, method: str, endpoint: str, params: Dict = None, 
                         data: Dict = None, files: Dict = None, oauth1: bool = False) -> Dict[str, Any]:
        """
        Realiza una solicitud a la API de Twitter/X
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint de la API
            params: Parámetros de consulta
            data: Datos para enviar en el cuerpo
            files: Archivos para subir
            oauth1: Si es True, usa OAuth 1.0a en lugar de Bearer Token
            
        Returns:
            Respuesta de la API o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar autenticación
            if oauth1 and (not self.access_token or not self.access_token_secret):
                return {"status": "error", "message": "No hay tokens de acceso para autenticación OAuth 1.0a"}
            
            if not oauth1 and not self.bearer_token:
                if not self._get_bearer_token():
                    return {"status": "error", "message": "No se pudo obtener bearer token"}
            
            # Preparar URL
            if endpoint.startswith("http"):
                url = endpoint  # URL completa
            else:
                url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
            
            # Preparar headers
            headers = {}
            
            if oauth1:
                # Usar OAuth 1.0a para solicitudes que requieren autenticación de usuario
                oauth_headers = self._create_oauth1_header(method, url, params)
                headers.update(oauth_headers)
            else:
                # Usar Bearer Token para solicitudes que no requieren autenticación de usuario
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            
            if not files and method.upper() in ["POST", "PUT", "PATCH"]:
                headers["Content-Type"] = "application/json"
            
            # Realizar solicitud
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                if files:
                    response = requests.post(url, headers=headers, params=params, data=data, files=files)
                else:
                    response = requests.post(url, headers=headers, params=params, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, params=params, json=data)
            else:
                return {"status": "error", "message": f"Método no soportado: {method}"}
            
            # Actualizar límites de tasa si están en la respuesta
            if 'x-rate-limit-remaining' in response.headers:
                self.rate_limit_remaining = int(response.headers['x-rate-limit-remaining'])
            
            if 'x-rate-limit-reset' in response.headers:
                self.rate_limit_reset = datetime.fromtimestamp(int(response.headers['x-rate-limit-reset']))
            
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
            logger.error(f"Error en solicitud a API de Twitter/X: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_user_info(self) -> Dict[str, Any]:
        """
        Obtiene información del usuario autenticado
        
        Returns:
            Información del usuario
        """
        try:
            # Usar endpoint de usuario autenticado
            response = self._make_api_request(
                "GET",
                "users/me",
                params={"user.fields": "id,name,username,description,profile_image_url,public_metrics"}
            )
            
            return response
        except Exception as e:
            logger.error(f"Error al obtener información del usuario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def create_tweet(self, text: str, media_ids: List[str] = None, reply_to: str = None) -> Dict[str, Any]:
        """
        Crea un tweet
        
        Args:
            text: Texto del tweet (máximo 280 caracteres)
            media_ids: Lista de IDs de medios a adjuntar (opcional)
            reply_to: ID del tweet al que se responde (opcional)
            
        Returns:
            Información del tweet creado o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar longitud del texto
            if len(text) > 280:
                logger.warning(f"Texto demasiado largo ({len(text)} caracteres), se truncará a 280 caracteres")
                text = text[:277] + "..."
            
            # Preparar datos del tweet
            tweet_data = {
                "text": text
            }
            
            # Añadir medios si existen
            if media_ids and len(media_ids) > 0:
                tweet_data["media"] = {"media_ids": media_ids}
            
            # Añadir respuesta si existe
            if reply_to:
                tweet_data["reply"] = {"in_reply_to_tweet_id": reply_to}
            
            # Crear tweet
            response = self._make_api_request(
                "POST",
                "tweets",
                data=tweet_data,
                oauth1=True  # Usar OAuth 1.0a para publicación
            )
            
            if response.get("status") != "success":
                return response
            
            tweet_id = response.get("data", {}).get("data", {}).get("id", "")
            
            logger.info(f"Tweet creado correctamente: {tweet_id}")
            return {
                "status": "success",
                "tweet_id": tweet_id,
                "text": text,
                "media_ids": media_ids,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al crear tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def upload_media(self, media_path: str) -> Dict[str, Any]:
        """
        Sube un medio (imagen o video) para usar en tweets
        
        Args:
            media_path: Ruta al archivo de medio
            
        Returns:
            ID del medio subido o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            if not os.path.exists(media_path):
                return {"status": "error", "message": f"Archivo no encontrado: {media_path}"}
            
            # Determinar tipo de medio
            media_type = ""
            if media_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                media_type = "image"
            elif media_path.lower().endswith((".mp4", ".mov")):
                media_type = "video"
            else:
                return {"status": "error", "message": f"Tipo de archivo no soportado: {media_path}"}
            
            # Para imágenes, subir directamente
            if media_type == "image":
                with open(media_path, "rb") as media_file:
                    files = {"media": media_file}
                    
                    # Usar endpoint de carga de medios
                    response = self._make_api_request(
                        "POST",
                        self.upload_url,
                        files=files,
                        oauth1=True  # Usar OAuth 1.0a para carga de medios
                    )
                    
                    if response.get("status") != "success":
                        return response
                    
                    media_id = response.get("data", {}).get("media_id_string", "")
                    
                    logger.info(f"Medio subido correctamente: {media_id}")
                    return {
                        "status": "success",
                        "media_id": media_id,
                        "media_type": media_type,
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Para videos, usar carga en trozos
            elif media_type == "video":
                # Paso 1: Iniciar carga
                file_size = os.path.getsize(media_path)
                
                init_data = {
                    "command": "INIT",
                    "total_bytes": str(file_size),
                    "media_type": "video/mp4",
                    "media_category": "tweet_video"
                }
                
                init_response = self._make_api_request(
                    "POST",
                    self.upload_url,
                    data=init_data,
                    oauth1=True
                )
                
                if init_response.get("status") != "success":
                    return init_response
                
                media_id = init_response.get("data", {}).get("media_id_string", "")
                
                # Paso 2: Subir trozos
                chunk_size = 1024 * 1024  # 1MB
                
                with open(media_path, "rb") as media_file:
                    chunk_index = 0
                    
                    while True:
                        chunk = media_file.read(chunk_size)
                        if not chunk:
                            break
                        
                        append_data = {
                            "command": "APPEND",
                            "media_id": media_id,
                            "segment_index": str(chunk_index)
                        }
                        
                        files = {
                            "media": chunk
                        }
                        
                        append_response = self._make_api_request(
                            "POST",
                            self.upload_url,
                            data=append_data,
                            files=files,
                            oauth1=True
                        )
                        
                        if append_response.get("status") != "success":
                            return append_response
                        
                        chunk_index += 1
                
                # Paso 3: Finalizar carga
                finalize_data = {
                    "command": "FINALIZE",
                    "media_id": media_id
                }
                
                finalize_response = self._make_api_request(
                    "POST",
                    self.upload_url,
                    data=finalize_data,
                    oauth1=True
                )
                
                if finalize_response.get("status") != "success":
                    return finalize_response
                
                # Paso 4: Verificar estado de procesamiento
                status_data = {
                    "command": "STATUS",
                    "media_id": media_id
                }
                
                max_checks = 10
                checks = 0
                processing_info = finalize_response.get("data", {}).get("processing_info", {})
                
                while processing_info.get("state") == "pending" and checks < max_checks:
                    # Esperar el tiempo recomendado
                    wait_time = processing_info.get("check_after_secs", 1)
                    time.sleep(wait_time)
                    
                    # Verificar estado
                    status_response = self._make_api_request(
                        "GET",
                        self.upload_url,
                        params=status_data,
                        oauth1=True
                    )
                    
                    if status_response.get("status") != "success":
                        return status_response
                    
                    processing_info = status_response.get("data", {}).get("processing_info", {})
                    checks += 1
                
                if processing_info.get("state") == "failed":
                    error_message = processing_info.get("error", {}).get("message", "Error desconocido")
                    return {"status": "error", "message": f"Error al procesar video: {error_message}"}
                
                logger.info(f"Video subido correctamente: {media_id}")
                return {
                    "status": "success",
                    "media_id": media_id,
                    "media_type": media_type,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error al subir medio: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Obtiene información de un tweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Información del tweet
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"tweets/{tweet_id}",
                params={
                    "tweet.fields": "id,text,created_at,public_metrics,entities,attachments,author_id",
                    "expansions": "author_id,attachments.media_keys",
                    "media.fields": "type,url,preview_image_url,public_metrics",
                    "user.fields": "id,name,username,profile_image_url"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            tweet_data = response.get("data", {}).get("data", {})
            includes = response.get("data", {}).get("includes", {})
            
            # Procesar información
            tweet_info = {
                "tweet_id": tweet_data.get("id", ""),
                "text": tweet_data.get("text", ""),
                "created_at": tweet_data.get("created_at", ""),
                "metrics": tweet_data.get("public_metrics", {}),
                "author_id": tweet_data.get("author_id", ""),
                "media_keys": tweet_data.get("attachments", {}).get("media_keys", []),
                "entities": tweet_data.get("entities", {})
            }
            
            # Añadir información de autor
            users = includes.get("users", [])
            for user in users:
                if user.get("id") == tweet_info["author_id"]:
                    tweet_info["author"] = {
                        "id": user.get("id", ""),
                        "name": user.get("name", ""),
                        "username": user.get("username", ""),
                        "profile_image_url": user.get("profile_image_url", "")
                    }
                    break
            
            # Añadir información de medios
            media = includes.get("media", [])
            tweet_info["media"] = []
            
            for media_item in media:
                if media_item.get("media_key") in tweet_info["media_keys"]:
                    tweet_info["media"].append({
                        "type": media_item.get("type", ""),
                        "url": media_item.get("url", ""),
                        "preview_image_url": media_item.get("preview_image_url", ""),
                        "metrics": media_item.get("public_metrics", {})
                    })
            
            return {"status": "success", "tweet_info": tweet_info}
        except Exception as e:
            logger.error(f"Error al obtener tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def delete_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Elimina un tweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "DELETE",
                f"tweets/{tweet_id}",
                oauth1=True  # Usar OAuth 1.0a para eliminación
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Tweet {tweet_id} eliminado correctamente")
            return {"status": "success", "message": "Tweet eliminado correctamente"}
        except Exception as e:
            logger.error(f"Error al eliminar tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_tweets(self, user_id: str = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Obtiene tweets de un usuario
        
        Args:
            user_id: ID del usuario (si es None, usa el usuario autenticado)
            max_results: Número máximo de tweets a obtener
            
        Returns:
            Lista de tweets
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Si no se proporciona user_id, usar el propio
            if not user_id:
                if not self.user_id:
                    user_info = self._get_user_info()
                    if user_info.get("status") != "success":
                        return user_info
                    
                    self.user_id = user_info.get("data", {}).get("data", {}).get("id")
                
                user_id = self.user_id
            
            response = self._make_api_request(
                "GET",
                f"users/{user_id}/tweets",
                params={
                    "max_results": max_results,
                    "tweet.fields": "id,text,created_at,public_metrics,entities,attachments",
                    "expansions": "attachments.media_keys",
                    "media.fields": "type,url,preview_image_url"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            tweets_data = response.get("data", {}).get("data", [])
            includes = response.get("data", {}).get("includes", {})
            
            # Procesar tweets
            tweets = []
            for tweet in tweets_data:
                tweet_info = {
                    "tweet_id": tweet.get("id", ""),
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", ""),
                    "metrics": tweet.get("public_metrics", {}),
                    "entities": tweet.get("entities", {}),
                    "media_keys": tweet.get("attachments", {}).get("media_keys", [])
                }
                
                # Añadir información de medios
                media = includes.get("media", [])
                tweet_info["media"] = []
                
                for media_item in media:
                    if media_item.get("media_key") in tweet_info["media_keys"]:
                        tweet_info["media"].append({
                            "type": media_item.get("type", ""),
                            "url": media_item.get("url", ""),
                            "preview_image_url": media_item.get("preview_image_url", "")
                        })
                
                tweets.append(tweet_info)
            
            return {"status": "success", "tweets": tweets}
        except Exception as e:
                        logger.error(f"Error al obtener tweets del usuario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_tweet_metrics(self, tweet_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas detalladas de un tweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Métricas del tweet
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            response = self._make_api_request(
                "GET",
                f"tweets/{tweet_id}",
                params={
                    "tweet.fields": "public_metrics,non_public_metrics,organic_metrics,promoted_metrics"
                },
                oauth1=True  # Algunas métricas requieren autenticación OAuth 1.0a
            )
            
            if response.get("status") != "success":
                return response
            
            metrics_data = response.get("data", {}).get("data", {})
            
            # Extraer métricas
            metrics = {
                "public": metrics_data.get("public_metrics", {}),
                "non_public": metrics_data.get("non_public_metrics", {}),
                "organic": metrics_data.get("organic_metrics", {}),
                "promoted": metrics_data.get("promoted_metrics", {})
            }
            
            return {"status": "success", "metrics": metrics}
        except Exception as e:
            logger.error(f"Error al obtener métricas del tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_tweet_replies(self, tweet_id: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Obtiene respuestas a un tweet
        
        Args:
            tweet_id: ID del tweet
            max_results: Número máximo de respuestas a obtener
            
        Returns:
            Lista de respuestas
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Usar búsqueda para encontrar respuestas
            query = f"conversation_id:{tweet_id}"
            
            response = self._make_api_request(
                "GET",
                "tweets/search/recent",
                params={
                    "query": query,
                    "max_results": max_results,
                    "tweet.fields": "id,text,created_at,public_metrics,author_id,in_reply_to_user_id",
                    "expansions": "author_id",
                    "user.fields": "id,name,username,profile_image_url"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            replies_data = response.get("data", {}).get("data", [])
            includes = response.get("data", {}).get("includes", {})
            
            # Procesar respuestas
            replies = []
            for reply in replies_data:
                # Excluir el tweet original
                if reply.get("id") == tweet_id:
                    continue
                
                reply_info = {
                    "tweet_id": reply.get("id", ""),
                    "text": reply.get("text", ""),
                    "created_at": reply.get("created_at", ""),
                    "metrics": reply.get("public_metrics", {}),
                    "author_id": reply.get("author_id", ""),
                    "in_reply_to_user_id": reply.get("in_reply_to_user_id", "")
                }
                
                # Añadir información de autor
                users = includes.get("users", [])
                for user in users:
                    if user.get("id") == reply_info["author_id"]:
                        reply_info["author"] = {
                            "id": user.get("id", ""),
                            "name": user.get("name", ""),
                            "username": user.get("username", ""),
                            "profile_image_url": user.get("profile_image_url", "")
                        }
                        break
                
                replies.append(reply_info)
            
            return {"status": "success", "replies": replies}
        except Exception as e:
            logger.error(f"Error al obtener respuestas al tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def like_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Da like a un tweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar si hay usuario autenticado
            if not self.user_id:
                user_info = self._get_user_info()
                if user_info.get("status") != "success":
                    return {"status": "error", "message": "No se pudo obtener información del usuario"}
                
                self.user_id = user_info.get("data", {}).get("data", {}).get("id")
            
            # Crear like
            data = {
                "tweet_id": tweet_id
            }
            
            response = self._make_api_request(
                "POST",
                f"users/{self.user_id}/likes",
                data=data,
                oauth1=True  # Usar OAuth 1.0a para likes
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Like dado al tweet {tweet_id}")
            return {"status": "success", "message": "Like dado correctamente"}
        except Exception as e:
            logger.error(f"Error al dar like al tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def unlike_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Quita like a un tweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar si hay usuario autenticado
            if not self.user_id:
                user_info = self._get_user_info()
                if user_info.get("status") != "success":
                    return {"status": "error", "message": "No se pudo obtener información del usuario"}
                
                self.user_id = user_info.get("data", {}).get("data", {}).get("id")
            
            # Eliminar like
            response = self._make_api_request(
                "DELETE",
                f"users/{self.user_id}/likes/{tweet_id}",
                oauth1=True  # Usar OAuth 1.0a para unlikes
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Like quitado al tweet {tweet_id}")
            return {"status": "success", "message": "Like quitado correctamente"}
        except Exception as e:
            logger.error(f"Error al quitar like al tweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def retweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Retweetea un tweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar si hay usuario autenticado
            if not self.user_id:
                user_info = self._get_user_info()
                if user_info.get("status") != "success":
                    return {"status": "error", "message": "No se pudo obtener información del usuario"}
                
                self.user_id = user_info.get("data", {}).get("data", {}).get("id")
            
            # Crear retweet
            data = {
                "tweet_id": tweet_id
            }
            
            response = self._make_api_request(
                "POST",
                f"users/{self.user_id}/retweets",
                data=data,
                oauth1=True  # Usar OAuth 1.0a para retweets
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Tweet {tweet_id} retweeteado correctamente")
            return {"status": "success", "message": "Tweet retweeteado correctamente"}
        except Exception as e:
            logger.error(f"Error al retweetear: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def unretweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Quita un retweet
        
        Args:
            tweet_id: ID del tweet
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar si hay usuario autenticado
            if not self.user_id:
                user_info = self._get_user_info()
                if user_info.get("status") != "success":
                    return {"status": "error", "message": "No se pudo obtener información del usuario"}
                
                self.user_id = user_info.get("data", {}).get("data", {}).get("id")
            
            # Eliminar retweet
            response = self._make_api_request(
                "DELETE",
                f"users/{self.user_id}/retweets/{tweet_id}",
                oauth1=True  # Usar OAuth 1.0a para unretweets
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Retweet quitado al tweet {tweet_id}")
            return {"status": "success", "message": "Retweet quitado correctamente"}
        except Exception as e:
            logger.error(f"Error al quitar retweet: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_followers(self, user_id: str = None, max_results: int = 100) -> Dict[str, Any]:
        """
        Obtiene seguidores de un usuario
        
        Args:
            user_id: ID del usuario (si es None, usa el usuario autenticado)
            max_results: Número máximo de seguidores a obtener
            
        Returns:
            Lista de seguidores
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Si no se proporciona user_id, usar el propio
            if not user_id:
                if not self.user_id:
                    user_info = self._get_user_info()
                    if user_info.get("status") != "success":
                        return user_info
                    
                    self.user_id = user_info.get("data", {}).get("data", {}).get("id")
                
                user_id = self.user_id
            
            response = self._make_api_request(
                "GET",
                f"users/{user_id}/followers",
                params={
                    "max_results": max_results,
                    "user.fields": "id,name,username,description,profile_image_url,public_metrics"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            followers_data = response.get("data", {}).get("data", [])
            
            # Procesar seguidores
            followers = []
            for follower in followers_data:
                follower_info = {
                    "id": follower.get("id", ""),
                    "name": follower.get("name", ""),
                    "username": follower.get("username", ""),
                    "description": follower.get("description", ""),
                    "profile_image_url": follower.get("profile_image_url", ""),
                    "metrics": follower.get("public_metrics", {})
                }
                
                followers.append(follower_info)
            
            return {"status": "success", "followers": followers}
        except Exception as e:
            logger.error(f"Error al obtener seguidores: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_user_following(self, user_id: str = None, max_results: int = 100) -> Dict[str, Any]:
        """
        Obtiene usuarios seguidos por un usuario
        
        Args:
            user_id: ID del usuario (si es None, usa el usuario autenticado)
            max_results: Número máximo de usuarios a obtener
            
        Returns:
            Lista de usuarios seguidos
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Si no se proporciona user_id, usar el propio
            if not user_id:
                if not self.user_id:
                    user_info = self._get_user_info()
                    if user_info.get("status") != "success":
                        return user_info
                    
                    self.user_id = user_info.get("data", {}).get("data", {}).get("id")
                
                user_id = self.user_id
            
            response = self._make_api_request(
                "GET",
                f"users/{user_id}/following",
                params={
                    "max_results": max_results,
                    "user.fields": "id,name,username,description,profile_image_url,public_metrics"
                }
            )
            
            if response.get("status") != "success":
                return response
            
            following_data = response.get("data", {}).get("data", [])
            
            # Procesar usuarios seguidos
            following = []
            for user in following_data:
                user_info = {
                    "id": user.get("id", ""),
                    "name": user.get("name", ""),
                    "username": user.get("username", ""),
                    "description": user.get("description", ""),
                    "profile_image_url": user.get("profile_image_url", ""),
                    "metrics": user.get("public_metrics", {})
                }
                
                following.append(user_info)
            
            return {"status": "success", "following": following}
        except Exception as e:
            logger.error(f"Error al obtener usuarios seguidos: {str(e)}")
            return {"status": "error", "message": str(e)}