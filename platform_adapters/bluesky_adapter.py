"""
Bluesky Adapter - Adaptador para Bluesky

Este módulo proporciona una interfaz unificada para interactuar con la API de Bluesky (AT Protocol),
gestionando la publicación de contenido, análisis de métricas, gestión de comentarios,
y otras funcionalidades específicas de Bluesky.
"""

import os
import sys
import json
import logging
import time
import requests
import base64
import mimetypes
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bluesky_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BlueskyAdapter")

class BlueskyAdapter:
    """
    Adaptador para interactuar con la API de Bluesky (AT Protocol)
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el adaptador de Bluesky
        
        Args:
            config_path: Ruta al archivo de configuración con credenciales
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.api_base_url = "https://bsky.social/xrpc"
        self.session = None
        self.auth_token = None
        self.auth_expiry = None
        self.did = None  # Decentralized Identifier
        self.handle = None  # User handle
        self.rate_limit_remaining = 100  # Valor inicial estimado
        self.rate_limit_reset = datetime.now() + timedelta(minutes=15)
        self.initialize_api()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo
        
        Returns:
            Configuración de Bluesky
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('bluesky', {})
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def initialize_api(self) -> bool:
        """
        Inicializa la conexión con la API de Bluesky
        
        Returns:
            True si se inicializó correctamente, False en caso contrario
        """
        try:
            # Verificar si hay credenciales
            if not self.config:
                logger.error("No se encontró configuración para Bluesky")
                return False
            
            # Verificar si el token actual es válido
            if self.auth_token and self.auth_expiry and datetime.now() < self.auth_expiry:
                logger.info("Token de autenticación actual válido hasta " + self.auth_expiry.isoformat())
                return True
            
            # Obtener credenciales
            identifier = self.config.get('identifier')  # handle o email
            password = self.config.get('app_password')  # app password
            
            if not identifier or not password:
                logger.error("Faltan credenciales en la configuración")
                return False
            
            # Crear sesión
            self.session = requests.Session()
            
            # Autenticar
            auth_response = self.session.post(
                f"{self.api_base_url}/com.atproto.server.createSession",
                json={
                    "identifier": identifier,
                    "password": password
                }
            )
            
            if auth_response.status_code != 200:
                logger.error(f"Error de autenticación: {auth_response.text}")
                return False
            
            auth_data = auth_response.json()
            self.auth_token = auth_data.get("accessJwt")
            self.did = auth_data.get("did")
            self.handle = auth_data.get("handle")
            
            # Establecer token en la sesión
            self.session.headers.update({
                "Authorization": f"Bearer {self.auth_token}"
            })
            
            # Establecer expiración (por defecto 2 horas)
            self.auth_expiry = datetime.now() + timedelta(hours=2)
            
            logger.info(f"API de Bluesky inicializada correctamente para {self.handle}, token válido hasta {self.auth_expiry.isoformat()}")
            return True
        except Exception as e:
            logger.error(f"Error al inicializar API de Bluesky: {str(e)}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """
        Verifica si hay límite de tasa disponible
        
        Returns:
            True si hay límite disponible, False en caso contrario
        """
        # Verificar si es tiempo de reiniciar el límite
        if datetime.now() > self.rate_limit_reset:
            self.rate_limit_remaining = 100  # Valor estimado
            self.rate_limit_reset = datetime.now() + timedelta(minutes=15)
            logger.info("Límite de tasa de Bluesky API reiniciado")
        
                # Verificar si hay límite disponible
        if self.rate_limit_remaining <= 0:
            logger.warning(f"Límite de tasa alcanzado, se reiniciará en {(self.rate_limit_reset - datetime.now()).total_seconds()} segundos")
            return False
        
        # Decrementar límite
        self.rate_limit_remaining -= 1
        return True
    
    def _make_api_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """
        Realiza una solicitud a la API de Bluesky
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint de la API
            data: Datos para enviar en el cuerpo
            
        Returns:
            Respuesta de la API o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar token
            if not self.auth_token or (self.auth_expiry and datetime.now() >= self.auth_expiry):
                if not self.initialize_api():
                    return {"status": "error", "message": "No se pudo inicializar la API"}
            
            # Preparar URL
            url = f"{self.api_base_url}/{endpoint}"
            
            # Realizar solicitud
            if method.upper() == "GET":
                response = self.session.get(url, params=data)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, json=data)
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
            logger.error(f"Error en solicitud a API de Bluesky: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def create_post(self, text: str, images: List[str] = None, reply_to: str = None) -> Dict[str, Any]:
        """
        Crea un post en Bluesky
        
        Args:
            text: Texto del post
            images: Lista de rutas a imágenes para adjuntar (opcional)
            reply_to: URI del post al que se responde (opcional)
            
        Returns:
            Información del post creado o error
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Verificar autenticación
            if not self.did or not self.auth_token:
                if not self.initialize_api():
                    return {"status": "error", "message": "No se pudo inicializar la API"}
            
            # Preparar facets (enlaces, menciones, etc.)
            facets = self._extract_facets(text)
            
            # Preparar post
            post_data = {
                "repo": self.did,
                "collection": "app.bsky.feed.post",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": text,
                    "createdAt": datetime.now().isoformat(),
                    "langs": ["es"]
                }
            }
            
            # Añadir facets si existen
            if facets:
                post_data["record"]["facets"] = facets
            
            # Añadir reply si existe
            if reply_to:
                # Extraer información del post al que se responde
                reply_parts = reply_to.split("/")
                if len(reply_parts) >= 2:
                    reply_did = reply_parts[0]
                    reply_rkey = reply_parts[1]
                    
                    post_data["record"]["reply"] = {
                        "root": {
                            "uri": f"at://{reply_did}/app.bsky.feed.post/{reply_rkey}",
                            "cid": reply_rkey  # Simplificado, en realidad se necesita el CID correcto
                        },
                        "parent": {
                            "uri": f"at://{reply_did}/app.bsky.feed.post/{reply_rkey}",
                            "cid": reply_rkey  # Simplificado, en realidad se necesita el CID correcto
                        }
                    }
            
            # Añadir imágenes si existen
            if images and len(images) > 0:
                image_blobs = []
                
                for image_path in images:
                    if not os.path.exists(image_path):
                        logger.warning(f"Imagen no encontrada: {image_path}")
                        continue
                    
                    # Subir imagen
                    upload_result = self._upload_blob(image_path)
                    if upload_result.get("status") != "success":
                        logger.warning(f"Error al subir imagen: {upload_result.get('message')}")
                        continue
                    
                    blob_data = upload_result.get("data", {})
                    
                    # Añadir blob a la lista
                    image_blobs.append({
                        "image": blob_data.get("blob"),
                        "alt": os.path.basename(image_path)  # Usar nombre de archivo como texto alternativo
                    })
                
                # Añadir imágenes al post
                if image_blobs:
                    post_data["record"]["embed"] = {
                        "$type": "app.bsky.embed.images",
                        "images": image_blobs
                    }
            
            # Crear post
            response = self._make_api_request(
                "POST",
                "com.atproto.repo.createRecord",
                data=post_data
            )
            
            if response.get("status") != "success":
                return response
            
            post_uri = response.get("data", {}).get("uri", "")
            post_cid = response.get("data", {}).get("cid", "")
            
            logger.info(f"Post creado correctamente: {post_uri}")
            return {
                "status": "success",
                "post_uri": post_uri,
                "post_cid": post_cid,
                "text": text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al crear post: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _extract_facets(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae facets (enlaces, menciones, etc.) del texto
        
        Args:
            text: Texto del post
            
        Returns:
            Lista de facets
        """
        facets = []
        
        # Buscar URLs
        import re
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            start, end = match.span()
            facets.append({
                "index": {
                    "byteStart": start,
                    "byteEnd": end
                },
                "features": [
                    {
                        "$type": "app.bsky.richtext.facet#link",
                        "uri": match.group()
                    }
                ]
            })
        
        # Buscar menciones (@usuario)
        mention_pattern = r'@([a-zA-Z0-9.-]+)'
        for match in re.finditer(mention_pattern, text):
            start, end = match.span()
            handle = match.group(1)
            
            # Obtener DID del usuario mencionado
            did = self._resolve_handle(handle)
            if did:
                facets.append({
                    "index": {
                        "byteStart": start,
                        "byteEnd": end
                    },
                    "features": [
                        {
                            "$type": "app.bsky.richtext.facet#mention",
                            "did": did
                        }
                    ]
                })
        
        return facets
    
    def _resolve_handle(self, handle: str) -> str:
        """
        Resuelve un handle de usuario a su DID
        
        Args:
            handle: Handle del usuario
            
        Returns:
            DID del usuario o cadena vacía
        """
        try:
            response = self._make_api_request(
                "GET",
                "com.atproto.identity.resolveHandle",
                data={"handle": handle}
            )
            
            if response.get("status") != "success":
                return ""
            
            return response.get("data", {}).get("did", "")
        except Exception as e:
            logger.error(f"Error al resolver handle: {str(e)}")
            return ""
    
    def _upload_blob(self, file_path: str) -> Dict[str, Any]:
        """
        Sube un archivo como blob
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Información del blob o error
        """
        try:
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"Archivo no encontrado: {file_path}"}
            
            # Determinar tipo MIME
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Leer archivo
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            # Subir blob
            headers = {
                "Content-Type": mime_type,
                "Authorization": f"Bearer {self.auth_token}"
            }
            
            response = requests.post(
                f"{self.api_base_url}/com.atproto.repo.uploadBlob",
                headers=headers,
                data=file_data
            )
            
            if response.status_code >= 400:
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
            
            blob_data = response.json()
            
            return {
                "status": "success",
                "data": blob_data
            }
        except Exception as e:
            logger.error(f"Error al subir blob: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def delete_post(self, post_uri: str) -> Dict[str, Any]:
        """
        Elimina un post
        
        Args:
            post_uri: URI del post
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Extraer información del post
            uri_parts = post_uri.split("/")
            if len(uri_parts) < 2:
                return {"status": "error", "message": "URI de post inválido"}
            
            repo = self.did
            collection = "app.bsky.feed.post"
            rkey = uri_parts[-1]
            
            # Eliminar post
            response = self._make_api_request(
                "POST",
                "com.atproto.repo.deleteRecord",
                data={
                    "repo": repo,
                    "collection": collection,
                    "rkey": rkey
                }
            )
            
            if response.get("status") != "success":
                return response
            
            logger.info(f"Post {post_uri} eliminado correctamente")
            return {"status": "success", "message": "Post eliminado correctamente"}
        except Exception as e:
            logger.error(f"Error al eliminar post: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_post(self, post_uri: str) -> Dict[str, Any]:
        """
        Obtiene información de un post
        
        Args:
            post_uri: URI del post
            
        Returns:
            Información del post
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Extraer información del post
            uri_parts = post_uri.split("/")
            if len(uri_parts) < 2:
                return {"status": "error", "message": "URI de post inválido"}
            
            # Obtener post
            response = self._make_api_request(
                "GET",
                "app.bsky.feed.getPostThread",
                data={
                    "uri": post_uri,
                    "depth": 0
                }
            )
            
            if response.get("status") != "success":
                return response
            
            thread = response.get("data", {}).get("thread", {})
            post = thread.get("post", {})
            
            # Procesar información
            post_info = {
                "uri": post.get("uri", ""),
                "cid": post.get("cid", ""),
                "author": {
                    "did": post.get("author", {}).get("did", ""),
                    "handle": post.get("author", {}).get("handle", ""),
                    "display_name": post.get("author", {}).get("displayName", "")
                },
                "text": post.get("record", {}).get("text", ""),
                "created_at": post.get("record", {}).get("createdAt", ""),
                "like_count": post.get("likeCount", 0),
                "reply_count": post.get("replyCount", 0),
                "repost_count": post.get("repostCount", 0)
            }
            
            return {"status": "success", "post_info": post_info}
        except Exception as e:
            logger.error(f"Error al obtener post: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_profile(self, handle: str = None) -> Dict[str, Any]:
        """
        Obtiene información del perfil de un usuario
        
        Args:
            handle: Handle del usuario (opcional, usa el propio si no se especifica)
            
        Returns:
            Información del perfil
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Si no se proporciona handle, usar el propio
            if not handle:
                handle = self.handle
            
            # Obtener perfil
            response = self._make_api_request(
                "GET",
                "app.bsky.actor.getProfile",
                data={"actor": handle}
            )
            
            if response.get("status") != "success":
                return response
            
            profile_data = response.get("data", {})
            
            # Procesar información
            profile_info = {
                "did": profile_data.get("did", ""),
                "handle": profile_data.get("handle", ""),
                "display_name": profile_data.get("displayName", ""),
                "description": profile_data.get("description", ""),
                "avatar": profile_data.get("avatar", ""),
                "banner": profile_data.get("banner", ""),
                "follower_count": profile_data.get("followersCount", 0),
                "following_count": profile_data.get("followsCount", 0),
                "post_count": profile_data.get("postsCount", 0),
                "indexed_at": profile_data.get("indexedAt", "")
            }
            
            return {"status": "success", "profile_info": profile_info}
        except Exception as e:
            logger.error(f"Error al obtener perfil: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def like_post(self, post_uri: str) -> Dict[str, Any]:
        """
        Da like a un post
        
        Args:
            post_uri: URI del post
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Extraer información del post
            uri_parts = post_uri.split("/")
            if len(uri_parts) < 2:
                return {"status": "error", "message": "URI de post inválido"}
            
            # Crear like
            response = self._make_api_request(
                "POST",
                "com.atproto.repo.createRecord",
                data={
                    "repo": self.did,
                    "collection": "app.bsky.feed.like",
                    "record": {
                        "$type": "app.bsky.feed.like",
                        "subject": {
                            "uri": post_uri,
                            "cid": uri_parts[-1]  # Simplificado, en realidad se necesita el CID correcto
                        },
                        "createdAt": datetime.now().isoformat()
                    }
                }
            )
            
            if response.get("status") != "success":
                return response
            
            like_uri = response.get("data", {}).get("uri", "")
            
            logger.info(f"Like creado correctamente: {like_uri}")
            return {
                "status": "success",
                "like_uri": like_uri,
                "post_uri": post_uri,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al dar like: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def repost(self, post_uri: str) -> Dict[str, Any]:
        """
        Repostea un post
        
        Args:
            post_uri: URI del post
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Extraer información del post
            uri_parts = post_uri.split("/")
            if len(uri_parts) < 2:
                return {"status": "error", "message": "URI de post inválido"}
            
            # Crear repost
            response = self._make_api_request(
                "POST",
                "com.atproto.repo.createRecord",
                data={
                    "repo": self.did,
                    "collection": "app.bsky.feed.repost",
                    "record": {
                        "$type": "app.bsky.feed.repost",
                        "subject": {
                            "uri": post_uri,
                            "cid": uri_parts[-1]  # Simplificado, en realidad se necesita el CID correcto
                        },
                        "createdAt": datetime.now().isoformat()
                    }
                }
            )
            
            if response.get("status") != "success":
                return response
            
            repost_uri = response.get("data", {}).get("uri", "")
            
            logger.info(f"Repost creado correctamente: {repost_uri}")
            return {
                "status": "success",
                "repost_uri": repost_uri,
                "post_uri": post_uri,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al repostear: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def follow_user(self, handle: str) -> Dict[str, Any]:
        """
        Sigue a un usuario
        
        Args:
            handle: Handle del usuario
            
        Returns:
            Resultado de la operación
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Resolver DID del usuario
            did = self._resolve_handle(handle)
            if not did:
                return {"status": "error", "message": f"No se pudo resolver el handle: {handle}"}
            
            # Crear follow
            response = self._make_api_request(
                "POST",
                "com.atproto.repo.createRecord",
                data={
                    "repo": self.did,
                    "collection": "app.bsky.graph.follow",
                    "record": {
                        "$type": "app.bsky.graph.follow",
                        "subject": did,
                        "createdAt": datetime.now().isoformat()
                    }
                }
            )
            
            if response.get("status") != "success":
                return response
            
            follow_uri = response.get("data", {}).get("uri", "")
            
            logger.info(f"Follow creado correctamente: {follow_uri}")
            return {
                "status": "success",
                "follow_uri": follow_uri,
                "handle": handle,
                "did": did,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al seguir usuario: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_timeline(self, limit: int = 20) -> Dict[str, Any]:
        """
        Obtiene la línea de tiempo del usuario
        
        Args:
            limit: Número máximo de posts
            
        Returns:
            Lista de posts
        """
        if not self._check_rate_limit():
            return {"status": "error", "message": "Límite de tasa alcanzado"}
        
        try:
            # Obtener timeline
            response = self._make_api_request(
                "GET",
                "app.bsky.feed.getTimeline",
                data={"limit": limit}
            )
            
            if response.get("status") != "success":
                return response
            
            feed_data = response.get("data", {}).get("feed", [])
            
            # Procesar posts
            posts = []
            for item in feed_data:
                post = item.get("post", {})
                
                posts.append({
                    "uri": post.get("uri", ""),
                    "cid": post.get("cid", ""),
                    "author": {
                        "did": post.get("author", {}).get("did", ""),
                        "handle": post.get("author", {}).get("handle", ""),
                        "display_name": post.get("author", {}).get("displayName", "")
                    },
                    "text": post.get("record", {}).get("text", ""),
                    "created_at": post.get("record", {}).get("createdAt", ""),
                    "like_count": post.get("likeCount", 0),
                    "reply_count": post.get("replyCount", 0),
                    "repost_count": post.get("repostCount", 0)
                })
            
            return {"status": "success", "posts": posts}
        except Exception as e:
            logger.error(f"Error al obtener timeline: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_post_engagement(self, post_uri: str) -> Dict[str, Any]:
        """
        Obtiene métricas de engagement de un post
        
        Args:
            post_uri: URI del post
            
        Returns:
            Métricas de engagement
        """
        try:
            # Obtener información del post
            post_info_response = self.get_post(post_uri)
            if post_info_response.get("status") != "success":
                return post_info_response
            
            post_info = post_info_response.get("post_info", {})
            
            # Calcular engagement
            like_count = post_info.get("like_count", 0)
            reply_count = post_info.get("reply_count", 0)
            repost_count = post_info.get("repost_count", 0)
            
            total_engagement = like_count + reply_count + repost_count
            
            # Obtener perfil para calcular engagement rate
            profile_response = self.get_profile()
            if profile_response.get("status") != "success":
                follower_count = 1  # Valor predeterminado si no se puede obtener
            else:
                follower_count = profile_response.get("profile_info", {}).get("follower_count", 1)
            
            engagement_rate = (total_engagement / follower_count) * 100 if follower_count > 0 else 0
            
            return {
                "status": "success",
                "post_uri": post_uri,
                "metrics": {
                    "like_count": like_count,
                    "reply_count": reply_count,
                    "repost_count": repost_count,
                    "total_engagement": total_engagement,
                    "follower_count": follower_count,
                    "engagement_rate": engagement_rate
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al obtener engagement: {str(e)}")
            return {"status": "error", "message": str(e)}