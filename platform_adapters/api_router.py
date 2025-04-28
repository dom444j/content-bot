"""
API Router - Manejador común para adaptadores de plataformas

Este módulo proporciona una interfaz unificada para interactuar con múltiples plataformas
a través de sus adaptadores específicos. Centraliza las solicitudes API, maneja errores,
gestiona límites de tasa, y proporciona funciones comunes para publicación, análisis y
gestión de contenido en todas las plataformas soportadas.
"""

import os
import sys
import json
import logging
import time
import importlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import threading
import queue
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_router.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("APIRouter")

class APIRouter:
    """
    Enrutador de API para gestionar múltiples adaptadores de plataformas
    """
    
    def __init__(self, config_path: str = "config/platforms.json"):
        """
        Inicializa el enrutador de API
        
        Args:
            config_path: Ruta al archivo de configuración con credenciales
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.adapters = {}
        self.adapter_status = {}
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker_threads = []
        self.max_workers = 5
        self.running = False
        self.initialize_adapters()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo
        
        Returns:
            Configuración de plataformas
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def initialize_adapters(self) -> bool:
        """
        Inicializa todos los adaptadores de plataformas configurados
        
        Returns:
            True si al menos un adaptador se inicializó correctamente, False en caso contrario
        """
        try:
            # Lista de adaptadores disponibles
            available_adapters = {
                "youtube": "youtube_adapter.YouTubeAdapter",
                "tiktok": "tiktok_adapter.TikTokAdapter",
                "instagram": "instagram_adapter.InstagramAdapter",
                "threads": "threads_adapter.ThreadsAdapter",
                "bluesky": "bluesky_adapter.BlueskyAdapter",
                "twitter": "x_adapter.XAdapter"
            }
            
            # Inicializar adaptadores configurados
            success_count = 0
            
            for platform, adapter_info in available_adapters.items():
                if platform in self.config:
                    try:
                        # Dividir el nombre del módulo y la clase
                        module_name, class_name = adapter_info.split(".")
                        
                        # Importar dinámicamente el módulo
                        module_path = f"platform_adapters.{module_name}"
                        module = importlib.import_module(module_path)
                        
                        # Obtener la clase del adaptador
                        adapter_class = getattr(module, class_name)
                        
                        # Instanciar el adaptador
                        adapter_instance = adapter_class(self.config_path)
                        
                        # Guardar el adaptador
                        self.adapters[platform] = adapter_instance
                        self.adapter_status[platform] = {
                            "initialized": True,
                            "last_error": None,
                            "error_count": 0,
                            "last_request": None,
                            "request_count": 0,
                            "enabled": True
                        }
                        
                        logger.info(f"Adaptador para {platform} inicializado correctamente")
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error al inicializar adaptador para {platform}: {str(e)}")
                        self.adapter_status[platform] = {
                            "initialized": False,
                            "last_error": str(e),
                            "error_count": 1,
                            "last_request": None,
                            "request_count": 0,
                            "enabled": False
                        }
            
            if success_count > 0:
                logger.info(f"{success_count} adaptadores inicializados correctamente")
                return True
            else:
                logger.error("No se pudo inicializar ningún adaptador")
                return False
        except Exception as e:
            logger.error(f"Error al inicializar adaptadores: {str(e)}")
            return False
    
    def start_workers(self) -> bool:
        """
        Inicia los hilos de trabajo para procesar solicitudes en segundo plano
        
        Returns:
            True si se iniciaron correctamente, False en caso contrario
        """
        if self.running:
            logger.warning("Los trabajadores ya están en ejecución")
            return True
        
        try:
            self.running = True
            
            # Crear y iniciar hilos de trabajo
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._request_worker,
                    name=f"APIRouter-Worker-{i}",
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)
            
            logger.info(f"{self.max_workers} trabajadores iniciados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al iniciar trabajadores: {str(e)}")
            self.running = False
            return False
    
    def stop_workers(self) -> bool:
        """
        Detiene los hilos de trabajo
        
        Returns:
            True si se detuvieron correctamente, False en caso contrario
        """
        if not self.running:
            logger.warning("Los trabajadores no están en ejecución")
            return True
        
        try:
            # Señalizar a los trabajadores que deben detenerse
            self.running = False
            
            # Agregar tareas de terminación para cada trabajador
            for _ in range(self.max_workers):
                self.request_queue.put(None)
            
            # Esperar a que los trabajadores terminen
            for worker in self.worker_threads:
                worker.join(timeout=5.0)
            
            # Limpiar lista de trabajadores
            self.worker_threads = []
            
            logger.info("Trabajadores detenidos correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al detener trabajadores: {str(e)}")
            return False
    
    def _request_worker(self) -> None:
        """
        Función de trabajo para procesar solicitudes en segundo plano
        """
        while self.running:
            try:
                # Obtener una tarea de la cola
                task = self.request_queue.get(timeout=1.0)
                
                # Si la tarea es None, terminar
                if task is None:
                    break
                
                # Extraer información de la tarea
                task_id = task.get("task_id")
                platform = task.get("platform")
                method = task.get("method")
                args = task.get("args", [])
                kwargs = task.get("kwargs", {})
                
                # Verificar si el adaptador existe y está habilitado
                if platform not in self.adapters or not self.adapter_status[platform]["enabled"]:
                    response = {
                        "task_id": task_id,
                        "status": "error",
                        "message": f"Adaptador para {platform} no disponible o deshabilitado"
                    }
                    self.response_queue.put(response)
                    continue
                
                # Obtener el adaptador
                adapter = self.adapters[platform]
                
                # Verificar si el método existe
                if not hasattr(adapter, method):
                    response = {
                        "task_id": task_id,
                        "status": "error",
                        "message": f"Método {method} no disponible en adaptador para {platform}"
                    }
                    self.response_queue.put(response)
                    continue
                
                # Ejecutar el método
                adapter_method = getattr(adapter, method)
                result = adapter_method(*args, **kwargs)
                
                # Actualizar estadísticas del adaptador
                self.adapter_status[platform]["last_request"] = datetime.now()
                self.adapter_status[platform]["request_count"] += 1
                
                # Verificar resultado
                if isinstance(result, dict) and result.get("status") == "error":
                    self.adapter_status[platform]["error_count"] += 1
                    self.adapter_status[platform]["last_error"] = result.get("message")
                
                # Enviar respuesta
                response = {
                    "task_id": task_id,
                    "platform": platform,
                    "method": method,
                    "result": result
                }
                self.response_queue.put(response)
            except queue.Empty:
                # No hay tareas en la cola, continuar
                continue
            except Exception as e:
                # Error al procesar la tarea
                logger.error(f"Error en worker: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Intentar enviar respuesta de error
                try:
                    if 'task_id' in locals():
                        response = {
                            "task_id": task_id,
                            "status": "error",
                            "message": f"Error interno: {str(e)}"
                        }
                        self.response_queue.put(response)
                except:
                    pass
    
    def execute(self, platform: str, method: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un método en un adaptador específico
        
        Args:
            platform: Nombre de la plataforma
            method: Nombre del método a ejecutar
            *args: Argumentos posicionales para el método
            **kwargs: Argumentos con nombre para el método
            
        Returns:
            Resultado de la ejecución del método
        """
        try:
            # Verificar si el adaptador existe
            if platform not in self.adapters:
                return {"status": "error", "message": f"Adaptador para {platform} no disponible"}
            
            # Verificar si el adaptador está habilitado
            if not self.adapter_status[platform]["enabled"]:
                return {"status": "error", "message": f"Adaptador para {platform} deshabilitado"}
            
            # Obtener el adaptador
            adapter = self.adapters[platform]
            
            # Verificar si el método existe
            if not hasattr(adapter, method):
                return {"status": "error", "message": f"Método {method} no disponible en adaptador para {platform}"}
            
            # Ejecutar el método
            adapter_method = getattr(adapter, method)
            result = adapter_method(*args, **kwargs)
            
            # Actualizar estadísticas del adaptador
            self.adapter_status[platform]["last_request"] = datetime.now()
            self.adapter_status[platform]["request_count"] += 1
            
            # Verificar resultado
            if isinstance(result, dict) and result.get("status") == "error":
                self.adapter_status[platform]["error_count"] += 1
                self.adapter_status[platform]["last_error"] = result.get("message")
            
            return result
        except Exception as e:
            logger.error(f"Error al ejecutar {method} en {platform}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def execute_async(self, platform: str, method: str, *args, **kwargs) -> str:
        """
        Ejecuta un método en un adaptador específico de forma asíncrona
        
        Args:
            platform: Nombre de la plataforma
            method: Nombre del método a ejecutar
            *args: Argumentos posicionales para el método
            **kwargs: Argumentos con nombre para el método
            
        Returns:
            ID de la tarea
        """
        try:
            # Verificar si los trabajadores están en ejecución
            if not self.running:
                self.start_workers()
            
            # Generar ID de tarea
            task_id = f"{platform}-{method}-{datetime.now().timestamp()}"
            
            # Crear tarea
            task = {
                "task_id": task_id,
                "platform": platform,
                "method": method,
                "args": args,
                "kwargs": kwargs,
                "timestamp": datetime.now().isoformat()
            }
            
            # Agregar tarea a la cola
            self.request_queue.put(task)
            
            return task_id
        except Exception as e:
            logger.error(f"Error al programar tarea asíncrona: {str(e)}")
            return None
    
    def get_result(self, task_id: str, timeout: float = 0.0) -> Dict[str, Any]:
        """
        Obtiene el resultado de una tarea asíncrona
        
        Args:
            task_id: ID de la tarea
            timeout: Tiempo máximo de espera en segundos (0 para no esperar)
            
        Returns:
            Resultado de la tarea o None si no está disponible
        """
        try:
            # Buscar en la cola de respuestas
            start_time = time.time()
            
            while True:
                # Verificar timeout
                if timeout > 0 and time.time() - start_time > timeout:
                    return {"status": "pending", "message": "Tiempo de espera agotado"}
                
                # Intentar obtener una respuesta sin bloquear
                try:
                    response = self.response_queue.get(block=False)
                    
                    # Verificar si es la respuesta buscada
                    if response.get("task_id") == task_id:
                        return response.get("result", {"status": "error", "message": "Resultado no disponible"})
                    
                    # No es la respuesta buscada, devolverla a la cola
                    self.response_queue.put(response)
                except queue.Empty:
                    # No hay respuestas en la cola
                    if timeout <= 0:
                        return {"status": "pending", "message": "Resultado no disponible"}
                    
                    # Esperar un poco antes de intentar de nuevo
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error al obtener resultado: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def execute_all(self, method: str, platforms: List[str] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta un método en múltiples adaptadores
        
        Args:
            method: Nombre del método a ejecutar
            platforms: Lista de plataformas (si es None, usa todas las disponibles)
            *args: Argumentos posicionales para el método
            **kwargs: Argumentos con nombre para el método
            
        Returns:
            Diccionario con resultados por plataforma
        """
        try:
            # Si no se especifican plataformas, usar todas las disponibles
            if platforms is None:
                platforms = list(self.adapters.keys())
            
            # Ejecutar método en cada plataforma
            results = {}
            
            for platform in platforms:
                # Verificar si el adaptador existe
                if platform not in self.adapters:
                    results[platform] = {"status": "error", "message": f"Adaptador para {platform} no disponible"}
                    continue
                
                # Verificar si el adaptador está habilitado
                if not self.adapter_status[platform]["enabled"]:
                    results[platform] = {"status": "error", "message": f"Adaptador para {platform} deshabilitado"}
                    continue
                
                # Ejecutar método
                result = self.execute(platform, method, *args, **kwargs)
                results[platform] = result
            
            return results
        except Exception as e:
            logger.error(f"Error al ejecutar {method} en múltiples plataformas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def execute_all_async(self, method: str, platforms: List[str] = None, *args, **kwargs) -> Dict[str, str]:
        """
        Ejecuta un método en múltiples adaptadores de forma asíncrona
        
        Args:
            method: Nombre del método a ejecutar
            platforms: Lista de plataformas (si es None, usa todas las disponibles)
            *args: Argumentos posicionales para el método
            **kwargs: Argumentos con nombre para el método
            
        Returns:
            Diccionario con IDs de tareas por plataforma
        """
        try:
            # Si no se especifican plataformas, usar todas las disponibles
            if platforms is None:
                platforms = list(self.adapters.keys())
            
            # Ejecutar método en cada plataforma
            task_ids = {}
            
            for platform in platforms:
                # Verificar si el adaptador existe
                if platform not in self.adapters:
                    continue
                
                # Verificar si el adaptador está habilitado
                if not self.adapter_status[platform]["enabled"]:
                    continue
                
                # Ejecutar método
                task_id = self.execute_async(platform, method, *args, **kwargs)
                task_ids[platform] = task_id
            
            return task_ids
        except Exception as e:
            logger.error(f"Error al programar tareas asíncronas: {str(e)}")
            return {}
    
    def get_all_results(self, task_ids: Dict[str, str], timeout: float = 0.0) -> Dict[str, Any]:
        """
        Obtiene los resultados de múltiples tareas asíncronas
        
        Args:
            task_ids: Diccionario con IDs de tareas por plataforma
            timeout: Tiempo máximo de espera en segundos (0 para no esperar)
            
        Returns:
            Diccionario con resultados por plataforma
        """
        try:
            # Obtener resultados para cada tarea
            results = {}
            
            for platform, task_id in task_ids.items():
                result = self.get_result(task_id, timeout)
                results[platform] = result
            
            return results
        except Exception as e:
            logger.error(f"Error al obtener resultados: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def publish_content(self, content_data: Dict[str, Any], platforms: List[str] = None) -> Dict[str, Any]:
        """
        Publica contenido en múltiples plataformas
        
        Args:
            content_data: Datos del contenido a publicar
            platforms: Lista de plataformas (si es None, usa todas las disponibles)
            
        Returns:
            Diccionario con resultados por plataforma
        """
        try:
            # Si no se especifican plataformas, usar todas las disponibles
            if platforms is None:
                platforms = list(self.adapters.keys())
            
            # Publicar en cada plataforma
            results = {}
            
            for platform in platforms:
                # Verificar si el adaptador existe
                if platform not in self.adapters:
                    results[platform] = {"status": "error", "message": f"Adaptador para {platform} no disponible"}
                    continue
                
                # Verificar si el adaptador está habilitado
                if not self.adapter_status[platform]["enabled"]:
                    results[platform] = {"status": "error", "message": f"Adaptador para {platform} deshabilitado"}
                    continue
                
                # Obtener datos específicos para la plataforma
                platform_data = content_data.get(platform, {})
                
                # Combinar con datos comunes
                combined_data = {**content_data.get("common", {}), **platform_data}
                
                # Determinar método de publicación según el tipo de contenido
                content_type = combined_data.get("content_type", "video")
                
                if content_type == "video":
                    # Publicar video
                    video_path = combined_data.get("video_path")
                    title = combined_data.get("title")
                    description = combined_data.get("description")
                    tags = combined_data.get("tags", [])
                    thumbnail_path = combined_data.get("thumbnail_path")
                    
                    if platform == "youtube":
                        result = self.execute(platform, "upload_video", video_path, title, description, tags, thumbnail_path)
                    elif platform == "tiktok":
                        result = self.execute(platform, "upload_video", video_path, description, tags)
                    elif platform == "instagram":
                        result = self.execute(platform, "upload_reel", video_path, description, tags)
                    elif platform == "twitter":
                        result = self.execute(platform, "upload_media", video_path)
                        if result.get("status") == "success":
                            media_id = result.get("media_id")
                            result = self.execute(platform, "create_tweet", description, [media_id])
                    else:
                        result = {"status": "error", "message": f"Publicación de video no implementada para {platform}"}
                
                elif content_type == "image":
                    # Publicar imagen
                    image_path = combined_data.get("image_path")
                    description = combined_data.get("description")
                    tags = combined_data.get("tags", [])
                    
                    if platform == "instagram":
                        result = self.execute(platform, "upload_image", image_path, description, tags)
                    elif platform == "twitter":
                        result = self.execute(platform, "upload_media", image_path)
                        if result.get("status") == "success":
                            media_id = result.get("media_id")
                            result = self.execute(platform, "create_tweet", description, [media_id])
                    else:
                        result = {"status": "error", "message": f"Publicación de imagen no implementada para {platform}"}
                
                elif content_type == "text":
                    # Publicar texto
                    text = combined_data.get("text")
                    
                    if platform == "twitter":
                        result = self.execute(platform, "create_tweet", text)
                    elif platform == "threads":
                        result = self.execute(platform, "create_thread", text)
                    elif platform == "bluesky":
                        result = self.execute(platform, "create_post", text)
                    else:
                        result = {"status": "error", "message": f"Publicación de texto no implementada para {platform}"}
                
                else:
                    result = {"status": "error", "message": f"Tipo de contenido {content_type} no soportado"}
                
                results[platform] = result
            
            return results
        except Exception as e:
            logger.error(f"Error al publicar contenido: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def publish_content_async(self, content_data: Dict[str, Any], platforms: List[str] = None) -> Dict[str, str]:
        """
        Publica contenido en múltiples plataformas de forma asíncrona
        
        Args:
            content_data: Datos del contenido a publicar
            platforms: Lista de plataformas (si es None, usa todas las disponibles)
            
        Returns:
            Diccionario con IDs de tareas por plataforma
        """
        try:
            # Crear tarea para publicación
            task_id = self.execute_async("_internal", "publish_content", content_data, platforms)
            
            return {"task_id": task_id}
        except Exception as e:
            logger.error(f"Error al programar publicación asíncrona: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_platform_metrics(self, platforms: List[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene métricas de múltiples plataformas
        
        Args:
            platforms: Lista de plataformas (si es None, usa todas las disponibles)
            days: Número de días para obtener métricas
            
        Returns:
            Diccionario con métricas por plataforma
        """
        try:
            # Si no se especifican plataformas, usar todas las disponibles
            if platforms is None:
                platforms = list(self.adapters.keys())
            
            # Obtener métricas para cada plataforma
            results = {}
            
            for platform in platforms:
                # Verificar si el adaptador existe
                if platform not in self.adapters:
                    results[platform] = {"status": "error", "message": f"Adaptador para {platform} no disponible"}
                    continue
                
                # Verificar si el adaptador está habilitado
                if not self.adapter_status[platform]["enabled"]:
                    results[platform] = {"status": "error", "message": f"Adaptador para {platform} deshabilitado"}
                    continue
                
                # Determinar método para obtener métricas según la plataforma
                if platform == "youtube":
                    result = self.execute(platform, "get_channel_analytics", days=days)
                elif platform == "tiktok":
                    result = self.execute(platform, "get_account_analytics", days=days)
                elif platform == "instagram":
                    result = self.execute(platform, "get_account_insights", days=days)
                elif platform == "twitter":
                    result = self.execute(platform, "get_user_metrics", days=days)
                else:
                    result = {"status": "error", "message": f"Obtención de métricas no implementada para {platform}"}
                
                results[platform] = result
            
            return results
        except Exception as e:
            logger.error(f"Error al obtener métricas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_adapter_status(self, platform: str = None) -> Dict[str, Any]:
        """
        Obtiene el estado de los adaptadores
        
        Args:
            platform: Nombre de la plataforma (si es None, devuelve todos)
            
        Returns:
            Estado del adaptador o de todos los adaptadores
        """
        try:
            if platform is not None:
                # Verificar si el adaptador existe
                if platform not in self.adapter_status:
                    return {"status": "error", "message": f"Adaptador para {platform} no disponible"}
                
                return {
                    "status": "success",
                    "platform": platform,
                    "adapter_status": self.adapter_status[platform]
                }
            else:
                return {
                    "status": "success",
                    "adapters": self.adapter_status
                }
        except Exception as e:
            logger.error(f"Error al obtener estado de adaptadores: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def enable_adapter(self, platform: str) -> Dict[str, Any]:
        """
        Habilita un adaptador
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si el adaptador existe
            if platform not in self.adapter_status:
                return {"status": "error", "message": f"Adaptador para {platform} no disponible"}
            
            # Habilitar adaptador
            self.adapter_status[platform]["enabled"] = True
            
            return {
                "status": "success",
                "message": f"Adaptador para {platform} habilitado correctamente"
            }
        except Exception as e:
            logger.error(f"Error al habilitar adaptador: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def disable_adapter(self, platform: str) -> Dict[str, Any]:
        """
        Deshabilita un adaptador
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si el adaptador existe
            if platform not in self.adapter_status:
                return {"status": "error", "message": f"Adaptador para {platform} no disponible"}
            
            # Deshabilitar adaptador
            self.adapter_status[platform]["enabled"] = False
            
            return {
                "status": "success",
                "message": f"Adaptador para {platform} deshabilitado correctamente"
            }
        except Exception as e:
            logger.error(f"Error al deshabilitar adaptador: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def reinitialize_adapter(self, platform: str) -> Dict[str, Any]:
        """
        Reinicializa un adaptador
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si el adaptador existe en la configuración
            if platform not in self.config:
                return {"status": "error", "message": f"Configuración para {platform} no disponible"}
            
            # Obtener información del adaptador
            adapter_info = None
            
            if platform == "youtube":
                adapter_info = "youtube_adapter.YouTubeAdapter"
            elif platform == "tiktok":
                adapter_info = "tiktok_adapter.TikTokAdapter"
            elif platform == "instagram":
                adapter_info = "instagram_adapter.InstagramAdapter"
            elif platform == "threads":
                adapter_info = "threads_adapter.ThreadsAdapter"
            elif platform == "bluesky":
                adapter_info = "bluesky_adapter.BlueskyAdapter"
            elif platform == "twitter":
                adapter_info = "x_adapter.XAdapter"
            
            if adapter_info is None:
                return {"status": "error", "message": f"Adaptador para {platform} no soportado"}
            
            try:
                # Dividir el nombre del módulo y la clase
                module_name, class_name = adapter_info.split(".")
                
                # Importar dinámicamente el módulo
                module_path = f"platform_adapters.{module_name}"
                module = importlib.import_module(module_path)
                
                # Obtener la clase del adaptador
                adapter_class = getattr(module, class_name)
                
                                # Instanciar el adaptador
                adapter_instance = adapter_class(self.config_path)
                
                # Guardar el adaptador
                self.adapters[platform] = adapter_instance
                self.adapter_status[platform] = {
                    "initialized": True,
                    "last_error": None,
                    "error_count": 0,
                    "last_request": None,
                    "request_count": 0,
                    "enabled": True
                }
                
                logger.info(f"Adaptador para {platform} reinicializado correctamente")
                return {
                    "status": "success",
                    "message": f"Adaptador para {platform} reinicializado correctamente"
                }
            except Exception as e:
                logger.error(f"Error al reinicializar adaptador para {platform}: {str(e)}")
                self.adapter_status[platform] = {
                    "initialized": False,
                    "last_error": str(e),
                    "error_count": 1,
                    "last_request": None,
                    "request_count": 0,
                    "enabled": False
                }
                return {
                    "status": "error",
                    "message": f"Error al reinicializar adaptador para {platform}: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error al reinicializar adaptador: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_supported_platforms(self) -> Dict[str, Any]:
        """
        Obtiene la lista de plataformas soportadas
        
        Returns:
            Lista de plataformas soportadas
        """
        try:
            # Lista de adaptadores disponibles
            available_adapters = {
                "youtube": "YouTube",
                "tiktok": "TikTok",
                "instagram": "Instagram",
                "threads": "Threads",
                "bluesky": "Bluesky",
                "twitter": "Twitter/X"
            }
            
            # Obtener estado de cada plataforma
            platforms = []
            
            for platform_id, platform_name in available_adapters.items():
                platform_info = {
                    "id": platform_id,
                    "name": platform_name,
                    "configured": platform_id in self.config,
                    "initialized": platform_id in self.adapters,
                    "enabled": platform_id in self.adapter_status and self.adapter_status[platform_id]["enabled"]
                }
                
                platforms.append(platform_info)
            
            return {
                "status": "success",
                "platforms": platforms
            }
        except Exception as e:
            logger.error(f"Error al obtener plataformas soportadas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_platform_features(self, platform: str) -> Dict[str, Any]:
        """
        Obtiene las características soportadas por una plataforma
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Características soportadas
        """
        try:
            # Verificar si el adaptador existe
            if platform not in self.adapters:
                return {"status": "error", "message": f"Adaptador para {platform} no disponible"}
            
            # Obtener el adaptador
            adapter = self.adapters[platform]
            
            # Obtener métodos públicos del adaptador
            methods = []
            
            for method_name in dir(adapter):
                # Excluir métodos privados y especiales
                if method_name.startswith('_'):
                    continue
                
                # Obtener el método
                method = getattr(adapter, method_name)
                
                # Verificar si es un método
                if callable(method):
                    # Obtener docstring
                    doc = method.__doc__ or ""
                    
                    # Agregar información del método
                    methods.append({
                        "name": method_name,
                        "description": doc.strip()
                    })
            
            return {
                "status": "success",
                "platform": platform,
                "features": methods
            }
        except Exception as e:
            logger.error(f"Error al obtener características de {platform}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def reload_config(self) -> Dict[str, Any]:
        """
        Recarga la configuración desde el archivo
        
        Returns:
            Resultado de la operación
        """
        try:
            # Cargar configuración
            self.config = self._load_config()
            
            # Reinicializar adaptadores
            self.initialize_adapters()
            
            return {
                "status": "success",
                "message": "Configuración recargada correctamente"
            }
        except Exception as e:
            logger.error(f"Error al recargar configuración: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de las colas de solicitudes y respuestas
        
        Returns:
            Estado de las colas
        """
        try:
            return {
                "status": "success",
                "request_queue_size": self.request_queue.qsize(),
                "response_queue_size": self.response_queue.qsize(),
                "workers_running": self.running,
                "active_workers": len(self.worker_threads)
            }
        except Exception as e:
            logger.error(f"Error al obtener estado de colas: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def clear_queues(self) -> Dict[str, Any]:
        """
        Limpia las colas de solicitudes y respuestas
        
        Returns:
            Resultado de la operación
        """
        try:
            # Limpiar cola de solicitudes
            while not self.request_queue.empty():
                try:
                    self.request_queue.get(block=False)
                except queue.Empty:
                    break
            
            # Limpiar cola de respuestas
            while not self.response_queue.empty():
                try:
                    self.response_queue.get(block=False)
                except queue.Empty:
                    break
            
            return {
                "status": "success",
                "message": "Colas limpiadas correctamente"
            }
        except Exception as e:
            logger.error(f"Error al limpiar colas: {str(e)}")
            return {"status": "error", "message": str(e)}