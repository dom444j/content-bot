"""
PublishManager - Gestor de publicación de contenido

Este módulo se encarga de coordinar la publicación de contenido en diferentes plataformas,
gestionando la programación, publicación inmediata y monitoreo del estado de publicación.
"""

import logging
import time
import datetime
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum, auto

# Configuración de logging
logger = logging.getLogger(__name__)

class PublishStatus(Enum):
    """Enumeración de estados de publicación."""
    SCHEDULED = auto()      # Programado para publicación futura
    PUBLISHING = auto()     # En proceso de publicación
    PUBLISHED = auto()      # Publicado correctamente
    FAILED = auto()         # Falló la publicación
    CANCELED = auto()       # Publicación cancelada
    DELETED = auto()        # Publicación eliminada de la plataforma

class PublishManager:
    """
    Gestor de publicación de contenido para el Orchestrator.
    
    Esta clase se encarga de coordinar la publicación de contenido en diferentes plataformas,
    gestionando la programación, publicación inmediata y monitoreo del estado.
    """
    
    def __init__(self, config_manager=None, persistence_manager=None, task_manager=None, 
                 content_manager=None, platform_adapters=None):
        """
        Inicializa el gestor de publicación.
        
        Args:
            config_manager: Gestor de configuración
            persistence_manager: Gestor de persistencia
            task_manager: Gestor de tareas
            content_manager: Gestor de contenido
            platform_adapters: Diccionario de adaptadores de plataforma
        """
        self.config = config_manager
        self.persistence = persistence_manager
        self.task_manager = task_manager
        self.content_manager = content_manager
        self.platform_adapters = platform_adapters or {}
        
        # Diccionario de publicaciones por ID
        self.publications = {}
        
        # Hilo para publicaciones programadas
        self.scheduler_thread = None
        self.scheduler_stop_event = threading.Event()
        
        # Cargar publicaciones desde persistencia si está disponible
        self._load_publications_from_persistence()
        
        # Iniciar hilo de programación
        self._start_scheduler()
        
        logger.info("PublishManager inicializado")
    
    def _load_publications_from_persistence(self) -> None:
        """
        Carga publicaciones desde el sistema de persistencia.
        """
        if not self.persistence:
            logger.debug("No hay gestor de persistencia configurado")
            return
        
        try:
            publication_data = self.persistence.load_collection("publications")
            if publication_data:
                for item in publication_data:
                    # Convertir strings a enumeraciones
                    if "status" in item and isinstance(item["status"], str):
                        try:
                            item["status"] = PublishStatus[item["status"]]
                        except KeyError:
                            logger.warning(f"Estado de publicación inválido: {item['status']}")
                            item["status"] = PublishStatus.FAILED
                    
                    self.publications[item["id"]] = item
                
                logger.info(f"Cargadas {len(publication_data)} publicaciones desde persistencia")
        
        except Exception as e:
            logger.error(f"Error al cargar publicaciones desde persistencia: {str(e)}")
    
    def _save_publication_to_persistence(self, publication_id: str) -> None:
        """
        Guarda una publicación en el sistema de persistencia.
        
        Args:
            publication_id: ID de la publicación a guardar
        """
        if not self.persistence:
            return
        
        try:
            publication = self.publications.get(publication_id)
            if publication:
                # Convertir enumeraciones a strings para serialización
                serialized = publication.copy()
                if "status" in serialized and isinstance(serialized["status"], PublishStatus):
                    serialized["status"] = serialized["status"].name
                
                self.persistence.save_document("publications", publication_id, serialized)
        
        except Exception as e:
            logger.error(f"Error al guardar publicación {publication_id} en persistencia: {str(e)}")
    
    def _start_scheduler(self) -> None:
        """
        Inicia el hilo de programación para publicaciones programadas.
        """
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.debug("El hilo de programación ya está en ejecución")
            return
        
        self.scheduler_stop_event.clear()
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="PublishScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        logger.debug("Hilo de programación iniciado")
    
    def _scheduler_loop(self) -> None:
        """
        Bucle principal del hilo de programación.
        """
        logger.debug("Bucle de programación iniciado")
        
        while not self.scheduler_stop_event.is_set():
            try:
                now = datetime.datetime.now()
                
                # Buscar publicaciones programadas que deben ejecutarse
                for pub_id, publication in list(self.publications.items()):
                    if publication["status"] != PublishStatus.SCHEDULED:
                        continue
                    
                    scheduled_time = publication.get("scheduled_time")
                    if not scheduled_time:
                        continue
                    
                    try:
                        scheduled_datetime = datetime.datetime.fromisoformat(scheduled_time)
                        
                        # Si es hora de publicar
                        if scheduled_datetime <= now:
                            logger.info(f"Ejecutando publicación programada {pub_id}")
                            
                            # Actualizar estado
                            publication["status"] = PublishStatus.PUBLISHING
                            self._save_publication_to_persistence(pub_id)
                            
                            # Crear tarea para publicación
                            if self.task_manager:
                                from brain.orchestrator.core.task import TaskType, TaskPriority
                                
                                self.task_manager.create_task(
                                    task_type=TaskType.CONTENT_PUBLISHING,
                                    priority=TaskPriority.HIGH,
                                    channel_id=publication.get("channel_id"),
                                    data={
                                        "publication_id": pub_id,
                                        "content_id": publication.get("content_id"),
                                        "platform": publication.get("platform"),
                                        "params": publication.get("params", {})
                                    },
                                    callback=self._on_publish_task_complete
                                )
                            else:
                                # Si no hay task_manager, publicar directamente
                                threading.Thread(
                                    target=self._publish_content_direct,
                                    args=(pub_id,),
                                    daemon=True
                                ).start()
                    
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error al procesar fecha programada para publicación {pub_id}: {str(e)}")
                        publication["status"] = PublishStatus.FAILED
                        publication["error"] = f"Error en fecha programada: {str(e)}"
                        self._save_publication_to_persistence(pub_id)
            
            except Exception as e:
                logger.error(f"Error en bucle de programación: {str(e)}")
            
            # Esperar antes de la siguiente iteración
            time.sleep(10)  # Verificar cada 10 segundos
    
    def _publish_content_direct(self, publication_id: str) -> None:
        """
        Publica contenido directamente (sin usar el task_manager).
        
        Args:
            publication_id: ID de la publicación
        """
        publication = self.publications.get(publication_id)
        if not publication:
            logger.error(f"Publicación {publication_id} no encontrada")
            return
        
        try:
            # Obtener información necesaria
            content_id = publication.get("content_id")
            platform = publication.get("platform")
            params = publication.get("params", {})
            
            # Verificar que tenemos toda la información necesaria
            if not content_id or not platform:
                raise ValueError(f"Falta información necesaria: content_id={content_id}, platform={platform}")
            
            # Verificar que tenemos el adaptador de plataforma
            if platform not in self.platform_adapters:
                raise ValueError(f"Adaptador de plataforma no disponible: {platform}")
            
            # Obtener contenido
            if self.content_manager:
                content = self.content_manager.get_content(content_id)
                if not content:
                    raise ValueError(f"Contenido {content_id} no encontrado")
            else:
                # Si no hay content_manager, usar información de la publicación
                content = publication.get("content_data", {})
            
            # Publicar en plataforma
            platform_adapter = self.platform_adapters[platform]
            result = platform_adapter.publish_content(content, params)
            
            # Actualizar publicación con resultado
            publication["status"] = PublishStatus.PUBLISHED
            publication["platform_data"] = result
            publication["published_at"] = datetime.datetime.now().isoformat()
            publication["error"] = None
            
            # Actualizar contenido si hay content_manager
            if self.content_manager:
                self.content_manager.mark_as_published(content_id, {
                    "platform": platform,
                    "platform_data": result,
                    "publication_id": publication_id
                })
            
            logger.info(f"Contenido {content_id} publicado correctamente en {platform}")
        
        except Exception as e:
            logger.error(f"Error al publicar contenido: {str(e)}")
            
            # Actualizar publicación con error
            publication["status"] = PublishStatus.FAILED
            publication["error"] = str(e)
        
        # Guardar cambios
        self._save_publication_to_persistence(publication_id)
    
    def _on_publish_task_complete(self, task) -> None:
        """
        Callback para cuando se completa una tarea de publicación.
        
        Args:
            task: Tarea completada
        """
        publication_id = task.data.get("publication_id")
        if not publication_id or publication_id not in self.publications:
            logger.warning(f"Publicación {publication_id} no encontrada para tarea completada")
            return
        
        publication = self.publications[publication_id]
        
        if task.status == "COMPLETED":
            # Actualizar publicación con resultados
            publication["status"] = PublishStatus.PUBLISHED
            publication["platform_data"] = task.result.get("platform_data", {})
            publication["published_at"] = datetime.datetime.now().isoformat()
            publication["error"] = None
            
            # Actualizar contenido si hay content_manager
            content_id = publication.get("content_id")
            platform = publication.get("platform")
            if self.content_manager and content_id:
                self.content_manager.mark_as_published(content_id, {
                    "platform": platform,
                    "platform_data": task.result.get("platform_data", {}),
                    "publication_id": publication_id
                })
            
            logger.info(f"Publicación {publication_id} completada correctamente")
        else:
            # Marcar como fallida
            publication["status"] = PublishStatus.FAILED
            publication["error"] = task.error
            
            logger.error(f"Publicación {publication_id} fallida: {task.error}")
        
        # Persistir cambios
        self._save_publication_to_persistence(publication_id)
    
    def schedule_publication(self, content_id: str, platform: str, 
                           scheduled_time: Union[str, datetime.datetime],
                           params: Dict[str, Any] = None) -> str:
        """
        Programa la publicación de contenido para una fecha futura.
        
        Args:
            content_id: ID del contenido a publicar
            platform: Plataforma donde publicar
            scheduled_time: Fecha y hora programada (formato ISO o datetime)
            params: Parámetros específicos para la publicación
            
        Returns:
            str: ID de la publicación programada
        """
        # Verificar que el contenido existe si hay content_manager
        if self.content_manager:
            content = self.content_manager.get_content(content_id)
            if not content:
                logger.error(f"Contenido {content_id} no encontrado")
                raise ValueError(f"Contenido {content_id} no encontrado")
            
            # Verificar que el contenido está aprobado
            if content.get("status") != "APPROVED":
                logger.warning(f"Contenido {content_id} no está aprobado para publicación (estado: {content.get('status')})")
        
        # Convertir datetime a string ISO si es necesario
        if isinstance(scheduled_time, datetime.datetime):
            scheduled_time = scheduled_time.isoformat()
        
        # Crear ID único para la publicación
        publication_id = str(uuid.uuid4())
        
        # Obtener channel_id del contenido si está disponible
        channel_id = None
        if self.content_manager:
            content = self.content_manager.get_content(content_id)
            if content:
                channel_id = content.get("channel_id")
        
        # Crear estructura de la publicación
        publication = {
            "id": publication_id,
            "content_id": content_id,
            "platform": platform,
            "channel_id": channel_id,
            "status": PublishStatus.SCHEDULED,
            "scheduled_time": scheduled_time,
            "params": params or {},
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "published_at": None,
            "platform_data": None,
            "error": None
        }
        
        # Guardar en diccionario local
        self.publications[publication_id] = publication
        
        # Persistir
        self._save_publication_to_persistence(publication_id)
        
        logger.info(f"Publicación {publication_id} programada para {scheduled_time} en {platform}")
        return publication_id
    
    def publish_now(self, content_id: str, platform: str, params: Dict[str, Any] = None) -> str:
        """
        Publica contenido inmediatamente.
        
        Args:
            content_id: ID del contenido a publicar
            platform: Plataforma donde publicar
            params: Parámetros específicos para la publicación
            
        Returns:
            str: ID de la publicación
        """
        # Verificar que el contenido existe si hay content_manager
        if self.content_manager:
            content = self.content_manager.get_content(content_id)
            if not content:
                logger.error(f"Contenido {content_id} no encontrado")
                raise ValueError(f"Contenido {content_id} no encontrado")
            
            # Verificar que el contenido está aprobado
            if content.get("status") != "APPROVED":
                logger.warning(f"Contenido {content_id} no está aprobado para publicación (estado: {content.get('status')})")
        
        # Crear ID único para la publicación
        publication_id = str(uuid.uuid4())
        
        # Obtener channel_id del contenido si está disponible
        channel_id = None
        if self.content_manager:
            content = self.content_manager.get_content(content_id)
            if content:
                channel_id = content.get("channel_id")
        
        # Crear estructura de la publicación
        publication = {
            "id": publication_id,
            "content_id": content_id,
            "platform": platform,
            "channel_id": channel_id,
            "status": PublishStatus.PUBLISHING,
            "scheduled_time": None,
            "params": params or {},
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "published_at": None,
            "platform_data": None,
            "error": None
        }
        
        # Guardar en diccionario local
        self.publications[publication_id] = publication
        
        # Persistir
        self._save_publication_to_persistence(publication_id)
        
        # Crear tarea para publicación si hay task_manager
        if self.task_manager:
            from brain.orchestrator.core.task import TaskType, TaskPriority
            
            self.task_manager.create_task(
                task_type=TaskType.CONTENT_PUBLISHING,
                priority=TaskPriority.HIGH,
                channel_id=channel_id,
                data={
                    "publication_id": publication_id,
                    "content_id": content_id,
                    "platform": platform,
                    "params": params or {}
                },
                callback=self._on_publish_task_complete
            )
        else:
            # Si no hay task_manager, publicar directamente
            threading.Thread(
                target=self._publish_content_direct,
                args=(publication_id,),
                daemon=True
            ).start()
        
        logger.info(f"Publicación {publication_id} iniciada inmediatamente en {platform}")
        return publication_id
    
    def cancel_publication(self, publication_id: str) -> bool:
        """
        Cancela una publicación programada.
        
        Args:
            publication_id: ID de la publicación
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        publication = self.publications.get(publication_id)
        if not publication:
            logger.warning(f"Intento de cancelar publicación inexistente: {publication_id}")
            return False
        
        # Solo se pueden cancelar publicaciones programadas
        if publication["status"] != PublishStatus.SCHEDULED:
            logger.warning(f"No se puede cancelar publicación {publication_id} en estado {publication['status'].name}")
            return False
        
        # Actualizar estado
        publication["status"] = PublishStatus.CANCELED
        publication["updated_at"] = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_publication_to_persistence(publication_id)
        
        logger.info(f"Publicación {publication_id} cancelada")
        return True
    
    def get_publication(self, publication_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de una publicación.
        
        Args:
            publication_id: ID de la publicación
            
        Returns:
            Optional[Dict[str, Any]]: Información de la publicación o None si no existe
        """
        publication = self.publications.get(publication_id)
        if not publication:
            return None
        
        # Crear copia para no modificar el original
        result = publication.copy()
        
        # Convertir enumeraciones a strings para serialización
        if "status" in result and isinstance(result["status"], PublishStatus):
            result["status"] = result["status"].name
        
        return result
    
    def get_publications_by_content(self, content_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene todas las publicaciones para un contenido específico.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            List[Dict[str, Any]]: Lista de publicaciones
        """
        results = []
        
        for pub_id, publication in self.publications.items():
            if publication["content_id"] == content_id:
                # Crear copia para no modificar el original
                result = publication.copy()
                
                # Convertir enumeraciones a strings para serialización
                if "status" in result and isinstance(result["status"], PublishStatus):
                    result["status"] = result["status"].name
                
                results.append(result)
        
        return results
    
    def get_publications_by_channel(self, channel_id: str, 
                                  status: Union[PublishStatus, str, List[Union[PublishStatus, str]]] = None) -> List[Dict[str, Any]]:
        """
        Obtiene todas las publicaciones para un canal específico,
        opcionalmente filtradas por estado.
        
        Args:
            channel_id: ID del canal
            status: Estado o lista de estados para filtrar (opcional)
            
        Returns:
            List[Dict[str, Any]]: Lista de publicaciones
        """
        results = []
        
        # Convertir status a lista de PublishStatus si se proporciona
        status_list = []
        if status:
            if isinstance(status, (PublishStatus, str)):
                status_list = [status]
            else:
                status_list = status
            
            # Convertir strings a enums
            for i, s in enumerate(status_list):
                if isinstance(s, str):
                    try:
                        status_list[i] = PublishStatus[s.upper()]
                    except KeyError:
                        logger.warning(f"Estado de publicación inválido: {s}")
                        status_list[i] = None
            
            # Eliminar estados inválidos
            status_list = [s for s in status_list if s is not None]
        
        # Filtrar publicaciones
        for pub_id, publication in self.publications.items():
            if publication["channel_id"] != channel_id:
                continue
            
            if status_list and publication["status"] not in status_list:
                continue
            
            # Crear copia para no modificar el original
            result = publication.copy()
            
            # Convertir enumeraciones a strings para serialización
            if "status" in result and isinstance(result["status"], PublishStatus):
                result["status"] = result["status"].name
            
            results.append(result)
        
        return results
    
    def get_publication_stats(self, channel_id: str = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de publicaciones, opcionalmente filtradas por canal.
        
        Args:
            channel_id: ID del canal (opcional)
            
        Returns:
            Dict[str, Any]: Estadísticas de publicaciones
        """
        stats = {
            "total": 0,
            "by_status": {status.name: 0 for status in PublishStatus},
            "by_platform": {}
        }
        
        # Filtrar por canal si se proporciona
        publications = self.publications.values()
        if channel_id:
            publications = [p for p in publications if p["channel_id"] == channel_id]
        
        # Calcular estadísticas
        for publication in publications:
            stats["total"] += 1
            
            # Por estado
            status = publication["status"]
            if isinstance(status, PublishStatus):
                stats["by_status"][status.name] += 1
            
            # Por plataforma
            platform = publication.get("platform")
            if platform:
                if platform not in stats["by_platform"]:
                    stats["by_platform"][platform] = 0
                stats["by_platform"][platform] += 1
        
        return stats
    
    def search_publications(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Busca publicaciones según criterios específicos.
        
        Args:
            query: Diccionario con criterios de búsqueda
            
        Returns:
            List[Dict[str, Any]]: Lista de publicaciones que coinciden con los criterios
        """
        results = []
        
        for pub_id, publication in self.publications.items():
            match = True
            
            # Verificar cada criterio
            for key, value in query.items():
                if key not in publication:
                    match = False
                    break
                
                # Manejar casos especiales
                if key == "status" and isinstance(publication[key], PublishStatus):
                    if isinstance(value, str):
                        if publication[key].name != value.upper():
                            match = False
                            break
                    elif publication[key] != value:
                        match = False
                        break
                
                # Búsqueda en parámetros
                elif key == "params" and isinstance(value, dict):
                    pub_params = publication.get("params", {})
                    for param_key, param_value in value.items():
                        if param_key not in pub_params or pub_params[param_key] != param_value:
                            match = False
                            break
                
                # Comparación directa para otros campos
                elif publication[key] != value:
                    match = False
                    break
            
            if match:
                # Crear copia para no modificar el original
                result = publication.copy()
                
                # Convertir enumeraciones a strings para serialización
                if "status" in result and isinstance(result["status"], PublishStatus):
                    result["status"] = result["status"].name
                
                results.append(result)
        
        return results
    
    def bulk_update_publications(self, query: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Actualiza múltiples publicaciones que coinciden con los criterios.
        
        Args:
            query: Diccionario con criterios de búsqueda
            updates: Cambios a aplicar
            
        Returns:
            int: Número de publicaciones actualizadas
        """
        # Buscar publicaciones que coinciden con los criterios
        matching_publications = self.search_publications(query)
        updated_count = 0
        
        # Actualizar cada publicación
        for publication in matching_publications:
            pub_id = publication["id"]
            
            # Obtener publicación original
            original_pub = self.publications.get(pub_id)
            if not original_pub:
                continue
            
            # Aplicar actualizaciones
            for key, value in updates.items():
                # No permitir actualizar campos críticos
                if key in ["id", "content_id", "created_at"]:
                    continue
                
                # Manejar casos especiales
                if key == "status" and isinstance(value, str):
                    try:
                        original_pub[key] = PublishStatus[value.upper()]
                    except KeyError:
                        logger.warning(f"Estado de publicación inválido: {value}")
                        continue
                else:
                    original_pub[key] = value
            
            # Actualizar timestamp
            original_pub["updated_at"] = datetime.datetime.now().isoformat()
            
            # Persistir cambios
            self._save_publication_to_persistence(pub_id)
            
            updated_count += 1
        
        return updated_count
    
    def delete_publication(self, publication_id: str) -> bool:
        """
        Elimina una publicación de la plataforma y del sistema.
        
        Args:
            publication_id: ID de la publicación
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario
        """
        publication = self.publications.get(publication_id)
        if not publication:
            logger.warning(f"Intento de eliminar publicación inexistente: {publication_id}")
            return False
        
        # Solo se pueden eliminar publicaciones publicadas
        if publication["status"] not in [PublishStatus.PUBLISHED, PublishStatus.FAILED]:
            logger.warning(f"No se puede eliminar publicación {publication_id} en estado {publication['status'].name}")
            return False
        
        try:
            # Intentar eliminar de la plataforma si está publicada
            if publication["status"] == PublishStatus.PUBLISHED:
                platform = publication.get("platform")
                platform_data = publication.get("platform_data", {})
                
                if platform and platform in self.platform_adapters:
                    platform_adapter = self.platform_adapters[platform]
                    
                    # Verificar si el adaptador tiene método de eliminación
                    if hasattr(platform_adapter, "delete_content") and callable(platform_adapter.delete_content):
                        platform_adapter.delete_content(platform_data)
            
            # Actualizar estado
            publication["status"] = PublishStatus.DELETED
            publication["updated_at"] = datetime.datetime.now().isoformat()
            
            # Persistir
            self._save_publication_to_persistence(publication_id)
            
            logger.info(f"Publicación {publication_id} eliminada")
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar publicación {publication_id}: {str(e)}")
            return False
    
    def generate_publication_report(self, channel_id: str = None) -> Dict[str, Any]:
        """
        Genera un informe detallado sobre las publicaciones, opcionalmente filtrado por canal.
        
        Args:
            channel_id: ID del canal (opcional)
            
        Returns:
            Dict[str, Any]: Informe detallado
        """
        # Obtener estadísticas básicas
        stats = self.get_publication_stats(channel_id)
        
        # Filtrar publicaciones por canal si se proporciona
        publications = list(self.publications.values())
        if channel_id:
            publications = [p for p in publications if p["channel_id"] == channel_id]
        
        # Calcular métricas adicionales
        now = datetime.datetime.now()
        
        # Publicaciones por período de tiempo
        time_periods = {
            "last_24h": 0,
            "last_7d": 0,
            "last_30d": 0,
            "last_90d": 0
        }
        
        for publication in publications:
            published_at = publication.get("published_at")
            if not published_at:
                continue
            
            try:
                published_date = datetime.datetime.fromisoformat(published_at)
                delta = now - published_date
                
                if delta.days < 1:
                    time_periods["last_24h"] += 1
                if delta.days < 7:
                    time_periods["last_7d"] += 1
                if delta.days < 30:
                    time_periods["last_30d"] += 1
                if delta.days < 90:
                    time_periods["last_90d"] += 1
            except (ValueError, TypeError):
                logger.warning(f"Formato de fecha inválido para publicación {publication.get('id')}: {published_at}")
        
        # Tiempo promedio entre programación y publicación
        schedule_to_publish_times = []
        for publication in publications:
            scheduled_time = publication.get("scheduled_time")
            published_at = publication.get("published_at")
            
            if not scheduled_time or not published_at:
                continue
            
            try:
                scheduled_date = datetime.datetime.fromisoformat(scheduled_time)
                published_date = datetime.datetime.fromisoformat(published_at)
                delta = published_date - scheduled_date
                schedule_to_publish_times.append(delta.total_seconds())
            except (ValueError, TypeError):
                continue
        
        avg_schedule_to_publish = None
        if schedule_to_publish_times:
            avg_schedule_to_publish = sum(schedule_to_publish_times) / len(schedule_to_publish_times)
        
        # Tasa de éxito de publicaciones
        success_rate = 0
        if publications:
            successful_pubs = sum(1 for p in publications if p["status"] == PublishStatus.PUBLISHED)
            success_rate = (successful_pubs / len(publications)) * 100
        
        # Construir informe completo
        report = {
            "stats": stats,
            "time_periods": time_periods,
            "avg_schedule_to_publish_seconds": avg_schedule_to_publish,
            "success_rate_percent": success_rate,
            "total_publication_count": len(publications),
            "generated_at": now.isoformat()
        }
        
        if channel_id:
            report["channel_id"] = channel_id
        
        return report
    
    def stop(self) -> None:
        """
        Detiene el gestor de publicación y sus hilos.
        """
        logger.info("Deteniendo PublishManager...")
        
        # Detener hilo de programación
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_stop_event.set()
            self.scheduler_thread.join(timeout=5)
            if self.scheduler_thread.is_alive():
                logger.warning("No se pudo detener el hilo de programación limpiamente")
        
        logger.info("PublishManager detenido")