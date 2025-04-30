"""
ContentManager - Gestor de creación de contenido

Este módulo se encarga de coordinar la creación de contenido para diferentes plataformas,
gestionando la generación de guiones, imágenes, videos y otros elementos multimedia.
"""

import logging
import time
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum, auto

# Configuración de logging
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Enumeración de tipos de contenido."""
    VIDEO = auto()         # Video completo
    SHORT = auto()         # Video corto (shorts, reels, tiktok)
    POST = auto()          # Publicación de texto
    IMAGE = auto()         # Imagen o gráfico
    CAROUSEL = auto()      # Carrusel de imágenes
    STORY = auto()         # Historia temporal
    LIVESTREAM = auto()    # Transmisión en vivo
    PODCAST = auto()       # Episodio de podcast
    ARTICLE = auto()       # Artículo o blog

class ContentStatus(Enum):
    """Enumeración de estados de contenido."""
    DRAFT = auto()         # Borrador inicial
    GENERATING = auto()    # En proceso de generación
    REVIEW = auto()        # Listo para revisión
    APPROVED = auto()      # Aprobado para publicación
    REJECTED = auto()      # Rechazado (requiere cambios)
    PUBLISHED = auto()     # Publicado
    ARCHIVED = auto()      # Archivado
    FAILED = auto()        # Falló en generación

class ContentManager:
    """
    Gestor de creación de contenido para el Orchestrator.
    
    Esta clase se encarga de coordinar la creación de diferentes tipos de contenido,
    gestionando el flujo desde la idea inicial hasta el contenido listo para publicar.
    """
    
    def __init__(self, config_manager=None, persistence_manager=None, task_manager=None):
        """
        Inicializa el gestor de contenido.
        
        Args:
            config_manager: Gestor de configuración
            persistence_manager: Gestor de persistencia
            task_manager: Gestor de tareas
        """
        self.config = config_manager
        self.persistence = persistence_manager
        self.task_manager = task_manager
        self.content_items = {}  # Diccionario de contenidos por ID
        
        # Cargar contenidos desde persistencia si está disponible
        self._load_content_from_persistence()
        
        logger.info("ContentManager inicializado")
    
    def _load_content_from_persistence(self) -> None:
        """
        Carga contenidos desde el sistema de persistencia.
        """
        if not self.persistence:
            logger.debug("No hay gestor de persistencia configurado")
            return
        
        try:
            content_data = self.persistence.load_collection("content")
            if content_data:
                for item in content_data:
                    # Convertir strings a enumeraciones
                    if "type" in item and isinstance(item["type"], str):
                        item["type"] = ContentType[item["type"]]
                    if "status" in item and isinstance(item["status"], str):
                        item["status"] = ContentStatus[item["status"]]
                    
                    self.content_items[item["id"]] = item
                
                logger.info(f"Cargados {len(content_data)} elementos de contenido desde persistencia")
        
        except Exception as e:
            logger.error(f"Error al cargar contenidos desde persistencia: {str(e)}")
    
    def _save_content_to_persistence(self, content_id: str) -> None:
        """
        Guarda un elemento de contenido en el sistema de persistencia.
        
        Args:
            content_id: ID del contenido a guardar
        """
        if not self.persistence:
            return
        
        try:
            content = self.content_items.get(content_id)
            if content:
                # Convertir enumeraciones a strings para serialización
                serialized = content.copy()
                if "type" in serialized and isinstance(serialized["type"], ContentType):
                    serialized["type"] = serialized["type"].name
                if "status" in serialized and isinstance(serialized["status"], ContentStatus):
                    serialized["status"] = serialized["status"].name
                
                self.persistence.save_document("content", content_id, serialized)
        
        except Exception as e:
            logger.error(f"Error al guardar contenido {content_id} en persistencia: {str(e)}")
    
    def create_content(self, channel_id: str, content_type: Union[ContentType, str], 
                      params: Dict[str, Any] = None) -> str:
        """
        Crea un nuevo elemento de contenido.
        
        Args:
            channel_id: ID del canal para el que se crea el contenido
            content_type: Tipo de contenido a crear
            params: Parámetros específicos para la creación
            
        Returns:
            str: ID del contenido creado
        """
        # Convertir string a enum si es necesario
        if isinstance(content_type, str):
            try:
                content_type = ContentType[content_type.upper()]
            except KeyError:
                logger.error(f"Tipo de contenido inválido: {content_type}")
                raise ValueError(f"Tipo de contenido inválido: {content_type}")
        
        # Crear ID único para el contenido
        content_id = str(uuid.uuid4())
        
        # Crear estructura base del contenido
        content = {
            "id": content_id,
            "channel_id": channel_id,
            "type": content_type,
            "status": ContentStatus.DRAFT,
            "params": params or {},
            "assets": {},
            "metadata": {},
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "published_at": None,
            "error": None
        }
        
        # Guardar en diccionario local
        self.content_items[content_id] = content
        
        # Persistir
        self._save_content_to_persistence(content_id)
        
        # Crear tarea para generación de contenido si hay task_manager
        if self.task_manager:
            from brain.orchestrator.core.task import TaskType, TaskPriority
            
            self.task_manager.create_task(
                task_type=TaskType.CONTENT_CREATION,
                priority=TaskPriority.NORMAL,
                channel_id=channel_id,
                data={
                    "content_id": content_id,
                    "content_type": content_type.name,
                    "params": params or {}
                },
                callback=self._on_content_generation_complete
            )
        
        logger.info(f"Contenido {content_id} creado para canal {channel_id}: {content_type.name}")
        return content_id
    
    def _on_content_generation_complete(self, task) -> None:
        """
        Callback para cuando se completa una tarea de generación de contenido.
        
        Args:
            task: Tarea completada
        """
        content_id = task.data.get("content_id")
        if not content_id or content_id not in self.content_items:
            logger.warning(f"Contenido {content_id} no encontrado para tarea completada")
            return
        
        content = self.content_items[content_id]
        
        if task.status == "COMPLETED":
            # Actualizar contenido con resultados
            content["status"] = ContentStatus.REVIEW
            content["assets"] = task.result.get("assets", {})
            content["metadata"] = task.result.get("metadata", {})
            content["updated_at"] = datetime.datetime.now().isoformat()
            
            logger.info(f"Generación de contenido {content_id} completada")
        else:
            # Marcar como fallido
            content["status"] = ContentStatus.FAILED
            content["error"] = task.error
            content["updated_at"] = datetime.datetime.now().isoformat()
            
            logger.error(f"Generación de contenido {content_id} fallida: {task.error}")
        
        # Persistir cambios
        self._save_content_to_persistence(content_id)
    
    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un elemento de contenido.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            Optional[Dict[str, Any]]: Información del contenido o None si no existe
        """
        content = self.content_items.get(content_id)
        if not content:
            return None
        
        # Crear copia para no modificar el original
        result = content.copy()
        
        # Convertir enumeraciones a strings para serialización
        if "type" in result and isinstance(result["type"], ContentType):
            result["type"] = result["type"].name
        if "status" in result and isinstance(result["status"], ContentStatus):
            result["status"] = result["status"].name
        
        return result
    
    def update_content(self, content_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualiza un elemento de contenido existente.
        
        Args:
            content_id: ID del contenido
            updates: Cambios a aplicar
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario
        """
        content = self.content_items.get(content_id)
        if not content:
            logger.warning(f"Intento de actualizar contenido inexistente: {content_id}")
            return False
        
        # Aplicar actualizaciones
        for key, value in updates.items():
            if key in ["id", "created_at"]:
                # No permitir cambiar estos campos
                continue
            
            # Manejar campos especiales
            if key == "type" and isinstance(value, str):
                try:
                    content[key] = ContentType[value.upper()]
                except KeyError:
                    logger.warning(f"Tipo de contenido inválido: {value}")
                    continue
            elif key == "status" and isinstance(value, str):
                try:
                    content[key] = ContentStatus[value.upper()]
                except KeyError:
                    logger.warning(f"Estado de contenido inválido: {value}")
                    continue
            else:
                content[key] = value
        
        # Actualizar timestamp
        content["updated_at"] = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_content_to_persistence(content_id)
        
        logger.info(f"Contenido {content_id} actualizado")
        return True
    
    def delete_content(self, content_id: str) -> bool:
        """
        Elimina un elemento de contenido.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario
        """
        if content_id not in self.content_items:
            logger.warning(f"Intento de eliminar contenido inexistente: {content_id}")
            return False
        
        # Eliminar del diccionario
        del self.content_items[content_id]
        
        # Eliminar de persistencia
        if self.persistence:
            try:
                self.persistence.delete_document("content", content_id)
            except Exception as e:
                logger.error(f"Error al eliminar contenido {content_id} de persistencia: {str(e)}")
        
        logger.info(f"Contenido {content_id} eliminado")
        return True
    
    def approve_content(self, content_id: str) -> bool:
        """
        Aprueba un elemento de contenido para publicación.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            bool: True si se aprobó correctamente, False en caso contrario
        """
        content = self.content_items.get(content_id)
        if not content:
            logger.warning(f"Intento de aprobar contenido inexistente: {content_id}")
            return False
        
        # Verificar que el contenido está en revisión
        if content["status"] != ContentStatus.REVIEW:
            logger.warning(f"No se puede aprobar contenido {content_id} en estado {content['status'].name}")
            return False
        
        # Actualizar estado
        content["status"] = ContentStatus.APPROVED
        content["updated_at"] = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_content_to_persistence(content_id)
        
        logger.info(f"Contenido {content_id} aprobado para publicación")
        return True
    
    def reject_content(self, content_id: str, reason: str = None) -> bool:
        """
        Rechaza un elemento de contenido.
        
        Args:
            content_id: ID del contenido
            reason: Motivo del rechazo (opcional)
            
        Returns:
            bool: True si se rechazó correctamente, False en caso contrario
        """
        content = self.content_items.get(content_id)
        if not content:
            logger.warning(f"Intento de rechazar contenido inexistente: {content_id}")
            return False
        
        # Verificar que el contenido está en revisión
        if content["status"] != ContentStatus.REVIEW:
            logger.warning(f"No se puede rechazar contenido {content_id} en estado {content['status'].name}")
            return False
        
        # Actualizar estado
        content["status"] = ContentStatus.REJECTED
        content["updated_at"] = datetime.datetime.now().isoformat()
        
        # Añadir motivo si se proporciona
        if reason:
            if "metadata" not in content:
                content["metadata"] = {}
            
            content["metadata"]["rejection_reason"] = reason
        
        # Persistir
        self._save_content_to_persistence(content_id)
        
        logger.info(f"Contenido {content_id} rechazado: {reason or 'Sin motivo especificado'}")
        return True
    
    def mark_as_published(self, content_id: str, platform_data: Dict[str, Any] = None) -> bool:
        """
        Marca un elemento de contenido como publicado.
        
        Args:
            content_id: ID del contenido
            platform_data: Datos específicos de la plataforma (URLs, IDs, etc.)
            
        Returns:
            bool: True si se marcó correctamente, False en caso contrario
        """
        content = self.content_items.get(content_id)
        if not content:
            logger.warning(f"Intento de marcar como publicado contenido inexistente: {content_id}")
            return False
        
        # Verificar que el contenido está aprobado
        if content["status"] != ContentStatus.APPROVED:
            logger.warning(f"No se puede marcar como publicado contenido {content_id} en estado {content['status'].name}")
            return False
        
        # Actualizar estado
        content["status"] = ContentStatus.PUBLISHED
        content["published_at"] = datetime.datetime.now().isoformat()
        content["updated_at"] = datetime.datetime.now().isoformat()
        
        # Añadir datos de plataforma si se proporcionan
        if platform_data:
            if "metadata" not in content:
                content["metadata"] = {}
            
            content["metadata"]["platform_data"] = platform_data
        
        # Persistir
        self._save_content_to_persistence(content_id)
        
        logger.info(f"Contenido {content_id} marcado como publicado")
        return True
    
    def archive_content(self, content_id: str) -> bool:
        """
        Archiva un elemento de contenido.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            bool: True si se archivó correctamente, False en caso contrario
        """
        content = self.content_items.get(content_id)
        if not content:
            logger.warning(f"Intento de archivar contenido inexistente: {content_id}")
            return False
        
        # Actualizar estado
        content["status"] = ContentStatus.ARCHIVED
        content["updated_at"] = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_content_to_persistence(content_id)
        
        logger.info(f"Contenido {content_id} archivado")
        return True
    
    def get_content_by_channel(self, channel_id: str, status: Union[ContentStatus, str, List[Union[ContentStatus, str]]] = None) -> List[Dict[str, Any]]:
        """
        Obtiene todos los elementos de contenido para un canal específico,
        opcionalmente filtrados por estado.
        
        Args:
            channel_id: ID del canal
            status: Estado o lista de estados para filtrar (opcional)
            
        Returns:
            List[Dict[str, Any]]: Lista de elementos de contenido
        """
        results = []
        
        # Convertir status a lista de ContentStatus si se proporciona
        status_list = []
        if status:
            if isinstance(status, (ContentStatus, str)):
                status_list = [status]
            else:
                status_list = status
            
            # Convertir strings a enums
            for i, s in enumerate(status_list):
                if isinstance(s, str):
                    try:
                        status_list[i] = ContentStatus[s.upper()]
                    except KeyError:
                        logger.warning(f"Estado de contenido inválido: {s}")
                        status_list[i] = None
            
            # Eliminar estados inválidos
            status_list = [s for s in status_list if s is not None]
        
        # Filtrar contenidos
        for content_id, content in self.content_items.items():
            if content["channel_id"] != channel_id:
                continue
            
            if status_list and content["status"] not in status_list:
                continue
            
            # Crear copia para no modificar el original
            result = content.copy()
            
            # Convertir enumeraciones a strings para serialización
            if "type" in result and isinstance(result["type"], ContentType):
                result["type"] = result["type"].name
            if "status" in result and isinstance(result["status"], ContentStatus):
                result["status"] = result["status"].name
            
            results.append(result)
        
        return results
    
    def get_content_stats(self, channel_id: str = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de contenido, opcionalmente filtradas por canal.
        
        Args:
            channel_id: ID del canal (opcional)
            
        Returns:
            Dict[str, Any]: Estadísticas de contenido
        """
        stats = {
            "total": 0,
            "by_status": {status.name: 0 for status in ContentStatus},
            "by_type": {content_type.name: 0 for content_type in ContentType}
        }
        
        # Filtrar por canal si se proporciona
        contents = self.content_items.values()
        if channel_id:
            contents = [c for c in contents if c["channel_id"] == channel_id]
        
        # Calcular estadísticas
        for content in contents:
            stats["total"] += 1
            
            # Por estado
            status = content["status"]
            if isinstance(status, ContentStatus):
                stats["by_status"][status.name] += 1
            
            # Por tipo
            content_type = content["type"]
            if isinstance(content_type, ContentType):
                stats["by_type"][content_type.name] += 1
        
        return stats
    
    def search_content(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Busca contenido según criterios específicos.
        
        Args:
            query: Diccionario con criterios de búsqueda
            
        Returns:
            List[Dict[str, Any]]: Lista de contenidos que coinciden con los criterios
        """
        results = []
        
        for content_id, content in self.content_items.items():
            match = True
            
            # Verificar cada criterio
            for key, value in query.items():
                if key not in content:
                    match = False
                    break
                
                # Manejar casos especiales
                if key == "type" and isinstance(content[key], ContentType):
                    if isinstance(value, str):
                        if content[key].name != value.upper():
                            match = False
                            break
                    elif content[key] != value:
                        match = False
                        break
                
                elif key == "status" and isinstance(content[key], ContentStatus):
                    if isinstance(value, str):
                        if content[key].name != value.upper():
                            match = False
                            break
                    elif content[key] != value:
                        match = False
                        break
                
                # Búsqueda en metadatos
                elif key == "metadata" and isinstance(value, dict):
                    content_metadata = content.get("metadata", {})
                    for meta_key, meta_value in value.items():
                        if meta_key not in content_metadata or content_metadata[meta_key] != meta_value:
                            match = False
                            break
                
                # Comparación directa para otros campos
                elif content[key] != value:
                    match = False
                    break
            
            if match:
                # Crear copia para no modificar el original
                result = content.copy()
                
                # Convertir enumeraciones a strings para serialización
                if "type" in result and isinstance(result["type"], ContentType):
                    result["type"] = result["type"].name
                if "status" in result and isinstance(result["status"], ContentStatus):
                    result["status"] = result["status"].name
                
                results.append(result)
        
        return results
    
    def bulk_update_content(self, query: Dict[str, Any], updates: Dict[str, Any]) -> int:
        """
        Actualiza múltiples elementos de contenido que coinciden con los criterios.
        
        Args:
            query: Diccionario con criterios de búsqueda
            updates: Cambios a aplicar
            
        Returns:
            int: Número de elementos actualizados
        """
        # Buscar contenidos que coinciden con los criterios
        matching_contents = self.search_content(query)
        updated_count = 0
        
        # Actualizar cada contenido
        for content in matching_contents:
            content_id = content["id"]
            if self.update_content(content_id, updates):
                updated_count += 1
        
        return updated_count
    
    def generate_content_report(self, channel_id: str = None) -> Dict[str, Any]:
        """
        Genera un informe detallado sobre el contenido, opcionalmente filtrado por canal.
        
        Args:
            channel_id: ID del canal (opcional)
            
        Returns:
            Dict[str, Any]: Informe detallado
        """
        # Obtener estadísticas básicas
        stats = self.get_content_stats(channel_id)
        
        # Filtrar contenidos por canal si se proporciona
        contents = list(self.content_items.values())
        if channel_id:
            contents = [c for c in contents if c["channel_id"] == channel_id]
        
        # Calcular métricas adicionales
        now = datetime.datetime.now()
        
        # Contenido por período de tiempo
        time_periods = {
            "last_24h": 0,
            "last_7d": 0,
            "last_30d": 0,
            "last_90d": 0
        }
        
        for content in contents:
            created_at = content.get("created_at")
            if not created_at:
                continue
            
            try:
                created_date = datetime.datetime.fromisoformat(created_at)
                delta = now - created_date
                
                if delta.days < 1:
                    time_periods["last_24h"] += 1
                if delta.days < 7:
                    time_periods["last_7d"] += 1
                if delta.days < 30:
                    time_periods["last_30d"] += 1
                if delta.days < 90:
                    time_periods["last_90d"] += 1
            except (ValueError, TypeError):
                logger.warning(f"Formato de fecha inválido para contenido {content.get('id')}: {created_at}")
        
        # Tiempo promedio entre creación y publicación
        creation_to_publish_times = []
        for content in contents:
            created_at = content.get("created_at")
            published_at = content.get("published_at")
            
            if not created_at or not published_at:
                continue
            
            try:
                created_date = datetime.datetime.fromisoformat(created_at)
                published_date = datetime.datetime.fromisoformat(published_at)
                delta = published_date - created_date
                creation_to_publish_times.append(delta.total_seconds())
            except (ValueError, TypeError):
                continue
        
        avg_creation_to_publish = None
        if creation_to_publish_times:
            avg_creation_to_publish = sum(creation_to_publish_times) / len(creation_to_publish_times)
        
        # Construir informe completo
        report = {
            "stats": stats,
            "time_periods": time_periods,
            "avg_creation_to_publish_seconds": avg_creation_to_publish,
            "total_content_count": len(contents),
            "generated_at": now.isoformat()
        }
        
        if channel_id:
            report["channel_id"] = channel_id
        
        return report