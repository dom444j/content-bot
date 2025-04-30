"""
Channel - Gestión de canales para el Orchestrator

Este módulo implementa la gestión de canales para el Orchestrator, permitiendo
crear, actualizar, eliminar y consultar canales en diferentes plataformas.
"""

import uuid
import logging
import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict

# Configuración de logging
logger = logging.getLogger(__name__)

class ChannelStatus(Enum):
    """Enumeración de estados de canales."""
    ACTIVE = auto()         # Canal activo y funcionando normalmente
    SHADOWBANNED = auto()   # Canal detectado como shadowbanned
    RECOVERING = auto()     # Canal en proceso de recuperación de shadowban
    PAUSED = auto()         # Canal pausado temporalmente
    ARCHIVED = auto()       # Canal archivado (inactivo permanentemente)
    PENDING = auto()        # Canal en proceso de creación/verificación
    SUSPENDED = auto()      # Canal suspendido por la plataforma

class ChannelType(Enum):
    """Enumeración de tipos de canales."""
    YOUTUBE = auto()        # Canal de YouTube
    TIKTOK = auto()         # Cuenta de TikTok
    INSTAGRAM = auto()      # Cuenta de Instagram
    THREADS = auto()        # Cuenta de Threads
    BLUESKY = auto()        # Cuenta de Bluesky
    X = auto()              # Cuenta de Twitter/X
    FACEBOOK = auto()       # Página de Facebook
    LINKEDIN = auto()       # Perfil de LinkedIn
    PINTEREST = auto()      # Cuenta de Pinterest
    TWITCH = auto()         # Canal de Twitch

class ChannelNiche(Enum):
    """Enumeración de nichos de canales."""
    TECH = auto()           # Tecnología
    FINANCE = auto()        # Finanzas
    HEALTH = auto()         # Salud y bienestar
    GAMING = auto()         # Videojuegos
    EDUCATION = auto()      # Educación
    ENTERTAINMENT = auto()  # Entretenimiento
    LIFESTYLE = auto()      # Estilo de vida
    TRAVEL = auto()         # Viajes
    FOOD = auto()           # Comida y cocina
    FASHION = auto()        # Moda y belleza
    SPORTS = auto()         # Deportes
    BUSINESS = auto()       # Negocios y emprendimiento
    SCIENCE = auto()        # Ciencia
    SPIRITUALITY = auto()   # Espiritualidad y religión
    POLITICS = auto()       # Política
    COMEDY = auto()         # Comedia y humor
    MUSIC = auto()          # Música
    ART = auto()            # Arte y diseño
    DIY = auto()            # Bricolaje y manualidades
    PETS = auto()           # Mascotas y animales

@dataclass
class Channel:
    """
    Clase que representa un canal en el sistema.
    
    Attributes:
        id: Identificador único del canal
        name: Nombre del canal
        type: Tipo de canal (plataforma)
        niche: Nicho del canal
        status: Estado actual del canal
        credentials: Credenciales de acceso (encriptadas)
        platform_id: ID del canal en la plataforma
        platform_url: URL del canal en la plataforma
        metrics: Métricas del canal
        content_strategy: Estrategia de contenido
        monetization_strategy: Estrategia de monetización
        created_at: Timestamp de creación
        updated_at: Timestamp de última actualización
        last_published_at: Timestamp de última publicación
        shadowban_history: Historial de shadowbans
        recovery_plan: Plan de recuperación (si está en recuperación)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: ChannelType = ChannelType.YOUTUBE
    niche: ChannelNiche = ChannelNiche.TECH
    status: ChannelStatus = ChannelStatus.PENDING
    credentials: Dict[str, Any] = field(default_factory=dict)
    platform_id: Optional[str] = None
    platform_url: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    content_strategy: Dict[str, Any] = field(default_factory=dict)
    monetization_strategy: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_published_at: Optional[str] = None
    shadowban_history: List[Dict[str, Any]] = field(default_factory=list)
    recovery_plan: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el canal a un diccionario serializable.
        
        Returns:
            Dict[str, Any]: Representación del canal como diccionario
        """
        result = asdict(self)
        # Convertir enumeraciones a strings para serialización
        result["type"] = self.type.name
        result["niche"] = self.niche.name
        result["status"] = self.status.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Channel':
        """
        Crea un canal a partir de un diccionario.
        
        Args:
            data: Diccionario con datos del canal
            
        Returns:
            Channel: Instancia de canal
        """
        # Convertir strings a enumeraciones
        if "type" in data and isinstance(data["type"], str):
            data["type"] = ChannelType[data["type"]]
        if "niche" in data and isinstance(data["niche"], str):
            data["niche"] = ChannelNiche[data["niche"]]
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ChannelStatus[data["status"]]
        
        return cls(**data)
    
    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """
        Actualiza las métricas del canal.
        
        Args:
            new_metrics: Nuevas métricas a añadir/actualizar
        """
        self.metrics.update(new_metrics)
        self.updated_at = datetime.datetime.now().isoformat()
    
    def add_shadowban_event(self, event: Dict[str, Any]) -> None:
        """
        Añade un evento de shadowban al historial.
        
        Args:
            event: Datos del evento de shadowban
        """
        if "detected_at" not in event:
            event["detected_at"] = datetime.datetime.now().isoformat()
        
        self.shadowban_history.append(event)
        self.updated_at = datetime.datetime.now().isoformat()
    
    def set_recovery_plan(self, plan: Dict[str, Any]) -> None:
        """
        Establece un plan de recuperación para el canal.
        
        Args:
            plan: Plan de recuperación
        """
        self.recovery_plan = plan
        self.status = ChannelStatus.RECOVERING
        self.updated_at = datetime.datetime.now().isoformat()
    
    def clear_recovery_plan(self) -> None:
        """
        Elimina el plan de recuperación y marca el canal como activo.
        """
        self.recovery_plan = None
        self.status = ChannelStatus.ACTIVE
        self.updated_at = datetime.datetime.now().isoformat()

class ChannelManager:
    """
    Gestor de canales para el Orchestrator.
    
    Esta clase se encarga de crear, actualizar, eliminar y consultar canales,
    así como de gestionar su estado y persistencia.
    """
    
    def __init__(self, persistence_manager=None):
        """
        Inicializa el gestor de canales.
        
        Args:
            persistence_manager: Gestor de persistencia para guardar canales
        """
        self.channels = {}  # Diccionario de canales por ID
        self.persistence = persistence_manager
        
        # Cargar canales desde persistencia si está disponible
        self._load_channels_from_persistence()
        
        logger.info("ChannelManager inicializado")
    
    def _load_channels_from_persistence(self) -> None:
        """
        Carga canales desde el sistema de persistencia.
        """
        if not self.persistence:
            logger.debug("No hay gestor de persistencia configurado")
            return
        
        try:
            channels_data = self.persistence.load_collection("channels")
            if channels_data:
                for channel_data in channels_data:
                    channel = Channel.from_dict(channel_data)
                    self.channels[channel.id] = channel
                
                logger.info(f"Cargados {len(channels_data)} canales desde persistencia")
        
        except Exception as e:
            logger.error(f"Error al cargar canales desde persistencia: {str(e)}")
    
    def _save_channel_to_persistence(self, channel: Channel) -> None:
        """
        Guarda un canal en el sistema de persistencia.
        
        Args:
            channel: Canal a guardar
        """
        if not self.persistence:
            return
        
        try:
            self.persistence.save_document("channels", channel.id, channel.to_dict())
        except Exception as e:
            logger.error(f"Error al guardar canal {channel.id} en persistencia: {str(e)}")
    
    def create_channel(self, name: str, channel_type: ChannelType, niche: ChannelNiche,
                      credentials: Optional[Dict[str, Any]] = None,
                      platform_id: Optional[str] = None,
                      platform_url: Optional[str] = None,
                      content_strategy: Optional[Dict[str, Any]] = None,
                      monetization_strategy: Optional[Dict[str, Any]] = None) -> Channel:
        """
        Crea un nuevo canal.
        
        Args:
            name: Nombre del canal
            channel_type: Tipo de canal (plataforma)
            niche: Nicho del canal
            credentials: Credenciales de acceso (opcional)
            platform_id: ID del canal en la plataforma (opcional)
            platform_url: URL del canal en la plataforma (opcional)
            content_strategy: Estrategia de contenido (opcional)
            monetization_strategy: Estrategia de monetización (opcional)
            
        Returns:
            Channel: Canal creado
        """
        channel = Channel(
            name=name,
            type=channel_type,
            niche=niche,
            credentials=credentials or {},
            platform_id=platform_id,
            platform_url=platform_url,
            content_strategy=content_strategy or {},
            monetization_strategy=monetization_strategy or {}
        )
        
        # Guardar canal
        self.channels[channel.id] = channel
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Canal {channel.id} creado: {name}, tipo {channel_type.name}, nicho {niche.name}")
        
        return channel
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """
        Obtiene un canal por su ID.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Optional[Channel]: Canal encontrado o None si no existe
        """
        return self.channels.get(channel_id)
    
    def update_channel(self, channel_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualiza un canal existente.
        
        Args:
            channel_id: ID del canal
            updates: Diccionario con los campos a actualizar
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de actualizar canal inexistente: {channel_id}")
            return False
        
        # Actualizar campos
        for key, value in updates.items():
            if hasattr(channel, key):
                # Manejar enumeraciones
                if key == "type" and isinstance(value, str):
                    try:
                        value = ChannelType[value]
                    except KeyError:
                        logger.warning(f"Tipo de canal inválido: {value}")
                        continue
                elif key == "niche" and isinstance(value, str):
                    try:
                        value = ChannelNiche[value]
                    except KeyError:
                        logger.warning(f"Nicho de canal inválido: {value}")
                        continue
                elif key == "status" and isinstance(value, str):
                    try:
                        value = ChannelStatus[value]
                    except KeyError:
                        logger.warning(f"Estado de canal inválido: {value}")
                        continue
                
                setattr(channel, key, value)
            else:
                logger.warning(f"Campo desconocido en actualización de canal: {key}")
        
        # Actualizar timestamp
        channel.updated_at = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Canal {channel_id} actualizado")
        return True
    
    def delete_channel(self, channel_id: str) -> bool:
        """
        Elimina un canal.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            bool: True si la eliminación fue exitosa, False en caso contrario
        """
        if channel_id not in self.channels:
            logger.warning(f"Intento de eliminar canal inexistente: {channel_id}")
            return False
        
        # Eliminar canal
        del self.channels[channel_id]
        
        # Eliminar de persistencia
        if self.persistence:
            try:
                self.persistence.delete_document("channels", channel_id)
            except Exception as e:
                logger.error(f"Error al eliminar canal {channel_id} de persistencia: {str(e)}")
        
        logger.info(f"Canal {channel_id} eliminado")
        return True
    
    def get_all_channels(self, status: Optional[ChannelStatus] = None,
                        channel_type: Optional[ChannelType] = None,
                        niche: Optional[ChannelNiche] = None) -> List[Channel]:
        """
        Obtiene todos los canales, opcionalmente filtrados.
        
        Args:
            status: Filtrar por estado (opcional)
            channel_type: Filtrar por tipo (opcional)
            niche: Filtrar por nicho (opcional)
            
        Returns:
            List[Channel]: Lista de canales que cumplen los filtros
        """
        filtered_channels = []
        
        for channel in self.channels.values():
            # Aplicar filtros
            if status and channel.status != status:
                continue
            if channel_type and channel.type != channel_type:
                continue
            if niche and channel.niche != niche:
                continue
            
            filtered_channels.append(channel)
        
        return filtered_channels
    
    def get_channel_count(self, status: Optional[ChannelStatus] = None) -> int:
        """
        Obtiene el número de canales, opcionalmente filtrados por estado.
        
        Args:
            status: Estado para filtrar (opcional)
            
        Returns:
            int: Número de canales
        """
        if status:
            return sum(1 for channel in self.channels.values() if channel.status == status)
        return len(self.channels)
    
    def update_channel_status(self, channel_id: str, status: ChannelStatus) -> bool:
        """
        Actualiza el estado de un canal.
        
        Args:
            channel_id: ID del canal
            status: Nuevo estado
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de actualizar estado de canal inexistente: {channel_id}")
            return False
        
        # Actualizar estado
        channel.status = status
        channel.updated_at = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Estado de canal {channel_id} actualizado a {status.name}")
        return True
    
    def mark_channel_shadowbanned(self, channel_id: str, details: Dict[str, Any]) -> bool:
        """
        Marca un canal como shadowbanned y registra el evento.
        
        Args:
            channel_id: ID del canal
            details: Detalles del shadowban
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de marcar como shadowbanned canal inexistente: {channel_id}")
            return False
        
        # Añadir evento al historial
        event = {
            "detected_at": datetime.datetime.now().isoformat(),
            "details": details
        }
        channel.add_shadowban_event(event)
        
        # Actualizar estado
        channel.status = ChannelStatus.SHADOWBANNED
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.warning(f"Canal {channel_id} marcado como shadowbanned: {details.get('reason', 'Sin razón especificada')}")
        return True
    
    def set_recovery_plan(self, channel_id: str, plan: Dict[str, Any]) -> bool:
        """
        Establece un plan de recuperación para un canal shadowbanned.
        
        Args:
            channel_id: ID del canal
            plan: Plan de recuperación
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de establecer plan de recuperación para canal inexistente: {channel_id}")
            return False
        
        # Verificar que el canal está shadowbanned
        if channel.status != ChannelStatus.SHADOWBANNED:
            logger.warning(f"Intento de establecer plan de recuperación para canal no shadowbanned: {channel_id}")
            return False
        
        # Establecer plan de recuperación
        channel.set_recovery_plan(plan)
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Plan de recuperación establecido para canal {channel_id}")
        return True
    
    def mark_channel_recovered(self, channel_id: str) -> bool:
        """
        Marca un canal como recuperado de shadowban.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de marcar como recuperado canal inexistente: {channel_id}")
            return False
        
        # Verificar que el canal está en recuperación
        if channel.status != ChannelStatus.RECOVERING:
            logger.warning(f"Intento de marcar como recuperado canal no en recuperación: {channel_id}")
            return False
        
        # Limpiar plan de recuperación y actualizar estado
        channel.clear_recovery_plan()
        
        # Añadir evento al historial
        event = {
            "recovered_at": datetime.datetime.now().isoformat(),
            "details": {
                "days_in_recovery": (datetime.datetime.now() - 
                                    datetime.datetime.fromisoformat(channel.shadowban_history[-1]["detected_at"])).days
            }
        }
        channel.shadowban_history[-1].update(event)
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Canal {channel_id} marcado como recuperado de shadowban")
        return True
    
    def update_channel_metrics(self, channel_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Actualiza las métricas de un canal.
        
        Args:
            channel_id: ID del canal
            metrics: Nuevas métricas a añadir/actualizar
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de actualizar métricas de canal inexistente: {channel_id}")
            return False
        
        # Actualizar métricas
        channel.update_metrics(metrics)
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.debug(f"Métricas actualizadas para canal {channel_id}")
        return True
    
    def update_content_strategy(self, channel_id: str, strategy: Dict[str, Any]) -> bool:
        """
        Actualiza la estrategia de contenido de un canal.
        
        Args:
            channel_id: ID del canal
            strategy: Nueva estrategia de contenido
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de actualizar estrategia de canal inexistente: {channel_id}")
            return False
        
        # Actualizar estrategia
        channel.content_strategy.update(strategy)
        channel.updated_at = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Estrategia de contenido actualizada para canal {channel_id}")
        return True
    
    def update_monetization_strategy(self, channel_id: str, strategy: Dict[str, Any]) -> bool:
        """
        Actualiza la estrategia de monetización de un canal.
        
        Args:
            channel_id: ID del canal
            strategy: Nueva estrategia de monetización
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de actualizar estrategia de monetización de canal inexistente: {channel_id}")
            return False
        
        # Actualizar estrategia
        channel.monetization_strategy.update(strategy)
        channel.updated_at = datetime.datetime.now().isoformat()
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Estrategia de monetización actualizada para canal {channel_id}")
        return True
    
    def record_publication(self, channel_id: str, publication_data: Dict[str, Any]) -> bool:
        """
        Registra una nueva publicación en un canal.
        
        Args:
            channel_id: ID del canal
            publication_data: Datos de la publicación
            
        Returns:
            bool: True si el registro fue exitoso, False en caso contrario
        """
        channel = self.channels.get(channel_id)
        if not channel:
            logger.warning(f"Intento de registrar publicación en canal inexistente: {channel_id}")
            return False
        
        # Actualizar timestamp de última publicación
        channel.last_published_at = datetime.datetime.now().isoformat()
        
        # Añadir a métricas si no existe
        if "publications" not in channel.metrics:
            channel.metrics["publications"] = []
        
        # Añadir publicación a métricas
        channel.metrics["publications"].append(publication_data)
        
        # Persistir
        self._save_channel_to_persistence(channel)
        
        logger.info(f"Publicación registrada para canal {channel_id}: {publication_data.get('title', 'Sin título')}")
        return True
    
    def get_channels_by_platform(self, platform: ChannelType) -> List[Channel]:
        """
        Obtiene todos los canales de una plataforma específica.
        
        Args:
            platform: Tipo de plataforma
            
        Returns:
            List[Channel]: Lista de canales de la plataforma
        """
        return [channel for channel in self.channels.values() if channel.type == platform]
    
    def get_active_channels(self) -> List[Channel]:
        """
        Obtiene todos los canales activos.
        
        Returns:
            List[Channel]: Lista de canales activos
        """
        return [channel for channel in self.channels.values() if channel.status == ChannelStatus.ACTIVE]
    
    def get_channels_needing_recovery(self) -> List[Channel]:
        """
        Obtiene todos los canales que necesitan recuperación.
        
        Returns:
            List[Channel]: Lista de canales shadowbanned o en recuperación
        """
        return [
            channel for channel in self.channels.values() 
            if channel.status in [ChannelStatus.SHADOWBANNED, ChannelStatus.RECOVERING]
        ]
    
    def export_channels_data(self) -> Dict[str, Any]:
        """
        Exporta todos los datos de canales para respaldo o migración.
        
        Returns:
            Dict[str, Any]: Datos de todos los canales
        """
        return {
            "channels": [channel.to_dict() for channel in self.channels.values()],
            "exported_at": datetime.datetime.now().isoformat(),
            "count": len(self.channels)
        }
    
    def import_channels_data(self, data: Dict[str, Any]) -> int:
        """
        Importa datos de canales desde un respaldo o migración.
        
        Args:
            data: Datos de canales a importar
            
        Returns:
            int: Número de canales importados
        """
        if "channels" not in data:
            logger.error("Datos de importación inválidos: falta clave 'channels'")
            return 0
        
        imported_count = 0
        
        for channel_data in data["channels"]:
            try:
                channel = Channel.from_dict(channel_data)
                
                # Evitar sobrescribir canales existentes
                if channel.id in self.channels:
                    logger.warning(f"Canal {channel.id} ya existe, omitiendo importación")
                    continue
                
                # Guardar canal
                self.channels[channel.id] = channel
                
                # Persistir
                self._save_channel_to_persistence(channel)
                
                imported_count += 1
            
            except Exception as e:
                logger.error(f"Error al importar canal: {str(e)}")
        
        logger.info(f"Importados {imported_count} canales")
        return imported_count