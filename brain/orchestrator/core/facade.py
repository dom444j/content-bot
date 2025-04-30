"""
Facade - Patrón de diseño Facade para el Orchestrator

Este módulo implementa el patrón de diseño Facade para proporcionar una interfaz
simplificada al sistema complejo del Orchestrator. Actúa como punto de entrada
único para los clientes externos, delegando las operaciones a los componentes
especializados internos.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Configuración de logging
logger = logging.getLogger(__name__)

class OrchestratorFacade:
    """
    Implementación del patrón Facade para el Orchestrator.
    
    Esta clase proporciona una interfaz simplificada para interactuar con
    el sistema complejo del Orchestrator, ocultando los detalles de implementación
    y delegando las operaciones a los componentes especializados.
    """
    
    def __init__(self, orchestrator):
        """
        Inicializa la fachada con una referencia al Orchestrator.
        
        Args:
            orchestrator: Instancia del Orchestrator principal
        """
        self.orchestrator = orchestrator
        logger.info("OrchestratorFacade inicializado")
    
    def start_system(self) -> bool:
        """
        Inicia el sistema Orchestrator.
        
        Returns:
            bool: True si el inicio fue exitoso, False en caso contrario
        """
        try:
            self.orchestrator.start()
            return True
        except Exception as e:
            logger.error(f"Error al iniciar el sistema: {str(e)}")
            return False
    
    def stop_system(self) -> bool:
        """
        Detiene el sistema Orchestrator.
        
        Returns:
            bool: True si la detención fue exitosa, False en caso contrario
        """
        try:
            self.orchestrator.stop()
            return True
        except Exception as e:
            logger.error(f"Error al detener el sistema: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Dict[str, Any]: Diccionario con información del estado del sistema
        """
        try:
            return self.orchestrator.get_system_status()
        except Exception as e:
            logger.error(f"Error al obtener estado del sistema: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def create_channel(self, channel_data: Dict[str, Any]) -> Optional[str]:
        """
        Crea un nuevo canal en el sistema.
        
        Args:
            channel_data: Datos del canal a crear
            
        Returns:
            Optional[str]: ID del canal creado o None si hubo un error
        """
        try:
            return self.orchestrator.create_channel(channel_data)
        except Exception as e:
            logger.error(f"Error al crear canal: {str(e)}")
            return None
    
    def update_channel(self, channel_id: str, channel_data: Dict[str, Any]) -> bool:
        """
        Actualiza un canal existente.
        
        Args:
            channel_id: ID del canal a actualizar
            channel_data: Nuevos datos del canal
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        try:
            return self.orchestrator.channel_manager.update_channel(channel_id, channel_data)
        except Exception as e:
            logger.error(f"Error al actualizar canal {channel_id}: {str(e)}")
            return False
    
    def get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un canal específico.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Optional[Dict[str, Any]]: Datos del canal o None si no existe o hubo un error
        """
        try:
            channel = self.orchestrator.channel_manager.get_channel(channel_id)
            return channel.__dict__ if channel else None
        except Exception as e:
            logger.error(f"Error al obtener canal {channel_id}: {str(e)}")
            return None
    
    def list_channels(self, filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Lista los canales del sistema, opcionalmente filtrados.
        
        Args:
            filter_params: Parámetros de filtrado (opcional)
            
        Returns:
            List[Dict[str, Any]]: Lista de canales
        """
        try:
            channels = self.orchestrator.channel_manager.get_all_channels()
            
            if filter_params:
                # Aplicar filtros si se proporcionan
                filtered_channels = []
                for channel in channels:
                    match = True
                    for key, value in filter_params.items():
                        if hasattr(channel, key) and getattr(channel, key) != value:
                            match = False
                            break
                    if match:
                        filtered_channels.append(channel.__dict__)
                return filtered_channels
            
            return [channel.__dict__ for channel in channels]
        except Exception as e:
            logger.error(f"Error al listar canales: {str(e)}")
            return []
    
    def run_pipeline(self, channel_id: str) -> Optional[str]:
        """
        Ejecuta el pipeline completo para un canal específico.
        
        Args:
            channel_id: ID del canal para ejecutar el pipeline
            
        Returns:
            Optional[str]: ID de la tarea creada o None si hubo un error
        """
        try:
            return self.orchestrator.run_pipeline(channel_id)
        except Exception as e:
            logger.error(f"Error al ejecutar pipeline para canal {channel_id}: {str(e)}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de una tarea específica.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Optional[Dict[str, Any]]: Estado de la tarea o None si no existe o hubo un error
        """
        try:
            task = self.orchestrator.task_manager.get_task(task_id)
            return task.__dict__ if task else None
        except Exception as e:
            logger.error(f"Error al obtener estado de tarea {task_id}: {str(e)}")
            return None
    
    def list_tasks(self, filter_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Lista las tareas del sistema, opcionalmente filtradas.
        
        Args:
            filter_params: Parámetros de filtrado (opcional)
            
        Returns:
            List[Dict[str, Any]]: Lista de tareas
        """
        try:
            tasks = self.orchestrator.task_manager.get_all_tasks()
            
            if filter_params:
                # Aplicar filtros si se proporcionan
                filtered_tasks = []
                for task in tasks:
                    match = True
                    for key, value in filter_params.items():
                        if hasattr(task, key) and getattr(task, key) != value:
                            match = False
                            break
                    if match:
                        filtered_tasks.append(task.__dict__)
                return filtered_tasks
            
            return [task.__dict__ for task in tasks]
        except Exception as e:
            logger.error(f"Error al listar tareas: {str(e)}")
            return []
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea en ejecución.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si la cancelación fue exitosa, False en caso contrario
        """
        try:
            return self.orchestrator.task_manager.cancel_task(task_id)
        except Exception as e:
            logger.error(f"Error al cancelar tarea {task_id}: {str(e)}")
            return False
    
    def get_metrics(self, metric_names: Optional[List[str]] = None, 
                   time_range: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Obtiene métricas del sistema, opcionalmente filtradas por nombre y rango de tiempo.
        
        Args:
            metric_names: Lista de nombres de métricas a obtener (opcional)
            time_range: Rango de tiempo para filtrar métricas (opcional)
            
        Returns:
            Dict[str, Any]: Diccionario con métricas
        """
        try:
            return self.orchestrator.metrics_collector.get_metrics(
                metric_names=metric_names,
                time_range=time_range
            )
        except Exception as e:
            logger.error(f"Error al obtener métricas: {str(e)}")
            return {"error": str(e)}