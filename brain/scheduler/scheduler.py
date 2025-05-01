"""
Scheduler - Sistema de Planificación de Tareas

Este módulo actúa como punto de entrada principal para el sistema de planificación refactorizado.
Proporciona una interfaz simplificada que redirige a la implementación modular en scheduler/core/scheduler_facade.py.

Características:
- Planificación de creación de contenido
- Programación de publicaciones en horarios óptimos
- Gestión de tareas recurrentes (análisis, engagement)
- Coordinación de múltiples canales y plataformas
- Reintentos automáticos y manejo de errores
- Persistencia de tareas y recuperación ante fallos
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import warnings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'scheduler.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Scheduler')

# Importar la implementación refactorizada
try:
    from brain.scheduler.core.scheduler_facade import SchedulerFacade
    from brain.scheduler.core.task_model import TaskStatus, TaskPriority, TaskType
    from brain.scheduler.core.config import SchedulerConfig
    from brain.scheduler.utils.monitoring import SchedulerMonitor
except ImportError as e:
    logger.error(f"Error al importar módulos refactorizados: {e}")
    logger.warning("Usando implementación heredada como fallback")
    # Importar implementación heredada (código antiguo) si la nueva no está disponible
    from brain.scheduler_legacy import Scheduler as SchedulerFacade
    from brain.scheduler_legacy import TaskStatus, TaskPriority
    from brain.scheduler_legacy import retry_with_backoff
    
    # Definir TaskType si no existe en el código legacy
    from enum import Enum
    class TaskType(Enum):
        CONTENT_CREATION = "content_creation"
        PUBLISH = "publish"
        ANALYZE = "analyze"
        MONETIZE = "monetize"
        MAINTENANCE = "maintenance"
    
    warnings.warn(
        "Usando implementación heredada del Scheduler. Se recomienda migrar a la nueva estructura.",
        DeprecationWarning, stacklevel=2
    )


class Scheduler:
    """
    Clase principal que actúa como fachada para el sistema de planificación refactorizado.
    
    Esta clase proporciona compatibilidad hacia atrás mientras redirige las llamadas
    a la nueva implementación modular.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el planificador con configuración opcional.
        
        Args:
            config: Diccionario de configuración para el planificador
        """
        logger.info("Inicializando sistema de planificación")
        self.config = config or {}
        
        # Inicializar la implementación refactorizada
        try:
            scheduler_config = SchedulerConfig(**self.config)
            self._scheduler = SchedulerFacade(config=scheduler_config)
            self._monitor = SchedulerMonitor(self._scheduler)
            self._is_legacy = False
            logger.info("Sistema de planificación refactorizado inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar sistema refactorizado: {e}")
            logger.warning("Inicializando sistema heredado como fallback")
            # Fallback a la implementación heredada
            self._scheduler = SchedulerFacade(self.config)
            self._is_legacy = True
            warnings.warn(
                "Usando implementación heredada del Scheduler. Se recomienda migrar a la nueva estructura.",
                DeprecationWarning, stacklevel=2
            )
    
    def schedule_task(self, 
                     task_type: str, 
                     task_data: Dict[str, Any], 
                     execution_time: Optional[datetime.datetime] = None,
                     priority: int = None,
                     dependencies: List[str] = None,
                     retry_policy: Dict[str, Any] = None) -> str:
        """
        Programa una nueva tarea para ejecución.
        
        Args:
            task_type: Tipo de tarea (ej. "content_creation", "publish", "analyze")
            task_data: Datos específicos para la tarea
            execution_time: Momento programado para la ejecución (None = inmediato)
            priority: Prioridad de la tarea (valores más altos = mayor prioridad)
            dependencies: Lista de IDs de tareas que deben completarse antes
            retry_policy: Política de reintentos para la tarea
            
        Returns:
            ID único de la tarea programada
        """
        logger.debug(f"Programando tarea de tipo {task_type}")
        
        # Convertir prioridad si es necesario
        if priority is not None and not self._is_legacy:
            if isinstance(priority, int) and not isinstance(priority, TaskPriority):
                # Convertir entero a enum TaskPriority
                priority_map = {
                    1: TaskPriority.LOWEST,
                    2: TaskPriority.LOW,
                    3: TaskPriority.NORMAL,
                    4: TaskPriority.HIGH,
                    5: TaskPriority.HIGHEST
                }
                priority = priority_map.get(priority, TaskPriority.NORMAL)
        
        # Delegar a la implementación refactorizada
        return self._scheduler.schedule_task(
            task_type=task_type,
            task_data=task_data,
            execution_time=execution_time,
            priority=priority,
            dependencies=dependencies,
            retry_policy=retry_policy
        )
    
    def schedule_recurring_task(self,
                               task_type: str,
                               task_data: Dict[str, Any],
                               schedule_pattern: str,
                               priority: int = None,
                               retry_policy: Dict[str, Any] = None) -> str:
        """
        Programa una tarea recurrente según un patrón cron.
        
        Args:
            task_type: Tipo de tarea
            task_data: Datos específicos para la tarea
            schedule_pattern: Patrón cron (ej. "0 9 * * *" = todos los días a las 9am)
            priority: Prioridad de la tarea
            retry_policy: Política de reintentos
            
        Returns:
            ID único de la tarea recurrente
        """
        logger.debug(f"Programando tarea recurrente de tipo {task_type} con patrón {schedule_pattern}")
        
        # Convertir prioridad si es necesario (igual que en schedule_task)
        if priority is not None and not self._is_legacy:
            if isinstance(priority, int) and not isinstance(priority, TaskPriority):
                priority_map = {
                    1: TaskPriority.LOWEST,
                    2: TaskPriority.LOW,
                    3: TaskPriority.NORMAL,
                    4: TaskPriority.HIGH,
                    5: TaskPriority.HIGHEST
                }
                priority = priority_map.get(priority, TaskPriority.NORMAL)
        
        # Delegar a la implementación refactorizada
        return self._scheduler.schedule_recurring_task(
            task_type=task_type,
            task_data=task_data,
            schedule_pattern=schedule_pattern,
            priority=priority,
            retry_policy=retry_policy
        )
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea programada.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            True si la tarea fue cancelada, False si no se encontró
        """
        logger.debug(f"Cancelando tarea {task_id}")
        return self._scheduler.cancel_task(task_id)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de una tarea.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Diccionario con información de estado de la tarea
        """
        return self._scheduler.get_task_status(task_id)
    
    def get_pending_tasks(self, 
                         task_type: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene lista de tareas pendientes, opcionalmente filtradas por tipo.
        
        Args:
            task_type: Filtrar por tipo de tarea (None = todos los tipos)
            limit: Número máximo de tareas a devolver
            
        Returns:
            Lista de diccionarios con información de tareas pendientes
        """
        return self._scheduler.get_pending_tasks(task_type=task_type, limit=limit)
    
    def get_tasks_by_status(self, 
                           status: Union[str, List[str]], 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene tareas filtradas por estado.
        
        Args:
            status: Estado o lista de estados para filtrar
            limit: Número máximo de tareas a devolver
            
        Returns:
            Lista de diccionarios con información de tareas
        """
        # Convertir estado si es necesario
        if not self._is_legacy and isinstance(status, str):
            status_map = {
                "pending": TaskStatus.PENDING,
                "running": TaskStatus.RUNNING,
                "completed": TaskStatus.COMPLETED,
                "failed": TaskStatus.FAILED,
                "cancelled": TaskStatus.CANCELLED
            }
            status = status_map.get(status.lower(), status)
        
        return self._scheduler.get_tasks_by_status(status=status, limit=limit)
    
    def pause(self) -> bool:
        """
        Pausa temporalmente la ejecución de tareas.
        
        Returns:
            True si el planificador fue pausado correctamente
        """
        logger.info("Pausando planificador")
        return self._scheduler.pause()
    
    def resume(self) -> bool:
        """
        Reanuda la ejecución de tareas después de una pausa.
        
        Returns:
            True si el planificador fue reanudado correctamente
        """
        logger.info("Reanudando planificador")
        return self._scheduler.resume()
    
    def shutdown(self, wait: bool = True) -> bool:
        """
        Detiene el planificador y libera recursos.
        
        Args:
            wait: Si es True, espera a que las tareas en ejecución terminen
            
        Returns:
            True si el planificador fue detenido correctamente
        """
        logger.info(f"Deteniendo planificador (wait={wait})")
        return self._scheduler.shutdown(wait=wait)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del planificador.
        
        Returns:
            Diccionario con métricas (tareas completadas, fallidas, tiempo medio, etc.)
        """
        if hasattr(self, '_monitor') and not self._is_legacy:
            return self._monitor.get_metrics()
        return self._scheduler.get_metrics() if hasattr(self._scheduler, 'get_metrics') else {}
    
    def visualize_schedule(self, 
                          output_format: str = 'html', 
                          start_time: Optional[datetime.datetime] = None,
                          end_time: Optional[datetime.datetime] = None,
                          output_file: Optional[str] = None) -> Optional[str]:
        """
        Genera una visualización del calendario de tareas programadas.
        
        Args:
            output_format: Formato de salida ('html', 'png', 'svg')
            start_time: Tiempo de inicio para la visualización
            end_time: Tiempo de fin para la visualización
            output_file: Ruta del archivo de salida (None = retorna como string)
            
        Returns:
            Ruta al archivo generado o contenido como string (según output_file)
        """
        if not self._is_legacy and hasattr(self._scheduler, 'visualize_schedule'):
            return self._scheduler.visualize_schedule(
                output_format=output_format,
                start_time=start_time,
                end_time=end_time,
                output_file=output_file
            )
        logger.warning("La visualización de calendario no está disponible en esta versión")
        return None
    
    def optimize_schedule(self, 
                         optimization_strategy: str = 'balanced',
                         constraints: Dict[str, Any] = None) -> bool:
        """
        Optimiza el calendario de tareas según una estrategia y restricciones.
        
        Args:
            optimization_strategy: Estrategia de optimización ('balanced', 'performance', 'resource')
            constraints: Restricciones adicionales para la optimización
            
        Returns:
            True si la optimización fue exitosa
        """
        if not self._is_legacy and hasattr(self._scheduler, 'optimize_schedule'):
            return self._scheduler.optimize_schedule(
                optimization_strategy=optimization_strategy,
                constraints=constraints or {}
            )
        logger.warning("La optimización de calendario no está disponible en esta versión")
        return False
    
    def register_callback(self, 
                         event_type: str, 
                         callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Registra una función de callback para eventos del planificador.
        
        Args:
            event_type: Tipo de evento ('task_completed', 'task_failed', etc.)
            callback: Función a llamar cuando ocurra el evento
            
        Returns:
            ID del callback registrado
        """
        if not self._is_legacy and hasattr(self._scheduler, 'register_callback'):
            return self._scheduler.register_callback(event_type=event_type, callback=callback)
        logger.warning("El registro de callbacks no está disponible en esta versión")
        return ""
    
    def unregister_callback(self, callback_id: str) -> bool:
        """
        Elimina un callback previamente registrado.
        
        Args:
            callback_id: ID del callback a eliminar
            
        Returns:
            True si el callback fue eliminado correctamente
        """
        if not self._is_legacy and hasattr(self._scheduler, 'unregister_callback'):
            return self._scheduler.unregister_callback(callback_id=callback_id)
        logger.warning("La eliminación de callbacks no está disponible en esta versión")
        return False


# Crear instancia global para uso sencillo
default_scheduler = Scheduler()


# Funciones de conveniencia para uso directo
def schedule_task(task_type: str, 
                 task_data: Dict[str, Any], 
                 execution_time: Optional[datetime.datetime] = None,
                 priority: int = None,
                 dependencies: List[str] = None,
                 retry_policy: Dict[str, Any] = None) -> str:
    """Programa una tarea usando el planificador predeterminado."""
    return default_scheduler.schedule_task(
        task_type=task_type,
        task_data=task_data,
        execution_time=execution_time,
        priority=priority,
        dependencies=dependencies,
        retry_policy=retry_policy
    )


def schedule_recurring_task(task_type: str,
                           task_data: Dict[str, Any],
                           schedule_pattern: str,
                           priority: int = None,
                           retry_policy: Dict[str, Any] = None) -> str:
    """Programa una tarea recurrente usando el planificador predeterminado."""
    return default_scheduler.schedule_recurring_task(
        task_type=task_type,
        task_data=task_data,
        schedule_pattern=schedule_pattern,
        priority=priority,
        retry_policy=retry_policy
    )


def cancel_task(task_id: str) -> bool:
    """Cancela una tarea usando el planificador predeterminado."""
    return default_scheduler.cancel_task(task_id)


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Obtiene el estado de una tarea usando el planificador predeterminado."""
    return default_scheduler.get_task_status(task_id)


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar el planificador con opciones personalizadas
    custom_scheduler = Scheduler({
        "max_workers": 4,
        "persistence_enabled": True,
        "persistence_path": "data/scheduler_state.json",
        "retry_defaults": {
            "max_retries": 3,
            "retry_delay": 5,
            "backoff_factor": 2
        }
    })
    
    # Programar una tarea simple
    task_id = custom_scheduler.schedule_task(
        task_type="content_creation",
        task_data={
            "channel_id": "channel123",
            "content_type": "short_video",
            "theme": "tech_news"
        },
        execution_time=datetime.datetime.now() + datetime.timedelta(minutes=5),
        priority=TaskPriority.HIGH if not custom_scheduler._is_legacy else 4
    )
    
    print(f"Tarea programada con ID: {task_id}")
    
    # Programar una tarea recurrente
    recurring_id = custom_scheduler.schedule_recurring_task(
        task_type="analytics",
        task_data={
            "channel_id": "channel123",
            "metrics": ["views", "engagement", "conversion"]
        },
        schedule_pattern="0 */3 * * *",  # Cada 3 horas
        priority=TaskPriority.NORMAL if not custom_scheduler._is_legacy else 3
    )
    
    print(f"Tarea recurrente programada con ID: {recurring_id}")
    
    # Obtener métricas
    metrics = custom_scheduler.get_metrics()
    print(f"Métricas del planificador: {metrics}")
    
    # Visualizar calendario
    visualization = custom_scheduler.visualize_schedule(
        output_format="html",
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now() + datetime.timedelta(days=1),
        output_file="scheduler_visualization.html"
    )
    
    if visualization:
        print(f"Visualización generada en: {visualization}")