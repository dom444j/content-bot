"""
Clase base para ejecutores de tareas.

Este módulo define la interfaz común que deben implementar todos los
ejecutores de tareas en el sistema de planificación.
"""

import logging
import abc
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
import time

from ..core.task_model import Task, TaskStatus

logger = logging.getLogger('Scheduler.Executor.Base')

class BaseExecutor(abc.ABC):
    """
    Clase base abstracta para todos los ejecutores de tareas.
    
    Define la interfaz común y proporciona funcionalidad básica compartida
    por todos los tipos de ejecutores.
    """
    
    def __init__(self, task_queue, config=None):
        """
        Inicializa el ejecutor base.
        
        Args:
            task_queue: Cola de tareas compartida
            config: Configuración específica del ejecutor
        """
        self.task_queue = task_queue
        self.config = config or {}
        self.running = False
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.active_tasks = {}  # task_id -> info
        self.active_tasks_lock = threading.RLock()
        
        # Registro de handlers para tipos de tareas
        self.task_handlers = {}
        
        logger.debug(f"Inicializado {self.__class__.__name__}")
    
    def register_task_handler(self, task_type: str, handler: Callable[[Task], Any]) -> None:
        """
        Registra un handler para un tipo específico de tarea.
        
        Args:
            task_type: Tipo de tarea
            handler: Función que maneja la tarea
        """
        self.task_handlers[task_type] = handler
        logger.debug(f"Handler registrado para tipo de tarea: {task_type}")
    
    def start(self) -> None:
        """
        Inicia el ejecutor y sus workers.
        """
        if self.running:
            logger.warning(f"{self.__class__.__name__} ya está en ejecución")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Iniciar thread de worker
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.__class__.__name__}Worker",
            daemon=True
        )
        self.worker_thread.start()
        
        logger.info(f"{self.__class__.__name__} iniciado")
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Detiene el ejecutor.
        
        Args:
            wait: Si es True, espera a que todas las tareas terminen
        """
        if not self.running:
            return
        
        logger.info(f"Deteniendo {self.__class__.__name__}...")
        self.running = False
        self.stop_event.set()
        
        if wait and self.worker_thread:
            self.worker_thread.join()
            logger.info(f"{self.__class__.__name__} detenido correctamente")
    
    def _worker_loop(self) -> None:
        """
        Bucle principal del worker que obtiene y ejecuta tareas.
        """
        logger.debug(f"Worker loop iniciado para {self.__class__.__name__}")
        
        while self.running and not self.stop_event.is_set():
            try:
                # Obtener siguiente tarea
                task = self.task_queue.get_next_task(block=True, timeout=1.0)
                
                if task is None:
                    # No hay tareas disponibles o no es tiempo de ejecutar
                    continue
                
                # Verificar si este ejecutor debe manejar esta tarea
                if task.executor_type != self._get_executor_type():
                    # No es para este ejecutor, devolver a la cola
                    logger.debug(f"Tarea {task.task_id} no es para este ejecutor ({task.executor_type} != {self._get_executor_type()})")
                    self.task_queue.add_task(task)
                    continue
                
                # Ejecutar la tarea
                self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Error en worker loop de {self.__class__.__name__}: {str(e)}", exc_info=True)
                time.sleep(1)  # Evitar bucle infinito de errores
    
    @abc.abstractmethod
    def _get_executor_type(self) -> str:
        """
        Obtiene el tipo de ejecutor.
        
        Returns:
            str: Tipo de ejecutor (ej: 'local', 'thread', 'process')
        """
        pass
    
    @abc.abstractmethod
    def _execute_task(self, task: Task) -> None:
        """
        Ejecuta una tarea.
        
        Args:
            task: Tarea a ejecutar
        """
        pass
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea en ejecución.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        with self.active_tasks_lock:
            if task_id not in self.active_tasks:
                return False
            
            # Implementación específica de cancelación
            result = self._cancel_task_impl(task_id)
            
            if result:
                # Eliminar de tareas activas
                del self.active_tasks[task_id]
                logger.info(f"Tarea {task_id} cancelada")
            
            return result
    
    @abc.abstractmethod
    def _cancel_task_impl(self, task_id: str) -> bool:
        """
        Implementación específica de cancelación de tarea.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        pass
    
    def get_active_tasks(self) -> List[str]:
        """
        Obtiene las tareas actualmente en ejecución.
        
        Returns:
            List[str]: Lista de IDs de tareas activas
        """
        with self.active_tasks_lock:
            return list(self.active_tasks.keys())
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de una tarea activa.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Dict o None: Información de estado de la tarea o None si no existe
        """
        with self.active_tasks_lock:
            return self.active_tasks.get(task_id)
    
    def _update_task_status(self, task_id: str, status: TaskStatus, result: Any = None, error: Optional[str] = None) -> None:
        """
        Actualiza el estado de una tarea.
        
        Args:
            task_id: ID de la tarea
            status: Nuevo estado
            result: Resultado (opcional)
            error: Error (opcional)
        """
        # Actualizar en la cola de tareas
        self.task_queue.update_task(
            task_id=task_id,
            status=status,
            result=result,
            error=error
        )
        
        # Actualizar en tareas activas
        with self.active_tasks_lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = status
                if result is not None:
                    self.active_tasks[task_id]['result'] = result
                if error is not None:
                    self.active_tasks[task_id]['error'] = error
                
                # Si es un estado terminal, eliminar de tareas activas
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    del self.active_tasks[task_id]