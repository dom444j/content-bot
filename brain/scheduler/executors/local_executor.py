"""
Ejecutor local para tareas síncronas.

Este módulo implementa un ejecutor que ejecuta tareas de forma síncrona
en el mismo hilo, ideal para tareas simples y rápidas.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
import traceback

from ..core.task_model import Task, TaskStatus
from .base_executor import BaseExecutor

logger = logging.getLogger('Scheduler.Executor.Local')

class LocalExecutor(BaseExecutor):
    """
    Ejecutor que procesa tareas de forma síncrona en el mismo hilo.
    
    Ideal para:
    - Tareas simples y rápidas
    - Operaciones que no bloquean (no I/O bound ni CPU bound)
    - Pruebas y depuración
    
    Limitaciones:
    - No paralelismo (una tarea a la vez)
    - Bloquea el hilo principal durante la ejecución
    - No adecuado para tareas largas o que puedan fallar
    """
    
    def __init__(self, task_queue, config=None):
        """
        Inicializa el ejecutor local.
        
        Args:
            task_queue: Cola de tareas compartida
            config: Configuración específica del ejecutor
        """
        super().__init__(task_queue, config)
        logger.info("LocalExecutor inicializado")
    
    def _get_executor_type(self) -> str:
        """
        Obtiene el tipo de ejecutor.
        
        Returns:
            str: 'local'
        """
        return 'local'
    
    def _execute_task(self, task: Task) -> None:
        """
        Ejecuta una tarea de forma síncrona.
        
        Args:
            task: Tarea a ejecutar
        """
        logger.debug(f"Ejecutando tarea {task.task_id} ({task.task_type}) localmente")
        
        # Actualizar estado y registrar en tareas activas
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        with self.active_tasks_lock:
            self.active_tasks[task.task_id] = {
                'task': task,
                'status': TaskStatus.RUNNING,
                'start_time': task.started_at
            }
        
        # Buscar handler para el tipo de tarea
        handler = self.task_handlers.get(task.task_type)
        
        if handler is None:
            error_msg = f"No hay handler registrado para el tipo de tarea: {task.task_type}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Ejecutar la tarea
        try:
            result = handler(task)
            
            # Actualizar estado a completado
            task.completed_at = time.time()
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            self._update_task_status(task.task_id, TaskStatus.COMPLETED, result=result)
            
            logger.info(f"Tarea {task.task_id} completada exitosamente")
            
        except Exception as e:
            # Capturar y registrar error
            error_msg = f"Error al ejecutar tarea {task.task_id}: {str(e)}"
            error_trace = traceback.format_exc()
            
            logger.error(f"{error_msg}\n{error_trace}")
            
            # Actualizar estado a fallido
            task.completed_at = time.time()
            task.error = error_msg
            task.status = TaskStatus.FAILED
            
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
    
    def _cancel_task_impl(self, task_id: str) -> bool:
        """
        Implementación de cancelación de tarea.
        
        Para el ejecutor local, no se puede cancelar una tarea en ejecución
        ya que se ejecuta de forma síncrona.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: False (no se puede cancelar)
        """
        logger.warning(f"No se puede cancelar la tarea {task_id} en el ejecutor local")
        return False