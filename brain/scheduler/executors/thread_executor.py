"""
Ejecutor basado en hilos para tareas asíncronas.

Este módulo implementa un ejecutor que ejecuta tareas en hilos separados,
ideal para tareas que involucran operaciones de I/O (como llamadas a APIs,
operaciones de red o acceso a archivos).
"""

import logging
import threading
import queue
import time
import concurrent.futures
from typing import Dict, Any, Optional, List, Callable

from ..core.task_model import Task, TaskStatus
from .base_executor import BaseExecutor

logger = logging.getLogger('Scheduler.Executor.Thread')

class ThreadExecutor(BaseExecutor):
    """
    Ejecutor que procesa tareas en hilos separados utilizando un ThreadPoolExecutor.
    
    Ideal para:
    - Tareas I/O bound (llamadas a APIs, operaciones de red, acceso a archivos)
    - Operaciones que requieren espera pero no consumen mucha CPU
    - Tareas que pueden ejecutarse en paralelo sin bloquear el hilo principal
    
    Ventajas:
    - Paralelismo para operaciones de I/O
    - Bajo overhead comparado con procesos
    - Comparte memoria con el proceso principal
    
    Limitaciones:
    - No aprovecha múltiples núcleos para tareas CPU-bound (GIL de Python)
    - Posibles problemas de concurrencia si las tareas comparten estado
    """
    
    def __init__(self, task_queue, config=None):
        """
        Inicializa el ejecutor de hilos.
        
        Args:
            task_queue: Cola de tareas compartida
            config: Configuración específica del ejecutor
        """
        super().__init__(task_queue, config)
        
        # Obtener número máximo de workers de la configuración
        self.max_workers = self.config.get('max_workers', 10)
        
        # Inicializar ThreadPoolExecutor
        self.thread_pool = None
        self.futures = {}  # task_id -> future
        self.futures_lock = threading.RLock()
        
        logger.info(f"ThreadExecutor inicializado con {self.max_workers} workers")
    
    def start(self):
        """
        Inicia el ejecutor y su pool de hilos.
        """
        if self.running:
            logger.warning("ThreadExecutor ya está en ejecución")
            return
        
        # Inicializar ThreadPoolExecutor
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="TaskThread"
        )
        
        # Iniciar el ejecutor base
        super().start()
        
        logger.info(f"ThreadExecutor iniciado con {self.max_workers} workers")
    
    def shutdown(self, wait=True):
        """
        Detiene el ejecutor y su pool de hilos.
        
        Args:
            wait: Si es True, espera a que todas las tareas terminen
        """
        if not self.running:
            return
        
        logger.info("Deteniendo ThreadExecutor...")
        
        # Detener el ejecutor base
        super().shutdown(wait=False)  # No esperar aquí, lo haremos después
        
        # Detener ThreadPoolExecutor
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)
            self.thread_pool = None
        
        # Limpiar futures
        with self.futures_lock:
            self.futures.clear()
        
        logger.info("ThreadExecutor detenido correctamente")
    
    def _get_executor_type(self) -> str:
        """
        Obtiene el tipo de ejecutor.
        
        Returns:
            str: 'thread'
        """
        return 'thread'
    
    def _execute_task(self, task: Task) -> None:
        """
        Ejecuta una tarea en un hilo separado.
        
        Args:
            task: Tarea a ejecutar
        """
        logger.debug(f"Programando tarea {task.task_id} ({task.task_type}) para ejecución en hilo")
        
        # Actualizar estado y registrar en tareas activas
        task.status = TaskStatus.QUEUED
        
        with self.active_tasks_lock:
            self.active_tasks[task.task_id] = {
                'task': task,
                'status': TaskStatus.QUEUED,
                'queued_at': time.time()
            }
        
        # Buscar handler para el tipo de tarea
        handler = self.task_handlers.get(task.task_type)
        
        if handler is None:
            error_msg = f"No hay handler registrado para el tipo de tarea: {task.task_type}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Enviar tarea al ThreadPoolExecutor
        try:
            # Crear future para la tarea
            future = self.thread_pool.submit(self._task_wrapper, task, handler)
            
            # Registrar future
            with self.futures_lock:
                self.futures[task.task_id] = future
            
            # Añadir callback para cuando termine
            future.add_done_callback(
                lambda f, task_id=task.task_id: self._task_completed_callback(task_id, f)
            )
            
            logger.debug(f"Tarea {task.task_id} enviada al ThreadPoolExecutor")
            
        except Exception as e:
            error_msg = f"Error al programar tarea {task.task_id} en ThreadPoolExecutor: {str(e)}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
    
    def _task_wrapper(self, task: Task, handler: Callable) -> Any:
        """
        Wrapper para ejecutar la tarea y capturar excepciones.
        
        Args:
            task: Tarea a ejecutar
            handler: Función handler para la tarea
            
        Returns:
            Any: Resultado de la tarea
        """
        # Actualizar estado a RUNNING
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        with self.active_tasks_lock:
            if task.task_id in self.active_tasks:
                self.active_tasks[task.task_id]['status'] = TaskStatus.RUNNING
                self.active_tasks[task.task_id]['start_time'] = task.started_at
        
        self._update_task_status(task.task_id, TaskStatus.RUNNING)
        
        logger.info(f"Iniciando ejecución de tarea {task.task_id} ({task.task_type}) en hilo")
        
        try:
            # Ejecutar handler
            result = handler(task)
            return result
            
        except Exception as e:
            # Propagar excepción para que sea capturada por el callback
            logger.error(f"Error en ejecución de tarea {task.task_id}: {str(e)}", exc_info=True)
            raise
    
    def _task_completed_callback(self, task_id: str, future: concurrent.futures.Future) -> None:
        """
        Callback invocado cuando una tarea termina.
        
        Args:
            task_id: ID de la tarea
            future: Future de la tarea
        """
        # Eliminar future de la lista
        with self.futures_lock:
            if task_id in self.futures:
                del self.futures[task_id]
        
        # Verificar si la tarea fue cancelada
        if future.cancelled():
            logger.info(f"Tarea {task_id} fue cancelada")
            self._update_task_status(task_id, TaskStatus.CANCELLED)
            return
        
        # Verificar si hubo excepción
        exception = future.exception()
        if exception:
            error_msg = f"Tarea {task_id} falló con error: {str(exception)}"
            logger.error(error_msg)
            self._update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Obtener resultado
        try:
            result = future.result()
            logger.info(f"Tarea {task_id} completada exitosamente")
            self._update_task_status(task_id, TaskStatus.COMPLETED, result=result)
            
        except Exception as e:
            # Esto no debería ocurrir ya que las excepciones son capturadas por future.exception()
            error_msg = f"Error inesperado al obtener resultado de tarea {task_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
    
    def _cancel_task_impl(self, task_id: str) -> bool:
        """
        Implementación específica de cancelación de tarea.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si se canceló correctamente, False en caso contrario
        """
        with self.futures_lock:
            if task_id not in self.futures:
                return False
            
            future = self.futures[task_id]
            
            # Intentar cancelar el future
            cancelled = future.cancel()
            
            if cancelled:
                logger.info(f"Tarea {task_id} cancelada exitosamente")
                # La actualización de estado se hará en el callback
            else:
                logger.warning(f"No se pudo cancelar la tarea {task_id}, posiblemente ya está en ejecución")
            
            return cancelled
    
    def get_active_tasks(self) -> List[str]:
        """
        Obtiene las tareas actualmente en ejecución o en cola.
        
        Returns:
            List[str]: Lista de IDs de tareas activas
        """
        with self.active_tasks_lock, self.futures_lock:
            # Combinar tareas activas y futures
            active_tasks = list(self.active_tasks.keys())
            return active_tasks