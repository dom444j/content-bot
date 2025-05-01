"""
Ejecutor basado en procesos para tareas intensivas en CPU.

Este módulo implementa un ejecutor que ejecuta tareas en procesos separados,
ideal para tareas que requieren alto procesamiento de CPU y pueden beneficiarse
de la ejecución en paralelo en múltiples núcleos.
"""

import logging
import threading
import time
import concurrent.futures
import pickle
import traceback
from typing import Dict, Any, Optional, List, Callable, Tuple

from ..core.task_model import Task, TaskStatus
from .base_executor import BaseExecutor

logger = logging.getLogger('Scheduler.Executor.Process')

class ProcessExecutor(BaseExecutor):
    """
    Ejecutor que procesa tareas en procesos separados utilizando un ProcessPoolExecutor.
    
    Ideal para:
    - Tareas CPU-bound (procesamiento de imágenes, video, cálculos intensivos)
    - Operaciones que pueden beneficiarse de múltiples núcleos
    - Tareas que requieren aislamiento del proceso principal
    
    Ventajas:
    - Paralelismo real en múltiples núcleos (bypass del GIL)
    - Aislamiento de memoria y recursos
    - Mayor estabilidad ante fallos (un proceso que falla no afecta a otros)
    
    Limitaciones:
    - Mayor overhead que los hilos
    - Comunicación más costosa entre procesos
    - Limitaciones en la serialización de objetos complejos
    """
    
    def __init__(self, task_queue, config=None):
        """
        Inicializa el ejecutor de procesos.
        
        Args:
            task_queue: Cola de tareas compartida
            config: Configuración específica del ejecutor
        """
        super().__init__(task_queue, config)
        
        # Obtener número máximo de workers de la configuración
        self.max_workers = self.config.get('max_workers', 5)
        
        # Inicializar ProcessPoolExecutor
        self.process_pool = None
        self.futures = {}  # task_id -> future
        self.futures_lock = threading.RLock()
        
        logger.info(f"ProcessExecutor inicializado con {self.max_workers} workers")
    
    def start(self):
        """
        Inicia el ejecutor y su pool de procesos.
        """
        if self.running:
            logger.warning("ProcessExecutor ya está en ejecución")
            return
        
        # Inicializar ProcessPoolExecutor
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )
        
        # Iniciar el ejecutor base
        super().start()
        
        logger.info(f"ProcessExecutor iniciado con {self.max_workers} workers")
    
    def shutdown(self, wait=True):
        """
        Detiene el ejecutor y su pool de procesos.
        
        Args:
            wait: Si es True, espera a que todas las tareas terminen
        """
        if not self.running:
            return
        
        logger.info("Deteniendo ProcessExecutor...")
        
        # Detener el ejecutor base
        super().shutdown(wait=False)  # No esperar aquí, lo haremos después
        
        # Detener ProcessPoolExecutor
        if self.process_pool:
            self.process_pool.shutdown(wait=wait)
            self.process_pool = None
        
        # Limpiar futures
        with self.futures_lock:
            self.futures.clear()
        
        logger.info("ProcessExecutor detenido correctamente")
    
    def _get_executor_type(self) -> str:
        """
        Obtiene el tipo de ejecutor.
        
        Returns:
            str: 'process'
        """
        return 'process'
    
    def register_task_handler(self, task_type: str, handler: Callable[[Task], Any]) -> None:
        """
        Registra un handler para un tipo específico de tarea.
        
        Nota: Para el ProcessExecutor, los handlers deben ser funciones
        globales o métodos estáticos que puedan ser serializados con pickle.
        
        Args:
            task_type: Tipo de tarea
            handler: Función que maneja la tarea
        """
        # Verificar que el handler sea serializable
        try:
            pickle.dumps(handler)
        except (TypeError, AttributeError) as e:
            logger.error(f"El handler para {task_type} no es serializable: {str(e)}")
            logger.error("Los handlers para ProcessExecutor deben ser funciones globales o métodos estáticos")
            raise ValueError(f"Handler no serializable para ProcessExecutor: {str(e)}")
        
        super().register_task_handler(task_type, handler)
    
    def _execute_task(self, task: Task) -> None:
        """
        Ejecuta una tarea en un proceso separado.
        
        Args:
            task: Tarea a ejecutar
        """
        logger.debug(f"Programando tarea {task.task_id} ({task.task_type}) para ejecución en proceso")
        
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
        
        # Verificar que la tarea sea serializable
        try:
            pickle.dumps(task)
        except (TypeError, AttributeError) as e:
            error_msg = f"La tarea {task.task_id} no es serializable: {str(e)}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
            return
        
        # Enviar tarea al ProcessPoolExecutor
        try:
            # Crear future para la tarea
            future = self.process_pool.submit(process_task_wrapper, task, handler)
            
            # Registrar future
            with self.futures_lock:
                self.futures[task.task_id] = future
            
            # Añadir callback para cuando termine
            future.add_done_callback(
                lambda f, task_id=task.task_id: self._task_completed_callback(task_id, f)
            )
            
            # Actualizar estado a RUNNING
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            with self.active_tasks_lock:
                if task.task_id in self.active_tasks:
                    self.active_tasks[task.task_id]['status'] = TaskStatus.RUNNING
                    self.active_tasks[task.task_id]['start_time'] = task.started_at
            
            self._update_task_status(task.task_id, TaskStatus.RUNNING)
            
            logger.debug(f"Tarea {task.task_id} enviada al ProcessPoolExecutor")
            
        except Exception as e:
            error_msg = f"Error al programar tarea {task.task_id} en ProcessPoolExecutor: {str(e)}"
            logger.error(error_msg)
            self._update_task_status(task.task_id, TaskStatus.FAILED, error=error_msg)
    
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
            
            # El resultado puede ser una tupla (success, result/error)
            if isinstance(result, tuple) and len(result) == 2:
                success, data = result
                
                if success:
                    logger.info(f"Tarea {task_id} completada exitosamente")
                    self._update_task_status(task_id, TaskStatus.COMPLETED, result=data)
                else:
                    logger.error(f"Tarea {task_id} falló: {data}")
                    self._update_task_status(task_id, TaskStatus.FAILED, error=data)
            else:
                # Resultado directo
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
            # Nota: La cancelación de procesos puede no ser inmediata
            cancelled = future.cancel()
            
            if cancelled:
                logger.info(f"Tarea {task_id} cancelada exitosamente")
                # La actualización de estado se hará en el callback
            else:
                logger.warning(f"No se pudo cancelar la tarea {task_id}, posiblemente ya está en ejecución")
            
            return cancelled


# Función global para ejecutar en procesos separados
def process_task_wrapper(task: Task, handler: Callable) -> Tuple[bool, Any]:
    """
    Wrapper para ejecutar la tarea en un proceso separado.
    
    Esta función debe ser global para poder ser serializada con pickle.
    
    Args:
        task: Tarea a ejecutar
        handler: Función handler para la tarea
        
    Returns:
        Tuple[bool, Any]: (éxito, resultado/error)
    """
    try:
        # Configurar logging en el proceso hijo
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(f'ProcessTask-{task.task_id}')
        
        logger.info(f"Iniciando ejecución de tarea {task.task_id} ({task.task_type}) en proceso separado")
        
        # Ejecutar handler
        result = handler(task)
        
        logger.info(f"Tarea {task.task_id} completada exitosamente en proceso separado")
        
        return (True, result)
        
    except Exception as e:
        error_msg = f"Error en ejecución de tarea {task.task_id} en proceso separado: {str(e)}"
        error_trace = traceback.format_exc()
        full_error = f"{error_msg}\n{error_trace}"
        
        # Intentar loggear el error
        try:
            logger.error(full_error)
        except:
            pass
        
        return (False, full_error)