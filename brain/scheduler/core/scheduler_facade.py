"""
Fachada principal para el sistema de planificación de tareas.

Este módulo implementa el patrón Facade para proporcionar una interfaz
simplificada al sistema de planificación, ocultando la complejidad interna
y facilitando su uso desde otros componentes del sistema.
"""

import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import uuid

from .task_queue import TaskQueue
from .task_model import Task, TaskStatus, TaskPriority
from .config import SchedulerConfig

# Importaciones de ejecutores (se implementarán después)
from ..executors.local_executor import LocalExecutor
from ..executors.thread_executor import ThreadExecutor
from ..executors.process_executor import ProcessExecutor
from ..executors.distributed_executor import DistributedExecutor
from ..executors.cron_executor import CronExecutor

# Importaciones de middleware (se implementarán después)
from ..middleware.audit_middleware import AuditMiddleware
from ..middleware.throttling_middleware import ThrottlingMiddleware
from ..middleware.validation_middleware import ValidationMiddleware

# Importaciones de estrategias (se implementarán después)
from ..strategies.retry_strategy import RetryStrategy
from ..strategies.circuit_breaker import CircuitBreaker

logger = logging.getLogger('Scheduler.Facade')

class SchedulerFacade:
    """
    Fachada que proporciona una interfaz unificada para todas las operaciones
    del sistema de planificación de tareas.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la fachada del planificador con la configuración especificada.
        
        Args:
            config: Configuración personalizada (opcional)
        """
        # Inicializar configuración
        self.config = SchedulerConfig(config)
        
        # Inicializar cola de tareas
        self.task_queue = TaskQueue()
        
        # Inicializar ejecutores
        self.executors = {
            'local': LocalExecutor(self.task_queue),
            'thread': ThreadExecutor(self.task_queue),
            'process': ProcessExecutor(self.task_queue),
            'distributed': DistributedExecutor(self.task_queue, self.config),
            'cron': CronExecutor(self.task_queue)
        }
        
        # Inicializar middleware
        self.middleware = [
            ValidationMiddleware(),
            AuditMiddleware(),
            ThrottlingMiddleware(self.config.get('throttling', {}))
        ]
        
        # Inicializar estrategias
        self.retry_strategy = RetryStrategy(self.config.get('retry', {}))
        self.circuit_breaker = CircuitBreaker(self.config.get('circuit_breaker', {}))
        
        # Iniciar ejecutores según configuración
        self._start_executors()
        
        logger.info("SchedulerFacade inicializada")
    
    def _start_executors(self) -> None:
        """
        Inicia los ejecutores según la configuración.
        """
        for executor_type, enabled in self.config.get('executors', {}).items():
            if enabled and executor_type in self.executors:
                self.executors[executor_type].start()
                logger.info(f"Ejecutor {executor_type} iniciado")
    
    def schedule_task(self, 
                     task_type: str, 
                     task_data: Dict[str, Any], 
                     execution_time: Optional[datetime.datetime] = None,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     executor_type: str = 'thread',
                     retry_policy: Optional[Dict[str, Any]] = None,
                     timeout: Optional[float] = None,
                     task_id: Optional[str] = None) -> str:
        """
        Programa una tarea para su ejecución.
        
        Args:
            task_type: Tipo de tarea (ej: 'content_creation', 'publish', etc.)
            task_data: Datos necesarios para la ejecución de la tarea
            execution_time: Momento de ejecución (si es None, se ejecuta inmediatamente)
            priority: Prioridad de la tarea
            executor_type: Tipo de ejecutor a utilizar
            retry_policy: Política de reintentos personalizada
            timeout: Tiempo máximo de ejecución en segundos
            task_id: ID personalizado para la tarea (opcional)
            
        Returns:
            str: ID de la tarea programada
        """
        # Crear objeto Task
        task = Task(
            task_id=task_id or str(uuid.uuid4()),
            task_type=task_type,
            task_data=task_data,
            execution_time=execution_time or datetime.datetime.now(),
            priority=priority,
            executor_type=executor_type,
            retry_policy=retry_policy or self.config.get('default_retry_policy', {}),
            timeout=timeout or self.config.get('default_timeout', 300)
        )
        
        # Aplicar middleware de validación
        for middleware in self.middleware:
            if hasattr(middleware, 'validate'):
                middleware.validate(task)
        
        # Añadir a la cola
        task_id = self.task_queue.add_task(task)
        
        logger.info(f"Tarea programada: {task_id} - Tipo: {task_type} - Ejecutor: {executor_type}")
        
        return task_id
    
    def schedule_recurring_task(self,
                               task_type: str,
                               task_data: Dict[str, Any],
                               schedule_pattern: str,
                               priority: TaskPriority = TaskPriority.NORMAL,
                               executor_type: str = 'thread',
                               retry_policy: Optional[Dict[str, Any]] = None,
                               timeout: Optional[float] = None,
                               task_id: Optional[str] = None) -> str:
        """
        Programa una tarea recurrente.
        
        Args:
            task_type: Tipo de tarea
            task_data: Datos de la tarea
            schedule_pattern: Patrón de programación (formato cron o expresión de schedule)
            priority: Prioridad de la tarea
            executor_type: Tipo de ejecutor
            retry_policy: Política de reintentos
            timeout: Tiempo máximo de ejecución
            task_id: ID personalizado
            
        Returns:
            str: ID de la tarea recurrente
        """
        # Verificar que el ejecutor cron esté disponible
        if not self.executors.get('cron'):
            raise ValueError("El ejecutor cron no está disponible")
        
        # Crear tarea recurrente
        task = Task(
            task_id=task_id or str(uuid.uuid4()),
            task_type=task_type,
            task_data=task_data,
            priority=priority,
            executor_type='cron',  # Forzar uso del ejecutor cron
            retry_policy=retry_policy or self.config.get('default_retry_policy', {}),
            timeout=timeout or self.config.get('default_timeout', 300),
            recurring=True,
            schedule_pattern=schedule_pattern
        )
        
        # Aplicar middleware de validación
        for middleware in self.middleware:
            if hasattr(middleware, 'validate'):
                middleware.validate(task)
        
        # Registrar en el ejecutor cron
        self.executors['cron'].register_recurring_task(task)
        
        logger.info(f"Tarea recurrente programada: {task.task_id} - Patrón: {schedule_pattern}")
        
        return task.task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea programada.
        
        Args:
            task_id: ID de la tarea a cancelar
            
        Returns:
            bool: True si la cancelación fue exitosa, False en caso contrario
        """
        task = self.task_queue.get_task(task_id)
        
        if not task:
            logger.warning(f"Intento de cancelar tarea inexistente: {task_id}")
            return False
        
        # Si es recurrente, cancelar en el ejecutor cron
        if task.recurring and task.executor_type == 'cron':
            self.executors['cron'].cancel_recurring_task(task_id)
        
        # Cancelar en la cola
        result = self.task_queue.cancel_task(task_id)
        
        if result:
            logger.info(f"Tarea cancelada: {task_id}")
        
        return result
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Obtiene el estado actual de una tarea.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            TaskStatus o None: Estado de la tarea o None si no existe
        """
        task = self.task_queue.get_task(task_id)
        return task.status if task else None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Obtiene el resultado de una tarea completada.
        
        Args:
            task_id: ID de la tarea
            
        Returns:
            Any o None: Resultado de la tarea o None si no está disponible
        """
        task = self.task_queue.get_task(task_id)
        return task.result if task and task.status == TaskStatus.COMPLETED else None
    
    def get_all_tasks(self) -> List[Task]:
        """
        Obtiene todas las tareas en el sistema.
        
        Returns:
            List[Task]: Lista de todas las tareas
        """
        return self.task_queue.get_all_tasks()
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Obtiene todas las tareas con un estado específico.
        
        Args:
            status: Estado de las tareas a obtener
            
        Returns:
            List[Task]: Lista de tareas con el estado especificado
        """
        return self.task_queue.get_tasks_by_status(status)
    
    def get_tasks_by_type(self, task_type: str) -> List[Task]:
        """
        Obtiene todas las tareas de un tipo específico.
        
        Args:
            task_type: Tipo de las tareas a obtener
            
        Returns:
            List[Task]: Lista de tareas del tipo especificado
        """
        return self.task_queue.get_tasks_by_type(task_type)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Detiene el planificador y todos sus ejecutores.
        
        Args:
            wait: Si es True, espera a que todas las tareas en ejecución terminen
        """
        for executor_name, executor in self.executors.items():
            if hasattr(executor, 'shutdown'):
                logger.info(f"Deteniendo ejecutor {executor_name}...")
                executor.shutdown(wait=wait)
        
        logger.info("SchedulerFacade detenida")