"""
Middleware base para el sistema de planificación.

Este módulo define la interfaz común para todos los middleware
que pueden interceptar y modificar tareas antes y después de su ejecución.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union

from ..core.task_model import Task, TaskStatus

logger = logging.getLogger('Scheduler.Middleware.Base')

class BaseMiddleware(ABC):
    """
    Clase base para todos los middleware del sistema de planificación.
    
    Los middleware permiten interceptar tareas antes y después de su ejecución
    para realizar operaciones como validación, auditoría, control de tasas, etc.
    
    Para crear un nuevo middleware, extiende esta clase e implementa los métodos
    process_task_pre_execution y/o process_task_post_execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el middleware.
        
        Args:
            config: Configuración específica del middleware
        """
        self.config = config or {}
        self.next_middleware = None
        logger.debug(f"Inicializando {self.__class__.__name__}")
    
    def set_next(self, middleware: 'BaseMiddleware') -> 'BaseMiddleware':
        """
        Establece el siguiente middleware en la cadena.
        
        Args:
            middleware: Siguiente middleware
            
        Returns:
            El middleware añadido (para encadenamiento)
        """
        self.next_middleware = middleware
        return middleware
    
    def process_task(self, task: Task, context: Dict[str, Any] = None) -> Optional[Task]:
        """
        Procesa una tarea antes de su ejecución.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            La tarea procesada o None si debe ser rechazada
        """
        # Procesar con este middleware
        processed_task = self.process_task_pre_execution(task, context or {})
        
        # Si la tarea fue rechazada, detener la cadena
        if processed_task is None:
            return None
        
        # Continuar con el siguiente middleware si existe
        if self.next_middleware:
            return self.next_middleware.process_task(processed_task, context)
        
        return processed_task
    
    def process_task_result(self, task: Task, result: Any, error: Optional[str] = None, 
                           context: Dict[str, Any] = None) -> Tuple[Any, Optional[str]]:
        """
        Procesa el resultado de una tarea después de su ejecución.
        
        Args:
            task: Tarea ejecutada
            result: Resultado de la ejecución
            error: Error si la tarea falló
            context: Contexto adicional
            
        Returns:
            Tupla (resultado procesado, error procesado)
        """
        # Procesar con este middleware
        processed_result, processed_error = self.process_task_post_execution(
            task, result, error, context or {}
        )
        
        # Continuar con el siguiente middleware si existe
        if self.next_middleware:
            return self.next_middleware.process_task_result(
                task, processed_result, processed_error, context
            )
        
        return processed_result, processed_error
    
    @abstractmethod
    def process_task_pre_execution(self, task: Task, context: Dict[str, Any]) -> Optional[Task]:
        """
        Procesa una tarea antes de su ejecución.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            La tarea procesada o None si debe ser rechazada
        """
        pass
    
    @abstractmethod
    def process_task_post_execution(self, task: Task, result: Any, error: Optional[str],
                                   context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        Procesa el resultado de una tarea después de su ejecución.
        
        Args:
            task: Tarea ejecutada
            result: Resultado de la ejecución
            error: Error si la tarea falló
            context: Contexto adicional
            
        Returns:
            Tupla (resultado procesado, error procesado)
        """
        pass