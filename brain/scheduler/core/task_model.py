"""
Modelos de datos para el sistema de planificación de tareas.

Este módulo define las clases y enumeraciones necesarias para representar
tareas, sus estados, prioridades y metadatos asociados.
"""

import datetime
import uuid
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Union
import json

class TaskStatus(Enum):
    """Posibles estados de una tarea."""
    PENDING = auto()      # Esperando ser ejecutada
    RUNNING = auto()      # En ejecución
    COMPLETED = auto()    # Completada exitosamente
    FAILED = auto()       # Falló en su ejecución
    CANCELLED = auto()    # Cancelada antes de ejecutarse
    RETRYING = auto()     # Fallida y esperando reintento
    TIMEOUT = auto()      # Excedió el tiempo máximo de ejecución

class TaskPriority(Enum):
    """Niveles de prioridad para las tareas."""
    CRITICAL = 0    # Máxima prioridad, ejecución inmediata
    HIGH = 1        # Alta prioridad
    NORMAL = 2      # Prioridad normal (por defecto)
    LOW = 3         # Baja prioridad
    BACKGROUND = 4  # Mínima prioridad, ejecutar cuando haya recursos disponibles

class Task:
    """
    Representa una tarea programada en el sistema.
    
    Attributes:
        task_id: Identificador único de la tarea
        task_type: Tipo de tarea (ej: 'content_creation', 'publish')
        task_data: Datos necesarios para la ejecución
        status: Estado actual de la tarea
        priority: Prioridad de la tarea
        execution_time: Momento programado para la ejecución
        created_at: Momento de creación de la tarea
        started_at: Momento en que comenzó la ejecución
        completed_at: Momento en que finalizó la ejecución
        executor_type: Tipo de ejecutor a utilizar
        retry_count: Número de reintentos realizados
        retry_policy: Política de reintentos
        timeout: Tiempo máximo de ejecución en segundos
        result: Resultado de la ejecución
        error: Información de error si falló
        recurring: Si es una tarea recurrente
        schedule_pattern: Patrón de programación para tareas recurrentes
        parent_id: ID de la tarea padre (para tareas dependientes)
        dependencies: IDs de tareas de las que depende
        metadata: Metadatos adicionales
    """
    
    def __init__(self,
                task_type: str,
                task_data: Dict[str, Any],
                task_id: Optional[str] = None,
                status: TaskStatus = TaskStatus.PENDING,
                priority: TaskPriority = TaskPriority.NORMAL,
                execution_time: Optional[datetime.datetime] = None,
                executor_type: str = 'thread',
                retry_policy: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None,
                recurring: bool = False,
                schedule_pattern: Optional[str] = None,
                parent_id: Optional[str] = None,
                dependencies: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Inicializa una nueva tarea.
        
        Args:
            task_type: Tipo de tarea
            task_data: Datos de la tarea
            task_id: ID único (generado automáticamente si es None)
            status: Estado inicial
            priority: Prioridad
            execution_time: Momento de ejecución
            executor_type: Tipo de ejecutor
            retry_policy: Política de reintentos
            timeout: Tiempo máximo de ejecución
            recurring: Si es recurrente
            schedule_pattern: Patrón de programación
            parent_id: ID de tarea padre
            dependencies: IDs de tareas dependientes
            metadata: Metadatos adicionales
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.task_data = task_data
        self.status = status
        self.priority = priority
        self.execution_time = execution_time or datetime.datetime.now()
        self.created_at = datetime.datetime.now()
        self.started_at = None
        self.completed_at = None
        self.executor_type = executor_type
        self.retry_count = 0
        self.retry_policy = retry_policy or {}
        self.timeout = timeout or 300  # 5 minutos por defecto
        self.result = None
        self.error = None
        self.recurring = recurring
        self.schedule_pattern = schedule_pattern
        self.parent_id = parent_id
        self.dependencies = dependencies or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la tarea a un diccionario serializable.
        
        Returns:
            Dict[str, Any]: Representación de la tarea como diccionario
        """
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'task_data': self.task_data,
            'status': self.status.name,
            'priority': self.priority.name,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'executor_type': self.executor_type,
            'retry_count': self.retry_count,
            'retry_policy': self.retry_policy,
            'timeout': self.timeout,
            'result': str(self.result) if self.result is not None else None,
            'error': str(self.error) if self.error is not None else None,
            'recurring': self.recurring,
            'schedule_pattern': self.schedule_pattern,
            'parent_id': self.parent_id,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Crea una tarea a partir de un diccionario.
        
        Args:
            data: Diccionario con los datos de la tarea
            
        Returns:
            Task: Objeto Task creado
        """
        # Convertir strings a objetos datetime
        execution_time = None
        if data.get('execution_time'):
            execution_time = datetime.datetime.fromisoformat(data['execution_time'])
        
        # Convertir strings a enumeraciones
        status = TaskStatus[data.get('status', 'PENDING')]
        priority = TaskPriority[data.get('priority', 'NORMAL')]
        
        # Crear la tarea
        task = cls(
            task_type=data['task_type'],
            task_data=data['task_data'],
            task_id=data['task_id'],
            status=status,
            priority=priority,
            execution_time=execution_time,
            executor_type=data.get('executor_type', 'thread'),
            retry_policy=data.get('retry_policy'),
            timeout=data.get('timeout'),
            recurring=data.get('recurring', False),
            schedule_pattern=data.get('schedule_pattern'),
            parent_id=data.get('parent_id'),
            dependencies=data.get('dependencies', []),
            metadata=data.get('metadata', {})
        )
        
        # Establecer campos adicionales
        if data.get('created_at'):
            task.created_at = datetime.datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            task.started_at = datetime.datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            task.completed_at = datetime.datetime.fromisoformat(data['completed_at'])
        
        task.retry_count = data.get('retry_count', 0)
        task.result = data.get('result')
        task.error = data.get('error')
        
        return task
    
    def __str__(self) -> str:
        """
        Representación en string de la tarea.
        
        Returns:
            str: Representación legible de la tarea
        """
        return (f"Task(id={self.task_id}, type={self.task_type}, "
                f"status={self.status.name}, priority={self.priority.name})")
    
    def __repr__(self) -> str:
        """
        Representación oficial de la tarea.
        
        Returns:
            str: Representación oficial para debugging
        """
        return self.__str__()